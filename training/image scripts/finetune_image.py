#!/usr/bin/env python3
"""
ImageGuard v2 — Phase 2 Fine-Tune on Hard + OOD Datasets
Progressive strategy: 70% original clean data + 30% hard data
Lower LR + partial backbone freeze to prevent catastrophic forgetting
"""

import os, re, json, random, warnings
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# 1. CONFIG — Key differences from Phase 1
# ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    img_size:         int   = 224
    batch_size:       int   = 64
    epochs:           int   = 7          # short — already pre-trained
    patience:         int   = 3          # tighter early stopping
    lr:               float = 5e-5       # 4x lower than Phase 1
    weight_decay:     float = 1e-4
    grad_accum_steps: int   = 2
    max_grad_norm:    float = 1.0
    amp:              bool  = True
    num_workers:      int   = 8
    seed:             int   = 42
    freeze_ratio:     float = 0.45        # freeze bottom 60% of spatial backbone
    hard_data_ratio:  float = 0.30       # 30% hard data in each training batch

cfg = CFG()

BASE_DIR      = Path.home() / 'deepfake_project/image'
ORIG_DATA_DIR = BASE_DIR / 'data'
HARD_DATA_DIR = BASE_DIR / 'test_data'   # re-use downloaded test sets as hard train
CKPT_DIR      = BASE_DIR / 'checkpoints'
LOG_DIR       = BASE_DIR / 'logs'
ORIG_CKPT     = CKPT_DIR / 'imageguard_v2_best.pt'
FINE_CKPT     = CKPT_DIR / 'imageguard_v2_finetuned.pt'

for d in [CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON = cfg.amp and DEVICE.type == 'cuda'

print("=" * 65)
print("  ImageGuard v2 — Phase 2 Fine-Tune")
print("=" * 65)
print(f"Device     : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"LR         : {cfg.lr}  (was 2e-4 in Phase 1)")
print(f"Hard ratio : {cfg.hard_data_ratio*100:.0f}% hard + "
      f"{(1-cfg.hard_data_ratio)*100:.0f}% original")
print(f"Freeze     : bottom {cfg.freeze_ratio*100:.0f}% of spatial backbone")


# ─────────────────────────────────────────────────────────────
# 2. SCANNER
# ─────────────────────────────────────────────────────────────
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
REAL_TOKENS = {'real','original','authentic','pristine','genuine',
               'true','youtube','celeb'}
FAKE_TOKENS = {'fake','deepfake','deepfakes','ai','generated','synthetic',
               'faceswap','face2face','faceshifter','neuraltextures',
               'deepfakedetection','manipulated','altered','synthesis'}
SKIP_ORIG   = {'ff_greatgame','deepfake60k','celebdf'}

# Only use datasets where test AUC was weakest (edit this list after testing)
# Default: all hard datasets included — remove any that scored > 0.97
HARD_DATASETS_TO_USE = {
    'wilddeepfake',
    'openforensics',
    'dfd',
    'dfd_frames_cropped',  # face-cropped DFD frames
    'deepfake_2025',
    'ciplab_faces',
}

def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(norm(part).split())
        if t & REAL_TOKENS: return 0
        if t & FAKE_TOKENS: return 1
    return None

def scan_dir(root, skip_set=None, only_set=None):
    pairs = []
    for droot in sorted(root.iterdir()):
        if not droot.is_dir(): continue
        if skip_set and droot.name in skip_set: continue
        if only_set and droot.name not in only_set: continue
        r, f = 0, 0
        for p in droot.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                label = infer_label(p)
                if label == 0: r += 1; pairs.append((str(p), 0))
                elif label == 1: f += 1; pairs.append((str(p), 1))
        print(f"  {droot.name:<35} real={r:>7,}  fake={f:>7,}")
    return pairs

print("\nScanning original training data (sample)...")
orig_pairs = scan_dir(ORIG_DATA_DIR, skip_set=SKIP_ORIG)
random.shuffle(orig_pairs)

print("\nScanning hard/OOD datasets...")
hard_pairs = scan_dir(HARD_DATA_DIR, only_set=HARD_DATASETS_TO_USE)
random.shuffle(hard_pairs)

print(f"\nOriginal pool : {len(orig_pairs):,}")
print(f"Hard pool     : {len(hard_pairs):,}")

# Build mixed training set: 70% original, 30% hard
n_hard     = len(hard_pairs)
n_orig_use = int(n_hard * (cfg.hard_data_ratio / (1 - cfg.hard_data_ratio)))
n_orig_use = min(n_orig_use, len(orig_pairs))
orig_sample = random.sample(orig_pairs, n_orig_use)
mixed_train  = orig_sample + hard_pairs
random.shuffle(mixed_train)

print(f"\nMixed train   : {len(orig_sample):,} original + "
      f"{len(hard_pairs):,} hard = {len(mixed_train):,} total")


# ─────────────────────────────────────────────────────────────
# 3. VIDEO-SAFE DEV / EVAL SPLIT FROM HARD DATA ONLY
# ─────────────────────────────────────────────────────────────
# WITH THIS — stratified split by label:
real_hard = [(p,l) for p,l in hard_pairs if l==0]
fake_hard = [(p,l) for p,l in hard_pairs if l==1]
random.shuffle(real_hard); random.shuffle(fake_hard)

def stratified_split(pairs, dev_frac=0.12, eval_frac=0.12):
    n     = len(pairs)
    n_dev = max(int(n * dev_frac), 50)
    n_ev  = max(int(n * eval_frac), 50)
    return pairs[n_dev+n_ev:], pairs[:n_dev], pairs[n_dev:n_dev+n_ev]

real_train, real_dev, real_eval = stratified_split(real_hard)
fake_train, fake_dev, fake_eval = stratified_split(fake_hard)

dev_pairs   = real_dev  + fake_dev
eval_pairs  = real_eval + fake_eval
hard_train  = real_train + fake_train
random.shuffle(dev_pairs); random.shuffle(eval_pairs)

# Rebuild mixed_train with corrected hard split
dev_paths  = set(p for p,_ in dev_pairs)
eval_paths = set(p for p,_ in eval_pairs)
mixed_train = [(p,l) for p,l in mixed_train
               if p not in dev_paths and p not in eval_paths]

print(f"Split         : Train {len(mixed_train):,} | "
      f"Dev {len(dev_pairs):,} | Eval {len(eval_pairs):,}")


# ─────────────────────────────────────────────────────────────
# 4. AUGMENTATIONS — Heavier compression sim for hard data
# ─────────────────────────────────────────────────────────────
train_tfms = A.Compose([
    A.Resize(cfg.img_size, cfg.img_size),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.ImageCompression(quality_lower=30, quality_upper=95, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.15, hue=0.05, p=0.4),
    A.ToGray(p=0.05, num_output_channels=3),
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])
eval_tfms = A.Compose([
    A.Resize(cfg.img_size, cfg.img_size),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────
# 5. DATASET
# ─────────────────────────────────────────────────────────────
class ImageGuardDataset(Dataset):
    def __init__(self, pairs, transforms):
        self.pairs = pairs
        self.transforms = transforms
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (self.transforms(image=img)['image'],
                torch.tensor(float(label), dtype=torch.float32))

def build_sampler(pairs):
    lbls = [l for _, l in pairs]
    n0 = max(lbls.count(0), 1); n1 = max(lbls.count(1), 1)
    sw = torch.DoubleTensor([1.0/n0 if l==0 else 1.0/n1 for l in lbls])
    return WeightedRandomSampler(sw, len(sw), replacement=True)

train_loader = DataLoader(ImageGuardDataset(mixed_train, train_tfms),
                          batch_size=cfg.batch_size,
                          sampler=build_sampler(mixed_train),
                          num_workers=cfg.num_workers, pin_memory=True,
                          prefetch_factor=2)
dev_loader   = DataLoader(ImageGuardDataset(dev_pairs, eval_tfms),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)
eval_loader  = DataLoader(ImageGuardDataset(eval_pairs, eval_tfms),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)


# ─────────────────────────────────────────────────────────────
# 6. MODEL
# ─────────────────────────────────────────────────────────────
def rgb_to_fft(x):
    gray = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
    fft  = torch.fft.fft2(gray)
    mag  = torch.log1p(torch.abs(fft))
    mn   = mag.amin(dim=(2,3), keepdim=True)
    mx   = mag.amax(dim=(2,3), keepdim=True)
    return (mag - mn) / (mx - mn + 1e-6)

class ImageGuardV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = timm.create_model('tf_efficientnet_b4_ns',
                        pretrained=False, num_classes=0, global_pool='avg')
        self.freq    = timm.create_model('mobilenetv3_small_050',
                        pretrained=False, in_chans=1,
                        num_classes=0, global_pool='avg')
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size)
            d_s = self.spatial(dummy).shape[1]
            d_f = self.freq(rgb_to_fft(dummy)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(d_s + d_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 128),       nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.head(
            torch.cat([self.spatial(x), self.freq(rgb_to_fft(x))], dim=1)
        ).squeeze(1)

# Load Phase 1 checkpoint
print(f"\nLoading Phase 1 checkpoint: {ORIG_CKPT}")
model = ImageGuardV2().to(DEVICE)
ckpt  = torch.load(ORIG_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Phase 1 Dev AUC was: {ckpt['dev_auc']:.4f}")

# Freeze bottom 60% of spatial backbone
all_params = list(model.spatial.parameters())
n_freeze   = int(len(all_params) * cfg.freeze_ratio)
for i, p in enumerate(all_params):
    p.requires_grad = (i >= n_freeze)
frozen = sum(1 for p in model.parameters() if not p.requires_grad)
total  = sum(1 for p in model.parameters())
print(f"Frozen {frozen}/{total} param groups "
      f"(bottom {cfg.freeze_ratio*100:.0f}% of spatial backbone)")


# ─────────────────────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-7)
criterion = nn.BCEWithLogitsLoss()
scaler    = GradScaler(enabled=AMP_ON)

def eval_model(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ON):
                prob = torch.sigmoid(model(x))
            ps.append(prob.cpu().numpy()); ys.append(y.numpy())
    y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, y_prob)
                     if len(np.unique(y_true)) > 1 else 0.5),
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1':  float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }

best_auc   = 0.0
no_improve = 0
log_path   = LOG_DIR / 'finetune_log.csv'
with open(log_path, 'w') as f:
    f.write('epoch,train_loss,dv_auc,dv_acc,dv_f1\n')

print("\n" + "=" * 65)
print("  Phase 2 Fine-Tuning")
print("=" * 65)

for epoch in range(1, cfg.epochs + 1):
    model.train(); optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for step, (x, y) in enumerate(train_loader, 1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with autocast(device_type=DEVICE.type, enabled=AMP_ON):
            loss = criterion(model(x), y) / cfg.grad_accum_steps
        scaler.scale(loss).backward()
        if step % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item()) * cfg.grad_accum_steps

    if len(train_loader) % cfg.grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()
    avg_loss = running_loss / max(len(train_loader), 1)
    m = eval_model(model, dev_loader)

    with open(log_path, 'a') as f:
        f.write(f"{epoch},{avg_loss:.4f},{m['auc']:.4f},"
                f"{m['acc']:.4f},{m['f1']:.4f}\n")

    flag = ''
    if m['auc'] > best_auc:
        best_auc = m['auc']; no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dev_auc': m['auc'], 'dev_acc': m['acc'],
            'phase': 2,
            'config': {'img_size': cfg.img_size,
                       'hard_ratio': cfg.hard_data_ratio},
        }, FINE_CKPT)
        flag = '  ✓ saved'
    else:
        no_improve += 1

    print(f"Ep {epoch:02d}/{cfg.epochs} | Loss {avg_loss:.4f} | "
          f"AUC {m['auc']:.4f} | Acc {m['acc']:.4f} | "
          f"F1 {m['f1']:.4f}{flag}")

    if no_improve >= cfg.patience:
        print(f"\nEarly stopping at epoch {epoch}."); break

print(f"\nBest Fine-Tune Dev AUC : {best_auc:.4f}")

# Final evaluation
ckpt = torch.load(FINE_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
tm = eval_model(model, eval_loader)

model.eval(); ev_preds, ev_true = [], []
with torch.no_grad():
    for x, y in eval_loader:
        p = (torch.sigmoid(model(x.to(DEVICE)))>=0.5).cpu().long().numpy()
        ev_preds.extend(p); ev_true.extend(y.long().numpy())

print("\n" + "=" * 55)
print("  PHASE 2 FINAL EVAL RESULTS")
print("=" * 55)
print(f"  AUC       : {tm['auc']:.4f}")
print(f"  Accuracy  : {tm['acc']:.4f}")
print(f"  F1        : {tm['f1']:.4f}")
print(f"  Precision : {tm['precision']:.4f}")
print(f"  Recall    : {tm['recall']:.4f}")
print("=" * 55)
print(classification_report(ev_true, ev_preds,
                             target_names=['Real','Fake']))

out = LOG_DIR / 'finetune_test_metrics.json'
with open(out, 'w') as f:
    json.dump(tm, f, indent=2)
print(f"\nFine-tuned checkpoint : {FINE_CKPT}")
print(f"Metrics               : {out}")
