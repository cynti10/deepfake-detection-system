#!/usr/bin/env python3
"""
ImageGuard v2 — EfficientNet-B4-NS + FFT-MobileNetV3 Dual Branch
Best-of-both deepfake detector
Datasets: ff140k, ff_frames, celebdf_frames, deepdetect,
          deepfake_real_200k, ai_faces, hard_fakes
"""

import os, re, json, random, hashlib, warnings
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
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
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    img_size:         int   = 224
    batch_size:       int   = 64       # reduce to 32 if CUDA OOM
    epochs:           int   = 20
    patience:         int   = 5
    lr:               float = 2e-4
    weight_decay:     float = 1e-4
    grad_accum_steps: int   = 2
    max_grad_norm:    float = 1.0
    amp:              bool  = True
    num_workers:      int   = 8
    seed:             int   = 42

cfg = CFG()

BASE_DIR  = Path.home() / 'deepfake_project/image'
DATA_DIR  = BASE_DIR / 'data'
CKPT_DIR  = BASE_DIR / 'checkpoints'
LOG_DIR   = BASE_DIR / 'logs'
CKPT_PATH = CKPT_DIR / 'imageguard_v2_finetuned.pt'

for d in [CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON  = cfg.amp and DEVICE.type == 'cuda'

print("=" * 65)
print("  ImageGuard v2 — EfficientNet-B4-NS + FFT-MobileNetV3")
print("=" * 65)
print(f"Device      : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"AMP         : {AMP_ON}")
print(f"Batch size  : {cfg.batch_size}")
print(f"Image size  : {cfg.img_size}")


# ─────────────────────────────────────────────────────────────
# 2. DATASET SCANNER
# ─────────────────────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

REAL_TOKENS = {
    'real', 'original', 'authentic', 'pristine',
    'genuine', 'true', 'youtube', 'celeb',
}
FAKE_TOKENS = {
    'fake', 'deepfake', 'deepfakes', 'ai', 'generated',
    'synthetic', 'faceswap', 'face2face', 'faceshifter',
    'neuraltextures', 'deepfakedetection', 'manipulated',
    'altered', 'synthesis',
}

SKIP_DATASETS = {'ff_greatgame', 'deepfake60k', 'celebdf'}


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(norm(part).split())
        if t & REAL_TOKENS:
            return 0
        if t & FAKE_TOKENS:
            return 1
    return None


print("\nScanning datasets...")
all_pairs = []

for droot in sorted(DATA_DIR.iterdir()):
    if not droot.is_dir():
        continue
    if droot.name in SKIP_DATASETS:
        print(f"  SKIPPED  : {droot.name:<35} (excluded)")
        continue
    r, f = 0, 0
    for p in droot.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = infer_label(p)
            if label == 0:
                r += 1
                all_pairs.append((str(p), 0))
            elif label == 1:
                f += 1
                all_pairs.append((str(p), 1))
    print(f"  {droot.name:<35} real={r:>7,}  fake={f:>7,}")

seen    = set()
deduped = []
for path, label in all_pairs:
    if path not in seen:
        seen.add(path)
        deduped.append((path, label))
all_pairs = deduped
random.shuffle(all_pairs)

real_n = sum(1 for _, l in all_pairs if l == 0)
fake_n = sum(1 for _, l in all_pairs if l == 1)
print(f"\nTotal   : {len(all_pairs):,}")
print(f"Real    : {real_n:,}")
print(f"Fake    : {fake_n:,}")
print(f"Ratio   : 1 : {fake_n / max(real_n, 1):.2f}")

assert len(all_pairs) > 10000, \
    f"Too few images ({len(all_pairs)}). Check dataset paths."


# ─────────────────────────────────────────────────────────────
# 3. VIDEO-SAFE SPLIT  70 / 15 / 15
# ─────────────────────────────────────────────────────────────
paths  = [p for p, _ in all_pairs]
labels = [l for _, l in all_pairs]
groups = [str(Path(p).parent) for p in paths]

unique_groups = list(set(groups))
random.shuffle(unique_groups)
n    = len(unique_groups)
tr_g = set(unique_groups[:int(n * 0.70)])
dv_g = set(unique_groups[int(n * 0.70):int(n * 0.85)])

train_pairs = [(p, l) for p, l, g in zip(paths, labels, groups) if g in tr_g]
dev_pairs   = [(p, l) for p, l, g in zip(paths, labels, groups) if g in dv_g]
eval_pairs  = [(p, l) for p, l, g in zip(paths, labels, groups)
               if g not in tr_g and g not in dv_g]

print(f"\nSplit   — Train : {len(train_pairs):,} | "
      f"Dev : {len(dev_pairs):,} | Eval : {len(eval_pairs):,}")

assert len(train_pairs) > 0, "Train split is empty"
assert len(dev_pairs)   > 0, "Dev split is empty"
assert len(eval_pairs)  > 0, "Eval split is empty"


# ─────────────────────────────────────────────────────────────
# 4. AUGMENTATIONS
# ─────────────────────────────────────────────────────────────
train_tfms = A.Compose([
    A.Resize(cfg.img_size, cfg.img_size),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
    A.ColorJitter(brightness=0.15, contrast=0.15,
                  saturation=0.1, hue=0.05, p=0.3),
    A.ToGray(p=0.05, num_output_channels=3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

eval_tfms = A.Compose([
    A.Resize(cfg.img_size, cfg.img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────
# 5. DATASET CLASS
# ─────────────────────────────────────────────────────────────
class ImageGuardDataset(Dataset):
    def __init__(self, pairs, transforms):
        self.pairs      = pairs
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)['image']
        return img, torch.tensor(float(label), dtype=torch.float32)


def build_sampler(pairs):
    lbls = [l for _, l in pairs]
    n0   = max(lbls.count(0), 1)
    n1   = max(lbls.count(1), 1)
    cw   = {0: 1.0 / n0, 1: 1.0 / n1}
    sw   = torch.DoubleTensor([cw[l] for l in lbls])
    return WeightedRandomSampler(sw, len(sw), replacement=True)


train_loader = DataLoader(
    ImageGuardDataset(train_pairs, train_tfms),
    batch_size      = cfg.batch_size,
    sampler         = build_sampler(train_pairs),
    num_workers     = cfg.num_workers,
    pin_memory      = True,
    prefetch_factor = 2,
)
dev_loader = DataLoader(
    ImageGuardDataset(dev_pairs, eval_tfms),
    batch_size  = cfg.batch_size,
    shuffle     = False,
    num_workers = cfg.num_workers,
    pin_memory  = True,
)
eval_loader = DataLoader(
    ImageGuardDataset(eval_pairs, eval_tfms),
    batch_size  = cfg.batch_size,
    shuffle     = False,
    num_workers = cfg.num_workers,
    pin_memory  = True,
)
print(f"Loaders — Train : {len(train_loader)} batches | "
      f"Dev : {len(dev_loader)} | Eval : {len(eval_loader)}")


# ─────────────────────────────────────────────────────────────
# 6. MODEL — EfficientNet-B4-NS + FFT-MobileNetV3
# ─────────────────────────────────────────────────────────────
def rgb_to_fft(x: torch.Tensor) -> torch.Tensor:
    gray = (0.2989 * x[:, 0:1]
          + 0.5870 * x[:, 1:2]
          + 0.1140 * x[:, 2:3])
    fft  = torch.fft.fft2(gray)
    mag  = torch.log1p(torch.abs(fft))
    mn   = mag.amin(dim=(2, 3), keepdim=True)
    mx   = mag.amax(dim=(2, 3), keepdim=True)
    return (mag - mn) / (mx - mn + 1e-6)


class ImageGuardV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = timm.create_model(
            'tf_efficientnet_b4_ns',
            pretrained  = True,
            num_classes = 0,
            global_pool = 'avg',
        )
        self.freq = timm.create_model(
            'mobilenetv3_small_050',
            pretrained  = True,
            in_chans    = 1,
            num_classes = 0,
            global_pool = 'avg',
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size)
            d_s   = self.spatial(dummy).shape[1]
            d_f   = self.freq(rgb_to_fft(dummy)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(d_s + d_f, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_s = self.spatial(x)
        f_f = self.freq(rgb_to_fft(x))
        return self.head(torch.cat([f_s, f_f], dim=1)).squeeze(1)


model = ImageGuardV2().to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

total_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
print(f"Trainable params : {total_params / 1e6:.2f}M")


# ─────────────────────────────────────────────────────────────
# 7. OPTIMISER / SCHEDULER / LOSS / SCALER
# ─────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = cfg.lr,
    weight_decay = cfg.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-6
)
criterion = nn.BCEWithLogitsLoss()
scaler    = GradScaler(device='cuda', enabled=AMP_ON)


# ─────────────────────────────────────────────────────────────
# 8. EVALUATION HELPER
# ─────────────────────────────────────────────────────────────
def eval_model(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ON):
                prob = torch.sigmoid(model(x))
            ps.append(prob.cpu().numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob)) \
          if len(np.unique(y_true)) > 1 else 0.5
    return {
        'auc':       auc,
        'acc':       float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────
# 9. CHECKPOINT RESUME
# ─────────────────────────────────────────────────────────────
best_auc    = 0.0
start_epoch = 1
no_improve  = 0

if CKPT_PATH.exists():
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt['model_state_dict']
    if hasattr(model, 'module'):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    best_auc    = ckpt['dev_auc']
    start_epoch = ckpt['epoch'] + 1
    print(f"\nResumed from epoch {ckpt['epoch']} — best AUC {best_auc:.4f}")
else:
    print("\nTraining from scratch")


# ─────────────────────────────────────────────────────────────
# 10. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
log_path = LOG_DIR / 'image_retrain_v3_log.csv'
with open(log_path, 'a') as f:
    if start_epoch == 1:
        f.write('epoch,train_loss,dv_auc,dv_acc,dv_f1,'
                'dv_precision,dv_recall\n')

print("\n" + "=" * 65)
print("  Starting Training")
print("=" * 65)

for epoch in range(start_epoch, cfg.epochs + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
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
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item()) * cfg.grad_accum_steps

    if len(train_loader) % cfg.grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()

    avg_loss = running_loss / max(len(train_loader), 1)
    m        = eval_model(model, dev_loader)

    with open(log_path, 'a') as f:
        f.write(f"{epoch},{avg_loss:.4f},{m['auc']:.4f},"
                f"{m['acc']:.4f},{m['f1']:.4f},"
                f"{m['precision']:.4f},{m['recall']:.4f}\n")

    flag = ''
    if m['auc'] > best_auc:
        best_auc   = m['auc']
        no_improve = 0
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     (model.module.state_dict()
                                     if hasattr(model, 'module')
                                     else model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'dev_auc':              m['auc'],
            'dev_acc':              m['acc'],
            'config': {
                'img_size': cfg.img_size,
                'model':    'tf_efficientnet_b4_ns+mobilenetv3_small_050',
            },
        }, CKPT_PATH)
        flag = '  ✓ saved'
    else:
        no_improve += 1

    print(f"Ep {epoch:02d}/{cfg.epochs} | "
          f"Loss {avg_loss:.4f} | "
          f"AUC {m['auc']:.4f} | "
          f"Acc {m['acc']:.4f} | "
          f"F1 {m['f1']:.4f} | "
          f"P {m['precision']:.4f} | "
          f"R {m['recall']:.4f}"
          f"{flag}")

    if no_improve >= cfg.patience:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no improvement for {cfg.patience} epochs).")
        break

print(f"\nBest Dev AUC : {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# 11. FINAL EVALUATION
# ─────────────────────────────────────────────────────────────
print("\nLoading best checkpoint for final evaluation...")
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
state = ckpt['model_state_dict']
if hasattr(model, 'module'):
    model.module.load_state_dict(state)
else:
    model.load_state_dict(state)

tm = eval_model(model, eval_loader)

model.eval()
ev_preds, ev_true = [], []
with torch.no_grad():
    for x, y in eval_loader:
        p = (torch.sigmoid(model(x.to(DEVICE))) >= 0.5).cpu().long().numpy()
        ev_preds.extend(p)
        ev_true.extend(y.long().numpy())

print("\n" + "=" * 55)
print("  FINAL EVAL RESULTS (held-out 15%)")
print("=" * 55)
print(f"  AUC       : {tm['auc']:.4f}")
print(f"  Accuracy  : {tm['acc']:.4f}")
print(f"  F1        : {tm['f1']:.4f}")
print(f"  Precision : {tm['precision']:.4f}")
print(f"  Recall    : {tm['recall']:.4f}")
print("=" * 55)
print(classification_report(ev_true, ev_preds,
                             target_names=['Real', 'Fake']))

out_json = LOG_DIR / 'test_metrics.json'
with open(out_json, 'w') as f:
    json.dump(tm, f, indent=2)

print(f"\nCheckpoint : {CKPT_PATH}")
print(f"Metrics    : {out_json}")
print(f"Train log  : {log_path}")
