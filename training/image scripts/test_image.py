#!/usr/bin/env python3
"""
ImageGuard v2 — External Test Evaluator
Runs inference on completely unseen test datasets
"""
import re, json, warnings
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report, confusion_matrix)
import torch
import torch.nn as nn
from torch.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON     = DEVICE.type == 'cuda'
CKPT_PATH  = Path.home() / 'deepfake_project/image/checkpoints/imageguard_v2_best.pt'
TEST_ROOT  = Path.home() / 'deepfake_project/image/test_data'
LOG_DIR    = Path.home() / 'deepfake_project/image/logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
REAL_TOKENS = {'real','original','authentic','pristine','genuine',
               'true','youtube','celeb'}
FAKE_TOKENS = {'fake','deepfake','deepfakes','ai','generated','synthetic',
               'faceswap','face2face','faceshifter','neuraltextures',
               'deepfakedetection','manipulated','altered','synthesis'}

# Datasets to skip (used in training — not valid for external testing)
SKIP_DATASETS = {'ff_greatgame','deepfake60k','celebdf'}


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(norm(part).split())
        if t & REAL_TOKENS: return 0
        if t & FAKE_TOKENS: return 1
    return None


# ── Model ─────────────────────────────────────────────────────────────
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
                                          pretrained=False, num_classes=0,
                                          global_pool='avg')
        self.freq    = timm.create_model('mobilenetv3_small_050',
                                          pretrained=False, in_chans=1,
                                          num_classes=0, global_pool='avg')
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            d_s   = self.spatial(dummy).shape[1]
            d_f   = self.freq(rgb_to_fft(dummy)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(d_s + d_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 128),       nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.head(torch.cat([self.spatial(x),
                                    self.freq(rgb_to_fft(x))], dim=1)).squeeze(1)

print("Loading checkpoint...")
model = ImageGuardV2().to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded epoch {ckpt['epoch']} — trained AUC {ckpt['dev_auc']:.4f}")


# ── Dataset + Loader ──────────────────────────────────────────────────
eval_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return eval_tfms(image=img)['image'], torch.tensor(float(label))


def run_test(name, pairs):
    if len(pairs) == 0:
        print(f"\n[{name}] SKIP — no labeled images found"); return None

    loader = DataLoader(TestDataset(pairs), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True)
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
    real_n = int((y_true == 0).sum())
    fake_n = int((y_true == 1).sum())
    auc = float(roc_auc_score(y_true, y_prob)) \
          if len(np.unique(y_true)) > 1 else 0.5

    metrics = {
        'dataset':   name,
        'n_total':   len(y_true),
        'n_real':    real_n,
        'n_fake':    fake_n,
        'auc':       auc,
        'acc':       float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  Images  : {len(y_true):,}  (Real: {real_n:,} | Fake: {fake_n:,})")
    print(f"  AUC     : {auc:.4f}")
    print(f"  Accuracy: {metrics['acc']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Precision:{metrics['precision']:.4f}")
    print(f"  Recall  : {metrics['recall']:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"              Pred Real  Pred Fake")
    print(f"  True Real :  {cm[0][0]:>8,}   {cm[0][1]:>8,}")
    print(f"  True Fake :  {cm[1][0]:>8,}   {cm[1][1]:>8,}")
    print(classification_report(y_true, y_pred,
                                 target_names=['Real','Fake']))
    return metrics


# ── Scan and Run All Test Datasets ────────────────────────────────────
all_results = []

for droot in sorted(TEST_ROOT.iterdir()):
    if not droot.is_dir() or droot.name in SKIP_DATASETS:
        continue
    pairs = []
    for p in droot.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = infer_label(p)
            if label is not None:
                pairs.append((str(p), label))
    result = run_test(droot.name, pairs)
    if result:
        all_results.append(result)

# ── Summary Table ─────────────────────────────────────────────────────
if all_results:
    print(f"\n{'='*70}")
    print("  SUMMARY — ImageGuard v2 External Test Results")
    print(f"{'='*70}")
    print(f"  {'Dataset':<30} {'N':>7} {'AUC':>7} {'Acc':>7} {'F1':>7}")
    print(f"  {'-'*58}")
    for r in all_results:
        print(f"  {r['dataset']:<30} {r['n_total']:>7,} "
              f"{r['auc']:>7.4f} {r['acc']:>7.4f} {r['f1']:>7.4f}")

    out = LOG_DIR / 'external_test_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out}")
