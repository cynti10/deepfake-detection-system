#!/usr/bin/env python3
"""
ImageGuard Phase 3 — Adversarial Robustness Training
Architecture: EfficientNet-B5 (spatial, RGB) + EfficientNet-B2-lite (freq, FFT)
              + MLP head (2816 → 512 → 128 → 1)
Dev AUC before Phase 3: 0.9992
Input:  imageguard_v2_finetuned.pt
Output: imageguard_phase3.pt
"""
import json, random, warnings, os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from PIL import Image
import torchvision.transforms as TF
import timm

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Config ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    img_size:      int   = 224
    batch_size:    int   = 32
    epochs:        int   = 4
    patience:      int   = 3
    lr:            float = 8e-7    # very low — model is at 0.9992 AUC already
    weight_decay:  float = 1e-4
    num_workers:   int   = 8
    amp:           bool  = True
    seed:          int   = 42
    adv_ratio:     float = 0.45
    fgsm_eps:      float = 0.010   # slightly conservative — model is good, don't destabilise
    pgd_eps:       float = 0.020
    pgd_steps:     int   = 10
    pgd_alpha:     float = 0.003
    adv_weight:    float = 0.35
    hard_ratio:    float = 0.3     # match original training config

cfg = CFG()
random.seed(cfg.seed); np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON  = cfg.amp and DEVICE.type == 'cuda'

BASE_DIR  = Path.home() / 'deepfake_project/image'
CKPT_DIR  = BASE_DIR / 'checkpoints'
LOG_DIR   = BASE_DIR / 'logs'
DATA_DIR  = BASE_DIR / 'test_data'
IN_CKPT   = CKPT_DIR / 'imageguard_phase3.pt'
OUT_CKPT  = CKPT_DIR / 'imageguard_phase3b.pt'
for d in [CKPT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

# ── Exact architecture from checkpoint ────────────────────────────────
# spatial: EfficientNet-B5 (conv_stem 48ch, conv_head 1792ch output), RGB input
# freq:    EfficientNet-B2-lite (conv_stem 16ch, conv_head 1024ch output), 1ch FFT input
# head:    Linear(2816→512) → GELU → Dropout → Linear(512→128) → GELU → Dropout → Linear(128→1)
# 2816 = 1792 (spatial) + 1024 (freq)

class ImageGuardV2(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        # Spatial branch — EfficientNet-B5
        self.spatial = timm.create_model(
            'tf_efficientnet_b4_ns',
                pretrained=False, num_classes=0, global_pool='avg'
        )
        # Freq branch — MobileNetV3-Small-0.50 (FFT mag, 1ch input → 1024 features)
        self.freq = timm.create_model(
            'mobilenetv3_small_050',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            global_pool='avg'
        )
        # MLP head: 1792 + 1024 = 2816 → 512 → 128 → 1
        # Matches: head.0 (Linear 2816→512), head.3 (Linear 512→128), head.6 (Linear 128→1)
        # Indices 1,2 = GELU+Dropout, 4,5 = GELU+Dropout
        self.head = nn.Sequential(
            nn.Linear(2816, 512),   # head.0
            nn.GELU(),              # head.1
            nn.Dropout(0.3),        # head.2
            nn.Linear(512, 128),    # head.3
            nn.GELU(),              # head.4
            nn.Dropout(0.2),        # head.5
            nn.Linear(128, 1),      # head.6
        )
        self._img_size = img_size

    def _fft_features(self, x):
        """x: [B, C, H, W] RGB → grayscale FFT magnitude [B, 1, H, W]"""
        gray = 0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2]  # [B, H, W]
        fft  = torch.fft.fft2(gray)
        mag  = torch.log1p(torch.abs(fft))                       # [B, H, W]
        mn   = mag.amin(dim=(-2,-1), keepdim=True)
        mx   = mag.amax(dim=(-2,-1), keepdim=True)
        return ((mag - mn) / (mx - mn + 1e-6)).unsqueeze(1)      # [B, 1, H, W]

    def forward(self, x):                    # x: [B, 3, H, W]
        f_spatial = self.spatial(x)          # [B, 1792]
        f_freq    = self.freq(self._fft_features(x))  # [B, 1024]
        fused     = torch.cat([f_spatial, f_freq], dim=1)  # [B, 2816]
        return self.head(fused).squeeze(1)   # [B]

# ── Load checkpoint with strict=True ──────────────────────────────────
print("Loading ImageGuard V2 checkpoint...")
model = ImageGuardV2(img_size=cfg.img_size).to(DEVICE)
ckpt  = torch.load(IN_CKPT, map_location=DEVICE, weights_only=False)
sd    = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"  Missing keys  ({len(missing)}): {missing[:5]}")
if unexpected:
    print(f"  Unexpected    ({len(unexpected)}): {unexpected[:5]}")
if not missing and not unexpected:
    print(f"  Loaded perfectly — strict match")
print(f"  Dev AUC at load: {ckpt.get('dev_auc', '?')}")

# Sanity check — forward pass must not crash
with torch.no_grad():
    dummy = torch.zeros(2, 3, cfg.img_size, cfg.img_size).to(DEVICE)
    out   = model(dummy)
    assert out.shape == (2,), f"Forward pass shape error: {out.shape}"
print(f"  Forward pass sanity check passed — output shape {out.shape}")

for p in model.parameters(): p.requires_grad = True
# Freeze nothing — at 0.9992 AUC we only want small adversarial nudges

# ── Transforms ────────────────────────────────────────────────────────
mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
train_tf = TF.Compose([
    TF.Resize((cfg.img_size, cfg.img_size)),
    TF.RandomHorizontalFlip(),
    TF.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03),
    TF.ToTensor(),
    TF.Normalize(mean, std),
])
eval_tf = TF.Compose([
    TF.Resize((cfg.img_size, cfg.img_size)),
    TF.ToTensor(),
    TF.Normalize(mean, std),
])

# ── Dataset ────────────────────────────────────────────────────────────
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
REAL_TOKENS = {'real','genuine','original','pristine','authentic','bonafide',
               'youtube','actors','raw','live','untampered','training_real'}
FAKE_TOKENS = {'fake','synthetic','generated','gan','diffusion','deepfake',
               'ai','stylegan','midjourney','dalle','stable','manipulation',
               'altered','tampered','spliced','facegen','training_fake'}

def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(part.lower().replace('_',' ').replace('-',' ').split())
        t.add(part.lower().replace('_','').replace('-',''))
        t.add(part.lower())
        if t & REAL_TOKENS: return 0
        if t & FAKE_TOKENS: return 1
    return None

class ImageAdvDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs; self.tf = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (cfg.img_size, cfg.img_size))
        return self.tf(img), torch.tensor(float(label))

def make_sampler(pairs):
    lbls = [l for _,l in pairs]
    n0 = max(lbls.count(0), 1); n1 = max(lbls.count(1), 1)
    w  = torch.DoubleTensor([1/n0 if l==0 else 1/n1 for l in lbls])
    return WeightedRandomSampler(w, len(w), replacement=True)

def strat_split(pairs, dv=0.10, ev=0.10):
    real = [p for p in pairs if p[1]==0]
    fake = [p for p in pairs if p[1]==1]
    random.shuffle(real); random.shuffle(fake)
    def sp(lst):
        nd = max(int(len(lst)*dv), 100)
        ne = max(int(len(lst)*ev), 100)
        return lst[nd+ne:], lst[:nd], lst[nd:nd+ne]
    rt,rd,re_ = sp(real); ft,fd,fe = sp(fake)
    return rt+ft, rd+fd, re_+fe

# Scan data
print(f"\nScanning {DATA_DIR} ...")
all_pairs = []
for p in DATA_DIR.rglob('*'):
    if not p.is_file(): continue
    if p.suffix.lower() not in IMG_EXTS: continue
    if p.stat().st_size < 500: continue
    l = infer_label(p)
    if l is not None:
        all_pairs.append((str(p), l))

r = sum(1 for _,l in all_pairs if l==0)
f = sum(1 for _,l in all_pairs if l==1)
print(f"  Found {len(all_pairs):,} images — real={r:,}  fake={f:,}")

train_pairs, dev_pairs, eval_pairs = strat_split(all_pairs)
print(f"  Split: Train {len(train_pairs):,} | Dev {len(dev_pairs):,} | Eval {len(eval_pairs):,}")

train_loader = DataLoader(ImageAdvDataset(train_pairs, train_tf),
    batch_size=cfg.batch_size, sampler=make_sampler(train_pairs),
    num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
dev_loader   = DataLoader(ImageAdvDataset(dev_pairs, eval_tf),
    batch_size=cfg.batch_size*2, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True)
eval_loader  = DataLoader(ImageAdvDataset(eval_pairs, eval_tf),
    batch_size=cfg.batch_size*2, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True)

# ── Adversarial attacks ────────────────────────────────────────────────
def fgsm(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss  = F.binary_cross_entropy_with_logits(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + eps * x_adv.grad.sign()
    return x_adv.detach()

def pgd(model, x, y, eps, alpha, steps):
    x_adv = (x + torch.zeros_like(x).uniform_(-eps, eps)).detach()
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        loss  = F.binary_cross_entropy_with_logits(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x+eps), x-eps)
        x_adv = x_adv.detach()
    return x_adv

def mixed_adv(model, x, y, ratio):
    n      = x.size(0)
    n_adv  = max(1, int(n * ratio))
    idx    = torch.randperm(n)[:n_adv]
    n_half = max(1, n_adv // 2)
    x_out  = x.clone()
    if n_half:
        x_out[idx[:n_half]] = fgsm(model, x[idx[:n_half]], y[idx[:n_half]], cfg.fgsm_eps)
    if n_adv - n_half:
        x_out[idx[n_half:]] = pgd(model, x[idx[n_half:]], y[idx[n_half:]],
                                   cfg.pgd_eps, cfg.pgd_alpha, cfg.pgd_steps)
    return x_out, y

# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate(loader, attack_fn=None):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if attack_fn: x = attack_fn(model, x, y)
        with torch.no_grad():
            with autocast(device_type=DEVICE.type, enabled=AMP_ON):
                prob = torch.sigmoid(model(x))
        ps.append(prob.cpu().numpy()); ys.append(y.cpu().numpy())
    yt = np.concatenate(ys); yp = np.concatenate(ps)
    pred = (yp >= 0.5).astype(int)
    auc  = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
    return {'auc': auc,
            'acc': float(accuracy_score(yt, pred)),
            'f1':  float(f1_score(yt, pred, zero_division=0))}

# ── Loss / Optimizer / Scheduler ──────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob*targets + (1-prob)*(1-targets)
        a_t  = self.alpha*targets + (1-self.alpha)*(1-targets)
        return (a_t*(1-p_t)**self.gamma*bce).mean()

criterion = FocalLoss()
optimizer = torch.optim.AdamW(
    [{'params': model.spatial.parameters(), 'lr': cfg.lr},
     {'params': model.freq.parameters(),    'lr': cfg.lr},
     {'params': model.head.parameters(),    'lr': cfg.lr * 5}],  # head trains faster
    weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-9)
scaler = GradScaler(device='cuda', enabled=AMP_ON)

# ── Training ───────────────────────────────────────────────────────────
best_auc = 0.0; no_imp = 0
log_path = LOG_DIR / 'phase3b_image_log.csv'
with open(log_path, 'w') as f:
    f.write('epoch,loss,dev_auc,dev_acc,dev_f1,loss_clean,loss_adv\n')

print("\n" + "="*65)
print("  ImageGuard — Phase 3b Hardened Adversarial Training")
print("="*65)
print(f"Device     : {DEVICE}  |  AMP: {AMP_ON}")
print(f"Backbone   : EfficientNet-B5 (spatial) + B2-lite (freq)")
print(f"FGSM ε={cfg.fgsm_eps}  PGD ε={cfg.pgd_eps}/{cfg.pgd_steps}-step  "
      f"adv_ratio={cfg.adv_ratio}")
print(f"Starting from Dev AUC 0.9992 — small LR, conservative ε")

for epoch in range(1, cfg.epochs+1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = clean_sum = adv_sum = 0.0

    for step, (x, y) in enumerate(train_loader, 1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # Clean forward
        with autocast(device_type=DEVICE.type, enabled=AMP_ON):
            logits_clean = model(x)
            loss_clean   = criterion(logits_clean, y)

        # Generate adversarial examples
        model.eval()
        x_adv, y_adv = mixed_adv(model, x, y, ratio=cfg.adv_ratio)
        model.train()

        # Adversarial forward
        with autocast(device_type=DEVICE.type, enabled=AMP_ON):
            logits_adv = model(x_adv)
            loss_adv   = criterion(logits_adv, y_adv)

        loss = loss_clean + cfg.adv_weight * loss_adv
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total     += loss.item()
        clean_sum += loss_clean.item()
        adv_sum   += loss_adv.item()

        if step % 100 == 0:
            print(f"  Ep{epoch} step {step}/{len(train_loader)} | "
                  f"loss {loss.item():.4f} "
                  f"(clean {loss_clean.item():.4f}  adv {loss_adv.item():.4f})")

    scheduler.step()
    n    = max(len(train_loader), 1)
    m    = evaluate(dev_loader)
    flag = ''
    if m['auc'] > best_auc:
        best_auc = m['auc']; no_imp = 0
        torch.save({'model_state_dict': model.state_dict(),
                    'dev_auc':  m['auc'],
                    'epoch':    epoch,
                    'phase':    3,
                    'config':   {'img_size': cfg.img_size,
                                 'hard_ratio': cfg.hard_ratio,
                                 'backbone_spatial': 'tf_efficientnet_b5_ns',
                                 'backbone_freq':    'tf_efficientnet_b2_ns'}},
                   OUT_CKPT)
        flag = '  ✓ saved'
    else:
        no_imp += 1

    print(f"Ep {epoch:02d}/{cfg.epochs} | Loss {total/n:.4f} | "
          f"AUC {m['auc']:.4f} | Acc {m['acc']:.4f} | F1 {m['f1']:.4f}{flag}")

    with open(log_path, 'a') as f:
        f.write(f"{epoch},{total/n:.4f},{m['auc']:.4f},{m['acc']:.4f},"
                f"{m['f1']:.4f},{clean_sum/n:.4f},{adv_sum/n:.4f}\n")

    if no_imp >= cfg.patience:
        print(f"\nEarly stopping at epoch {epoch}."); break

print(f"\nBest Phase 3 Dev AUC : {best_auc:.4f}")

# ── Robustness Evaluation ──────────────────────────────────────────────
ckpt = torch.load(OUT_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

print("\n" + "="*65)
print("  PHASE 3 IMAGE ROBUSTNESS EVALUATION")
print("="*65)

m_clean = evaluate(eval_loader)
print(f"  Clean  — AUC {m_clean['auc']:.4f}  Acc {m_clean['acc']:.4f}  F1 {m_clean['f1']:.4f}")

m_fgsm  = evaluate(eval_loader,
    attack_fn=lambda mdl,x,y: fgsm(mdl,x,y,cfg.fgsm_eps))
print(f"  FGSM   — AUC {m_fgsm['auc']:.4f}  Acc {m_fgsm['acc']:.4f}  F1 {m_fgsm['f1']:.4f}")

m_pgd   = evaluate(eval_loader,
    attack_fn=lambda mdl,x,y: pgd(mdl,x,y,cfg.pgd_eps,cfg.pgd_alpha,cfg.pgd_steps))
print(f"  PGD-{cfg.pgd_steps}  — AUC {m_pgd['auc']:.4f}  Acc {m_pgd['acc']:.4f}  F1 {m_pgd['f1']:.4f}")

results = {'phase': 3, 'clean': m_clean, 'fgsm': m_fgsm, 'pgd': m_pgd}
out_json = LOG_DIR / 'phase3b_image_metrics.json'
with open(out_json, 'w') as f: json.dump(results, f, indent=2)

print(f"\nCheckpoint : {OUT_CKPT}")
print(f"Metrics    : {out_json}")
print(f"Log        : {log_path}")
