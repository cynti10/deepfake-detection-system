#!/usr/bin/env python3
"""
Sentinel Video Detector — Phase 3: Adversarial Robustness Training
Architecture: tf_efficientnet_b3_ns (RGB) + mobilenetv3_small_050 (FFT) + Transformer
Attacks: FGSM + PGD-5 on frame sequences (spatial domain)
Mix: 35% adversarial per batch — direct video scanning, no manifest needed
Input: video_detector_final.pt
Output: video_detector_phase3.pt
"""
import json, random, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import timm

warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    seq_len:       int   = 16
    image_size:    int   = 224
    batch_size:    int   = 6
    grad_accum:    int   = 2
    epochs:        int   = 8          # more epochs with stronger attacks
    patience:      int   = 4          # more patience
    lr:            float = 3e-6       # slightly lower lr for stability
    weight_decay:  float = 1e-4
    num_workers:   int   = 4
    amp:           bool  = True
    seed:          int   = 42
    adv_ratio:     float = 0.50       # ↑ was 0.35 — half the batch is adversarial
    fgsm_eps:      float = 0.02       # ↑ was 0.01 — stronger FGSM training
    pgd_eps:       float = 0.03       # ↑ was 0.02 — stronger PGD training
    pgd_steps:     int   = 10         # ↑ was 5 — stronger iterative attack
    pgd_alpha:     float = 0.005
    adv_weight:    float = 1.0        # ↑ was 0.4 — equal weight to clean loss
    temp_weight:   float = 0.05
    video_cap:     int   = 5000
    max_fake_ratio: float = 2.0

cfg = CFG()
random.seed(cfg.seed); np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON  = cfg.amp and DEVICE.type == 'cuda'

BASE_DIR  = Path.home() / 'deepfake_project/video'
CKPT_DIR  = BASE_DIR / 'checkpoints'
LOG_DIR   = BASE_DIR / 'logs'
IN_CKPT   = CKPT_DIR / 'video_detector_final.pt'
OUT_CKPT  = CKPT_DIR / 'video_detector_phase3.pt'
for d in [CKPT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

# ── Label inference ────────────────────────────────────────────────────
VIDEO_EXTS  = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

REAL_TOKENS = {
    'real', 'genuine', 'original', 'pristine', 'authentic', 'bonafide',
    'youtube', 'actors', 'original_sequences', 'raw', 'c0', 'c23', 'c40',
    'real_videos', 'live', 'untampered',
    'celeb-real', 'youtube-real',
    'realfacevideos', 'original_faces',
    'train_real', 'test_real', 'val_real',
    'trainreal',  'testreal',  'valreal',
    'train-real', 'test-real', 'val-real',
}

FAKE_TOKENS = {
    'fake', 'deepfake', 'synthetic', 'manipulated', 'altered', 'generated',
    'spoof', 'faceswap', 'face2face', 'face_2_face', 'f2f', 'neuraltextures',
    'faceshifter', 'dfdc', 'celebdf', 'ff++', 'dfaker', 'fsgan', 'liae',
    'simswap', 'reface', 'roop', 'inswapper', 'diffswap', 'adversarial',
    'fake_videos', 'manipulated_sequences',
    'fakeface', 'fakefacevideos',
    'train_fake', 'test_fake', 'val_fake',
    'trainfake',  'testfake',  'valfake',
    'train-fake', 'test-fake', 'val-fake',
    'deeperforensics', 'fsshifter',
    'adv',
}

def norm_token(s: str) -> str:
    return s.lower().replace('_', ' ').replace('-', ' ')

def infer_video_label(path) -> int | None:
    parts = Path(path).parts
    for part in reversed(parts):
        normed = norm_token(part)
        tokens = set(normed.split()) | {normed, part.lower()}
        if tokens & REAL_TOKENS: return 0
        if tokens & FAKE_TOKENS: return 1
        for rt in REAL_TOKENS:
            if rt in normed: return 0
        for ft in FAKE_TOKENS:
            if ft in normed: return 1
    full = norm_token(str(path))
    for rt in REAL_TOKENS:
        if rt in full: return 0
    for ft in FAKE_TOKENS:
        if ft in full: return 1
    return None

# ── Video scanner ──────────────────────────────────────────────────────
VIDEO_ROOTS = [
    Path('/media/rit/New Volume/video_datasets/faceforensics'),
    Path('/media/rit/New Volume/video_datasets/deeperforensics'),
    Path('/media/rit/New Volume/video_datasets/celebdf'),
    Path.home() / 'deepfake_project/generation/video_gen',
    Path.home() / 'deepfake_project/video',
    Path.home() / 'deepfake_project/image/test_data',
]

def scan_videos(roots, cap=None, max_fake_ratio=2.0):
    pairs   = []
    skipped = []

    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob('*')):
            if not p.is_file(): continue
            if p.suffix.lower() not in VIDEO_EXTS: continue
            if p.stat().st_size < 10_000: continue
            label = infer_video_label(p)
            if label is None:
                skipped.append(str(p))
                continue
            pairs.append((str(p), label))

    if skipped:
        print(f"  Skipped {len(skipped):,} files with no label match")
        sample = skipped[:10]
        print("  Sample of unresolved paths:")
        for s in sample:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    … and {len(skipped)-10:,} more")

    # ── Balance: cap fakes at max_fake_ratio × real count ────────────────
    real_pairs = [p for p in pairs if p[1] == 0]
    fake_pairs = [p for p in pairs if p[1] == 1]

    max_fakes = int(len(real_pairs) * max_fake_ratio)
    if len(fake_pairs) > max_fakes:
        random.shuffle(fake_pairs)
        fake_pairs = fake_pairs[:max_fakes]
        print(f"  Class balance: kept {len(fake_pairs):,} fakes "
              f"(capped at {max_fake_ratio}× real={len(real_pairs):,})")

    pairs = real_pairs + fake_pairs

    # ── Apply global cap after balancing ─────────────────────────────────
    if cap and len(pairs) > cap:
        random.shuffle(pairs)
        pairs = pairs[:cap]

    r = sum(1 for _, l in pairs if l == 0)
    f = sum(1 for _, l in pairs if l == 1)
    print(f"  Total: {len(pairs):,} videos — real={r:,}  fake={f:,}  "
          f"ratio={f/max(r,1):.2f}:1")
    return pairs


# ── Architecture ───────────────────────────────────────────────────────
def rgb_to_fft_mag(x):
    if x.shape[1] == 3:
        gray = 0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2]
    else:
        gray = x[:,0]
    fft = torch.fft.fft2(gray)
    mag = torch.log1p(torch.abs(fft))
    mn  = mag.amin(dim=(-2,-1), keepdim=True)
    mx  = mag.amax(dim=(-2,-1), keepdim=True)
    return (mag - mn) / (mx - mn + 1e-6)

class SentinelVideoDetector(nn.Module):
    def __init__(self, rgb_backbone='tf_efficientnet_b3_ns',
                 d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.rgb_backbone = timm.create_model(rgb_backbone, pretrained=False,
                                              num_classes=0, global_pool='avg')
        self.fft_backbone = timm.create_model('mobilenetv3_small_050', pretrained=False,
                                              in_chans=1, num_classes=0, global_pool='avg')
        with torch.no_grad():
            d_rgb = self.rgb_backbone(torch.zeros(1,3,224,224)).shape[1]
            d_fft = self.fft_backbone(torch.zeros(1,1,224,224)).shape[1]
        feat_dim = d_rgb + d_fft
        self.frame_proj  = nn.Linear(feat_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.temporal    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.zeros(1,1,d_model))
        self.cls_head    = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        b, t, c, h, w = x.shape
        xr  = x.view(b*t, c, h, w)
        fr  = self.rgb_backbone(xr)
        fft = rgb_to_fft_mag(xr)
        ff  = self.fft_backbone(fft.unsqueeze(1))
        f   = torch.cat([fr, ff], dim=1)
        tok = self.frame_proj(f.view(b, t, -1))
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        out = self.temporal(seq)
        return self.cls_head(out[:,0]).squeeze(1)

    def forward_per_frame(self, x):
        b, t, c, h, w = x.shape
        xr  = x.view(b*t, c, h, w)
        fr  = self.rgb_backbone(xr)
        fft = rgb_to_fft_mag(xr)
        ff  = self.fft_backbone(fft.unsqueeze(1))
        f   = torch.cat([fr, ff], dim=1).view(b, t, -1)
        tok = self.frame_proj(f)
        return self.cls_head(tok).squeeze(-1)


# ── Load checkpoint ────────────────────────────────────────────────────
model = SentinelVideoDetector().to(DEVICE)
ckpt  = torch.load(IN_CKPT, map_location=DEVICE, weights_only=False)
sd    = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(sd, strict=True)
val_m = ckpt.get('val_metrics', {})
print(f"Loaded video_detector_final.pt — Val AUC {val_m.get('auc', '?')}")
for p in model.parameters(): p.requires_grad = True
print(f"All parameters unfrozen for Phase 3")


# ── Direct video dataset ───────────────────────────────────────────────
class DirectVideoDataset(Dataset):
    def __init__(self, pairs, seq_len=16, image_size=224, augment=False):
        self.pairs      = pairs
        self.seq_len    = seq_len
        self.image_size = image_size
        self.augment    = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self): return len(self.pairs)

    def _load_frames(self, path):
        blank = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        cap   = cv2.VideoCapture(path)
        if not cap.isOpened():
            return np.stack([blank]*self.seq_len, axis=0)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        idxs  = np.linspace(0, total-1, self.seq_len).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                frames.append(frames[-1].copy() if frames else blank.copy())
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size),
                               interpolation=cv2.INTER_AREA)
            x = frame.astype(np.float32) / 255.0
            if self.augment:
                if random.random() < 0.5:
                    x = x[:, ::-1, :].copy()
                if random.random() < 0.2:
                    x = np.clip(x * random.uniform(0.8, 1.2), 0, 1)
            x = (x - self.mean) / self.std
            frames.append(np.transpose(x, (2, 0, 1)).astype(np.float32))
        cap.release()
        while len(frames) < self.seq_len:
            frames.append(frames[-1].copy() if frames else blank.copy())
        return np.stack(frames[:self.seq_len], axis=0)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        frames = self._load_frames(path)
        return (torch.tensor(frames, dtype=torch.float32),
                torch.tensor(float(label), dtype=torch.float32))


def make_weighted_sampler(pairs):
    """
    Per-sample inverse-frequency weights so each batch
    sees ~50% real / ~50% fake regardless of dataset ratio.
    """
    labels     = [l for _, l in pairs]
    counts     = np.bincount(labels)                      # [n_real, n_fake]
    weights_cls = 1.0 / counts.astype(np.float32)        # inverse freq per class
    sample_wts  = torch.tensor([weights_cls[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sample_wts, num_samples=len(sample_wts), replacement=True)


# ── Scan, split, build loaders ─────────────────────────────────────────
print("\nScanning video data...")
all_pairs = scan_videos(VIDEO_ROOTS, cap=cfg.video_cap,
                        max_fake_ratio=cfg.max_fake_ratio)

if len(all_pairs) == 0:
    raise FileNotFoundError(
        "No labelled videos found.\n"
        "Check VIDEO_ROOTS and folder names contain tokens from "
        "REAL_TOKENS / FAKE_TOKENS.\n"
        f"Searched: {[str(r) for r in VIDEO_ROOTS]}")

real_p = [p for p in all_pairs if p[1]==0]
fake_p = [p for p in all_pairs if p[1]==1]
random.shuffle(real_p); random.shuffle(fake_p)

def vsplit(lst, dv=0.10, ev=0.10):
    nd = max(int(len(lst)*dv), 20)
    ne = max(int(len(lst)*ev), 20)
    return lst[nd+ne:], lst[:nd], lst[nd:nd+ne]

rt, rd, re_ = vsplit(real_p)
ft, fd, fe  = vsplit(fake_p)
train_pairs = rt + ft
dev_pairs   = rd + fd
eval_pairs  = re_ + fe
print(f"Split : Train {len(train_pairs):,} | Dev {len(dev_pairs):,} | Eval {len(eval_pairs):,}")

train_ds = DirectVideoDataset(train_pairs, cfg.seq_len, cfg.image_size, augment=True)

# WeightedRandomSampler guarantees balanced batches even with residual imbalance
train_sampler = make_weighted_sampler(train_pairs)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    sampler=train_sampler,           # replaces shuffle=True
    num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
dev_loader   = DataLoader(
    DirectVideoDataset(dev_pairs,  cfg.seq_len, cfg.image_size),
    batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True)
eval_loader  = DataLoader(
    DirectVideoDataset(eval_pairs, cfg.seq_len, cfg.image_size),
    batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True)


# ── Adversarial attacks ────────────────────────────────────────────────
def fgsm_video(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss  = F.binary_cross_entropy_with_logits(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + eps * x_adv.grad.sign()
    return x_adv.detach()

def pgd_video(model, x, y, eps, alpha, steps):
    x_adv = (x + torch.zeros_like(x).uniform_(-eps, eps)).detach()
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        loss  = F.binary_cross_entropy_with_logits(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = x_adv.detach()
    return x_adv

def mixed_adv_video(model, x, y, ratio=0.35):
    n      = x.size(0)
    n_adv  = max(1, int(n * ratio))
    idx    = torch.randperm(n)[:n_adv]
    n_half = max(1, n_adv // 2)
    i_fgsm = idx[:n_half]
    i_pgd  = idx[n_half:]
    x_out  = x.clone()
    if len(i_fgsm):
        x_out[i_fgsm] = fgsm_video(model, x[i_fgsm], y[i_fgsm], cfg.fgsm_eps)
    if len(i_pgd):
        x_out[i_pgd]  = pgd_video(model, x[i_pgd], y[i_pgd],
                                   cfg.pgd_eps, cfg.pgd_alpha, cfg.pgd_steps)
    return x_out, y

def temporal_consistency_loss(model, x):
    with torch.no_grad():
        frame_logits = model.forward_per_frame(x)
    return frame_logits.var(dim=1).mean()


# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate(loader, attack_fn=None):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if attack_fn:
            x = attack_fn(model, x, y)
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


# ── Loss, optimizer, scheduler ─────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    alpha=0.5 treats both classes equally — better for balanced batches.
    alpha=0.25 (original) down-weighted the minority (real) class too hard.
    """
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob*targets + (1-prob)*(1-targets)
        a_t  = self.alpha*targets + (1-self.alpha)*(1-targets)
        return (a_t * (1-p_t)**self.gamma * bce).mean()

criterion = FocalLoss(alpha=0.5, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-8)
scaler    = GradScaler(device='cuda', enabled=AMP_ON)


# ── Training loop ──────────────────────────────────────────────────────
best_auc = 0.0; no_imp = 0
log_path = LOG_DIR / 'phase3_video_log.csv'
with open(log_path, 'w') as f:
    f.write('epoch,loss,dev_auc,dev_acc,dev_f1,loss_clean,loss_adv,loss_temp\n')

print("\n" + "="*65)
print("  Sentinel Video — Phase 3 Adversarial Training")
print("="*65)
print(f"Device  : {DEVICE}  |  AMP: {AMP_ON}")
print(f"FGSM ε={cfg.fgsm_eps}  PGD ε={cfg.pgd_eps}/{cfg.pgd_steps}-step  "
      f"adv_ratio={cfg.adv_ratio}  videos={len(all_pairs):,}")

for epoch in range(1, cfg.epochs+1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = clean_sum = adv_sum = temp_sum = 0.0

    for step, (x, y) in enumerate(train_loader, 1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(device_type=DEVICE.type, enabled=AMP_ON):
            logits_clean = model(x)
            loss_clean   = criterion(logits_clean, y)
            loss_temp    = temporal_consistency_loss(model, x)

        model.eval()
        x_adv, y_adv = mixed_adv_video(model, x, y, ratio=cfg.adv_ratio)
        model.train()

        with autocast(device_type=DEVICE.type, enabled=AMP_ON):
            logits_adv = model(x_adv)
            loss_adv   = criterion(logits_adv, y_adv)

        loss = (loss_clean + cfg.adv_weight * loss_adv +
                cfg.temp_weight * loss_temp) / cfg.grad_accum

        scaler.scale(loss).backward()

        if step % cfg.grad_accum == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total     += loss.item() * cfg.grad_accum
        clean_sum += loss_clean.item()
        adv_sum   += loss_adv.item()
        temp_sum  += loss_temp.item()

        if step % 20 == 0:
            print(f"  Ep{epoch} step {step}/{len(train_loader)} | "
                  f"loss {loss.item()*cfg.grad_accum:.4f} "
                  f"(clean {loss_clean.item():.4f}  "
                  f"adv {loss_adv.item():.4f}  "
                  f"temp {loss_temp.item():.4f})")

    scheduler.step()
    n    = max(len(train_loader), 1)
    m    = evaluate(dev_loader)
    flag = ''
    if m['auc'] > best_auc:
        best_auc = m['auc']; no_imp = 0
        torch.save({'state_dict':  model.state_dict(),
                    'val_metrics': m,
                    'epoch':       epoch,
                    'phase':       3,
                    'args': {'rgb_backbone': 'tf_efficientnet_b3_ns',
                             'd_model': 512, 'nhead': 8, 'num_layers': 2,
                             'seq_len': cfg.seq_len,
                             'image_size': cfg.image_size}},
                   OUT_CKPT)
        flag = '  ✓ saved'
    else:
        no_imp += 1

    print(f"Ep {epoch:02d}/{cfg.epochs} | Loss {total/n:.4f} | "
          f"AUC {m['auc']:.4f} | Acc {m['acc']:.4f} | F1 {m['f1']:.4f}{flag}")

    with open(log_path, 'a') as f:
        f.write(f"{epoch},{total/n:.4f},{m['auc']:.4f},{m['acc']:.4f},"
                f"{m['f1']:.4f},{clean_sum/n:.4f},{adv_sum/n:.4f},{temp_sum/n:.4f}\n")

    if no_imp >= cfg.patience:
        print(f"\nEarly stopping at epoch {epoch}."); break

print(f"\nBest Phase 3 Dev AUC : {best_auc:.4f}")

# ── Final robustness evaluation ────────────────────────────────────────
ckpt = torch.load(OUT_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['state_dict'])

print("\n" + "="*65)
print("  PHASE 3 VIDEO ROBUSTNESS EVALUATION")
print("="*65)

m_clean = evaluate(eval_loader)
print(f"  Clean  — AUC {m_clean['auc']:.4f}  Acc {m_clean['acc']:.4f}  F1 {m_clean['f1']:.4f}")

m_fgsm  = evaluate(eval_loader,
    attack_fn=lambda mdl, x, y: fgsm_video(mdl, x, y, eps=cfg.fgsm_eps))
print(f"  FGSM   — AUC {m_fgsm['auc']:.4f}  Acc {m_fgsm['acc']:.4f}  F1 {m_fgsm['f1']:.4f}")

m_pgd   = evaluate(eval_loader,
    attack_fn=lambda mdl, x, y: pgd_video(mdl, x, y,
                                           eps=cfg.pgd_eps,
                                           alpha=cfg.pgd_alpha,
                                           steps=cfg.pgd_steps))
print(f"  PGD-{cfg.pgd_steps}  — AUC {m_pgd['auc']:.4f}  Acc {m_pgd['acc']:.4f}  F1 {m_pgd['f1']:.4f}")

results = {'phase': 3, 'clean': m_clean, 'fgsm': m_fgsm, 'pgd': m_pgd}
out_json = LOG_DIR / 'phase3_video_metrics.json'
with open(out_json, 'w') as f: json.dump(results, f, indent=2)

print(f"\nCheckpoint : {OUT_CKPT}")
print(f"Metrics    : {out_json}")
print(f"Log        : {log_path}")
