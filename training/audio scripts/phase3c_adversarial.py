#!/usr/bin/env python3
"""
AudioGuard RawNet3-FSAT — Phase 3c: PGD-Curriculum Hardening
Loads rawnet3_phase3c.pt (Phase 3 output)
Stronger attacks: FGSM ε=0.008, PGD-7 ε=0.015, F-SAT ε=0.015
Mix: 40% adversarial per batch + waveform consistency regularizer
"""
import re, json, warnings, random, math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torchaudio
import torchaudio.transforms as T
import subprocess, tempfile, os

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────
@dataclass
class CFG:
    sr:            int   = 16000
    n_samples:     int   = 64000
    batch_size:    int   = 24
    epochs:        int   = 6
    patience:      int   = 4
    lr:            float = 5e-7
    weight_decay:  float = 1e-4
    num_workers:   int   = 8
    amp:           bool  = True
    seed:          int   = 42
    # Adversarial params — stronger than Phase 3
    adv_ratio:     float = 0.30
    fgsm_eps:      float = 0.008    # 2.5× Phase 3
    pgd_eps:       float = 0.020    # 3× Phase 3
    pgd_steps:     int   = 20        # 7-step PGD
    pgd_alpha:     float = 0.003
    fsat_eps:      float = 0.018    # 1.5× Phase 3
    fsat_gamma:    float = 0.15     # slightly higher adv weight
    f_low:         int   = 4000
    f_high:        int   = 8000
    n_fft:         int   = 1024
    compress_prob: float = 0.20
    spec_weight:   float = 0.10     # waveform consistency regularizer weight

cfg = CFG()
random.seed(cfg.seed); np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON  = cfg.amp and DEVICE.type == 'cuda'

BASE_DIR  = Path.home() / 'deepfake_project/audio'
ORIG_DIR  = BASE_DIR / 'test_data'
CKPT_DIR  = BASE_DIR / 'checkpoints'
LOG_DIR   = BASE_DIR / 'logs'
IN_CKPT   = CKPT_DIR / 'rawnet3_phase3c.pt'    # Phase 3b as input
OUT_CKPT  = CKPT_DIR / 'rawnet3_phase3d.pt'   # Phase 3b output
for d in [CKPT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
REAL_TOKENS = {'real','genuine','authentic','bonafide','original',
               'true','live','lj','ljs','real_audio'}
FAKE_TOKENS = {'fake','spoof','synthetic','generated','tts','deepfake',
               'ai','cloned','vocoder','melgan','hifigan','waveglow',
               'parallel','wavegan','fastspeech','conformer','diffwave',
               'wavernn','ljspeech','jsut','multiband','fullband',
               'generated_audio','synthesis'}

def norm(s): return re.sub(r'[^a-z0-9]+',' ', s.lower()).strip()
def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(norm(part).split())
        if t & REAL_TOKENS: return 0
        if t & FAKE_TOKENS: return 1
    return None

# ── Waveform consistency regularizer ─────────────────────────────────
def waveform_consistency_loss(wav_batch):
    """
    Spectral regularizer for raw waveforms [B, 1, T].
    Penalizes PGD/FGSM artifacts: sudden temporal energy spikes
    and unnatural high-frequency energy concentration.
    """
    x = wav_batch.squeeze(1)                        # [B, T]
    delta = x[:, 1:] - x[:, :-1]                   # [B, T-1]
    temporal_roughness = delta.pow(2).mean()
    stft    = torch.stft(x, n_fft=cfg.n_fft,
                         hop_length=cfg.n_fft // 4,
                         return_complex=True)        # [B, F, t]
    mag     = stft.abs()
    n_bins  = mag.shape[1]
    top_bins = mag[:, n_bins * 3 // 4:, :]
    bot_bins = mag[:, :n_bins // 4, :]
    spectral_tilt = F.relu(top_bins.mean() - bot_bins.mean())
    return temporal_roughness + 0.5 * spectral_tilt

# ── Architecture ──────────────────────────────────────────────────────
class SincConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 125
        self.low_hz_  = nn.Parameter(torch.full((128, 1), 4000 / cfg.sr))
        self.band_hz_ = nn.Parameter(torch.full((128, 1), 4000 / cfg.sr))
        n = torch.arange(-62, 63, dtype=torch.float32)
        self.register_buffer('n_',      n.view(1, -1))
        self.register_buffer('window_', torch.hamming_window(125))
    @staticmethod
    def sinc(x):
        x = torch.where(x == 0, torch.full_like(x, 1e-6), x)
        return torch.sin(math.pi * x) / (math.pi * x)
    def forward(self, x):
        low  = torch.abs(self.low_hz_)
        high = low + torch.abs(self.band_hz_) + 1e-6
        f = 2 * (self.sinc(2 * high * self.n_) - self.sinc(2 * low * self.n_)) * self.window_
        f = f / (2 * f.abs().sum(dim=1, keepdim=True) + 1e-6)
        return F.conv1d(x, f.unsqueeze(1), padding=62)

class ResBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm1d(oc))
        self.skip = nn.Sequential(nn.Conv1d(ic, oc, 1, bias=False), nn.BatchNorm1d(oc))
        self.pool = nn.MaxPool1d(3)
    def forward(self, x):
        return self.pool(F.leaky_relu(self.conv(x) + self.skip(x), 0.1))

class ASP(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.Tanh(), nn.Conv1d(128, 256, 1))
    def forward(self, x):
        w   = torch.softmax(self.attn(x), dim=2)
        mu  = (w * x).sum(2)
        std = ((w * (x**2)).sum(2) - mu**2).clamp(1e-6).sqrt()
        return torch.cat([mu, std], 1)

class RawNet3FSAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc    = SincConv()
        self.bn0     = nn.BatchNorm1d(128)
        self.encoder = nn.Sequential(
            ResBlock(128, 128), ResBlock(128, 256),
            ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 256))
        self.asp       = ASP()
        self.bn_asp    = nn.BatchNorm1d(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x):
        x = self.sinc(x)
        x = F.leaky_relu(self.bn0(torch.abs(x)), 0.1)
        x = self.encoder(x)
        x = self.asp(x)
        x = self.bn_asp(x)
        return self.classifier(x).squeeze(1)

# ── Load Phase 3 checkpoint ───────────────────────────────────────────
model = RawNet3FSAT().to(DEVICE)
ckpt  = torch.load(IN_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
print(f"Loaded Phase 3 checkpoint — Dev AUC {ckpt['dev_auc']:.4f}  "
      f"Dev Acc {ckpt['dev_acc']:.4f}")

for p in model.parameters():
    p.requires_grad = True
print(f"All {sum(1 for p in model.parameters())} param groups unfrozen for Phase 3b")

# ── Waveform loader ───────────────────────────────────────────────────
def load_waveform(path):
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 100: return None
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != cfg.sr:     wav = T.Resample(sr, cfg.sr)(wav)
        wav = wav.squeeze(0).numpy()
        if len(wav) < cfg.n_samples:
            wav = np.pad(wav, (0, cfg.n_samples - len(wav)))
        else:
            wav = wav[:cfg.n_samples]
        pk = np.abs(wav).max()
        if pk > 1e-6: wav = wav / pk
        return wav.astype(np.float32)
    except:
        return None

# ── Compression augmentation ──────────────────────────────────────────
def compress_wav(wav_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    tmp_in_name = tmp_mp3_name = tmp_out_name = None   # ← fix: pre-initialize
    try:
        bitrate = random.choice(['32k', '48k', '64k', '96k', '128k'])
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            torchaudio.save(tmp_in.name,
                            torch.tensor(wav_np).unsqueeze(0), sr, format='wav')
            tmp_in_name = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
            tmp_mp3_name = tmp_mp3.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_name = tmp_out.name
        subprocess.run(['ffmpeg', '-y', '-i', tmp_in_name, '-b:a', bitrate, tmp_mp3_name],
                       capture_output=True, check=True)
        subprocess.run(['ffmpeg', '-y', '-i', tmp_mp3_name,
                        '-ar', str(sr), '-ac', '1', tmp_out_name],
                       capture_output=True, check=True)
        out, _ = torchaudio.load(tmp_out_name)
        out = out.squeeze(0).numpy()
        if len(out) < len(wav_np):
            out = np.pad(out, (0, len(wav_np) - len(out)))
        else:
            out = out[:len(wav_np)]
        pk = np.abs(out).max()
        return (out / pk).astype(np.float32) if pk > 1e-6 else wav_np
    except:
        return wav_np
    finally:
        for f in [tmp_in_name, tmp_mp3_name, tmp_out_name]:
            if f:                          # ← fix: only unlink if actually created
                try: os.unlink(f)
                except: pass

# ── Base augmentation ─────────────────────────────────────────────────
def augment(wav):
    wav = wav * random.uniform(0.6, 1.0)
    if random.random() < 0.4:
        wav += np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.015)
    if random.random() < 0.3:
        start = random.randint(0, cfg.n_samples // 8)
        wav = np.concatenate([wav[start:], np.zeros(start, dtype=np.float32)])
    if random.random() < cfg.compress_prob:
        wav = compress_wav(wav, cfg.sr)
    if random.random() < 0.20:
        delay_ms = random.randint(20, 180)
        delay    = int(delay_ms * cfg.sr / 1000)
        decay    = random.uniform(0.15, 0.45)
        if delay < len(wav):
            echo = np.concatenate([np.zeros(delay, dtype=np.float32), wav[:-delay]])
            wav  = np.clip(wav + decay * echo, -1.0, 1.0)
    if random.random() < 0.15:
        try:
            import torchaudio.functional as TAF
            w = torch.tensor(wav).unsqueeze(0)
            w = TAF.highpass_biquad(w, cfg.sr, 300.0)
            w = TAF.lowpass_biquad(w,  cfg.sr, 3400.0)
            wav = w.squeeze(0).numpy()
        except Exception:
            pass
    if random.random() < 0.10:
        try:
            import torchaudio.functional as TAF
            cutoff = random.uniform(4000.0, 7000.0)
            w   = torch.tensor(wav).unsqueeze(0)
            wav = TAF.lowpass_biquad(w, cfg.sr, cutoff).squeeze(0).numpy()
        except Exception:
            pass
    if random.random() < 0.08:
        wav = np.clip(wav + random.uniform(-0.03, 0.03), -1.0, 1.0)
    if random.random() < 0.08:
        thresh = random.uniform(0.6, 0.95)
        wav    = np.clip(wav, -thresh, thresh)
        wav    = wav / (thresh + 1e-6)
    pk = np.abs(wav).max()
    return wav / pk if pk > 1e-6 else wav

# ── Adversarial attacks ───────────────────────────────────────────────
def fgsm_attack(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss  = F.binary_cross_entropy_with_logits(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = (x + eps * x_adv.grad.sign()).clamp(-1.0, 1.0)
    return x_adv.detach()

def pgd_attack(model, x, y, eps, alpha, steps):
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(-1.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss  = F.binary_cross_entropy_with_logits(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(-1.0, 1.0)
    return x_adv.detach()

def fsat_attack(model, x, y, eps, steps=3):
    alpha = eps * 2 / steps
    r_l   = math.floor(cfg.f_low  * cfg.n_fft / cfg.sr)
    r_u   = math.ceil (cfg.f_high * cfg.n_fft / cfg.sr)
    T     = x.shape[-1]
    x_adv = (x + torch.zeros_like(x).uniform_(-eps, eps)).detach()
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        x2d   = x_adv.squeeze(1)
        stft  = torch.stft(x2d, n_fft=cfg.n_fft, return_complex=True)
        x_rec = torch.istft(stft.abs() * torch.exp(1j * stft.angle()),
                            n_fft=cfg.n_fft, length=T).unsqueeze(1)
        loss  = F.binary_cross_entropy_with_logits(model(x_rec), y)
        loss.backward()
        with torch.no_grad():
            g2d  = x_adv.grad.detach().squeeze(1)
            gs   = torch.stft(g2d, n_fft=cfg.n_fft, return_complex=True)
            msk  = torch.zeros_like(gs.abs()); msk[:, r_l:r_u, :] = 1.0
            gm   = torch.istft(gs * msk, n_fft=cfg.n_fft, length=T)
            delta = torch.clamp(x_adv.squeeze(1) + alpha * gm.sign()
                                - x.squeeze(1), -eps, eps)
            x_adv = (x.squeeze(1) + delta).unsqueeze(1).detach()
    return x_adv

def mixed_adversarial_batch(model, x, y, ratio=0.40):
    n      = x.size(0)
    n_adv  = max(1, int(n * ratio))
    idx    = torch.randperm(n)[:n_adv]
    n_each = max(1, n_adv // 3)
    i_fgsm = idx[:n_each]
    i_pgd  = idx[n_each:2*n_each]
    i_fsat = idx[2*n_each:]
    x_out  = x.clone()
    if len(i_fgsm):
        x_out[i_fgsm] = fgsm_attack(model, x[i_fgsm], y[i_fgsm], eps=cfg.fgsm_eps)
    if len(i_pgd):
        x_out[i_pgd]  = pgd_attack(model, x[i_pgd], y[i_pgd],
                                    eps=cfg.pgd_eps, alpha=cfg.pgd_alpha,
                                    steps=cfg.pgd_steps)
    if len(i_fsat):
        x_out[i_fsat] = fsat_attack(model, x[i_fsat], y[i_fsat], eps=cfg.fsat_eps)
    return x_out, y

# ── Dataset scan ──────────────────────────────────────────────────────
def scan_dir(root, only_set=None, skip_set=None, limit=None):
    pairs = []
    for droot in sorted(root.iterdir()):
        if not droot.is_dir(): continue
        if only_set and droot.name not in only_set: continue
        if skip_set and droot.name in skip_set:     continue
        r, f = 0, 0
        for p in droot.rglob('*'):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                if p.stat().st_size < 100: continue
                l = infer_label(p)
                if   l == 0: r += 1; pairs.append((str(p), 0))
                elif l == 1: f += 1; pairs.append((str(p), 1))
        print(f"  {droot.name:<45} real={r:>7,}  fake={f:>7,}")
    if limit: random.shuffle(pairs); pairs = pairs[:limit]
    return pairs

print("\n" + "="*65)
print("  AudioGuard RawNet3 — Phase 3c PGD-Curriculum Hardening")
print("="*65)
print(f"Device      : {DEVICE}  |  AMP: {AMP_ON}")
print(f"FGSM ε={cfg.fgsm_eps}  PGD ε={cfg.pgd_eps}/{cfg.pgd_steps}-step  "
      f"F-SAT ε={cfg.fsat_eps}  adv_ratio={cfg.adv_ratio}  "
      f"spec_weight={cfg.spec_weight}")

print("\nScanning data (cap 80k)...")
orig_pairs = scan_dir(ORIG_DIR, limit=80000)
mixed = orig_pairs[:]
random.shuffle(mixed)
print(f"Mixed   : {len(mixed):,} total samples")

def strat_split(pairs, dev_f=0.10, ev_f=0.10):
    real = [p for p in pairs if p[1]==0]
    fake = [p for p in pairs if p[1]==1]
    random.shuffle(real); random.shuffle(fake)
    def sp(lst):
        nd = max(int(len(lst)*dev_f), 30); ne = max(int(len(lst)*ev_f), 30)
        return lst[nd+ne:], lst[:nd], lst[nd:nd+ne]
    rt, rd, re_ = sp(real); ft, fd, fe = sp(fake)
    return rt+ft, rd+fd, re_+fe

train_pairs, dev_pairs, eval_pairs = strat_split(mixed)
dev_set  = {p for p,_ in dev_pairs}
eval_set = {p for p,_ in eval_pairs}
train_pairs = [(p,l) for p,l in train_pairs if p not in dev_set and p not in eval_set]
print(f"Split   : Train {len(train_pairs):,} | Dev {len(dev_pairs):,} | Eval {len(eval_pairs):,}")

# ── Dataset ───────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, pairs, augment_fn=None):
        self.pairs = pairs; self.aug = augment_fn
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        w = load_waveform(path)
        if w is None: w = np.zeros(cfg.n_samples, dtype=np.float32)
        if self.aug: w = self.aug(w)
        return torch.tensor(w).unsqueeze(0), torch.tensor(float(label))

def make_sampler(pairs):
    lbls = [l for _,l in pairs]
    n0 = max(lbls.count(0), 1); n1 = max(lbls.count(1), 1)
    w  = torch.DoubleTensor([1/n0 if l==0 else 1/n1 for l in lbls])
    return WeightedRandomSampler(w, len(w), replacement=True)

train_loader = DataLoader(AudioDataset(train_pairs, augment),
    batch_size=cfg.batch_size, sampler=make_sampler(train_pairs),
    num_workers=cfg.num_workers, pin_memory=True)
dev_loader   = DataLoader(AudioDataset(dev_pairs),
    batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True)
eval_loader  = DataLoader(AudioDataset(eval_pairs),
    batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True)

# ── Evaluation ────────────────────────────────────────────────────────
def evaluate(loader, attack_fn=None):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if attack_fn is not None:
            x = attack_fn(model, x, y)
        with torch.no_grad():
            with autocast(enabled=AMP_ON):
                prob = torch.sigmoid(model(x))
        ps.append(prob.cpu().numpy()); ys.append(y.cpu().numpy())
    yt = np.concatenate(ys); yp = np.concatenate(ps)
    pred = (yp >= 0.5).astype(int)
    auc  = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
    return {'auc': auc,
            'acc': float(accuracy_score(yt, pred)),
            'f1':  float(f1_score(yt, pred, zero_division=0))}

# ── Loss, optimizer, scheduler ───────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-8)
scaler    = GradScaler(enabled=AMP_ON)

# ── Training loop ─────────────────────────────────────────────────────
best_auc = 0.0; no_imp = 0
log_path = LOG_DIR / 'phase3c_log.csv'
with open(log_path, 'w') as f:
    f.write('epoch,loss,dev_auc,dev_acc,dev_f1,loss_clean,loss_adv,loss_spec\n')

print("\n" + "="*65)
print("  Training")
print("="*65)

for epoch in range(1, cfg.epochs + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = loss_clean_sum = loss_adv_sum = loss_spec_sum = 0.0

    for step, (x, y) in enumerate(train_loader, 1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # ── Forward on clean batch ────────────────────────────────────
        with autocast(enabled=AMP_ON):
            logits_clean = model(x)
            loss_clean   = criterion(logits_clean, y)

        # ── Build adversarial batch (float32 — attacks need full precision grads) ──
        model.eval()
        x_adv, y_adv = mixed_adversarial_batch(model, x, y, ratio=cfg.adv_ratio)
        model.train()

        # ── Forward on adversarial batch + spectral regularizer ──────
        with autocast(enabled=AMP_ON):
            logits_adv = model(x_adv)
            loss_adv   = criterion(logits_adv, y_adv)
            loss_spec  = waveform_consistency_loss(x_adv)

        # ── Combined loss ─────────────────────────────────────────────
        loss = loss_clean + cfg.fsat_gamma * loss_adv + cfg.spec_weight * loss_spec

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss     += loss.item()
        loss_clean_sum += loss_clean.item()
        loss_adv_sum   += loss_adv.item()
        loss_spec_sum  += loss_spec.item()

        if step % 50 == 0:
            print(f"  Ep{epoch} step {step}/{len(train_loader)} | "
                  f"loss {loss.item():.4f} "
                  f"(clean {loss_clean.item():.4f}  "
                  f"adv {loss_adv.item():.4f}  "
                  f"spec {loss_spec.item():.4f})")

    scheduler.step()
    n_steps  = max(len(train_loader), 1)
    avg_loss = total_loss / n_steps

    m    = evaluate(dev_loader)
    flag = ''
    if m['auc'] > best_auc:
        best_auc = m['auc']; no_imp = 0
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dev_auc':             m['auc'],
            'dev_acc':             m['acc'],
            'phase':               '3b',
            'config': {
                'sr': cfg.sr, 'n_samples': cfg.n_samples,
                'fgsm_eps': cfg.fgsm_eps, 'pgd_eps': cfg.pgd_eps,
                'pgd_steps': cfg.pgd_steps, 'fsat_eps': cfg.fsat_eps,
                'f_low': cfg.f_low, 'f_high': cfg.f_high,
                'spec_weight': cfg.spec_weight,
            }
        }, OUT_CKPT)
        flag = '  ✓ saved'
    else:
        no_imp += 1

    print(f"Ep {epoch:02d}/{cfg.epochs} | Loss {avg_loss:.4f} | "
          f"AUC {m['auc']:.4f} | Acc {m['acc']:.4f} | F1 {m['f1']:.4f}{flag}")

    with open(log_path, 'a') as f:
        f.write(f"{epoch},{avg_loss:.4f},{m['auc']:.4f},{m['acc']:.4f},"
                f"{m['f1']:.4f},{loss_clean_sum/n_steps:.4f},"
                f"{loss_adv_sum/n_steps:.4f},{loss_spec_sum/n_steps:.4f}\n")

    if no_imp >= cfg.patience:
        print(f"\nEarly stopping at epoch {epoch}."); break

print(f"\nBest Phase 3c Dev AUC : {best_auc:.4f}")

# ── Final robustness evaluation ───────────────────────────────────────
ckpt = torch.load(OUT_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

print("\n" + "="*65)
print("  PHASE 3c ROBUSTNESS EVALUATION")
print("="*65)

m_clean = evaluate(eval_loader)
print(f"  Clean      — AUC {m_clean['auc']:.4f}  Acc {m_clean['acc']:.4f}  F1 {m_clean['f1']:.4f}")

m_fgsm = evaluate(eval_loader,
    attack_fn=lambda mdl, x, y: fgsm_attack(mdl, x, y, eps=cfg.fgsm_eps))
print(f"  FGSM       — AUC {m_fgsm['auc']:.4f}  Acc {m_fgsm['acc']:.4f}  F1 {m_fgsm['f1']:.4f}")

m_pgd = evaluate(eval_loader,
    attack_fn=lambda mdl, x, y: pgd_attack(mdl, x, y,
                                            eps=cfg.pgd_eps,
                                            alpha=cfg.pgd_alpha,
                                            steps=cfg.pgd_steps))
print(f"  PGD-{cfg.pgd_steps}      — AUC {m_pgd['auc']:.4f}  Acc {m_pgd['acc']:.4f}  F1 {m_pgd['f1']:.4f}")

m_fsat = evaluate(eval_loader,
    attack_fn=lambda mdl, x, y: fsat_attack(mdl, x, y, eps=cfg.fsat_eps))
print(f"  F-SAT      — AUC {m_fsat['auc']:.4f}  Acc {m_fsat['acc']:.4f}  F1 {m_fsat['f1']:.4f}")

results = {
    'phase': '3b',
    'clean': m_clean, 'fgsm': m_fgsm,
    'pgd':   m_pgd,   'fsat': m_fsat
}
out_json = LOG_DIR / 'phase3c_metrics.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nCheckpoint : {OUT_CKPT}")
print(f"Metrics    : {out_json}")
print(f"Log        : {log_path}")
