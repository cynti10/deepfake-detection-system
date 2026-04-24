#!/usr/bin/env python3
import re, json, warnings, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              precision_score, recall_score, classification_report)
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings('ignore')

SR          = 16000
N_SAMPLES   = 64000
BATCH_SIZE  = 64
NUM_WORKERS = 4
MAX_PER_DS  = 5000
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON      = DEVICE.type == 'cuda'
CKPT_PATH   = Path.home() / 'deepfake_project/audio/checkpoints/rawnet3_fsat_best.pt'
TEST_ROOT   = Path.home() / 'deepfake_project/audio/test_data'
LOG_DIR     = Path.home() / 'deepfake_project/audio/logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

REAL_TOKENS = {
    'real', 'genuine', 'authentic', 'bonafide', 'original',
    'true', 'live', 'lj', 'ljs', 'real_audio',
}
FAKE_TOKENS = {
    'fake', 'spoof', 'synthetic', 'generated', 'tts', 'deepfake',
    'ai', 'cloned', 'vocoder', 'melgan', 'hifigan', 'waveglow',
    'parallel', 'wavegan', 'fastspeech', 'conformer', 'diffwave',
    'wavernn', 'ljspeech', 'jsut', 'multiband', 'fullband',
    'generated_audio', 'synthesis',
}

_load_fails = 0

def norm(s):
    return re.sub(r'[^a-z0-9]+',' ', s.lower()).strip()

def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(norm(part).split())
        if t & REAL_TOKENS: return 0
        if t & FAKE_TOKENS: return 1
    return None

def load_waveform(path):
    global _load_fails
    try:
        p = Path(path)
        if not p.exists() or not p.is_file() or p.stat().st_size < 100:
            _load_fails += 1
            return None
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = T.Resample(sr, SR)(wav)
        wav = wav.squeeze(0).numpy()
        if len(wav) < N_SAMPLES:
            wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
        else:
            wav = wav[:N_SAMPLES]
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak
        return wav.astype(np.float32)
    except Exception as e:
        _load_fails += 1
        return None


# ── Architecture ───────────────────────────────────────────────────────
class SincConv(nn.Module):
    def __init__(self, out_channels=128, kernel_size=125, sr=16000,
                 f_low=4000, f_high=8000):
        super().__init__()
        self.kernel_size = kernel_size
        self.low_hz_  = nn.Parameter(torch.full((out_channels,1), f_low/sr))
        self.band_hz_ = nn.Parameter(torch.full((out_channels,1),(f_high-f_low)/sr))
        n = torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32)
        self.register_buffer('n_',      n.view(1,-1))
        self.register_buffer('window_', torch.hamming_window(kernel_size))

    @staticmethod
    def sinc(x):
        x = torch.where(x==0, torch.full_like(x,1e-6), x)
        return torch.sin(np.pi*x)/(np.pi*x)

    def forward(self, x):
        low  = torch.abs(self.low_hz_)
        high = low + torch.abs(self.band_hz_) + 1e-6
        f    = 2*(self.sinc(2*high*self.n_) - self.sinc(2*low*self.n_)) * self.window_
        f    = f / (2*f.abs().sum(dim=1,keepdim=True) + 1e-6)
        return F.conv1d(x, f.unsqueeze(1), padding=self.kernel_size//2)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        return self.pool(F.leaky_relu(self.conv(x) + self.skip(x), 0.1))


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_ch=256, bottleneck=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_ch, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_ch, 1),
        )

    def forward(self, x):
        w   = torch.softmax(self.attn(x), dim=2)
        mu  = (w * x).sum(dim=2)
        std = ((w * (x**2)).sum(dim=2) - mu**2).clamp(min=1e-6).sqrt()
        return torch.cat([mu, std], dim=1)


class RawNet3FSAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc    = SincConv(128, 125, SR, 4000, 8000)
        self.bn0     = nn.BatchNorm1d(128)
        self.encoder = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.asp        = AttentiveStatsPool(256, 128)
        self.bn_asp     = nn.BatchNorm1d(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.sinc(x)
        x = F.leaky_relu(self.bn0(torch.abs(x)), 0.1)
        x = self.encoder(x)
        x = self.asp(x)
        x = self.bn_asp(x)
        return self.classifier(x).squeeze(1)


# ── Load checkpoint ────────────────────────────────────────────────────
print("="*62)
print("  AudioGuard RawNet3-FSAT — External Test")
print("="*62)
print(f"Device   : {DEVICE}")
print(f"torchaudio: {torchaudio.__version__}")

model = RawNet3FSAT().to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.eval()
print(f"Weights  : strict ✓")
print(f"Epoch    : {ckpt['epoch']}  |  Saved AUC: {ckpt['dev_auc']:.4f}")


# ── Sanity check: 1 file ───────────────────────────────────────────────
print("\nSanity check (1 file):")
for droot in sorted(TEST_ROOT.iterdir()):
    if not droot.is_dir(): continue
    for p in droot.rglob('*'):
        if p.suffix.lower() in AUDIO_EXTS:
            w = load_waveform(str(p))
            if w is None:
                print(f"  FAILED to load {p.name}"); continue
            x = torch.tensor(w).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit = model(x).item()
                prob  = torch.sigmoid(torch.tensor(logit)).item()
            print(f"  File  : {p.name}")
            print(f"  Wave  : min={w.min():.4f}  max={w.max():.4f}  zeros={( w==0).sum()}")
            print(f"  Logit : {logit:.4f}  →  P(fake)={prob:.4f}")
            break
    else: continue
    break


# ── Dataset ────────────────────────────────────────────────────────────
class AudioTestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        w = load_waveform(path)
        if w is None:
            w = np.zeros(N_SAMPLES, dtype=np.float32)
        return torch.tensor(w).unsqueeze(0), torch.tensor(float(label))


# ── Test runner ────────────────────────────────────────────────────────
def run_test(name, pairs):
    real_p = [p for p in pairs if p[1]==0]
    fake_p = [p for p in pairs if p[1]==1]
    n = min(len(real_p), len(fake_p), MAX_PER_DS//2)
    if n == 0:
        print(f"\n  SKIP {name} — 0 real or 0 fake  "
              f"(real={len(real_p)}, fake={len(fake_p)})")
        return None

    sampled = random.sample(real_p,n) + random.sample(fake_p,n)
    random.shuffle(sampled)

    loader = DataLoader(
        AudioTestDataset(sampled),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
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

    auc = float(roc_auc_score(y_true,y_prob)) if len(np.unique(y_true))>1 else 0.5
    acc = float(accuracy_score(y_true,y_pred))
    f1  = float(f1_score(y_true,y_pred,zero_division=0))
    pre = float(precision_score(y_true,y_pred,zero_division=0))
    rec = float(recall_score(y_true,y_pred,zero_division=0))

    status = "🟢" if auc>=0.95 else ("🟡" if auc>=0.90 else "🔴")
    print(f"\n{status} {name}")
    print(f"   Files     : {len(sampled):,}  ({n:,} real + {n:,} fake)")
    print(f"   AUC       : {auc:.4f}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1        : {f1:.4f}")
    print(f"   Precision : {pre:.4f}  |  Recall : {rec:.4f}")
    print(classification_report(y_true, y_pred,
                                 target_names=['Real','Fake'], digits=3))
    return {'dataset':name,'n':len(sampled),'auc':auc,'acc':acc,'f1':f1}


# ── Scan all datasets ──────────────────────────────────────────────────
print("\n" + "="*62)
all_results = []

for droot in sorted(TEST_ROOT.iterdir()):
    if not droot.is_dir(): continue
    pairs = []
    for p in droot.rglob('*'):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            l = infer_label(p)
            if l is not None:
                pairs.append((str(p), l))
    # Pre-filter broken/missing files
    pairs = [(p,l) for p,l in pairs
             if Path(p).exists() and Path(p).is_file()
             and Path(p).stat().st_size >= 100]
    real_n = sum(1 for _,l in pairs if l==0)
    fake_n = sum(1 for _,l in pairs if l==1)
    print(f"  {droot.name:<35} real={real_n:>6,}  fake={fake_n:>6,}")
    r = run_test(droot.name, pairs)
    if r: all_results.append(r)

print(f"\nTotal load failures : {_load_fails}")


# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  SUMMARY — AudioGuard RawNet3-FSAT")
print("="*62)
print(f"  {'Dataset':<30} {'AUC':>7} {'Acc':>7} {'F1':>7}  Status")
print(f"  {'-'*58}")
for r in all_results:
    s = "✅ GO" if r['auc']>=0.95 else ("⚠️  FT" if r['auc']>=0.90 else "🔴 NEED FT")
    print(f"  {r['dataset']:<30} {r['auc']:>7.4f} {r['acc']:>7.4f} {r['f1']:>7.4f}  {s}")
print()
print("  ✅ GO     = proceed to video model")
print("  ⚠️  FT    = optional fine-tune overnight")
print("  🔴 NEED FT = fine-tune before video model")

out = LOG_DIR / 'audio_test_results.json'
with open(out,'w') as f: json.dump(all_results,f,indent=2)
print(f"\nSaved: {out}")

