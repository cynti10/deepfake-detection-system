#!/usr/bin/env python3
"""
AudioGuard RawNet3-FSAT — Phase 2 Fine-Tune
70% original DeepFakeVox-HQ + 30% FoR hard data
Low LR + frozen sinc/encoder bottom layers
"""
import re, json, warnings, random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings('ignore')

@dataclass
class CFG:
    sr:           int   = 16000
    n_samples:    int   = 64000
    batch_size:   int   = 64
    epochs:       int   = 8
    patience:     int   = 3
    lr:           float = 2e-5       # 10x lower than original
    weight_decay: float = 1e-4
    num_workers:  int   = 8
    amp:          bool  = True
    seed:         int   = 42
    hard_ratio:   float = 0.40       # 40% hard (FoR) data — bigger shift needed
    freeze_ratio: float = 0.50       # freeze bottom 50% of encoder

cfg = CFG()
random.seed(cfg.seed); np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ON   = cfg.amp and DEVICE.type == 'cuda'

BASE_DIR  = Path.home() / 'deepfake_project/audio'
ORIG_DIR  = BASE_DIR / 'data'
HARD_DIR  = BASE_DIR / 'test_data'
CKPT_DIR  = BASE_DIR / 'checkpoints'
LOG_DIR   = BASE_DIR / 'logs'
ORIG_CKPT = CKPT_DIR / 'rawnet3_fsat_best.pt'
FINE_CKPT = CKPT_DIR / 'rawnet3_fsat_finetuned.pt'
for d in [CKPT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
REAL_TOKENS = {'real','genuine','authentic','bonafide','original',
               'true','live','lj','ljs','real_audio'}
FAKE_TOKENS = {'fake','spoof','synthetic','generated','tts','deepfake',
               'ai','cloned','vocoder','melgan','hifigan','waveglow',
               'parallel','wavegan','fastspeech','conformer','diffwave',
               'wavernn','ljspeech','jsut','multiband','fullband',
               'generated_audio','synthesis'}

def norm(s): return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
def infer_label(path):
    for part in reversed(Path(path).parts):
        t = set(norm(part).split())
        if t & REAL_TOKENS: return 0
        if t & FAKE_TOKENS: return 1
    return None

def load_waveform(path):
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 100: return None
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != cfg.sr: wav = T.Resample(sr, cfg.sr)(wav)
        wav = wav.squeeze(0).numpy()
        if len(wav) < cfg.n_samples:
            wav = np.pad(wav, (0, cfg.n_samples - len(wav)))
        else:
            wav = wav[:cfg.n_samples]
        pk = np.abs(wav).max()
        if pk > 1e-6: wav = wav / pk
        return wav.astype(np.float32)
    except: return None

def scan_dir(root, only_set=None, skip_set=None, limit=None):
    pairs = []
    for droot in sorted(root.iterdir()):
        if not droot.is_dir(): continue
        if only_set  and droot.name not in only_set:  continue
        if skip_set  and droot.name in skip_set:      continue
        r, f = 0, 0
        for p in droot.rglob('*'):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                if p.stat().st_size < 100: continue
                l = infer_label(p)
                if l == 0: r += 1; pairs.append((str(p), 0))
                elif l == 1: f += 1; pairs.append((str(p), 1))
        print(f"  {droot.name:<45} real={r:>7,}  fake={f:>7,}")
    if limit: random.shuffle(pairs); pairs = pairs[:limit]
    return pairs

print("="*65)
print("  AudioGuard RawNet3-FSAT — Phase 2 Fine-Tune")
print("="*65)
print(f"Device : {DEVICE}")

print("\nScanning original training data (sample 80k)...")
orig_pairs = scan_dir(ORIG_DIR, limit=80000)

print("\nScanning hard/FoR datasets...")
hard_pairs = scan_dir(HARD_DIR, only_set={'for_dataset','realvsfake_voice','wavefake'})

print(f"\nOriginal pool : {len(orig_pairs):,}")
print(f"Hard pool     : {len(hard_pairs):,}")

# 70/30 mix
n_hard    = len(hard_pairs)
n_orig    = int(n_hard * (1 - cfg.hard_ratio) / cfg.hard_ratio)
n_orig    = min(n_orig, len(orig_pairs))
orig_samp = random.sample(orig_pairs, n_orig)
mixed     = orig_samp + hard_pairs
random.shuffle(mixed)
print(f"Mixed train   : {n_orig:,} original + {n_hard:,} hard = {len(mixed):,}")

# Stratified split
def strat_split(pairs, dev_f=0.10, ev_f=0.10):
    real = [p for p in pairs if p[1]==0]
    fake = [p for p in pairs if p[1]==1]
    random.shuffle(real); random.shuffle(fake)
    def sp(lst):
        nd=max(int(len(lst)*dev_f),30); ne=max(int(len(lst)*ev_f),30)
        return lst[nd+ne:], lst[:nd], lst[nd:nd+ne]
    rt,rd,re_=sp(real); ft,fd,fe=sp(fake)
    return rt+ft, rd+fd, re_+fe

train_pairs, dev_pairs, eval_pairs = strat_split(mixed)
dev_set   = set(p for p,_ in dev_pairs)
eval_set  = set(p for p,_ in eval_pairs)
train_pairs = [(p,l) for p,l in train_pairs
               if p not in dev_set and p not in eval_set]
print(f"Split         : Train {len(train_pairs):,} | Dev {len(dev_pairs):,} | Eval {len(eval_pairs):,}")


# ── Augmentation ──────────────────────────────────────────────────────
def augment(wav):
    # Random gain
    gain = random.uniform(0.7, 1.0)
    wav = wav * gain
    # Additive noise
    if random.random() < 0.4:
        noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
        wav = wav + noise
    # Random crop + re-pad (simulates variable length)
    if random.random() < 0.3:
        start = random.randint(0, cfg.n_samples // 8)
        wav = np.concatenate([wav[start:], np.zeros(start, dtype=np.float32)])
    pk = np.abs(wav).max()
    if pk > 1e-6: wav = wav / pk
    return wav


# ── Architecture ──────────────────────────────────────────────────────
class SincConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size=125
        self.low_hz_  = nn.Parameter(torch.full((128,1),4000/cfg.sr))
        self.band_hz_ = nn.Parameter(torch.full((128,1),4000/cfg.sr))
        n = torch.arange(-62, 63, dtype=torch.float32)
        self.register_buffer('n_',      n.view(1,-1))
        self.register_buffer('window_', torch.hamming_window(125))
    @staticmethod
    def sinc(x):
        x=torch.where(x==0,torch.full_like(x,1e-6),x)
        return torch.sin(np.pi*x)/(np.pi*x)
    def forward(self,x):
        low=torch.abs(self.low_hz_); high=low+torch.abs(self.band_hz_)+1e-6
        f=2*(self.sinc(2*high*self.n_)-self.sinc(2*low*self.n_))*self.window_
        f=f/(2*f.abs().sum(dim=1,keepdim=True)+1e-6)
        return F.conv1d(x,f.unsqueeze(1),padding=62)

class ResBlock(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(ic,oc,3,padding=1,bias=False),nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.1,True),nn.Conv1d(oc,oc,3,padding=1,bias=False),nn.BatchNorm1d(oc))
        self.skip=nn.Sequential(nn.Conv1d(ic,oc,1,bias=False),nn.BatchNorm1d(oc))
        self.pool=nn.MaxPool1d(3)
    def forward(self,x): return self.pool(F.leaky_relu(self.conv(x)+self.skip(x),0.1))

class ASP(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.Sequential(nn.Conv1d(256,128,1),nn.Tanh(),nn.Conv1d(128,256,1))
    def forward(self,x):
        w=torch.softmax(self.attn(x),dim=2); mu=(w*x).sum(2)
        std=((w*(x**2)).sum(2)-mu**2).clamp(1e-6).sqrt()
        return torch.cat([mu,std],1)

class RawNet3FSAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc=SincConv(); self.bn0=nn.BatchNorm1d(128)
        self.encoder=nn.Sequential(ResBlock(128,128),ResBlock(128,256),
            ResBlock(256,256),ResBlock(256,256),ResBlock(256,256))
        self.asp=ASP(); self.bn_asp=nn.BatchNorm1d(512)
        self.classifier=nn.Sequential(nn.Linear(512,256),nn.LeakyReLU(0.1,True),
            nn.Dropout(0.3),nn.Linear(256,1))
    def forward(self,x):
        x=self.sinc(x); x=F.leaky_relu(self.bn0(torch.abs(x)),0.1)
        x=self.encoder(x); x=self.asp(x); x=self.bn_asp(x)
        return self.classifier(x).squeeze(1)


model=RawNet3FSAT().to(DEVICE)
ckpt=torch.load(ORIG_CKPT,map_location=DEVICE,weights_only=False)
model.load_state_dict(ckpt['model_state_dict'],strict=True)
print(f"\nLoaded Phase 1 — AUC {ckpt['dev_auc']:.4f}")

# Freeze bottom 50% of encoder
enc_params = list(model.encoder.parameters())
n_freeze   = int(len(enc_params) * cfg.freeze_ratio)
for i, p in enumerate(enc_params):
    p.requires_grad = (i >= n_freeze)
# Also freeze sinc filters — they encode frequency priors we want to keep
for p in model.sinc.parameters():
    p.requires_grad = False

frozen = sum(1 for p in model.parameters() if not p.requires_grad)
total  = sum(1 for p in model.parameters())
print(f"Frozen {frozen}/{total} param groups")


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
    lbls=[l for _,l in pairs]
    n0=max(lbls.count(0),1); n1=max(lbls.count(1),1)
    w=torch.DoubleTensor([1/n0 if l==0 else 1/n1 for l in lbls])
    return WeightedRandomSampler(w,len(w),replacement=True)

train_loader=DataLoader(AudioDataset(train_pairs,augment),
    batch_size=cfg.batch_size,sampler=make_sampler(train_pairs),
    num_workers=cfg.num_workers,pin_memory=True)
dev_loader=DataLoader(AudioDataset(dev_pairs),
    batch_size=cfg.batch_size,shuffle=False,
    num_workers=cfg.num_workers,pin_memory=True)
eval_loader=DataLoader(AudioDataset(eval_pairs),
    batch_size=cfg.batch_size,shuffle=False,
    num_workers=cfg.num_workers,pin_memory=True)


# ── Training ──────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-8)
criterion = nn.BCEWithLogitsLoss()
scaler    = GradScaler(device='cuda', enabled=AMP_ON)

def evaluate(loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x,y in loader:
            x=x.to(DEVICE,non_blocking=True)
            with autocast(device_type=DEVICE.type,enabled=AMP_ON):
                prob=torch.sigmoid(model(x))
            ps.append(prob.cpu().numpy()); ys.append(y.numpy())
    yt=np.concatenate(ys); yp=np.concatenate(ps); ypred=(yp>=0.5).astype(int)
    auc=float(roc_auc_score(yt,yp)) if len(np.unique(yt))>1 else 0.5
    return {'auc':auc,'acc':float(accuracy_score(yt,ypred)),
            'f1':float(f1_score(yt,ypred,zero_division=0))}

best_auc=0.0; no_imp=0
log_path=LOG_DIR/'finetune_audio_log.csv'
with open(log_path,'w') as f: f.write('epoch,loss,dev_auc,dev_acc,dev_f1\n')

print("\n"+"="*65)
print("  Training")
print("="*65)

for epoch in range(1,cfg.epochs+1):
    model.train(); optimizer.zero_grad(set_to_none=True)
    total_loss=0.0
    for step,(x,y) in enumerate(train_loader,1):
        x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
        with autocast(device_type=DEVICE.type,enabled=AMP_ON):
            loss=criterion(model(x),y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total_loss+=loss.item()
    scheduler.step()
    avg_loss=total_loss/max(len(train_loader),1)
    m=evaluate(dev_loader)
    flag=''
    if m['auc']>best_auc:
        best_auc=m['auc']; no_imp=0
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'dev_auc':m['auc'],'dev_acc':m['acc'],'phase':2},FINE_CKPT)
        flag='  ✓ saved'
    else: no_imp+=1
    print(f"Ep {epoch:02d}/{cfg.epochs} | Loss {avg_loss:.4f} | "
          f"AUC {m['auc']:.4f} | Acc {m['acc']:.4f} | F1 {m['f1']:.4f}{flag}")
    with open(log_path,'a') as f:
        f.write(f"{epoch},{avg_loss:.4f},{m['auc']:.4f},{m['acc']:.4f},{m['f1']:.4f}\n")
    if no_imp>=cfg.patience:
        print(f"\nEarly stopping at epoch {epoch}."); break

print(f"\nBest Fine-Tune Dev AUC : {best_auc:.4f}")

ckpt=torch.load(FINE_CKPT,map_location=DEVICE,weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
em=evaluate(eval_loader)
print("\n"+"="*55)
print("  PHASE 2 FINAL EVAL")
print("="*55)
print(f"  AUC       : {em['auc']:.4f}")
print(f"  Accuracy  : {em['acc']:.4f}")
print(f"  F1        : {em['f1']:.4f}")
out=LOG_DIR/'finetune_audio_metrics.json'
with open(out,'w') as f: json.dump(em,f,indent=2)
print(f"\nCheckpoint : {FINE_CKPT}")
print(f"Metrics    : {out}")
