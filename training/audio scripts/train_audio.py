#!/usr/bin/env python3
import os, sys, random, math, glob, csv, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
warnings.filterwarnings('ignore')

SEED=42; SR=16000; DURATION=4.0; N_SAMPLES=int(SR*DURATION)
BATCH_SIZE=24; EPOCHS=50; LR=1e-4; WEIGHT_DECAY=1e-4; PATIENCE=10
GAMMA=0.1; EPSILON=0.01; F_LOW=4000; F_HIGH=8000; N_FFT=1024

BASE_DIR=os.path.expanduser('~/deepfake_project/audio')
DATA_DIR=os.path.join(BASE_DIR,'data')
CKPT_DIR=os.path.join(BASE_DIR,'checkpoints')
LOG_DIR=os.path.join(BASE_DIR,'logs')
CKPT_PATH=os.path.join(CKPT_DIR,'rawnet3_fsat_best.pt')
for d in [CKPT_DIR,LOG_DIR]: os.makedirs(d,exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS=torch.cuda.device_count()
print(f"Device  : {DEVICE}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Parsers ────────────────────────────────────────────────────────
def parse_asvspoof2019(root):
    pairs=[]
    for audio_subdir,proto_file in [
        ('LA/LA/ASVspoof2019_LA_train/flac',
         'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        ('LA/LA/ASVspoof2019_LA_dev/flac',
         'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
    ]:
        audio_dir=os.path.join(root,audio_subdir)
        proto=os.path.join(root,proto_file)
        if not os.path.exists(proto):
            print(f"  [SKIP] {proto}"); continue
        with open(proto) as f:
            for line in f:
                parts=line.strip().split()
                if len(parts)<5: continue
                file_id,label_str=parts[1],parts[4]
                if label_str not in ('bonafide','spoof'): continue
                path=os.path.join(audio_dir,f'{file_id}.flac')
                if os.path.exists(path):
                    pairs.append((path,0 if label_str=='bonafide' else 1))
    print(f"  ASVspoof 2019 LA : {len(pairs):,}"); return pairs

def parse_wavefake(root):
    pairs=[]
    real_dir=os.path.join(root,'KAGGLE','AUDIO','REAL')
    fake_dir=os.path.join(root,'KAGGLE','AUDIO','FAKE')
    for ext in ('*.wav','*.flac','*.mp3'):
        for p in glob.glob(os.path.join(real_dir,'**',ext),recursive=True):
            pairs.append((p,0))
        for p in glob.glob(os.path.join(fake_dir,'**',ext),recursive=True):
            pairs.append((p,1))
    print(f"  WaveFake         : {len(pairs):,}"); return pairs

def parse_in_the_wild(root):
    pairs=[]
    meta_files=glob.glob(os.path.join(root,'**','meta.csv'),recursive=True)
    if not meta_files:
        print("  [SKIP] In-the-Wild meta.csv not found"); return pairs
    meta_path=meta_files[0]; audio_root=os.path.dirname(meta_path)
    with open(meta_path) as f:
        reader=csv.DictReader(f)
        for row in reader:
            fname=row.get('file',row.get('filename',''))
            label_str=row.get('label',row.get('type',''))
            path=os.path.join(audio_root,fname)
            if os.path.exists(path):
                pairs.append((path,0 if 'bona' in label_str.lower() else 1))
    print(f"  In-the-Wild      : {len(pairs):,}"); return pairs

print("\nParsing datasets...")
asv_pairs=parse_asvspoof2019(os.path.join(DATA_DIR,'asvspoof2019'))
wf_pairs=parse_wavefake(os.path.join(DATA_DIR,'wavefake'))
itw_pairs=parse_in_the_wild(os.path.join(DATA_DIR,'in_the_wild'))
all_pairs=asv_pairs+wf_pairs+itw_pairs
random.shuffle(all_pairs)
real_n=sum(1 for _,l in all_pairs if l==0)
fake_n=sum(1 for _,l in all_pairs if l==1)
print(f"Total : {len(all_pairs):,} | Real: {real_n:,} | Fake: {fake_n:,}")

paths=[p for p,_ in all_pairs]; labels=[l for _,l in all_pairs]
tr_p,tmp_p,tr_l,tmp_l=train_test_split(paths,labels,test_size=0.20,
                                        stratify=labels,random_state=SEED)
dv_p,ev_p,dv_l,ev_l=train_test_split(tmp_p,tmp_l,test_size=0.50,
                                      stratify=tmp_l,random_state=SEED)
train_pairs=list(zip(tr_p,tr_l))
dev_pairs=list(zip(dv_p,dv_l))
eval_pairs=list(zip(ev_p,ev_l))
print(f"Split — Train:{len(train_pairs):,} Dev:{len(dev_pairs):,} Eval:{len(eval_pairs):,}")

# ── Augmentation ───────────────────────────────────────────────────
class AudioRandAugment:
    def __init__(self,N=2,p=0.5,sr=SR):
        self.N=N; self.p=p; self.sr=sr
        self.augmentations=[
    self._gaussian_noise,self._background_noise,
    # self._time_stretch,   # disabled — librosa too slow
    # self._pitch_shift,    # disabled — librosa too slow
    self._time_mask,self._freq_mask,
    self._lowpass,self._highpass,
    self._reverb,self._compression,
    self._eq,self._gain_transition,
    self._bit_crush,self._aliasing,
        ]
    def __call__(self,x):
        if x.dim()==1: x=x.unsqueeze(0)
        for aug in random.sample(self.augmentations,min(self.N,len(self.augmentations))):
            if random.random()<self.p:
                try: x=aug(x)
                except: pass
        return x.squeeze(0)
    def _gaussian_noise(self,x):
        snr=random.uniform(10,40); sp=x.pow(2).mean().clamp(min=1e-10)
        return x+torch.randn_like(x)*(sp/(10**(snr/10))).sqrt()
    def _background_noise(self,x):
        return (x+torch.randn_like(x)*random.uniform(0.001,0.05)).clamp(-1,1)
    def _time_stretch(self,x):
        rate=random.uniform(0.8,1.2); orig=x.shape[-1]
        s=librosa.effects.time_stretch(x.numpy().squeeze(),rate=rate)
        s=torch.FloatTensor(s).unsqueeze(0)
        return s[...,:orig] if s.shape[-1]>orig else F.pad(s,(0,orig-s.shape[-1]))
    def _pitch_shift(self,x):
        s=librosa.effects.pitch_shift(x.numpy().squeeze(),sr=self.sr,
                                       n_steps=random.uniform(-3,3))
        return torch.FloatTensor(s).unsqueeze(0)
    def _time_mask(self,x):
        T=x.shape[-1]; ml=random.randint(int(T*.05),int(T*.15))
        st=random.randint(0,T-ml); x=x.clone(); x[...,st:st+ml]=0; return x
    def _freq_mask(self,x):
        stft=torch.stft(x.squeeze(),n_fft=512,return_complex=True)
        F_=stft.shape[0]; ml=random.randint(1,F_//8); st=random.randint(0,F_-ml)
        stft=stft.clone(); stft[st:st+ml,:]=0
        return torch.istft(stft,n_fft=512,length=x.shape[-1]).unsqueeze(0)
    def _lowpass(self,x):
        return AF.lowpass_biquad(x,self.sr,random.uniform(2000,6000))
    def _highpass(self,x):
        return AF.highpass_biquad(x,self.sr,random.uniform(100,500))
    def _reverb(self,x):
        d=random.randint(int(self.sr*.01),int(self.sr*.05))
        e=F.pad(x,(d,0))[...,:x.shape[-1]]
        return (x+random.uniform(.2,.5)*e).clamp(-1,1)
    def _compression(self,x):
        mu=random.choice([100,255])
        return AF.mu_law_encoding(x.clamp(-1,1),mu).float()/mu
    def _eq(self,x):
        return AF.equalizer_biquad(x,self.sr,random.uniform(300,4000),
                                    random.uniform(-6,6),random.uniform(.5,2))
    def _gain_transition(self,x):
        r=torch.linspace(random.uniform(.5,1.5),random.uniform(.5,1.5),x.shape[-1])
        return (x*r).clamp(-1,1)
    def _bit_crush(self,x):
        s=2**random.choice([8,12,16]); return (x*s).round()/s
    def _aliasing(self,x):
        f=random.choice([2,4]); d=x[...,::f]
        return F.interpolate(d.unsqueeze(0),size=x.shape[-1],
                             mode='linear',align_corners=False).squeeze(0)

# ── Dataset ────────────────────────────────────────────────────────
class AudioDeepfakeDataset(Dataset):
    def __init__(self,pairs,augment=False):
        self.pairs=pairs; self.augmentor=AudioRandAugment() if augment else None
    def __len__(self): return len(self.pairs)
    def _load(self,path):
        try:
            wav,sr=torchaudio.load(path)
        except:
            return torch.zeros(N_SAMPLES)
        if wav.shape[0]>1: wav=wav.mean(0,keepdim=True)
        if sr!=SR: wav=T.Resample(sr,SR)(wav)
        wav=wav.squeeze(0)
        if wav.shape[0]>=N_SAMPLES:
            st=random.randint(0,wav.shape[0]-N_SAMPLES) if self.augmentor else 0
            wav=wav[st:st+N_SAMPLES]
        else:
            wav=F.pad(wav,(0,N_SAMPLES-wav.shape[0]))
        return wav
    def __getitem__(self,idx):
        path,label=self.pairs[idx]; wav=self._load(path)
        if self.augmentor: wav=self.augmentor(wav.unsqueeze(0)).squeeze(0)
        wav=wav/(wav.abs().max()+1e-8)
        return wav,torch.tensor(label,dtype=torch.float32)

def build_sampler(pairs):
    lbls=[l for _,l in pairs]; cw=[1.0/lbls.count(0),1.0/lbls.count(1)]
    sw=torch.DoubleTensor([cw[l] for l in lbls])
    return WeightedRandomSampler(sw,len(sw),replacement=True)

train_ds=AudioDeepfakeDataset(train_pairs,augment=True)
dev_ds=AudioDeepfakeDataset(dev_pairs)
eval_ds=AudioDeepfakeDataset(eval_pairs)
sampler=build_sampler(train_pairs)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,sampler=sampler,
                        num_workers=8,pin_memory=True,prefetch_factor=2)
dev_loader=DataLoader(dev_ds,batch_size=BATCH_SIZE,shuffle=False,
                      num_workers=8,pin_memory=True)
eval_loader=DataLoader(eval_ds,batch_size=BATCH_SIZE,shuffle=False,
                       num_workers=8,pin_memory=True)
print(f"Loaders ready — Train:{len(train_loader)} Dev:{len(dev_loader)}")

# ── Model ──────────────────────────────────────────────────────────
class SincConv(nn.Module):
    def __init__(self,out_channels,kernel_size,sr=SR,min_low_hz=50,min_band_hz=50):
        super().__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size if kernel_size%2!=0 else kernel_size+1
        self.sr=sr; self.min_low_hz=min_low_hz; self.min_band_hz=min_band_hz
        low_hz=30.0; high_hz=sr/2-(min_low_hz+min_band_hz)
        mel=np.linspace(self._hz2mel(low_hz),self._hz2mel(high_hz),out_channels+1)
        hz=self._mel2hz(mel)
        self.low_hz_=nn.Parameter(torch.FloatTensor(hz[:-1]).view(-1,1))
        self.band_hz_=nn.Parameter(torch.FloatTensor(np.diff(hz)).view(-1,1))
        n=(self.kernel_size-1)//2   # integer half-length
        # n_ has shape (1, n): indices from -n to -1
        self.register_buffer('n_',
            torch.arange(1,n+1).float().view(1,-1)/sr)
        # window matches left half length n
        self.register_buffer('window_',
            torch.hamming_window(2*n+1)[:n])
    @staticmethod
    def _hz2mel(hz): return 2595*np.log10(1+hz/700)
    @staticmethod
    def _mel2hz(mel): return 700*(10**(mel/2595)-1)
    def forward(self,x):
        low=self.min_low_hz+torch.abs(self.low_hz_)
        high=torch.clamp(low+self.min_band_hz+torch.abs(self.band_hz_),
                         self.min_low_hz,self.sr/2)
        band=(high-low)[:,0]
        # n_ shape: (1, n) — broadcast with low/high (out_ch, 1)
        ftl=torch.matmul(low,self.n_)   # (out_ch, n)
        fth=torch.matmul(high,self.n_)  # (out_ch, n)
        bp_left=((torch.sin(2*math.pi*fth)-torch.sin(2*math.pi*ftl))
                 /(math.pi*self.n_))*self.window_   # (out_ch, n)
        bp_center=2*band.view(-1,1)                 # (out_ch, 1)
        bp_right=torch.flip(bp_left,dims=[1])       # (out_ch, n)
        bp=torch.cat([bp_left,bp_center,bp_right],dim=1)  # (out_ch, 2n+1)
        bp=bp/(2*band.unsqueeze(1))
        filters=bp.view(self.out_channels,1,self.kernel_size)
        return F.conv1d(x,filters,None,stride=1,
                        padding=self.kernel_size//2)

class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(in_ch,out_ch,kernel_size,stride=stride,
                      padding=kernel_size//2,bias=False),
            nn.BatchNorm1d(out_ch),nn.LeakyReLU(0.3),
            nn.Conv1d(out_ch,out_ch,kernel_size,padding=kernel_size//2,bias=False),
            nn.BatchNorm1d(out_ch))
        self.skip=(nn.Sequential(nn.Conv1d(in_ch,out_ch,1,stride=stride,bias=False),
                                  nn.BatchNorm1d(out_ch))
                   if in_ch!=out_ch or stride!=1 else nn.Identity())
        self.act=nn.LeakyReLU(0.3)
    def forward(self,x): return self.act(self.conv(x)+self.skip(x))

class AttentiveStatsPooling(nn.Module):
    def __init__(self,in_dim,bottleneck=128):
        super().__init__()
        self.attn=nn.Sequential(nn.Conv1d(in_dim,bottleneck,1),nn.Tanh(),
                                 nn.Conv1d(bottleneck,in_dim,1),nn.Softmax(dim=-1))
    def forward(self,x):
        w=self.attn(x); mean=(w*x).sum(-1)
        std=((w*x.pow(2)).sum(-1)-mean.pow(2)).clamp(min=1e-8).sqrt()
        return torch.cat([mean,std],dim=1)

class RawNet3(nn.Module):
    def __init__(self,sinc_filters=128,sinc_kernel=251,
                 channels=[128,256,256,256,256]):
        super().__init__()
        self.sinc=SincConv(sinc_filters,sinc_kernel)
        self.bn0=nn.BatchNorm1d(sinc_filters)
        self.act0=nn.LeakyReLU(0.3)
        layers=[]; in_ch=sinc_filters
        for out_ch,stride in zip(channels,[2,2,2,2,2]):
            layers.append(ResBlock(in_ch,out_ch,stride=stride)); in_ch=out_ch
        self.encoder=nn.Sequential(*layers)
        self.asp=AttentiveStatsPooling(channels[-1])
        self.bn_asp=nn.BatchNorm1d(channels[-1]*2)
        self.classifier=nn.Sequential(
            nn.Linear(channels[-1]*2,256),nn.LeakyReLU(0.3),
            nn.Dropout(0.5),nn.Linear(256,1))
    def forward(self,x):
        if x.dim()==1: x=x.unsqueeze(0)
        x=x.unsqueeze(1)
        x=self.act0(self.bn0(self.sinc(x)))
        x=self.encoder(x); x=self.bn_asp(self.asp(x))
        return self.classifier(x).squeeze(-1)

model=RawNet3().to(DEVICE)
if NUM_GPUS>1: model=nn.DataParallel(model)
print(f"RawNet3 params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

# ── F-SAT ──────────────────────────────────────────────────────────
def fsat_attack(model,waveform,labels,epsilon=EPSILON,
                f_low=F_LOW,f_high=F_HIGH,n_fft=N_FFT,steps=3,sr=SR):
    alpha=epsilon*2/steps
    r_l=math.floor(f_low*n_fft/sr); r_u=math.ceil(f_high*n_fft/sr)
    adv=(waveform+torch.zeros_like(waveform).uniform_(-epsilon,epsilon)).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        stft=torch.stft(adv.reshape(-1,adv.shape[-1]),n_fft=n_fft,
                        return_complex=True)
        adv_wav=torch.istft(stft.abs()*torch.exp(1j*stft.angle()),
                             n_fft=n_fft,length=waveform.shape[-1])
        loss=F.binary_cross_entropy_with_logits(model(adv_wav),labels)
        loss.backward()
        with torch.no_grad():
            g=adv.grad.detach()
            gs=torch.stft(g.reshape(-1,g.shape[-1]),n_fft=n_fft,return_complex=True)
            mask=torch.zeros_like(gs.abs()); mask[:,r_l:r_u,:]=1.0
            gm=torch.istft(gs*mask,n_fft=n_fft,length=waveform.shape[-1])
            adv=(waveform+torch.clamp(adv+alpha*gm.sign()-waveform,
                                       -epsilon,epsilon)).detach()
    return adv

def fsat_loss(model,waveform,labels):
    lc=F.binary_cross_entropy_with_logits(model(waveform),labels)
    lr=F.binary_cross_entropy_with_logits(
        model(fsat_attack(model,waveform,labels)),labels)
    return lc+GAMMA*lr,lc.item(),lr.item()

# ── Training ────────────────────────────────────────────────────────
optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)

def run_epoch(model,loader,optimizer=None):
    training=optimizer is not None
    model.train() if training else model.eval()
    total_loss=0; all_probs=[]; all_preds=[]; all_true=[]
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for wav,labels in tqdm(loader,leave=False,
                                desc='train' if training else 'eval'):
            wav=wav.to(DEVICE); labels=labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
                loss,lc,lr=fsat_loss(model,wav,labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step(); total_loss+=loss.item()
                logits=model(wav)
            else:
                logits=model(wav)
                total_loss+=F.binary_cross_entropy_with_logits(logits,labels).item()
            probs=torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs>0.5).astype(int))
            all_true.extend(labels.cpu().long().numpy())
    return (total_loss/len(loader),
            accuracy_score(all_true,all_preds),
            roc_auc_score(all_true,all_probs))

# ── Resume from checkpoint ─────────────────────────────────────────
if os.path.exists(CKPT_PATH):
    ckpt=torch.load(CKPT_PATH,map_location=DEVICE,weights_only=False)
    if hasattr(model,'module'): model.module.load_state_dict(ckpt['model_state_dict'])
    else: model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    best_auc=ckpt['dev_auc']
    start_epoch=ckpt['epoch']+1
    print(f"Resumed from epoch {ckpt['epoch']} — best AUC {best_auc:.4f}")
else:
    best_auc=0.0; start_epoch=1
    print("No checkpoint found — training from scratch")
no_improve=0

log_path=os.path.join(LOG_DIR,'training_log.csv')
# Append to existing log, don't overwrite
with open(log_path,'a') as f:
    if start_epoch==1: f.write('epoch,tr_loss,tr_acc,tr_auc,dv_acc,dv_auc\n')

print("\n"+"="*65)
print("  RawNet3 + RandAugment + F-SAT")
print("="*65)

for epoch in range(start_epoch,EPOCHS+1):
    tr_loss,tr_acc,tr_auc=run_epoch(model,train_loader,optimizer)
    _,dv_acc,dv_auc=run_epoch(model,dev_loader)
    scheduler.step()
    with open(log_path,'a') as f:
        f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{tr_auc:.4f},"
                f"{dv_acc:.4f},{dv_auc:.4f}\n")
    flag=''
    if dv_auc>best_auc:
        best_auc=dv_auc; no_improve=0
        torch.save({
            'epoch':epoch,
            'model_state_dict':(model.module.state_dict()
                                if hasattr(model,'module')
                                else model.state_dict()),
            'optimizer_state_dict':optimizer.state_dict(),
            'dev_auc':dv_auc,'dev_acc':dv_acc,
            'config':{'sr':SR,'n_samples':N_SAMPLES,
                      'f_low':F_LOW,'f_high':F_HIGH,
                      'epsilon':EPSILON,'gamma':GAMMA}
        },CKPT_PATH)
        flag=' ✓ saved'
    else:
        no_improve+=1
    print(f"Ep {epoch:02d} | Loss {tr_loss:.4f} | "
          f"Tr AUC {tr_auc:.4f} | Dv AUC {dv_auc:.4f} | "
          f"Dv Acc {dv_acc:.4f}{flag}")
    if no_improve>=PATIENCE:
        print(f"Early stopping at epoch {epoch}."); break

print(f"\nBest Dev AUC: {best_auc:.4f}")

# ── Final Evaluation ────────────────────────────────────────────────
ckpt=torch.load(CKPT_PATH,map_location=DEVICE,weights_only=False)
if hasattr(model,'module'): model.module.load_state_dict(ckpt['model_state_dict'])
else: model.load_state_dict(ckpt['model_state_dict'])
_,ev_acc,ev_auc=run_epoch(model,eval_loader)
ev_preds=[]; ev_true=[]
model.eval()
with torch.no_grad():
    for wav,labels in eval_loader:
        p=(torch.sigmoid(model(wav.to(DEVICE)))>0.5).cpu().long().numpy()
        ev_preds.extend(p); ev_true.extend(labels.long().numpy())
print("\n"+"="*55)
print(f"EVAL Accuracy : {ev_acc:.4f}")
print(f"EVAL ROC-AUC  : {ev_auc:.4f}")
print("="*55)
print(classification_report(ev_true,ev_preds,target_names=['Real','Fake']))
print(f"\nCheckpoint: {CKPT_PATH}")
ENDOFSCRIPT
