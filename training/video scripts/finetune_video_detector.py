#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import cv2, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import timm

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_frame_paths(s):
    return [p for p in str(s).split("|") if p]

def load_rgb(path, image_size):
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - np.array([0.485,0.456,0.406],dtype=np.float32)) / np.array([0.229,0.224,0.225],dtype=np.float32)
    return np.transpose(x, (2,0,1))

class VideoSequenceDataset(Dataset):
    def __init__(self, df, seq_len=16, image_size=224, augment=False):
        self.df=df.reset_index(drop=True); self.seq_len=seq_len
        self.image_size=image_size; self.augment=augment
    def __len__(self): return len(self.df)
    def _sample_paths(self, paths):
        if not paths: return []
        if len(paths) >= self.seq_len:
            return [paths[i] for i in np.linspace(0,len(paths)-1,self.seq_len).astype(int)]
        out=list(paths)
        while len(out)<self.seq_len: out.append(paths[-1])
        return out
    def __getitem__(self, idx):
        row=self.df.iloc[idx]; paths=self._sample_paths(parse_frame_paths(row["frame_paths"]))
        frames=[]
        for p in paths:
            x=load_rgb(p,self.image_size)
            if x is None: x=np.zeros((3,self.image_size,self.image_size),dtype=np.float32)
            if self.augment and random.random()<0.5: x=x[:,:,::-1].copy()
            if self.augment and random.random()<0.3: x=np.clip(x+np.random.uniform(-0.1,0.1),-2.5,2.5).astype(np.float32)
            frames.append(x)
        x=torch.tensor(np.stack(frames,axis=0),dtype=torch.float32)
        y=torch.tensor(float(row["label"]),dtype=torch.float32)
        meta={"video_id":str(row["video_id"]),"source_dataset":str(row["source_dataset"]),"sample_id":str(row["sample_id"])}
        return x,y,meta

def rgb_to_fft_mag(x):
    gray=0.2989*x[:,:,0:1]+0.5870*x[:,:,1:2]+0.1140*x[:,:,2:3]
    fft=torch.fft.fft2(gray); mag=torch.log1p(torch.abs(fft))
    mn=mag.amin(dim=(3,4),keepdim=True); mx=mag.amax(dim=(3,4),keepdim=True)
    return (mag-mn)/(mx-mn+1e-6)

class SentinelVideoDetector(nn.Module):
    def __init__(self, rgb_backbone="tf_efficientnet_b3_ns", d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.rgb_backbone=timm.create_model(rgb_backbone,pretrained=True,num_classes=0,global_pool="avg")
        self.fft_backbone=timm.create_model("mobilenetv3_small_050",pretrained=True,in_chans=1,num_classes=0,global_pool="avg")
        with torch.no_grad():
            d_rgb=self.rgb_backbone(torch.zeros(1,3,224,224)).shape[1]
            d_fft=self.fft_backbone(torch.zeros(1,1,224,224)).shape[1]
        self.frame_proj=nn.Linear(d_rgb+d_fft,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*4,dropout=dropout,batch_first=True)
        self.temporal=nn.TransformerEncoder(enc,num_layers=num_layers)
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model))
        self.cls_head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))
    def forward(self,x):
        b,t,c,h,w=x.shape; xr=x.view(b*t,c,h,w); fr=self.rgb_backbone(xr)
        xf=rgb_to_fft_mag(x).view(b*t,1,h,w); ff=self.fft_backbone(xf)
        tok=self.frame_proj(torch.cat([fr,ff],dim=1)).view(b,t,-1)
        cls=self.cls_token.expand(b,-1,-1)
        return self.cls_head(self.temporal(torch.cat([cls,tok],dim=1))[:,0]).squeeze(1)

def freeze_bottom_fraction(backbone, fraction=0.45):
    params=list(backbone.named_parameters()); n=int(len(params)*fraction)
    for _,p in params[:n]: p.requires_grad_(False)
    print(f"Frozen {n}/{len(params)} backbone param tensors (bottom {fraction*100:.0f}%)")

def compute_metrics(y_true, y_prob):
    y_true=np.asarray(y_true); y_prob=np.asarray(y_prob); y_pred=(y_prob>=0.5).astype(np.int32)
    return {"auc":float(roc_auc_score(y_true,y_prob)) if len(np.unique(y_true))>1 else 0.5,
            "acc":float(accuracy_score(y_true,y_pred)),"f1":float(f1_score(y_true,y_pred,zero_division=0)),
            "precision":float(precision_score(y_true,y_pred,zero_division=0)),
            "recall":float(recall_score(y_true,y_pred,zero_division=0))}

def evaluate(model, loader, device, amp_enabled, export_hard=False, hard_threshold=0.75):
    model.eval(); ys,ps,hard=[],[],[]
    with torch.no_grad():
        for x,y,meta in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with autocast(device_type=device.type,enabled=amp_enabled):
                prob=torch.sigmoid(model(x))
            y_cpu=y.cpu().numpy(); p_cpu=prob.cpu().numpy()
            ys.extend(y_cpu.tolist()); ps.extend(p_cpu.tolist())
            if export_hard:
                pred=(p_cpu>=0.5).astype(np.int32); conf=np.where(pred==1,p_cpu,1.0-p_cpu)
                for i in range(len(pred)):
                    if pred[i]!=int(y_cpu[i]) and conf[i]>=hard_threshold:
                        hard.append({"sample_id":meta["sample_id"][i],"video_id":meta["video_id"][i],
                                     "source_dataset":meta["source_dataset"][i],"label":int(y_cpu[i]),
                                     "pred":int(pred[i]),"confidence":float(conf[i])})
    return compute_metrics(ys,ps),hard

def build_mixed_train_df(original_manifest, hard_manifests, hard_ratio=0.30, seed=42):
    orig_df=pd.read_csv(original_manifest)
    hard_dfs=[pd.read_csv(p) for p in hard_manifests if Path(p).exists()]
    if not hard_dfs:
        print("No hard manifests loaded — using original data only."); return orig_df
    hard_df=pd.concat(hard_dfs,ignore_index=True)
    n_orig=min(int(len(hard_df)*(1.0-hard_ratio)/hard_ratio),len(orig_df))
    mixed=pd.concat([orig_df.sample(n=n_orig,random_state=seed),hard_df],ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)
    print(f"Original pool : {len(orig_df)}\nHard pool     : {len(hard_df)}\nMixed train   : {n_orig} original + {len(hard_df)} hard = {len(mixed)} total")
    return mixed

def main():
    parser=argparse.ArgumentParser(description="VideoGuard v1 — Phase 2 Fine-Tune")
    parser.add_argument("--train-manifest",required=True)
    parser.add_argument("--val-manifest",required=True)
    parser.add_argument("--test-manifest",required=True)
    parser.add_argument("--hard-manifests",nargs="*",default=[])
    parser.add_argument("--hard-ratio",type=float,default=0.30)
    parser.add_argument("--phase1-checkpoint",required=True)
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--rgb-backbone",default="tf_efficientnet_b3_ns")
    parser.add_argument("--d-model",type=int,default=512)
    parser.add_argument("--nhead",type=int,default=8)
    parser.add_argument("--num-layers",type=int,default=2)
    parser.add_argument("--epochs",type=int,default=7)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--seq-len",type=int,default=16)
    parser.add_argument("--image-size",type=int,default=224)
    parser.add_argument("--lr",type=float,default=5e-5)
    parser.add_argument("--weight-decay",type=float,default=5e-4)
    parser.add_argument("--grad-accum",type=int,default=2)
    parser.add_argument("--max-grad-norm",type=float,default=1.0)
    parser.add_argument("--patience",type=int,default=4)
    parser.add_argument("--freeze-fraction",type=float,default=0.45)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--amp",action="store_true")
    parser.add_argument("--export-hard-negatives",action="store_true")
    args=parser.parse_args()

    seed_everything(args.seed)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled=bool(args.amp and device.type=="cuda")

    print("="*65+"\n  VideoGuard v1 — Phase 2 Fine-Tune\n"+"="*65)
    print(f"Device     : {device}")
    if device.type=="cuda": print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"LR         : {args.lr}  (Phase 1 was 2e-4)")
    print(f"Hard ratio : {int(args.hard_ratio*100)}% hard + {int((1-args.hard_ratio)*100)}% original")
    print(f"Freeze     : bottom {int(args.freeze_fraction*100)}% of rgb_backbone")

    print("\nScanning original training data...")
    orig_df=pd.read_csv(args.train_manifest)
    for ds,grp in orig_df.groupby("source_dataset"):
        print(f"  {ds:<40} real={(grp['label']==0).sum():>6,}  fake={(grp['label']==1).sum():>6,}")

    mixed_df=build_mixed_train_df(args.train_manifest,args.hard_manifests,args.hard_ratio,args.seed) if args.hard_manifests else orig_df
    val_df=pd.read_csv(args.val_manifest); test_df=pd.read_csv(args.test_manifest)
    print(f"\nSplit  : Train {len(mixed_df):,} | Dev {len(val_df):,} | Eval {len(test_df):,}")

    labels=mixed_df["label"].values; counts=np.bincount(labels.astype(int))
    weights=1.0/counts[labels.astype(int)]
    sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.float32),len(mixed_df),replacement=True)

    pin=(device.type=="cuda")
    train_loader=DataLoader(VideoSequenceDataset(mixed_df,args.seq_len,args.image_size,True),
                            batch_size=args.batch_size,sampler=sampler,num_workers=args.num_workers,pin_memory=pin,drop_last=True)
    val_loader=DataLoader(VideoSequenceDataset(val_df,args.seq_len,args.image_size,False),
                          batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=pin)
    test_loader=DataLoader(VideoSequenceDataset(test_df,args.seq_len,args.image_size,False),
                           batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=pin)

    model=SentinelVideoDetector(args.rgb_backbone,args.d_model,args.nhead,args.num_layers).to(device)
    print(f"\nLoading Phase 1 checkpoint: {args.phase1_checkpoint}")
    ckpt=torch.load(args.phase1_checkpoint,map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    phase1_auc=ckpt.get("val_metrics",{}).get("auc","?")
    print(f"Phase 1 Val AUC was: {phase1_auc}")
    freeze_bottom_fraction(model.rgb_backbone,args.freeze_fraction)

    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    criterion=nn.BCEWithLogitsLoss(); scaler=GradScaler(enabled=amp_enabled)

    print("\n"+"="*65+"\n  Phase 2 Fine-Tuning\n"+"="*65)
    best_auc=float(phase1_auc) if isinstance(phase1_auc,float) else -1.0; wait=0; history=[]

    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running=0.0
        pbar=tqdm(enumerate(train_loader,start=1),total=len(train_loader),desc=f"Ep {epoch+1:02d}/{args.epochs}")
        for step,(x,y,_) in pbar:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with autocast(device_type=device.type,enabled=amp_enabled):
                loss=criterion(model(x),y)/args.grad_accum
            scaler.scale(loss).backward()
            if step%args.grad_accum==0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running+=float(loss.detach().item())*args.grad_accum
            pbar.set_postfix(loss=f"{running/step:.4f}")
        if len(train_loader)%args.grad_accum!=0:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        val_metrics,_=evaluate(model,val_loader,device,amp_enabled)
        train_loss=running/max(1,len(train_loader))
        print(f"Ep {epoch+1:02d}/{args.epochs} | Loss {train_loss:.4f} | AUC {val_metrics['auc']:.4f} | Acc {val_metrics['acc']:.4f} | F1 {val_metrics['f1']:.4f}",end="")
        epoch_ckpt=out_dir/f"epoch_{epoch+1:02d}_valauc_{val_metrics['auc']:.4f}.pt"
        torch.save({"state_dict":model.state_dict(),"args":vars(args),"epoch":epoch+1,"val_metrics":val_metrics},epoch_ckpt)
        history.append({"epoch":epoch+1,"train_loss":train_loss,"val_metrics":val_metrics,"checkpoint":str(epoch_ckpt)})
        if val_metrics["auc"]>best_auc:
            best_auc=val_metrics["auc"]; wait=0
            torch.save({"state_dict":model.state_dict(),"args":vars(args),"epoch":epoch+1,"val_metrics":val_metrics},out_dir/"best_model.pt")
            print("  ✓ saved")
        else:
            wait+=1; print()
            if wait>=args.patience: print("Early stopping triggered."); break

    best_ckpt=torch.load(out_dir/"best_model.pt",map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    test_metrics,hard=evaluate(model,test_loader,device,amp_enabled,export_hard=args.export_hard_negatives)
    print("\n"+"="*55+"\n  PHASE 2 FINAL EVAL RESULTS\n"+"="*55)
    for k,v in test_metrics.items(): print(f"  {k:<12}: {v:.4f}")
    print("="*55)
    print(f"\nFine-tuned checkpoint : {out_dir/'best_model.pt'}")
    with open(out_dir/"ft_train_history.json","w") as f: json.dump(history,f,indent=2)
    with open(out_dir/"ft_test_metrics.json","w") as f: json.dump(test_metrics,f,indent=2)
    if args.export_hard_negatives and hard:
        with open(out_dir/"ft_hard_negatives.jsonl","w") as f:
            for row in hard: f.write(json.dumps(row)+"\n")

if __name__=="__main__":
    main()
