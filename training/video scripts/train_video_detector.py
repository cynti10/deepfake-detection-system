import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import timm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_frame_paths(s):
    return [p for p in str(s).split("|") if p]


def load_rgb(path, image_size):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return x


def infer_label_from_path(path_str: str) -> int | None:
    """
    Infer binary label (1=real, 0=fake) from directory keywords.
    Returns None if the path contains neither keyword.
    """
    p = path_str.lower()
    # Check fake first so 'realfake' edge-cases fall to fake
    if "fake" in p:
        return 0
    if "real" in p:
        return 1
    return None


def load_manifest(csv_path: str) -> pd.DataFrame:
    """
    Load a manifest CSV and ensure every required column exists.
    Rows that are missing a label but have a path with 'real'/'fake'
    in the directory tree are repaired automatically instead of being dropped.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"frame_paths", "label", "video_id", "source_dataset", "sample_id"}
    missing_cols = required_cols - set(df.columns)

    # ── fill missing metadata columns with sensible defaults ─────────────────
    if "video_id" not in df.columns:
        df["video_id"] = df.index.astype(str)
    if "source_dataset" not in df.columns:
        df["source_dataset"] = "unknown"
    if "sample_id" not in df.columns:
        df["sample_id"] = df.index.astype(str)

    # ── fix missing labels via directory-name heuristic ──────────────────────
    if "label" not in df.columns:
        df["label"] = None

    null_mask = df["label"].isnull()
    if null_mask.any():
        inferred = df.loc[null_mask, "frame_paths"].apply(
            lambda s: infer_label_from_path(str(s).split("|")[0])
        )
        df.loc[null_mask, "label"] = inferred

    before = len(df)
    # Only drop rows where we truly cannot determine a label
    df = df.dropna(subset=["label"])
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[manifest] WARNING: dropped {dropped} rows with no label and "
              f"no recognisable real/fake path keyword.")

    repaired = null_mask.sum() - dropped
    if repaired > 0:
        print(f"[manifest] Repaired {repaired} rows using directory-name label inference.")

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


class VideoSequenceDataset(Dataset):
    def __init__(self, df, seq_len=16, image_size=224, augment=False):
        self.df         = df.reset_index(drop=True)
        self.seq_len    = seq_len
        self.image_size = image_size
        self.augment    = augment

    def __len__(self):
        return len(self.df)

    def _sample_paths(self, paths):
        if len(paths) == 0:
            return []
        if len(paths) >= self.seq_len:
            idx = np.linspace(0, len(paths) - 1, self.seq_len).astype(int)
            return [paths[i] for i in idx]
        out = list(paths)
        while len(out) < self.seq_len:
            out.append(paths[-1])
        return out

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        paths = self._sample_paths(parse_frame_paths(row["frame_paths"]))

        frames = []
        for p in paths:
            x = load_rgb(p, self.image_size)
            if x is None:
                x = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            if self.augment:
                # Horizontal flip
                if random.random() < 0.5:
                    x = x[:, :, ::-1].copy()
                # Brightness jitter
                if random.random() < 0.30:
                    x = np.clip(x * random.uniform(0.75, 1.25), 0, 1)
                # H.264 block artifact simulation
                if random.random() < 0.15:
                    H8, W8 = x.shape[1] // 8, x.shape[2] // 8
                    blk = np.random.uniform(-0.03, 0.03,
                          (3, H8, W8)).astype(np.float32)
                    blk = np.repeat(np.repeat(blk, 8, axis=1), 8, axis=2)
                    x   = np.clip(x + blk[:, :x.shape[1], :x.shape[2]], 0, 1)
            frames.append(x)

        frames_arr = np.stack(frames, axis=0)   # [T, C, H, W]

        # Motion-aware temporal delta maps
        deltas = np.zeros_like(frames_arr)
        for ti in range(1, len(frames_arr)):
            d          = np.abs(frames_arr[ti] - frames_arr[ti - 1])
            motion_mag = float(d.mean())
            if motion_mag < 0.005:
                deltas[ti] = 0.0
            elif motion_mag > 0.20:
                deltas[ti] = np.clip(d / (motion_mag + 1e-6), 0, 1)
            else:
                deltas[ti] = np.clip(d / 2.0, 0, 1)

        x    = torch.tensor(frames_arr, dtype=torch.float32)
        y    = torch.tensor(float(row["label"]), dtype=torch.float32)
        meta = {
            "video_id":       str(row["video_id"]),
            "source_dataset": str(row["source_dataset"]),
            "sample_id":      str(row["sample_id"]),
        }
        return x, y, meta


def rgb_to_fft_mag(x, training=False):
    # x: [B, T, C, H, W]
    gray = 0.2989 * x[:, :, 0:1] + 0.5870 * x[:, :, 1:2] + 0.1140 * x[:, :, 2:3]
    fft  = torch.fft.fft2(gray)
    mag  = torch.log1p(torch.abs(fft))
    mn   = mag.amin(dim=(3, 4), keepdim=True)
    mx   = mag.amax(dim=(3, 4), keepdim=True)
    mag  = (mag - mn) / (mx - mn + 1e-6)
    if training:
        B, T, _, H, W = mag.shape
        if torch.rand(1).item() < 0.30:
            n = max(1, int(H * 0.15))
            s = torch.randint(0, H - n, (1,)).item()
            mag = mag.clone()
            mag[:, :, :, s:s + n, :] = 0.0
        if torch.rand(1).item() < 0.30:
            n = max(1, int(W * 0.15))
            s = torch.randint(0, W - n, (1,)).item()
            mag = mag.clone()
            mag[:, :, :, :, s:s + n] = 0.0
    return mag


class SentinelVideoDetector(nn.Module):
    def __init__(self, rgb_backbone="tf_efficientnet_b3_ns",
                 d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.rgb_backbone = timm.create_model(
            rgb_backbone, pretrained=True, num_classes=0, global_pool="avg")
        self.fft_backbone = timm.create_model(
            "mobilenetv3_small_050", pretrained=True,
            in_chans=1, num_classes=0, global_pool="avg")

        with torch.no_grad():
            d_rgb    = self.rgb_backbone(torch.zeros(1, 3, 224, 224)).shape[1]
            d_fft    = self.fft_backbone(torch.zeros(1, 1, 224, 224)).shape[1]
            feat_dim = d_rgb + d_fft

        self.frame_proj = nn.Linear(feat_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.temporal  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_head  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        xr = x.view(b * t, c, h, w)
        fr = self.rgb_backbone(xr)

        xf = rgb_to_fft_mag(x, training=self.training).view(b * t, 1, h, w)
        ff = self.fft_backbone(xf)

        f   = torch.cat([fr, ff], dim=1)
        tok = self.frame_proj(f).view(b, t, -1)

        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        out = self.temporal(seq)
        return self.cls_head(out[:, 0]).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        import torch.nn.functional as F
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc":       auc,
        "acc":       float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


def evaluate(model, loader, device, amp_enabled,
             export_hard=False, hard_threshold=0.8):
    model.eval()
    ys, ps = [], []
    hard   = []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                prob   = torch.sigmoid(logits)

            y_cpu = y.detach().cpu().numpy()
            p_cpu = prob.detach().cpu().numpy()
            ys.extend(y_cpu.tolist())
            ps.extend(p_cpu.tolist())

            if export_hard:
                pred = (p_cpu >= 0.5).astype(np.int32)
                conf = np.where(pred == 1, p_cpu, 1.0 - p_cpu)
                for i in range(len(pred)):
                    if pred[i] != int(y_cpu[i]) and conf[i] >= hard_threshold:
                        hard.append({
                            "sample_id":      meta["sample_id"][i],
                            "video_id":       meta["video_id"][i],
                            "source_dataset": meta["source_dataset"][i],
                            "label":          int(y_cpu[i]),
                            "pred":           int(pred[i]),
                            "confidence":     float(conf[i]),
                        })

    return compute_metrics(ys, ps), hard


def main():
    parser = argparse.ArgumentParser(
        description="Train Sentinel video detector (RGB+Frequency+Temporal).")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest",   required=True)
    parser.add_argument("--test-manifest",  required=True)
    parser.add_argument("--out-dir",        required=True)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--seq-len",        type=int,   default=16)
    parser.add_argument("--image-size",     type=int,   default=224)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--grad-accum",     type=int,   default=2)
    parser.add_argument("--max-grad-norm",  type=float, default=1.0)
    parser.add_argument("--patience",       type=int,   default=5)
    parser.add_argument("--d-model",        type=int,   default=512)
    parser.add_argument("--nhead",          type=int,   default=8)
    parser.add_argument("--num-layers",     type=int,   default=2)
    parser.add_argument("--rgb-backbone",   default="tf_efficientnet_b3_ns")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--amp",            action="store_true")
    parser.add_argument("--export-hard-negatives", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    # ── load manifests with automatic label repair ────────────────────────────
    print("[manifest] Loading train manifest …")
    train_df = load_manifest(args.train_manifest)
    print(f"[manifest] Train samples: {len(train_df):,}  "
          f"(real={int((train_df['label']==1).sum()):,}, "
          f"fake={int((train_df['label']==0).sum()):,})")

    print("[manifest] Loading val manifest …")
    val_df   = load_manifest(args.val_manifest)
    print(f"[manifest] Val samples:   {len(val_df):,}")

    print("[manifest] Loading test manifest …")
    test_df  = load_manifest(args.test_manifest)
    print(f"[manifest] Test samples:  {len(test_df):,}")

    train_ds = VideoSequenceDataset(train_df, seq_len=args.seq_len,
                                    image_size=args.image_size, augment=True)
    val_ds   = VideoSequenceDataset(val_df,   seq_len=args.seq_len,
                                    image_size=args.image_size, augment=False)
    test_ds  = VideoSequenceDataset(test_df,  seq_len=args.seq_len,
                                    image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")

    model = SentinelVideoDetector(
        rgb_backbone=args.rgb_backbone,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler    = GradScaler(enabled=amp_enabled)

    best_auc = -1.0
    wait     = 0
    history  = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        pbar = tqdm(enumerate(train_loader, start=1),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (x, y, _) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss   = criterion(logits, y)
                scaled = loss / args.grad_accum

            scaler.scale(scaled).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach().item())
            pbar.set_postfix(loss=f"{running / max(1, step):.4f}")

        # flush remaining gradient accumulation
        if len(train_loader) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device, amp_enabled)
        epoch_ckpt = out_dir / f"epoch_{epoch+1:02d}_valauc_{val_metrics['auc']:.4f}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "args":       vars(args),
                "epoch":      epoch + 1,
                "val_metrics": val_metrics,
            },
            epoch_ckpt,
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss={running / max(1, len(train_loader)):.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history.append({
            "epoch":       epoch + 1,
            "train_loss":  running / max(1, len(train_loader)),
            "val_metrics": val_metrics,
            "checkpoint":  str(epoch_ckpt),
        })

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            wait     = 0
            torch.save(
                {
                    "state_dict":  model.state_dict(),
                    "args":        vars(args),
                    "epoch":       epoch + 1,
                    "val_metrics": val_metrics,
                },
                out_dir / "best_model.pt",
            )
            print("  -> new best checkpoint")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                break

    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    test_metrics, hard = evaluate(
        model, test_loader, device, amp_enabled,
        export_hard=args.export_hard_negatives,
    )

    print("Final test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    if args.export_hard_negatives:
        with open(out_dir / "hard_negatives_test.jsonl", "w", encoding="utf-8") as f:
            for row in hard:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
