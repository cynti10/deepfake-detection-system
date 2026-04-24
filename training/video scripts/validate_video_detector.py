import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from train_video_detector import SentinelVideoDetector, VideoSequenceDataset


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a Sentinel video checkpoint on any sequence manifest split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-json", default="validation_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = SentinelVideoDetector(
        rgb_backbone=ckpt_args.get("rgb_backbone", "tf_efficientnet_b3_ns"),
        d_model=int(ckpt_args.get("d_model", 512)),
        nhead=int(ckpt_args.get("nhead", 8)),
        num_layers=int(ckpt_args.get("num_layers", 2)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    df = pd.read_csv(args.manifest)
    ds = VideoSequenceDataset(
        df,
        seq_len=int(ckpt_args.get("seq_len", 16)),
        image_size=int(ckpt_args.get("image_size", 224)),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                prob = torch.sigmoid(model(x))
            ys.extend(y.numpy().tolist())
            ps.extend(prob.cpu().numpy().tolist())

    metrics = compute_metrics(ys, ps)
    metrics["checkpoint"] = str(Path(args.checkpoint))
    metrics["manifest"] = str(Path(args.manifest))
    metrics["samples"] = len(ys)

    print("Validation metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()

