import argparse
import csv
import os
import math
import json
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, log_loss


def read_csv_columns(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if v is None or v == "":
                cols[k].append(None)
                continue
            try:
                cols[k].append(float(v))
            except Exception:
                cols[k].append(v)
    return {k: np.array(v, dtype=object) for k, v in cols.items()}


def read_json_metrics(path):
    """Read metrics from a JSON file (flat dict of metric_name -> list of values)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    cols = defaultdict(list)
    for k, v in data.items():
        if isinstance(v, list):
            for val in v:
                try:
                    cols[k].append(float(val))
                except Exception:
                    cols[k].append(val)
        else:
            try:
                cols[k].append(float(v))
            except Exception:
                cols[k].append(v)
    return {k: np.array(v, dtype=object) for k, v in cols.items()}


def find_key(keys, candidates):
    kl = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in kl:
            return kl[c.lower()]
    return None


def plot_loss_acc(metrics, outdir):
    # Expect columns like epoch, train_loss, val_loss, train_acc, val_acc (or dv_auc, dv_acc, dv_f1)
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ matplotlib unavailable; skipping loss/accuracy plots. Writing CSV instead.")
        # Write CSV summary
        with open(os.path.join(outdir, "metrics_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            rows = list(zip(*[metrics[k] for k in metrics.keys()]))
            for row in rows:
                writer.writerow(dict(zip(metrics.keys(), row)))
        return

    keys = list(metrics.keys())
    epoch_key = find_key(keys, ["epoch", "epochs", "ep"])
    if epoch_key is None:
        n = len(next(iter(metrics.values())))
        epochs = list(range(1, n + 1))
    else:
        epochs = metrics[epoch_key].astype(float)

    # Loss curve
    train_loss_key = find_key(keys, ["train_loss", "loss_train", "trainloss", "loss"])
    val_loss_key = find_key(keys, ["val_loss", "loss_val", "eval_loss", "validation_loss"])
    if train_loss_key or val_loss_key:
        plt.figure(figsize=(10, 6))
        if train_loss_key:
            plt.plot(epochs, metrics[train_loss_key].astype(float), label="Train Loss", marker='o')
        if val_loss_key:
            plt.plot(epochs, metrics[val_loss_key].astype(float), label="Val Loss", marker='s')
        
        # Also plot alternative loss columns for adversarial training (video)
        loss_clean_key = find_key(keys, ["loss_clean"])
        loss_adv_key = find_key(keys, ["loss_adv"])
        if loss_clean_key:
            plt.plot(epochs, metrics[loss_clean_key].astype(float), label="Loss Clean", marker='^', alpha=0.7)
        if loss_adv_key:
            plt.plot(epochs, metrics[loss_adv_key].astype(float), label="Loss Adversarial", marker='d', alpha=0.7)
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved loss_curve.png")

    # Accuracy/AUC curve (handles both traditional val_acc and dv_acc patterns)
    train_acc_key = find_key(keys, ["train_acc", "train_accuracy", "accuracy_train"])
    val_acc_key = find_key(keys, ["val_acc", "eval_acc", "validation_accuracy", "accuracy_val", "dv_acc", "dev_acc"])
    val_auc_key = find_key(keys, ["dv_auc", "dev_auc", "val_auc"])
    val_f1_key = find_key(keys, ["dv_f1", "dev_f1", "val_f1"])
    
    if train_acc_key or val_acc_key or val_auc_key or val_f1_key:
        plt.figure(figsize=(10, 6))
        if train_acc_key:
            plt.plot(epochs, metrics[train_acc_key].astype(float), label="Train Accuracy", marker='o')
        if val_acc_key:
            plt.plot(epochs, metrics[val_acc_key].astype(float), label="Val Accuracy", marker='s')
        if val_auc_key:
            plt.plot(epochs, metrics[val_auc_key].astype(float), label="Val AUC", marker='^')
        if val_f1_key:
            plt.plot(epochs, metrics[val_f1_key].astype(float), label="Val F1", marker='d')
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("Validation Metrics Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "accuracy_curve.png"), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved accuracy_curve.png")


def plot_roc_and_metrics(roc_data, outdir, threshold=0.5):
    keys = list(roc_data.keys())
    y_true_key = find_key(keys, ["y_true", "label", "labels", "ground_truth"])
    y_prob_key = find_key(keys, ["y_prob", "prob", "score", "y_score", "fake_probability"])
    y_pred_key = find_key(keys, ["y_pred", "pred", "prediction"])  # optional

    # If this is a final metrics JSON (not raw predictions), just save it
    if y_true_key is None and y_prob_key is None:
        # Check if we have pre-computed metrics (auc, acc, f1, precision, recall)
        metric_keys = find_key(keys, ["auc"]), find_key(keys, ["acc", "accuracy"]), find_key(keys, ["f1"])
        if any(metric_keys):
            print(f"📈 Found pre-computed metrics in {outdir}")
            with open(os.path.join(outdir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
                for k, v in roc_data.items():
                    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                        val = v[0] if isinstance(v, (list, np.ndarray)) else v
                    else:
                        val = v
                    f.write(f"{k}: {val}\n")
            return dict(roc_data)
        print("ROC data must contain y_true and y_prob columns (or pre-computed metrics). Skipping.")
        return None

    y_true = np.array(roc_data[y_true_key].astype(float)).astype(int)
    y_prob = np.array(roc_data[y_prob_key].astype(float)).astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    if MATPLOTLIB_AVAILABLE:
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(outdir, "roc_curve.png"))
        plt.close()
    else:
        print("⚠️ matplotlib unavailable; saving ROC data as JSON instead.")
        with open(os.path.join(outdir, "roc_curve.json"), "w") as f:
            json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}, f, indent=2)

    # Predictions and other metrics
    if y_pred_key is not None:
        y_pred = np.array(roc_data[y_pred_key].astype(float)).astype(int)
    else:
        y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    try:
        ll = log_loss(y_true, y_prob)
    except Exception:
        ll = float("nan")

    summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc),
        "log_loss": float(ll),
    }

    # Save textual summary
    with open(os.path.join(outdir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return summary


def main():
    p = argparse.ArgumentParser(description="Plot training/eval metrics and ROC from CSVs/JSONs.")
    p.add_argument("--metrics", help="CSV path with per-epoch metrics (epoch, train_loss, val_loss, train_acc, val_acc)")
    p.add_argument("--roc", help="CSV path with columns y_true and y_prob (and optionally y_pred)")
    p.add_argument("--outdir", default="plots", help="Output directory for images and summaries")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold used for computing predicted labels from probabilities")
    p.add_argument("--auto", action="store_true", help="Auto-detect and plot logs for active models (imageguard_v2_finetuned, rawnet3_fsat_finetuned, video_best_model)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.auto:
        # Auto-detect and plot logs for models in app.py
        training_root = Path(__file__).parent.parent / "training"
        
        # Image logs (imageguard_v2_finetuned.pt)
        print("\n🖼️  ========== IMAGE MODEL ==========")
        image_logs_dir = training_root / "image scripts" / "logs"
        if image_logs_dir.exists():
            image_outdir = os.path.join(args.outdir, "image")
            os.makedirs(image_outdir, exist_ok=True)
            image_csv = image_logs_dir / "finetune_log.csv"
            image_json = image_logs_dir / "finetune_test_metrics.json"
            if image_csv.exists():
                print(f"📊 Processing image metrics: {image_csv}")
                metrics = read_csv_columns(str(image_csv))
                plot_loss_acc(metrics, image_outdir)
            if image_json.exists():
                print(f"📊 Processing image test metrics: {image_json}")
                roc_data = read_json_metrics(str(image_json))
                plot_roc_and_metrics(roc_data, image_outdir, threshold=args.threshold)
        
        # Audio logs (rawnet3_fsat_finetuned.pt)
        print("\n🎙️  ========== AUDIO MODEL ==========")
        audio_logs_dir = training_root / "audio scripts" / "logs"
        if audio_logs_dir.exists():
            audio_outdir = os.path.join(args.outdir, "audio")
            os.makedirs(audio_outdir, exist_ok=True)
            audio_csv = audio_logs_dir / "finetune_audio_log.csv"
            audio_json = audio_logs_dir / "finetune_audio_metrics.json"
            if audio_csv.exists():
                print(f"📊 Processing audio metrics: {audio_csv}")
                metrics = read_csv_columns(str(audio_csv))
                plot_loss_acc(metrics, audio_outdir)
            if audio_json.exists():
                print(f"📊 Processing audio test metrics: {audio_json}")
                roc_data = read_json_metrics(str(audio_json))
                plot_roc_and_metrics(roc_data, audio_outdir, threshold=args.threshold)
        
        # Video logs (video_best_model.pt)
        print("\n🎬 ========== VIDEO MODEL ==========")
        video_logs_dir = training_root / "video scripts" / "logs"
        if video_logs_dir.exists():
            video_outdir = os.path.join(args.outdir, "video")
            os.makedirs(video_outdir, exist_ok=True)
            video_csv = video_logs_dir / "phase3_video_log.csv"
            video_json = video_logs_dir / "phase3_video_metrics.json"
            if video_csv.exists():
                print(f"📊 Processing video metrics: {video_csv}")
                metrics = read_csv_columns(str(video_csv))
                plot_loss_acc(metrics, video_outdir)
            if video_json.exists():
                print(f"📊 Processing video test metrics: {video_json}")
                roc_data = read_json_metrics(str(video_json))
                plot_roc_and_metrics(roc_data, video_outdir, threshold=args.threshold)
        
        print(f"\n✅ All plots saved to: {args.outdir}/{{image,audio,video}}/")
        return

    if args.metrics:
        if not os.path.exists(args.metrics):
            print(f"Metrics CSV not found: {args.metrics}")
        else:
            metrics = read_csv_columns(args.metrics)
            plot_loss_acc(metrics, args.outdir)
            print("Saved loss/accuracy plots (if columns present) to:", args.outdir)

    if args.roc:
        if not os.path.exists(args.roc):
            print(f"ROC file not found: {args.roc}")
        else:
            if args.roc.endswith('.json'):
                roc_data = read_json_metrics(args.roc)
            else:
                roc_data = read_csv_columns(args.roc)
            summary = plot_roc_and_metrics(roc_data, args.outdir, threshold=args.threshold)
            if summary is not None:
                print("Saved ROC plot and metrics_summary.txt to:", args.outdir)


if __name__ == "__main__":
    main()
