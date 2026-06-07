#!/usr/bin/env python3
"""
Sentinel Demo Evaluator
Runs inference on ~50 samples per modality and produces:
  - Confusion matrix PNG
  - Confidence histogram PNG
  - Per-sample CSV
Usage: python demo_eval.py
"""
import random, warnings
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import seaborn as sns
import csv

warnings.filterwarnings('ignore')

# ── Paths — update these to your dataset locations ─────────────────────
VIDEO_REAL_DIR  = Path('/path/to/sdfvd2/real')
VIDEO_FAKE_DIR  = Path('/path/to/sdfvd2/fake')
IMAGE_REAL_DIR  = Path('/path/to/roop_akool/real')
IMAGE_FAKE_DIR  = Path('/path/to/roop_akool/fake')
AUDIO_REAL_DIR  = Path('/path/to/for-norm/real')
AUDIO_FAKE_DIR  = Path('/path/to/for-norm/fake')

VIDEO_CKPT = Path.home() / 'deepfake_project/video/checkpoints/video_detector_final.pt'
IMAGE_CKPT = Path.home() / 'deepfake_project/image/checkpoints/imageguard_phase3.pt'
AUDIO_CKPT = Path.home() / 'deepfake_project/audio/checkpoints/audio_phase3.pt'

OUT_DIR = Path.home() / 'deepfake_project/demo_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_DEMO   = 50     # samples per class per modality
SEQ_LEN  = 16
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Helpers ─────────────────────────────────────────────────────────────
def collect_files(directory, exts, n):
    files = [p for p in sorted(directory.rglob('*'))
             if p.is_file() and p.suffix.lower() in exts]
    random.shuffle(files)
    return files[:n]

def save_results_csv(name, paths, labels, probs):
    out = OUT_DIR / f'{name}_results.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'true_label', 'pred_prob', 'pred_label', 'correct'])
        for p, l, prob in zip(paths, labels, probs):
            pred = int(prob >= 0.5)
            w.writerow([Path(p).name, l, f'{prob:.4f}', pred, int(pred == l)])
    print(f"  Saved {out}")

def plot_results(name, labels, probs, modality_label):
    labels_arr = np.array(labels)
    probs_arr  = np.array(probs)
    preds      = (probs_arr >= 0.5).astype(int)
    auc        = roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.5

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f'{modality_label}  |  AUC = {auc:.4f}', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # ── Confusion matrix ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cm  = confusion_matrix(labels_arr, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Fake'])
    disp.plot(ax=ax1, colorbar=False, cmap='Blues')
    ax1.set_title('Confusion Matrix')

    # ── Confidence histogram ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    real_probs = probs_arr[labels_arr == 0]
    fake_probs = probs_arr[labels_arr == 1]
    bins = np.linspace(0, 1, 25)
    ax2.hist(real_probs, bins=bins, alpha=0.7, color='steelblue', label='Real')
    ax2.hist(fake_probs, bins=bins, alpha=0.7, color='tomato',    label='Fake')
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=1.2, label='Threshold 0.5')
    ax2.set_xlabel('Model Confidence (P(fake))')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.legend()

    plt.tight_layout()
    out = OUT_DIR / f'{name}_demo_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}  |  AUC={auc:.4f}")
    return auc

# ══════════════════════════════════════════════════════════════════════
# ── VIDEO ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
def load_video_frames(path):
    blank = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    cap   = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return np.stack([blank]*SEQ_LEN)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    idxs  = np.linspace(0, total-1, SEQ_LEN).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok or f is None:
            frames.append(frames[-1] if frames else blank); continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        x = (f.astype(np.float32)/255.0 - MEAN) / STD
        frames.append(np.transpose(x, (2,0,1)).astype(np.float32))
    cap.release()
    while len(frames) < SEQ_LEN: frames.append(frames[-1] if frames else blank)
    return np.stack(frames[:SEQ_LEN])   # [T,C,H,W]

def rgb_to_fft_mag(x):
    gray = 0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2]
    fft  = torch.fft.fft2(gray)
    mag  = torch.log1p(torch.abs(fft))
    mn   = mag.amin(dim=(-2,-1), keepdim=True)
    mx   = mag.amax(dim=(-2,-1), keepdim=True)
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
        self.frame_proj = nn.Linear(feat_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
              dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.temporal  = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.cls_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        b,t,c,h,w = x.shape
        xr  = x.view(b*t,c,h,w)
        fr  = self.rgb_backbone(xr)
        fft = rgb_to_fft_mag(xr)
        ff  = self.fft_backbone(fft.unsqueeze(1))
        f   = torch.cat([fr, ff], dim=1)
        tok = self.frame_proj(f.view(b,t,-1))
        cls = self.cls_token.expand(b,-1,-1)
        seq = torch.cat([cls, tok], dim=1)
        return self.cls_head(self.temporal(seq)[:,0]).squeeze(1)

def run_video_demo():
    print("\n── VIDEO DEMO ──────────────────────────────────────────")
    VIDEO_EXTS = {'.mp4','.avi','.mov','.mkv','.webm'}
    real_files = collect_files(VIDEO_REAL_DIR, VIDEO_EXTS, N_DEMO)
    fake_files = collect_files(VIDEO_FAKE_DIR, VIDEO_EXTS, N_DEMO)
    all_files  = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    random.shuffle(all_files)

    model = SentinelVideoDetector().to(DEVICE)
    ckpt  = torch.load(VIDEO_CKPT, map_location=DEVICE, weights_only=False)
    sd    = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd); model.eval()
    print(f"  Loaded VideoGuard — evaluating {len(all_files)} clips")

    labels, probs, paths = [], [], []
    for path, label in all_files:
        frames = load_video_frames(path)
        x = torch.tensor(frames).unsqueeze(0).to(DEVICE)   # [1,T,C,H,W]
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        labels.append(label); probs.append(prob); paths.append(str(path))
        status = '✓' if (int(prob>=0.5)==label) else '✗'
        print(f"  {status} {Path(path).name[:40]:<40} | label={'fake' if label else 'real'} | p={prob:.3f}")

    save_results_csv('video', paths, labels, probs)
    return plot_results('video', labels, probs, 'VideoGuard — SDFVD 2.0 Demo')

# ══════════════════════════════════════════════════════════════════════
# ── IMAGE ─────────────────────────────────════════════════════════════
# ══════════════════════════════════════════════════════════════════════
def load_image(path):
    img = cv2.imread(str(path))
    if img is None: return np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x   = (img.astype(np.float32)/255.0 - MEAN) / STD
    return np.transpose(x, (2,0,1)).astype(np.float32)

def run_image_demo():
    print("\n── IMAGE DEMO ──────────────────────────────────────────")
    IMG_EXTS   = {'.jpg','.jpeg','.png','.bmp','.webp'}
    real_files = collect_files(IMAGE_REAL_DIR, IMG_EXTS, N_DEMO)
    fake_files = collect_files(IMAGE_FAKE_DIR, IMG_EXTS, N_DEMO)
    all_files  = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    random.shuffle(all_files)

    # Load ImageGuard — adjust class name if yours differs
    ckpt  = torch.load(IMAGE_CKPT, map_location=DEVICE, weights_only=False)
    # Generic loader: use the stored args if available
    args  = ckpt.get('args', {})
    model = timm.create_model(
        args.get('rgb_backbone', 'tf_efficientnet_b3_ns'),
        pretrained=False, num_classes=1, global_pool='avg').to(DEVICE)
    # Try loading — if your ImageGuard has a custom class, replace this block
    try:
        sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"  Warning: {e} — loading with strict=False")
    model.eval()
    print(f"  Loaded ImageGuard — evaluating {len(all_files)} images")

    labels, probs, paths = [], [], []
    for path, label in all_files:
        x = torch.tensor(load_image(path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        labels.append(label); probs.append(prob); paths.append(str(path))
        status = '✓' if (int(prob>=0.5)==label) else '✗'
        print(f"  {status} {Path(path).name[:40]:<40} | label={'fake' if label else 'real'} | p={prob:.3f}")

    save_results_csv('image', paths, labels, probs)
    return plot_results('image', labels, probs, 'ImageGuard — Roop/Akool Demo')

# ══════════════════════════════════════════════════════════════════════
# ── AUDIO ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
def load_audio(path, sr=16000, max_sec=4):
    try:
        wav, _ = librosa.load(str(path), sr=sr, mono=True,
                              duration=max_sec)
        target = sr * max_sec
        if len(wav) < target:
            wav = np.pad(wav, (0, target - len(wav)))
        else:
            wav = wav[:target]
        return wav.astype(np.float32)
    except Exception:
        return np.zeros(sr * max_sec, dtype=np.float32)

def run_audio_demo():
    print("\n── AUDIO DEMO ──────────────────────────────────────────")
    AUDIO_EXTS = {'.wav','.flac','.mp3','.ogg','.m4a'}
    real_files = collect_files(AUDIO_REAL_DIR, AUDIO_EXTS, N_DEMO)
    fake_files = collect_files(AUDIO_FAKE_DIR, AUDIO_EXTS, N_DEMO)
    all_files  = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    random.shuffle(all_files)

    # Generic 1D-conv audio loader — replace with your actual RawNet3 class if needed
    ckpt  = torch.load(AUDIO_CKPT, map_location=DEVICE, weights_only=False)
    sd    = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    # Minimal wrapper — works if your model takes raw waveform [B, T]
    class AudioModelWrapper(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            # Replace with your actual RawNet3-FSAT instantiation
            # This is a placeholder that will raise clearly if wrong
            raise NotImplementedError(
                "Replace this with: from your_audio_module import RawNet3FSAT; "
                "self.model = RawNet3FSAT(...); self.model.load_state_dict(state_dict)")

    # ── Use your actual model class here ──────────────────────────────
    # from audio.model import RawNet3FSAT
    # model = RawNet3FSAT(config).to(DEVICE)
    # model.load_state_dict(sd); model.eval()
    # For now we'll signal clearly:
    print("  ⚠ Replace AudioModelWrapper with your RawNet3-FSAT class import")
    print("  Skipping audio inference — update AUDIO_CKPT path and model class")
    return None

# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    random.seed(42); np.random.seed(42)
    results = {}

    auc_v = run_video_demo()
    if auc_v: results['video'] = auc_v

    auc_i = run_image_demo()
    if auc_i: results['image'] = auc_i

    auc_a = run_audio_demo()
    if auc_a: results['audio'] = auc_a

    print("\n" + "="*55)
    print("  DEMO SUMMARY")
    print("="*55)
    for mod, auc in results.items():
        print(f"  {mod.upper():<8} AUC = {auc:.4f}")
    print(f"\nAll plots and CSVs saved to: {OUT_DIR}")