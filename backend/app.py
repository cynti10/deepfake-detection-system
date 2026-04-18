import os
import threading
import time
import uuid

import cv2
import librosa
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    import torch
    import torch.nn as nn
    from torch.amp import autocast
    import timm

    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)


CWD = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))

# Global model state
model_lock = threading.Lock()
model_load_errors = {}
active_models = {
    "image": None,
    "audio": None,
    "video": None,
}

# Model handles
image_torch_model = None
audio_torch_model = None
video_torch_model = None
video_cfg = {}


def _register_model_error(name, err):
    msg = str(err)
    model_load_errors[name] = msg
    print(f"❌ {name}: {msg}")


def _is_lfs_pointer(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join([f.readline() for _ in range(2)])
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _existing_non_pointer(paths):
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if _is_lfs_pointer(p):
            _register_model_error(os.path.basename(p), "Git LFS pointer detected. Run: git lfs pull")
            continue
        out.append(p)
    return out


def _load_torch_inference_model(path, model_tag):
    if not TORCH_AVAILABLE:
        return None

    # First try TorchScript modules for fully self-contained deployment.
    try:
        model = torch.jit.load(path, map_location=TORCH_DEVICE)
        model.eval()
        return model
    except Exception as jit_err:
        try:
            obj = torch.load(path, map_location=TORCH_DEVICE)

            if isinstance(obj, nn.Module):
                obj.eval()
                return obj

            if isinstance(obj, dict):
                maybe_model = obj.get("model")
                if isinstance(maybe_model, nn.Module):
                    maybe_model.eval()
                    return maybe_model

                if "model_state_dict" in obj or "state_dict" in obj:
                    _register_model_error(
                        os.path.basename(path),
                        (
                            f"{model_tag} checkpoint contains weights only (state_dict) without model architecture. "
                            "Export a scripted module (.pt) for inference deployment, or provide architecture code in backend."
                        ),
                    )
                    return None

                _register_model_error(
                    os.path.basename(path),
                    f"Unsupported checkpoint dictionary format for {model_tag}. Keys: {list(obj.keys())[:10]}",
                )
                return None

            _register_model_error(
                os.path.basename(path),
                f"Unsupported object type for {model_tag}: {type(obj).__name__}",
            )
            return None
        except Exception as load_err:
            _register_model_error(
                os.path.basename(path),
                f"TorchScript load failed ({jit_err}); torch.load failed ({load_err})",
            )
            return None


# ----------------------
# Torch model components
# ----------------------
if TORCH_AVAILABLE:

    def rgb_to_fft_mag(x):
        # x: [B, T, C, H, W]
        gray = 0.2989 * x[:, :, 0:1] + 0.5870 * x[:, :, 1:2] + 0.1140 * x[:, :, 2:3]
        fft = torch.fft.fft2(gray)
        mag = torch.log1p(torch.abs(fft))
        mn = mag.amin(dim=(3, 4), keepdim=True)
        mx = mag.amax(dim=(3, 4), keepdim=True)
        return (mag - mn) / (mx - mn + 1e-6)


    class SentinelVideoDetector(nn.Module):
        def __init__(
            self,
            rgb_backbone="tf_efficientnet_b3_ns",
            d_model=512,
            nhead=8,
            num_layers=2,
            dropout=0.1,
        ):
            super().__init__()
            self.rgb_backbone = timm.create_model(
                rgb_backbone, pretrained=False, num_classes=0, global_pool="avg"
            )
            self.fft_backbone = timm.create_model(
                "mobilenetv3_small_050", pretrained=False, in_chans=1, num_classes=0, global_pool="avg"
            )

            with torch.no_grad():
                d_rgb = self.rgb_backbone(torch.zeros(1, 3, 224, 224)).shape[1]
                d_fft = self.fft_backbone(torch.zeros(1, 1, 224, 224)).shape[1]
                feat_dim = d_rgb + d_fft

            self.frame_proj = nn.Linear(feat_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        def forward(self, x):
            # x: [B, T, C, H, W]
            b, t, c, h, w = x.shape
            xr = x.view(b * t, c, h, w)
            fr = self.rgb_backbone(xr)

            xf = rgb_to_fft_mag(x).view(b * t, 1, h, w)
            ff = self.fft_backbone(xf)

            f = torch.cat([fr, ff], dim=1)
            tok = self.frame_proj(f).view(b, t, -1)

            cls = self.cls_token.expand(b, -1, -1)
            seq = torch.cat([cls, tok], dim=1)
            out = self.temporal(seq)
            return self.cls_head(out[:, 0]).squeeze(1)


# ----------------------
# Preprocessing
# ----------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _image_to_tensor_rgb224(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return x


def prepare_audio_mel(audio_path):
    librosa.cache.clear()
    y, sr = librosa.load(audio_path, sr=22050, duration=3)

    target_len = 22050 * 3
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = mel.astype(np.float32)
    return mel[np.newaxis, ..., np.newaxis]


def prepare_audio_waveform(audio_path, sr=16000, duration_sec=4):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration_sec)
    target_len = sr * duration_sec
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def _to_fake_probability(val):
    # Accept logits or probabilities, and return fake probability in [0, 1].
    v = float(val)
    if 0.0 <= v <= 1.0:
        return v
    return 1.0 / (1.0 + np.exp(-v))


def _torch_output_to_fake_probability(out):
    if TORCH_AVAILABLE and isinstance(out, torch.Tensor):
        t = out.detach().float().cpu().flatten()
        if t.numel() == 1:
            return _to_fake_probability(float(t.item()))
        if t.numel() >= 2:
            probs = torch.softmax(t[:2], dim=0).numpy()
            # Convention for this project family: class index 0 is FAKE, 1 is REAL.
            return float(probs[0])
    arr = np.asarray(out).reshape(-1)
    if arr.size == 1:
        return _to_fake_probability(float(arr[0]))
    if arr.size >= 2:
        e = np.exp(arr[:2] - np.max(arr[:2]))
        probs = e / np.sum(e)
        return float(probs[0])
    return 0.5


def _result_from_fake_prob(fake_prob):
    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    conf = fake_prob if label == "FAKE" else 1.0 - fake_prob
    return label, conf


def _frame_indices(frame_count, num_frames):
    if frame_count <= 0:
        return []
    if frame_count >= num_frames:
        return np.linspace(0, frame_count - 1, num_frames).astype(int).tolist()
    idxs = list(range(frame_count))
    while len(idxs) < num_frames:
        idxs.append(idxs[-1])
    return idxs


def _extract_video_sequence(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = _frame_indices(frame_count, num_frames)

    seq = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        seq.append(_image_to_tensor_rgb224(frame))

    cap.release()

    if len(seq) == 0:
        return []
    while len(seq) < num_frames:
        seq.append(seq[-1])
    return seq[:num_frames]


def _predict_image_fake_prob_from_bgr(frame_bgr):
    with model_lock:
        if image_torch_model is not None and TORCH_AVAILABLE:
            x = _image_to_tensor_rgb224(frame_bgr)
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(TORCH_DEVICE)
            with torch.no_grad():
                with autocast(device_type=TORCH_DEVICE.type, enabled=TORCH_DEVICE.type == "cuda"):
                    out = image_torch_model(xt)
            return _torch_output_to_fake_probability(out)

    return None


# ----------------------
# Model loading
# ----------------------
print("--- INITIALIZING MODELS ---")

TORCH_DEVICE = None
if TORCH_AVAILABLE:
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    _register_model_error("torch_runtime", f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")

# Image model preference: finetuned .pt -> best .pt
image_pt_candidates = _existing_non_pointer(
    [
        os.path.join(MODEL_DIR, "imageguard_v2_finetuned.pt"),
        os.path.join(MODEL_DIR, "imageguard_v2_best.pt"),
    ]
)

if TORCH_AVAILABLE:
    for p in image_pt_candidates:
        print(f"Loading image model from: {p}")
        image_torch_model = _load_torch_inference_model(p, "image")
        if image_torch_model is not None:
            active_models["image"] = os.path.basename(p)
            print(f"✅ Image model loaded: {active_models['image']}")
            break

if active_models["image"] is None:
    print("⚠️ No image model loaded.")

# Audio model preference: finetuned .pt -> best .pt
audio_pt_candidates = _existing_non_pointer(
    [
        os.path.join(MODEL_DIR, "rawnet3_fsat_finetuned.pt"),
        os.path.join(MODEL_DIR, "rawnet3_fsat_best.pt"),
    ]
)

if TORCH_AVAILABLE:
    for p in audio_pt_candidates:
        print(f"Loading audio model from: {p}")
        audio_torch_model = _load_torch_inference_model(p, "audio")
        if audio_torch_model is not None:
            active_models["audio"] = os.path.basename(p)
            print(f"✅ Audio model loaded: {active_models['audio']}")
            break

if active_models["audio"] is None:
    print("⚠️ No audio model loaded.")

# Video model: use the single provided checkpoint
video_pt_candidates = _existing_non_pointer([os.path.join(MODEL_DIR, "video_best_model.pt")])

if TORCH_AVAILABLE:
    for p in video_pt_candidates:
        try:
            print(f"Loading video checkpoint from: {p}")
            ckpt = torch.load(p, map_location=TORCH_DEVICE)
            cfg = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
            model = SentinelVideoDetector(
                rgb_backbone=cfg.get("rgb_backbone", "tf_efficientnet_b3_ns"),
                d_model=int(cfg.get("d_model", 512)),
                nhead=int(cfg.get("nhead", 8)),
                num_layers=int(cfg.get("num_layers", 2)),
            ).to(TORCH_DEVICE)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            model.load_state_dict(state, strict=True)
            model.eval()
            video_torch_model = model
            video_cfg = {
                "seq_len": int(cfg.get("seq_len", 16)),
                "image_size": int(cfg.get("image_size", 224)),
            }
            active_models["video"] = os.path.basename(p)
            print(f"✅ Video model loaded: {active_models['video']}")
            break
        except Exception as e:
            _register_model_error(os.path.basename(p), e)

if active_models["video"] is None:
    print("⚠️ No video model loaded.")


# ----------------------
# API endpoints
# ----------------------
@app.route("/detect-image", methods=["POST"])
def detect_image():
    if active_models["image"] is None:
        return jsonify({"error": "Image model not active. Check /health for details."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    file_ext = os.path.splitext(secure_filename(file.filename))[1] or ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(path)

    try:
        frame = cv2.imread(path)
        if frame is None:
            return jsonify({"error": "Could not read image"}), 400

        fake_prob = _predict_image_fake_prob_from_bgr(frame)
        if fake_prob is None:
            return jsonify({"error": "Image model inference unavailable"}), 503

        label, conf = _result_from_fake_prob(fake_prob)
        return jsonify({"result": label, "confidence": float(conf)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        time.sleep(0.1)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


@app.route("/detect-video", methods=["POST"])
def detect_video():
    if active_models["video"] is None:
        return jsonify({"error": "Video model not active. Check /health for details."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    file_ext = os.path.splitext(secure_filename(file.filename))[1] or ".mp4"
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(path)

    try:
        seq_len = int(video_cfg.get("seq_len", 16))
        seq = _extract_video_sequence(path, num_frames=seq_len)
        if len(seq) == 0:
            return jsonify({"error": "No frames extracted"}), 500

        with model_lock:
            x = torch.tensor(np.stack(seq, axis=0), dtype=torch.float32).unsqueeze(0).to(TORCH_DEVICE)
            with torch.no_grad():
                with autocast(device_type=TORCH_DEVICE.type, enabled=TORCH_DEVICE.type == "cuda"):
                    logits = video_torch_model(x)
                    fake_prob = float(torch.sigmoid(logits).item())

        label, conf = _result_from_fake_prob(fake_prob)
        return jsonify({"result": label, "confidence": conf, "fake_probability": fake_prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        time.sleep(0.2)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


@app.route("/detect-audio", methods=["POST"])
def detect_audio():
    if active_models["audio"] is None:
        return jsonify({"error": "Audio model not active. Check /health for details."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    unique_name = f"{uuid.uuid4()}.wav"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(path)

    try:
        with model_lock:
            if audio_torch_model is not None and TORCH_AVAILABLE:
                fake_prob = None

                # Try waveform first (common for RawNet variants)
                try:
                    wav = prepare_audio_waveform(path)
                    xt = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(TORCH_DEVICE)
                    with torch.no_grad():
                        with autocast(device_type=TORCH_DEVICE.type, enabled=TORCH_DEVICE.type == "cuda"):
                            out = audio_torch_model(xt)
                    fake_prob = _torch_output_to_fake_probability(out)
                except Exception as wav_err:
                    _register_model_error("audio_waveform_infer", wav_err)

                # Fallback to mel tensor
                if fake_prob is None:
                    mel = prepare_audio_mel(path)
                    mel_t = torch.tensor(np.transpose(mel, (0, 3, 1, 2)), dtype=torch.float32).to(TORCH_DEVICE)
                    with torch.no_grad():
                        with autocast(device_type=TORCH_DEVICE.type, enabled=TORCH_DEVICE.type == "cuda"):
                            out = audio_torch_model(mel_t)
                    fake_prob = _torch_output_to_fake_probability(out)

                label, conf = _result_from_fake_prob(fake_prob)
                return jsonify({"result": label, "confidence": conf})

        return jsonify({"error": "Audio model inference unavailable"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "torch_available": TORCH_AVAILABLE,
            "torch_device": str(TORCH_DEVICE) if TORCH_AVAILABLE else None,
            "image_model": active_models["image"] or "MISSING",
            "audio_model": active_models["audio"] or "MISSING",
            "video_model": active_models["video"] or "MISSING",
            "errors": model_load_errors,
        }
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
