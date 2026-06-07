from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

REAL_TOKENS = {
    "real",
    "genuine",
    "original",
    "pristine",
    "authentic",
    "bonafide",
    "youtube",
    "actors",
    "raw",
    "live",
    "untampered",
    "training_real",
}
FAKE_TOKENS = {
    "fake",
    "synthetic",
    "generated",
    "gan",
    "diffusion",
    "deepfake",
    "ai",
    "stylegan",
    "midjourney",
    "dalle",
    "stable",
    "manipulation",
    "altered",
    "tampered",
    "spliced",
    "facegen",
    "training_fake",
    "spoof",
    "cloned",
    "tts",
}


@dataclass
class PipelineConfig:
    enabled: bool
    interval_sec: int
    min_pool_samples: int
    fine_tune_epochs: int
    max_auc_drop: float

    image_gen_per_cycle: int
    image_max_attempts_factor: int
    image_fgsm_eps: float
    image_lr: float
    image_gan_command: str

    audio_gen_per_cycle: int
    audio_max_attempts_factor: int
    audio_fgsm_eps: float
    audio_lr: float
    audio_gan_command: str

    video_gan_command: str

    image_source_dirs: list[str]
    audio_source_dirs: list[str]
    video_source_dirs: list[str]

    image_pool_dir: Path
    audio_pool_dir: Path
    video_pool_dir: Path

    model_dir: Path


class ImagePathDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, int]]):
        self.pairs = pairs
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            x = np.zeros((3, 224, 224), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            x = img.astype(np.float32) / 255.0
            x = (x - self.mean) / self.std
            x = np.transpose(x, (2, 0, 1))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(float(label), dtype=torch.float32)


class AudioPathDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, int]], n_samples: int = 64000, sr: int = 16000):
        self.pairs = pairs
        self.n_samples = n_samples
        self.sr = sr

    def __len__(self):
        return len(self.pairs)

    def _load_wav(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            try:
                arr = np.load(path).astype(np.float32)
                if arr.ndim > 1:
                    arr = np.squeeze(arr)
                if len(arr) < self.n_samples:
                    arr = np.pad(arr, (0, self.n_samples - len(arr)))
                else:
                    arr = arr[: self.n_samples]
                peak = float(np.max(np.abs(arr)))
                if peak > 1e-6:
                    arr = arr / peak
                return arr
            except Exception:
                pass

        try:
            wav, _ = librosa.load(path, sr=self.sr, duration=self.n_samples / self.sr)
            if len(wav) < self.n_samples:
                wav = np.pad(wav, (0, self.n_samples - len(wav)))
            else:
                wav = wav[: self.n_samples]
            wav = wav.astype(np.float32)
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-6:
                wav = wav / peak
            return wav
        except Exception:
            return np.zeros((self.n_samples,), dtype=np.float32)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        wav = self._load_wav(path)
        return torch.tensor(wav, dtype=torch.float32), torch.tensor(float(label), dtype=torch.float32)


class AdversarialBackgroundService:
    def __init__(
        self,
        cfg: PipelineConfig,
        model_lock: threading.Lock,
        torch_available: bool,
        torch_device: torch.device | None,
        get_models_fn: Callable[[], dict],
        swap_model_fn: Callable[[str, torch.nn.Module, str], None],
    ):
        self.cfg = cfg
        self.model_lock = model_lock
        self.torch_available = torch_available
        self.torch_device = torch_device
        self.get_models_fn = get_models_fn
        self.swap_model_fn = swap_model_fn

        self._stop_event = threading.Event()
        self._thread = None
        self._rnd = random.Random(42)

        self.status_lock = threading.Lock()
        self.status = {
            "enabled": bool(cfg.enabled),
            "started": False,
            "last_tick": None,
            "architecture": {
                "dual_loop": "gan_generator_plus_detector",
                "multimodal": ["image", "audio", "video", "metadata"],
                "auc_gate": float(cfg.max_auc_drop),
            },
            "image": {"state": "idle", "pool_size": 0, "message": "not started"},
            "audio": {"state": "idle", "pool_size": 0, "message": "not started"},
            "video": {"state": "idle", "pool_size": 0, "message": "not started"},
        }

        self._ensure_dirs()

    @staticmethod
    def from_env(
        base_dir: Path,
        model_dir: Path,
        model_lock: threading.Lock,
        torch_available: bool,
        torch_device: torch.device | None,
        get_models_fn: Callable[[], dict],
        swap_model_fn: Callable[[str, torch.nn.Module, str], None],
    ):
        cfg = PipelineConfig(
            enabled=os.getenv("ADV_PIPELINE_ENABLED", "true").lower() == "true",
            interval_sec=int(os.getenv("ADV_PIPELINE_INTERVAL_SEC", "120")),
            min_pool_samples=int(os.getenv("ADV_POOL_MIN_SAMPLES", "500")),
            fine_tune_epochs=int(os.getenv("ADV_FINE_TUNE_EPOCHS", "4")),
            max_auc_drop=float(os.getenv("ADV_MAX_AUC_DROP", "0.005")),
            image_gen_per_cycle=int(os.getenv("ADV_IMAGE_GEN_PER_CYCLE", "32")),
            image_max_attempts_factor=int(os.getenv("ADV_IMAGE_MAX_ATTEMPTS_FACTOR", "4")),
            image_fgsm_eps=float(os.getenv("ADV_IMAGE_FGSM_EPS", "0.010")),
            image_lr=float(os.getenv("ADV_IMAGE_FINE_TUNE_LR", "2e-6")),
            image_gan_command=os.getenv("ADV_IMAGE_GAN_COMMAND", "").strip(),
            audio_gen_per_cycle=int(os.getenv("ADV_AUDIO_GEN_PER_CYCLE", "32")),
            audio_max_attempts_factor=int(os.getenv("ADV_AUDIO_MAX_ATTEMPTS_FACTOR", "4")),
            audio_fgsm_eps=float(os.getenv("ADV_AUDIO_FGSM_EPS", "0.003")),
            audio_lr=float(os.getenv("ADV_AUDIO_FINE_TUNE_LR", "1e-6")),
            audio_gan_command=os.getenv("ADV_AUDIO_GAN_COMMAND", "").strip(),
            video_gan_command=os.getenv("ADV_VIDEO_GAN_COMMAND", "").strip(),
            image_source_dirs=_split_dirs(os.getenv("ADV_IMAGE_SOURCE_DIRS", "")),
            audio_source_dirs=_split_dirs(os.getenv("ADV_AUDIO_SOURCE_DIRS", "")),
            video_source_dirs=_split_dirs(os.getenv("ADV_VIDEO_SOURCE_DIRS", "")),
            image_pool_dir=Path(os.getenv("ADV_IMAGE_POOL_DIR", str(base_dir / "adversarial_pool" / "image"))),
            audio_pool_dir=Path(os.getenv("ADV_AUDIO_POOL_DIR", str(base_dir / "adversarial_pool" / "audio"))),
            video_pool_dir=Path(os.getenv("ADV_VIDEO_POOL_DIR", str(base_dir / "adversarial_pool" / "video"))),
            model_dir=model_dir,
        )
        return AdversarialBackgroundService(
            cfg=cfg,
            model_lock=model_lock,
            torch_available=torch_available,
            torch_device=torch_device,
            get_models_fn=get_models_fn,
            swap_model_fn=swap_model_fn,
        )

    def start(self):
        if not self.cfg.enabled:
            self._set_status_message("image", "disabled", "pipeline disabled by ADV_PIPELINE_ENABLED")
            self._set_status_message("audio", "disabled", "pipeline disabled by ADV_PIPELINE_ENABLED")
            self._set_status_message("video", "disabled", "pipeline disabled by ADV_PIPELINE_ENABLED")
            return

        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run_loop, name="adversarial-bg-pipeline", daemon=True)
        self._thread.start()
        with self.status_lock:
            self.status["started"] = True

    def stop(self):
        self._stop_event.set()

    def get_status(self):
        with self.status_lock:
            return copy.deepcopy(self.status)

    def _ensure_dirs(self):
        self.cfg.image_pool_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.audio_pool_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.video_pool_dir.mkdir(parents=True, exist_ok=True)

    def _run_loop(self):
        while not self._stop_event.is_set():
            now = datetime.utcnow().isoformat() + "Z"
            with self.status_lock:
                self.status["last_tick"] = now

            try:
                self._run_image_cycle()
            except Exception as e:
                self._set_status_message("image", "error", str(e))

            try:
                self._run_audio_cycle()
            except Exception as e:
                self._set_status_message("audio", "error", str(e))

            try:
                self._run_video_cycle()
            except Exception as e:
                self._set_status_message("video", "error", str(e))

            self._stop_event.wait(timeout=self.cfg.interval_sec)

    def _run_image_cycle(self):
        models = self.get_models_fn()
        model = models.get("image")
        if not self.torch_available or model is None or self.torch_device is None:
            self._set_pool_state("image", "waiting", "image model unavailable")
            return

        base_model = _clone_model(model)
        if base_model is None:
            self._set_pool_state("image", "error", "failed to clone image model for background cycle")
            return
        base_model = base_model.to(self.torch_device).eval()

        if not self.cfg.image_source_dirs:
            self._set_pool_state("image", "waiting", "ADV_IMAGE_SOURCE_DIRS not configured")
            return

        all_pairs = _scan_labeled_files(self.cfg.image_source_dirs, IMAGE_EXTS)
        if len(all_pairs) < 20:
            self._set_pool_state("image", "waiting", "not enough labeled image source data")
            return

        self._set_pool_state("image", "generating", f"source={len(all_pairs)}")
        made = 0
        if self.cfg.image_gan_command:
            made += self._run_gan_generator(
                modality="image",
                command=self.cfg.image_gan_command,
                pool_dir=self.cfg.image_pool_dir,
                fallback_pairs=all_pairs,
            )
        if made < self.cfg.image_gen_per_cycle:
            made += self._generate_image_adversarial_samples(base_model, all_pairs)
        pool_size = _pool_size(self.cfg.image_pool_dir)

        if made > 0:
            self._set_pool_state("image", "generated", f"added={made} pool={pool_size}")

        if pool_size < self.cfg.min_pool_samples:
            self._set_pool_state("image", "collecting", f"pool={pool_size}/{self.cfg.min_pool_samples}")
            return

        self._set_pool_state("image", "finetuning", f"pool={pool_size}")
        accepted, msg = self._fine_tune_image(base_model, all_pairs)
        if accepted:
            self._set_pool_state("image", "accepted", msg)
            _reset_pool(self.cfg.image_pool_dir)
        else:
            self._set_pool_state("image", "rejected", msg)

    def _run_audio_cycle(self):
        models = self.get_models_fn()
        model = models.get("audio")
        if not self.torch_available or model is None or self.torch_device is None:
            self._set_pool_state("audio", "waiting", "audio model unavailable")
            return

        base_model = _clone_model(model)
        if base_model is None:
            self._set_pool_state("audio", "error", "failed to clone audio model for background cycle")
            return
        base_model = base_model.to(self.torch_device).eval()

        if not self.cfg.audio_source_dirs:
            self._set_pool_state("audio", "waiting", "ADV_AUDIO_SOURCE_DIRS not configured")
            return

        all_pairs = _scan_labeled_files(self.cfg.audio_source_dirs, AUDIO_EXTS)
        if len(all_pairs) < 20:
            self._set_pool_state("audio", "waiting", "not enough labeled audio source data")
            return

        self._set_pool_state("audio", "generating", f"source={len(all_pairs)}")
        made = 0
        if self.cfg.audio_gan_command:
            made += self._run_gan_generator(
                modality="audio",
                command=self.cfg.audio_gan_command,
                pool_dir=self.cfg.audio_pool_dir,
                fallback_pairs=all_pairs,
            )
        if made < self.cfg.audio_gen_per_cycle:
            made += self._generate_audio_adversarial_samples(base_model, all_pairs)
        pool_size = _pool_size(self.cfg.audio_pool_dir)

        if made > 0:
            self._set_pool_state("audio", "generated", f"added={made} pool={pool_size}")

        if pool_size < self.cfg.min_pool_samples:
            self._set_pool_state("audio", "collecting", f"pool={pool_size}/{self.cfg.min_pool_samples}")
            return

        self._set_pool_state("audio", "finetuning", f"pool={pool_size}")
        accepted, msg = self._fine_tune_audio(base_model, all_pairs)
        if accepted:
            self._set_pool_state("audio", "accepted", msg)
            _reset_pool(self.cfg.audio_pool_dir)
        else:
            self._set_pool_state("audio", "rejected", msg)

    def _run_video_cycle(self):
        models = self.get_models_fn()
        video_model = models.get("video")
        pool_size = _pool_size(self.cfg.video_pool_dir)

        if video_model is None:
            self._set_pool_state(
                "video",
                "waiting",
                "video adversarial pipeline scaffold ready; starts automatically after video model is loaded",
                pool_size=pool_size,
            )
            return

        if not self.cfg.video_source_dirs:
            self._set_pool_state(
                "video",
                "waiting",
                "video model detected; set ADV_VIDEO_SOURCE_DIRS to enable background generation",
                pool_size=pool_size,
            )
            return

        labeled = _scan_labeled_files(self.cfg.video_source_dirs, VIDEO_EXTS)
        made = 0
        if self.cfg.video_gan_command:
            made = self._run_gan_generator(
                modality="video",
                command=self.cfg.video_gan_command,
                pool_dir=self.cfg.video_pool_dir,
                fallback_pairs=labeled,
            )
            pool_size = _pool_size(self.cfg.video_pool_dir)
        self._set_pool_state(
            "video",
            "ready",
            f"video model active; source_videos={len(labeled)}; generated={made}; pool={pool_size}",
            pool_size=pool_size,
        )

    def _run_gan_generator(self, modality: str, command: str, pool_dir: Path, fallback_pairs: list[tuple[str, int]]) -> int:
        env = os.environ.copy()
        env["ADV_MODALITY"] = modality
        env["ADV_POOL_DIR"] = str(pool_dir)
        env["ADV_TARGET_NEW_SAMPLES"] = str(
            self.cfg.image_gen_per_cycle if modality == "image" else self.cfg.audio_gen_per_cycle
        )
        env["ADV_SOURCE_COUNT"] = str(len(fallback_pairs))

        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.returncode != 0:
            self._set_pool_state(modality, "warning", f"GAN generator failed, fallback enabled: {proc.stderr.strip()}")
            return 0

        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        payload = None
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    payload = obj
                    break
            except Exception:
                continue

        if payload is None:
            self._set_pool_state(modality, "warning", "GAN output missing JSON payload, fallback enabled")
            return 0

        records = payload.get("generated", [])
        if not isinstance(records, list):
            self._set_pool_state(modality, "warning", "GAN output invalid generated schema, fallback enabled")
            return 0

        manifest_path = pool_dir / "pool_manifest.jsonl"
        added = 0
        for row in records:
            if not isinstance(row, dict):
                continue
            path = row.get("path")
            label = row.get("label")
            if path is None or label is None:
                continue
            if not os.path.exists(path):
                continue

            try:
                label_int = int(label)
            except Exception:
                continue

            rec = {
                "path": str(path),
                "label": label_int,
                "source": row.get("source", "gan_generator"),
                "attack": row.get("attack", "gan"),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "metadata": row.get("metadata", {}),
            }
            _append_jsonl(manifest_path, rec)
            added += 1

        return added

    def _generate_image_adversarial_samples(self, model, all_pairs: list[tuple[str, int]]) -> int:
        targets = self.cfg.image_gen_per_cycle
        max_attempts = max(1, targets * self.cfg.image_max_attempts_factor)
        pool_manifest = self.cfg.image_pool_dir / "pool_manifest.jsonl"
        added = 0

        candidates = all_pairs.copy()
        self._rnd.shuffle(candidates)
        attempts = 0

        for path, label in candidates:
            if added >= targets or attempts >= max_attempts:
                break
            attempts += 1

            x = _load_image_tensor(path)
            if x is None:
                continue

            y = torch.tensor([float(label)], dtype=torch.float32, device=self.torch_device)
            x = x.to(self.torch_device)

            with torch.no_grad():
                clean_prob = torch.sigmoid(model(x)).item()
            clean_pred = int(clean_prob >= 0.5)

            x_adv = x.detach().clone().requires_grad_(True)
            loss = F.binary_cross_entropy_with_logits(model(x_adv), y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                x_adv = x + self.cfg.image_fgsm_eps * x_adv.grad.sign()
                adv_prob = torch.sigmoid(model(x_adv)).item()

            adv_pred = int(adv_prob >= 0.5)
            if clean_pred == label and adv_pred != label:
                out_name = f"img_adv_{int(time.time() * 1000)}_{self._rnd.randint(1000, 9999)}_l{label}.png"
                out_path = self.cfg.image_pool_dir / out_name
                _save_image_tensor(x_adv.detach().cpu(), out_path)
                _append_jsonl(
                    pool_manifest,
                    {
                        "path": str(out_path),
                        "label": int(label),
                        "source": path,
                        "attack": "fgsm",
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "metadata": _extract_image_metadata(path),
                    },
                )
                added += 1

        return added

    def _generate_audio_adversarial_samples(self, model, all_pairs: list[tuple[str, int]]) -> int:
        targets = self.cfg.audio_gen_per_cycle
        max_attempts = max(1, targets * self.cfg.audio_max_attempts_factor)
        pool_manifest = self.cfg.audio_pool_dir / "pool_manifest.jsonl"
        added = 0

        candidates = all_pairs.copy()
        self._rnd.shuffle(candidates)
        attempts = 0

        for path, label in candidates:
            if added >= targets or attempts >= max_attempts:
                break
            attempts += 1

            wav = _load_audio(path)
            if wav is None:
                continue

            x = torch.tensor(wav, dtype=torch.float32, device=self.torch_device).unsqueeze(0)
            y = torch.tensor([float(label)], dtype=torch.float32, device=self.torch_device)

            with torch.no_grad():
                clean_prob = torch.sigmoid(model(x)).item()
            clean_pred = int(clean_prob >= 0.5)

            x_adv = x.detach().clone().requires_grad_(True)
            loss = F.binary_cross_entropy_with_logits(model(x_adv), y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                x_adv = torch.clamp(x + self.cfg.audio_fgsm_eps * x_adv.grad.sign(), -1.0, 1.0)
                adv_prob = torch.sigmoid(model(x_adv)).item()

            adv_pred = int(adv_prob >= 0.5)
            if clean_pred == label and adv_pred != label:
                out_name = f"aud_adv_{int(time.time() * 1000)}_{self._rnd.randint(1000, 9999)}_l{label}.npy"
                out_path = self.cfg.audio_pool_dir / out_name
                np.save(str(out_path), x_adv.detach().cpu().squeeze(0).numpy().astype(np.float32))
                _append_jsonl(
                    pool_manifest,
                    {
                        "path": str(out_path),
                        "label": int(label),
                        "source": path,
                        "attack": "fgsm",
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "metadata": _extract_audio_metadata(path),
                    },
                )
                added += 1

        return added

    def _fine_tune_image(self, model, original_pairs: list[tuple[str, int]]):
        adv_pairs = _load_pool_pairs(self.cfg.image_pool_dir)
        if len(adv_pairs) < self.cfg.min_pool_samples:
            return False, f"pool dropped below minimum during cycle ({len(adv_pairs)})"

        adv_real, adv_fake = _count_labels(adv_pairs)
        if adv_real == 0 or adv_fake == 0:
            return (
                False,
                f"pool class imbalance rejected: real={adv_real}, fake={adv_fake}. "
                "Need both classes before fine-tune.",
            )

        holdout = _build_holdout(original_pairs, max_per_class=250)
        if len(holdout) < 40:
            return False, "insufficient holdout for AUC gate"

        with torch.no_grad():
            baseline_auc = _eval_auc_image(model, holdout, self.torch_device)

        n_adv = len(adv_pairs)
        n_orig = min(len(original_pairs), int((0.70 / 0.30) * n_adv))
        if n_orig < 1:
            return False, "insufficient original data for 70/30 mix"

        orig_sample = self._rnd.sample(original_pairs, n_orig)
        train_pairs = orig_sample + adv_pairs
        self._rnd.shuffle(train_pairs)

        tr_real, tr_fake = _count_labels(train_pairs)
        if tr_real == 0 or tr_fake == 0:
            return (
                False,
                f"train mix invalid after sampling: real={tr_real}, fake={tr_fake}. "
                "Aborting fine-tune.",
            )

        candidate = _clone_model(model)
        if candidate is None:
            return False, "failed to clone image model"

        optimizer = torch.optim.AdamW(candidate.parameters(), lr=self.cfg.image_lr, weight_decay=1e-4)
        loader = _make_loader_image(train_pairs)

        candidate.train()
        for _ in range(max(1, self.cfg.fine_tune_epochs)):
            for x, y in loader:
                x = x.to(self.torch_device, non_blocking=True)
                y = y.to(self.torch_device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = candidate(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            candidate_auc = _eval_auc_image(candidate, holdout, self.torch_device)

        if candidate_auc + self.cfg.max_auc_drop < baseline_auc:
            return False, f"AUC gate rejected: baseline={baseline_auc:.4f} candidate={candidate_auc:.4f}"

        model_name = f"imageguard_pool_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"
        out_path = self.cfg.model_dir / model_name
        torch.save({"model_state_dict": candidate.state_dict(), "dev_auc": float(candidate_auc)}, out_path)
        self.swap_model_fn("image", candidate, model_name)
        return True, f"accepted baseline={baseline_auc:.4f} candidate={candidate_auc:.4f}"

    def _fine_tune_audio(self, model, original_pairs: list[tuple[str, int]]):
        adv_pairs = _load_pool_pairs(self.cfg.audio_pool_dir)
        if len(adv_pairs) < self.cfg.min_pool_samples:
            return False, f"pool dropped below minimum during cycle ({len(adv_pairs)})"

        adv_real, adv_fake = _count_labels(adv_pairs)
        if adv_real == 0 or adv_fake == 0:
            return (
                False,
                f"pool class imbalance rejected: real={adv_real}, fake={adv_fake}. "
                "Need both classes before fine-tune.",
            )

        holdout = _build_holdout(original_pairs, max_per_class=250)
        if len(holdout) < 40:
            return False, "insufficient holdout for AUC gate"

        with torch.no_grad():
            baseline_auc = _eval_auc_audio(model, holdout, self.torch_device)

        n_adv = len(adv_pairs)
        n_orig = min(len(original_pairs), int((0.70 / 0.30) * n_adv))
        if n_orig < 1:
            return False, "insufficient original data for 70/30 mix"

        orig_sample = self._rnd.sample(original_pairs, n_orig)
        train_pairs = orig_sample + adv_pairs
        self._rnd.shuffle(train_pairs)

        tr_real, tr_fake = _count_labels(train_pairs)
        if tr_real == 0 or tr_fake == 0:
            return (
                False,
                f"train mix invalid after sampling: real={tr_real}, fake={tr_fake}. "
                "Aborting fine-tune.",
            )

        candidate = _clone_model(model)
        if candidate is None:
            return False, "failed to clone audio model"

        optimizer = torch.optim.AdamW(candidate.parameters(), lr=self.cfg.audio_lr, weight_decay=1e-4)
        loader = _make_loader_audio(train_pairs)

        candidate.train()
        for _ in range(max(1, self.cfg.fine_tune_epochs)):
            for x, y in loader:
                x = x.to(self.torch_device, non_blocking=True)
                y = y.to(self.torch_device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = candidate(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            candidate_auc = _eval_auc_audio(candidate, holdout, self.torch_device)

        if candidate_auc + self.cfg.max_auc_drop < baseline_auc:
            return False, f"AUC gate rejected: baseline={baseline_auc:.4f} candidate={candidate_auc:.4f}"

        model_name = f"rawnet3_pool_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"
        out_path = self.cfg.model_dir / model_name
        torch.save({"model_state_dict": candidate.state_dict(), "dev_auc": float(candidate_auc)}, out_path)
        self.swap_model_fn("audio", candidate, model_name)
        return True, f"accepted baseline={baseline_auc:.4f} candidate={candidate_auc:.4f}"

    def _set_pool_state(self, modality: str, state: str, msg: str, pool_size: int | None = None):
        if pool_size is None:
            if modality == "image":
                pool_size = _pool_size(self.cfg.image_pool_dir)
            elif modality == "audio":
                pool_size = _pool_size(self.cfg.audio_pool_dir)
            else:
                pool_size = _pool_size(self.cfg.video_pool_dir)

        with self.status_lock:
            self.status[modality] = {
                "state": state,
                "pool_size": int(pool_size),
                "message": msg,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }

    def _set_status_message(self, modality: str, state: str, msg: str):
        self._set_pool_state(modality, state, msg)


def _clone_model(model):
    try:
        return copy.deepcopy(model)
    except Exception:
        return None


def _split_dirs(raw: str) -> list[str]:
    items = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            items.append(p)
    return items


def _scan_labeled_files(source_dirs: list[str], exts: set[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for root in source_dirs:
        rp = Path(root)
        if not rp.exists():
            continue
        for p in rp.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            if p.stat().st_size < 200:
                continue
            label = _infer_label(p)
            if label is None:
                continue
            out.append((str(p), int(label)))
    return out


def _infer_label(path: str | Path):
    for part in reversed(Path(path).parts):
        token = part.lower().replace("_", " ").replace("-", " ")
        words = set(token.split()) | {token}
        if words & REAL_TOKENS:
            return 0
        if words & FAKE_TOKENS:
            return 1
    return None


def _load_image_tensor(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


def _save_image_tensor(x_adv: torch.Tensor, out_path: Path):
    x = x_adv.squeeze(0).numpy()
    x = np.transpose(x, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = x * std + mean
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def _load_audio(path: str, sr: int = 16000, n_samples: int = 64000):
    try:
        wav, _ = librosa.load(path, sr=sr, duration=n_samples / sr)
        if len(wav) < n_samples:
            wav = np.pad(wav, (0, n_samples - len(wav)))
        else:
            wav = wav[:n_samples]
        wav = wav.astype(np.float32)
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-6:
            wav = wav / peak
        return wav
    except Exception:
        return None


def _extract_image_metadata(path: str) -> dict:
    try:
        img = cv2.imread(path)
        if img is None:
            return {}
        h, w = img.shape[:2]
        return {
            "width": int(w),
            "height": int(h),
            "channels": int(img.shape[2]) if img.ndim == 3 else 1,
        }
    except Exception:
        return {}


def _extract_audio_metadata(path: str) -> dict:
    try:
        duration = float(librosa.get_duration(path=path))
        sr = int(librosa.get_samplerate(path))
        return {
            "duration_sec": duration,
            "sample_rate": sr,
        }
    except Exception:
        return {}


def _extract_video_metadata(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = float(frames / fps) if fps > 0 else 0.0
        return {
            "fps": fps,
            "frame_count": frames,
            "width": width,
            "height": height,
            "duration_sec": duration,
        }
    finally:
        cap.release()


def _append_jsonl(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _pool_size(pool_dir: Path) -> int:
    return len(_load_pool_pairs(pool_dir))


def _load_pool_pairs(pool_dir: Path) -> list[tuple[str, int]]:
    manifest = pool_dir / "pool_manifest.jsonl"
    if not manifest.exists():
        return []

    out: list[tuple[str, int]] = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                p = row.get("path")
                l = int(row.get("label"))
                if p and os.path.exists(p):
                    out.append((str(p), l))
            except Exception:
                continue
    return out


def _build_holdout(pairs: list[tuple[str, int]], max_per_class: int = 250) -> list[tuple[str, int]]:
    real = [p for p in pairs if p[1] == 0]
    fake = [p for p in pairs if p[1] == 1]
    random.shuffle(real)
    random.shuffle(fake)
    return real[:max_per_class] + fake[:max_per_class]


def _count_labels(pairs: list[tuple[str, int]]) -> tuple[int, int]:
    n_real = sum(1 for _, label in pairs if int(label) == 0)
    n_fake = sum(1 for _, label in pairs if int(label) == 1)
    return n_real, n_fake


def _make_loader_image(pairs: list[tuple[str, int]]):
    labels = [l for _, l in pairs]
    n0 = max(labels.count(0), 1)
    n1 = max(labels.count(1), 1)
    weights = torch.DoubleTensor([1.0 / n0 if l == 0 else 1.0 / n1 for l in labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(ImagePathDataset(pairs), batch_size=16, sampler=sampler, num_workers=0, pin_memory=False)


def _make_loader_audio(pairs: list[tuple[str, int]]):
    labels = [l for _, l in pairs]
    n0 = max(labels.count(0), 1)
    n1 = max(labels.count(1), 1)
    weights = torch.DoubleTensor([1.0 / n0 if l == 0 else 1.0 / n1 for l in labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(AudioPathDataset(pairs), batch_size=12, sampler=sampler, num_workers=0, pin_memory=False)


def _eval_auc_image(model, holdout_pairs: list[tuple[str, int]], device: torch.device):
    loader = DataLoader(ImagePathDataset(holdout_pairs), batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(prob.tolist())
    if len(set(int(v) for v in ys)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(ys), np.array(ps)))


def _eval_auc_audio(model, holdout_pairs: list[tuple[str, int]], device: torch.device):
    loader = DataLoader(AudioPathDataset(holdout_pairs), batch_size=24, shuffle=False, num_workers=0)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(prob.tolist())
    if len(set(int(v) for v in ys)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(ys), np.array(ps)))


def _reset_pool(pool_dir: Path):
    manifest = pool_dir / "pool_manifest.jsonl"
    if manifest.exists():
        try:
            manifest.unlink()
        except Exception:
            pass

    for p in pool_dir.glob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass
