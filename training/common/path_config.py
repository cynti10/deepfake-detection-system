from __future__ import annotations

import os
from pathlib import Path


def resolve_base_dir(modality: str) -> Path:
    """Resolve modality base dir from env with sensible home-dir fallback.

    Env vars:
      - DEEPFAKE_IMAGE_BASE_DIR
      - DEEPFAKE_AUDIO_BASE_DIR
      - DEEPFAKE_VIDEO_BASE_DIR
    """
    key = f"DEEPFAKE_{modality.upper()}_BASE_DIR"
    default = Path.home() / "deepfake_project" / modality.lower()
    return Path(os.getenv(key, str(default)))


def build_default_paths(modality: str) -> dict[str, Path]:
    base = resolve_base_dir(modality)
    return {
        "base": base,
        "data": base / "data",
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
        "test_data": base / "test_data",
    }


def ensure_dirs(paths: dict[str, Path], keys: tuple[str, ...] = ("checkpoints", "logs")) -> None:
    for key in keys:
        path = paths.get(key)
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
