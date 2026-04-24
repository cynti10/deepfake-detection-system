import argparse
import hashlib
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
REAL_TOKENS = {"real", "original", "authentic", "genuine", "true"}
FAKE_TOKENS = {
    "fake",
    "deepfake",
    "deepfakes",
    "synth",
    "synthetic",
    "generated",
    "faceswap",
    "face2face",
    "faceshifter",
    "neuraltextures",
}
SPLIT_TOKENS = {"train", "val", "validation", "test"}


def normalize_token(s: str):
    out = []
    cur = []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def infer_label(path: Path):
    for part in reversed(path.parts):
        toks = set(normalize_token(part))
        if toks.intersection(REAL_TOKENS):
            return 0
        if toks.intersection(FAKE_TOKENS):
            return 1
    return None


def infer_source_dataset(video_path: Path, root: Path):
    rel = video_path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else root.name


def infer_identity_id(video_path: Path):
    # Use parent folders as identity proxy while skipping split/class tokens.
    parts = list(video_path.parts)
    for part in reversed(parts[:-1]):
        toks = set(normalize_token(part))
        if toks.intersection(REAL_TOKENS) or toks.intersection(FAKE_TOKENS) or toks.intersection(SPLIT_TOKENS):
            continue
        return part
    return video_path.stem


def infer_manipulation_type(video_path: Path, label: int):
    if label == 0:
        return "real"
    joined = " ".join(video_path.parts).lower()
    if "face2face" in joined:
        return "Face2Face"
    if "faceswap" in joined:
        return "FaceSwap"
    if "faceshifter" in joined:
        return "FaceShifter"
    if "neuraltextures" in joined:
        return "NeuralTextures"
    if "deepfakedetection" in joined:
        return "DeepFakeDetection"
    return "Deepfakes"


def hash_split(group_key: str, train_ratio: float, val_ratio: float):
    h = int(hashlib.md5(group_key.encode("utf-8")).hexdigest(), 16) % 10000
    x = h / 10000.0
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def probe_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return frame_count, fps


def collect_videos(roots):
    videos = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[WARN] root not found: {root}")
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                videos.append((root, p))
    return videos


def main():
    parser = argparse.ArgumentParser(description="Build unified train/val/test video manifests.")
    parser.add_argument("--roots", nargs="+", required=True, help="One or more dataset roots containing real/fake videos.")
    parser.add_argument("--out-dir", required=True, help="Output directory for manifests.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--probe-metadata", action="store_true", help="Probe fps/frame_count (slower).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    videos = collect_videos(args.roots)
    print(f"Discovered {len(videos)} video files.")

    for root, p in tqdm(videos, desc="Building manifest"):
        label = infer_label(p)
        if label is None:
            continue

        source_dataset = infer_source_dataset(p, root)
        identity_id = infer_identity_id(p)
        video_id = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16]

        if args.probe_metadata:
            n_frames, fps = probe_video(p)
        else:
            n_frames, fps = 0, 0.0

        manipulation = infer_manipulation_type(p, label)
        group_key = f"{source_dataset}::{video_id}"
        split = hash_split(group_key, args.train_ratio, args.val_ratio)

        rows.append(
            {
                "sample_id": f"vid_{video_id}",
                "source_dataset": source_dataset,
                "identity_id": identity_id,
                "video_id": video_id,
                "video_path": str(p),
                "label": int(label),
                "split": split,
                "manipulation_type": manipulation,
                "generator_hint": "unknown",
                "compression": "unknown",
                "num_frames": int(n_frames),
                "fps": float(fps),
            }
        )

    if not rows:
        raise RuntimeError("No labeled videos found. Check directory names for real/fake tokens.")

    df = pd.DataFrame(rows).drop_duplicates("sample_id").reset_index(drop=True)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    df.to_csv(out_dir / "video_master_manifest.csv", index=False)
    train_df.to_csv(out_dir / "video_train_manifest.csv", index=False)
    val_df.to_csv(out_dir / "video_val_manifest.csv", index=False)
    test_df.to_csv(out_dir / "video_test_manifest.csv", index=False)

    print("Saved:")
    print(out_dir / "video_master_manifest.csv")
    print(out_dir / "video_train_manifest.csv")
    print(out_dir / "video_val_manifest.csv")
    print(out_dir / "video_test_manifest.csv")

    print("\nCounts by split/label:")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()

