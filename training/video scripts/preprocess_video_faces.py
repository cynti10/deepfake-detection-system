import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def center_face_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


class FaceCropper:
    def __init__(self, mode="haar", device="cpu"):
        self.mode = mode
        self.mtcnn = None
        self.haar = None

        if mode == "mtcnn":
            try:
                from facenet_pytorch import MTCNN

                self.mtcnn = MTCNN(keep_all=False, device=device)
                print("Using MTCNN detector.")
            except Exception as e:
                print(f"[WARN] MTCNN unavailable ({e}). Falling back to haar.")
                self.mode = "haar"

        if self.mode == "haar":
            cascade = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            self.haar = cv2.CascadeClassifier(cascade)
            if self.haar.empty():
                print("[WARN] Haar cascade unavailable. Falling back to center crop.")
                self.mode = "none"

    def crop(self, frame_bgr):
        if self.mode == "mtcnn" and self.mtcnn is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame_bgr.shape[1], int(x2))
                y2 = min(frame_bgr.shape[0], int(y2))
                if x2 > x1 and y2 > y1:
                    return frame_bgr[y1:y2, x1:x2]

        if self.mode == "haar" and self.haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                return frame_bgr[y : y + h, x : x + w]

        return center_face_crop(frame_bgr)


def sample_indices(frame_count, frames_per_video):
    if frame_count <= 0:
        return []
    if frame_count <= frames_per_video:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, frames_per_video).astype(int).tolist()


def sha1_text(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from videos and build sequence manifest.")
    parser.add_argument("--video-manifest", required=True, help="Path to video_*_manifest.csv or video_master_manifest.csv")
    parser.add_argument("--out-root", required=True, help="Output root directory for extracted frames and sequence manifests")
    parser.add_argument("--frames-per-video", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--detector", choices=["mtcnn", "haar", "none"], default="haar")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (for mtcnn)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    frames_root = out_root / "frames"
    seq_root = out_root / "manifests"
    frames_root.mkdir(parents=True, exist_ok=True)
    seq_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.video_manifest)
    required = {
        "sample_id",
        "source_dataset",
        "identity_id",
        "video_id",
        "video_path",
        "label",
        "split",
        "manipulation_type",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    cropper = FaceCropper(mode=args.detector, device=args.device)

    seq_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        video_path = Path(row["video_path"])
        if not video_path.exists():
            continue

        split = str(row["split"])
        video_id = str(row["video_id"])
        out_dir = frames_root / split / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing:
            existing = sorted(out_dir.glob("*.jpg"))
            if len(existing) >= args.frames_per_video:
                frame_paths = [str(p) for p in existing[: args.frames_per_video]]
                seq_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "source_dataset": row["source_dataset"],
                        "identity_id": row["identity_id"],
                        "video_id": row["video_id"],
                        "label": int(row["label"]),
                        "split": split,
                        "manipulation_type": row["manipulation_type"],
                        "sequence_len": len(frame_paths),
                        "frame_paths": "|".join(frame_paths),
                        "sequence_id": sha1_text("|".join(frame_paths)),
                    }
                )
                continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_indices(frame_count, args.frames_per_video)

        saved = []
        for i, frame_idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            crop = cropper.crop(frame)
            crop = cv2.resize(crop, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

            out_path = out_dir / f"f_{i:03d}.jpg"
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            saved.append(str(out_path))

        cap.release()

        if len(saved) == 0:
            continue

        seq_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "identity_id": row["identity_id"],
                "video_id": row["video_id"],
                "label": int(row["label"]),
                "split": split,
                "manipulation_type": row["manipulation_type"],
                "sequence_len": len(saved),
                "frame_paths": "|".join(saved),
                "sequence_id": sha1_text("|".join(saved)),
            }
        )

    if not seq_rows:
        raise RuntimeError("No sequences extracted. Check manifest paths and video accessibility.")

    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(seq_root / "sequence_master_manifest.csv", index=False)
    seq_df[seq_df["split"] == "train"].to_csv(seq_root / "sequence_train_manifest.csv", index=False)
    seq_df[seq_df["split"] == "val"].to_csv(seq_root / "sequence_val_manifest.csv", index=False)
    seq_df[seq_df["split"] == "test"].to_csv(seq_root / "sequence_test_manifest.csv", index=False)

    print("Saved sequence manifests:")
    print(seq_root / "sequence_master_manifest.csv")
    print(seq_root / "sequence_train_manifest.csv")
    print(seq_root / "sequence_val_manifest.csv")
    print(seq_root / "sequence_test_manifest.csv")
    print("\nCounts:")
    print(seq_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()

