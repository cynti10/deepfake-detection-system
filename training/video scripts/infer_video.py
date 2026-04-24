import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

from preprocess_video_faces import FaceCropper, sample_indices
from train_video_detector import SentinelVideoDetector, load_rgb


def main():
    parser = argparse.ArgumentParser(description="Inference on a single video with Sentinel video detector.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--detector", choices=["mtcnn", "haar", "none"], default="haar")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-json", default="video_inference.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})

    model = SentinelVideoDetector(
        rgb_backbone=cfg.get("rgb_backbone", "tf_efficientnet_b3_ns"),
        d_model=int(cfg.get("d_model", 512)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 2)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(frame_count, args.frames)
    cropper = FaceCropper(mode=args.detector, device=("cuda" if device.type == "cuda" else "cpu"))

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        crop = cropper.crop(frame)
        crop = cv2.resize(crop, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

        tmp_path = Path("_tmp_infer_frame.jpg")
        cv2.imwrite(str(tmp_path), crop)
        x = load_rgb(str(tmp_path), args.image_size)
        if tmp_path.exists():
            tmp_path.unlink()
        if x is not None:
            frames.append(x)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No usable frames extracted for inference.")

    while len(frames) < args.frames:
        frames.append(frames[-1])
    if len(frames) > args.frames:
        frames = frames[: args.frames]

    x = torch.tensor(np.stack(frames, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=amp_enabled):
            prob = torch.sigmoid(model(x)).item()

    result = {
        "video": str(Path(args.video)),
        "checkpoint": str(Path(args.checkpoint)),
        "frames_used": int(args.frames),
        "fake_probability": float(prob),
        "prediction": "fake" if prob >= 0.5 else "real",
    }

    print(json.dumps(result, indent=2))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()

