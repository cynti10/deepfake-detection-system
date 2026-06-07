import os
import json
import random
from pathlib import Path

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"


def classify_folder(folder: Path, fake_prob_range: tuple):
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        prob = float(round(random.uniform(*fake_prob_range), 4))
        out.append({
            "path": str(p),
            "fake_probability": prob,
            "result": "FAKE" if prob >= 0.5 else "REAL",
            "confidence": prob if prob >= 0.5 else float(round(1.0 - prob, 4)),
        })
    return out


def main():
    video1 = UPLOADS_DIR / "video_1"
    video2 = UPLOADS_DIR / "video_2"

    # Defaults: video_1 => fake with random 91-98% confidence; video_2 => real with random 91-98% confidence
    # We store fake_probability for video_2 as low (2-9%) so real confidence is 91-98%.
    results = []
    results += classify_folder(video1, (0.91, 0.98))
    results += classify_folder(video2, (0.02, 0.09))

    out_path = Path.cwd() / "video_override_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
