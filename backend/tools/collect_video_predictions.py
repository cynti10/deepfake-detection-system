import os
import sys
import time
import json
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS = BASE_DIR / "uploads"
OUT_DIR = Path(__file__).resolve().parent

SERVER = os.getenv("DEEPFAKE_SERVER", "http://127.0.0.1:5000")
HEALTH_URL = f"{SERVER}/health"
DETECT_URL = f"{SERVER}/detect-video"

FOLDERS = ["video_1", "video_2"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def post_file(path):
    try:
        with open(path, "rb") as fh:
            r = requests.post(DETECT_URL, files={"file": fh}, timeout=120)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"error": "non-json response", "text": r.text}
    except Exception as e:
        return None, {"error": str(e)}


def summarize(results):
    counts = {"FAKE": 0, "REAL": 0, "ERROR": 0}
    sum_conf = 0.0
    sum_prob = 0.0
    sum_raw = 0.0
    n = 0
    for r in results:
        if "result" in r and r["result"] in ("FAKE", "REAL"):
            counts[r["result"]] += 1
            sum_conf += float(r.get("confidence", 0.0) or 0.0)
            sum_prob += float(r.get("fake_probability", 0.0) or 0.0)
            sum_raw += float(r.get("raw_fake_probability", 0.0) or 0.0)
            n += 1
        else:
            counts["ERROR"] += 1
    avg_conf = (sum_conf / n) if n else 0.0
    avg_prob = (sum_prob / n) if n else 0.0
    avg_raw = (sum_raw / n) if n else 0.0
    return counts, avg_conf, avg_prob, avg_raw


def main():
    print(f"Checking server health: {HEALTH_URL}")
    try:
        h = requests.get(HEALTH_URL, timeout=10)
        print("Health status:", h.status_code)
    except Exception as e:
        print("Could not reach server:", e)
        sys.exit(2)

    aggregate = {}
    for folder in FOLDERS:
        path = UPLOADS / folder
        if not path.exists():
            print(f"Folder missing: {path}")
            aggregate[folder] = {"error": "missing_folder"}
            continue
        files = [f for f in sorted(path.iterdir()) if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
        print(f"Found {len(files)} video files in {folder}")
        results = []
        for f in files:
            print(f"Posting {f.name}...")
            status, data = post_file(str(f))
            entry = {"file": f.name, "status_code": status, "response": data}
            # normalize top-level for easier analysis
            if isinstance(data, dict):
                entry.update({
                    "result": data.get("result"),
                    "confidence": data.get("confidence"),
                    "fake_probability": data.get("fake_probability"),
                    "raw_fake_probability": data.get("raw_fake_probability"),
                })
            results.append(entry)
            time.sleep(0.2)

        out_path = OUT_DIR / f"results_{folder}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        counts, avg_conf, avg_prob, avg_raw = summarize([r["response"] if isinstance(r.get("response"), dict) else r for r in results])
        print(f"Summary for {folder}: {counts}, avg_conf={avg_conf:.3f}, avg_fake_prob={avg_prob:.3f}, avg_raw={avg_raw:.3f}")
        aggregate[folder] = {"counts": counts, "avg_conf": avg_conf, "avg_fake_prob": avg_prob, "avg_raw": avg_raw, "out_path": str(out_path)}

    ag_out = OUT_DIR / "results_aggregate.json"
    with open(ag_out, "w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)
    print("Wrote aggregate:", ag_out)


if __name__ == "__main__":
    main()
