import subprocess
import sys
from pathlib import Path
import json

ROOT = Path(r"c:\D-drive\deepfake\deepfake-detection-system")
CHECKPOINT = ROOT / "backend" / "models" / "video_best_model.pt"
OUT = []

video_dirs = [
    ROOT / "backend" / "uploads" / "video_1",
    ROOT / "backend" / "uploads" / "video_1" / "video_2",
]

for d in video_dirs:
    if not d.exists():
        continue
    for p in sorted(d.glob("*.mp4")):
        cmd = [sys.executable, "infer_video.py", "--checkpoint", str(CHECKPOINT), "--video", str(p), "--out-json", str(p.with_suffix('.json'))]
        print("RUN:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        record = {"video": str(p), "stdout": out, "stderr": err, "returncode": proc.returncode}
        print(json.dumps(record, indent=2))
        OUT.append(record)

with open("batch_infer_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2)

print("Saved: batch_infer_results.json")
