import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
MANIFEST = MODEL_DIR / "models_manifest.json"


def download(url, dest: Path):
    req = Request(url, headers={"User-Agent": "python-urllib/3"})
    with urlopen(req) as resp, open(dest, "wb") as out:
        chunk = resp.read(8192)
        while chunk:
            out.write(chunk)
            chunk = resp.read(8192)


def main():
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        print("Copy models/models_manifest.json.example to models/models_manifest.json and add real URLs.")
        sys.exit(1)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data.get("models", []):
        fname = entry.get("filename")
        url = entry.get("url")
        if not fname or not url:
            print(f"Skipping malformed entry: {entry}")
            continue

        dest = MODEL_DIR / fname
        if dest.exists():
            print(f"Skipping existing: {dest.name}")
            continue

        print(f"Downloading {fname} from {url}...")
        try:
            download(url, dest)
            print(f"Saved to {dest}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    main()
