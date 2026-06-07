Setup for team — models, dependencies, and run steps

1) Recommended: use Git LFS for model checkpoints

- Install Git LFS (one-time per machine):
  - macOS / Linux: `git lfs install`
  - Windows: install from https://git-lfs.github.com/ then `git lfs install`

- Ensure `.gitattributes` is present in `backend/` (this repo includes one).

2) Populate the `backend/models` directory

- Option A (recommended): use the manifest and downloader:
  - Copy `backend/models/models_manifest.json.example` to `backend/models/models_manifest.json` and edit URLs.
  - Run:
    ```
    python scripts/download_models.py
    ```

- Option B (if you share models via file share or scp): place the `.pt` files directly into `backend/models/`.

3) Install Python dependencies

- Create a virtual environment and install requirements using the repository-level `requirements.txt`:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows (PowerShell)
  pip install -r requirements.txt
  ```

4) Run the backend server

```bash
python app.py
```

Notes
- If you prefer not to commit large `.pt` files, keep them out of Git and distribute via a shared cloud bucket.
- The `.gitattributes` file marks `models/**` as binary and suggests LFS tracking; to start tracking `.pt` with LFS run:
  ```bash
  git lfs track "*.pt"
  git add .gitattributes
  git commit -m "Track model checkpoints with Git LFS"
  ```
