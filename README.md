# Arti-fact

Arti-fact is a production-style deepfake detection platform with a React frontend and a PyTorch-powered Flask backend for image, audio, and video analysis.

## Features
- Multi-mode frontend with dedicated pages for:
  - Image deepfake detection
  - Audio deepfake detection
  - Video deepfake detection
- Real-time prediction output:
  - Verdict (`REAL` / `FAKE`)
  - Confidence score
- Drag-and-drop upload support with mode-aware file validation alerts
- Animated, cohesive pink-purple UI theme
- Backend health endpoint for model diagnostics
- Background adversarial hardening pipeline (configurable)
- Metadata forensics and cryptographic evidence signatures
- Gunicorn production entrypoint

## Project Structure
- `frontend/`: React app
- `backend/`: Flask API + models + production launcher
- `training/`: training and evaluation scripts for image/audio/video

## Requirements
### System
- Python 3.9+
- Node.js 18+ and npm

### Python dependencies
- Managed via root-level `requirements.txt`

### Frontend dependencies
- Managed via `frontend/package.json`

## Setup and Run
## 1. Clone and enter project
```bash
git clone <your-repo-url>
cd deepfake-detection-system
```

## 2. Python environment
```bash
conda create -n deepfake python=3.9 -y
conda activate deepfake
pip install -r requirements.txt
```

## 3. Frontend install
```bash
cd frontend
npm install
cd ..
```

## 4. Start backend (production-like)
```bash
cd backend
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

Backend will run on `http://localhost:5000` by default.

If port `5000` is already in use, free it and restart (Linux/macOS):

```bash
fuser -k 5000/tcp
cd backend
./start_prod.sh
```

## 5. Start frontend (new terminal)
```bash
cd frontend
npm start
```

Frontend will run on `http://localhost:3000`.

## API Endpoints
- `POST /detect-image`
- `POST /detect-audio`
- `POST /detect-video`
- `GET /health`
- `GET /adversarial-status`
- `POST /verify-evidence`
- `GET /edge-deployment-status`

All detect endpoints accept `multipart/form-data` with key: `file`.

## Environment Variables
Use `backend/.env.example` as reference:
- `FLASK_HOST`
- `FLASK_PORT`
- `FLASK_DEBUG`
- `MAX_CONTENT_LENGTH`
- `GUNICORN_WORKERS`
- `GUNICORN_THREADS`
- `GUNICORN_TIMEOUT`
- `ADV_PIPELINE_ENABLED`
- `ADV_IMAGE_SOURCE_DIRS`
- `ADV_AUDIO_SOURCE_DIRS`
- `FORENSICS_METADATA_ENABLED`
- `EVIDENCE_HMAC_SECRET`

Optional frontend API URL override:
- `REACT_APP_API_BASE_URL` (default: `http://localhost:5000`)

## Model Notes
Expected model files in `backend/models/`:
- `imageguard_v2_finetuned.pt` or `imageguard_v2_best.pt`
- `rawnet3_fsat_finetuned.pt` or `rawnet3_fsat_best.pt`
- `video_best_model.pt`

The backend can load TorchScript modules or state-dict checkpoints and attempts architecture reconstruction where applicable.

## Training Path Configuration
Many training scripts support a home-directory default path. To run on a different machine without editing code, set environment variables:
- `DEEPFAKE_IMAGE_BASE_DIR`
- `DEEPFAKE_AUDIO_BASE_DIR`

Example (PowerShell):

```powershell
$env:DEEPFAKE_IMAGE_BASE_DIR = "C:\data\deepfake\image"
$env:DEEPFAKE_AUDIO_BASE_DIR = "C:\data\deepfake\audio"
```

## Production Notes
- Flask dev server (`python app.py`) is for development only.
- Use Gunicorn (`backend/start_prod.sh`) for stable deployment behavior.
- Place backend behind Nginx or another reverse proxy in full production.

## Troubleshooting
- If frontend cannot connect: verify backend is running on port `5000`.
- If detections fail: check `http://localhost:5000/health` for model load status.
- If upload fails: verify file type matches selected detection mode.
- If adversarial pipeline does not advance: check `ADV_*_SOURCE_DIRS` and class balance in `backend/adversarial_pool/*/pool_manifest.jsonl`.

Quick backend checks:

```bash
curl http://127.0.0.1:5000/health
```

If the port is blocked by a stale process:

```bash
fuser -k 5000/tcp
```
