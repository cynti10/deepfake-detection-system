Speaker Notes — Multimodal Deepfake Detection

Audience: Non-Technical / General Public

1) Opening (30–45s)
- Greeting and hook: "Today I'll show you a system that checks whether media—images, audio, or video—is real or manipulated."
- Short example: show one clear real and one clear fake image/audio/video and ask the audience to guess.

2) What the system does (60s)
- Simple statement: "You upload media and the system returns REAL or FAKE with a confidence score." 
- Explain the outputs: label, confidence (0–100%), and a small "evidence" package (a file fingerprint and a signature) that proves the result was computed from that exact file.

3) Why this matters (60–90s)
- Trust: Deepfakes can mislead people, influence decisions, and harm reputations.
- Accessibility: Creating believable fakes is easier than ever—detection helps protect truth and verify sources.
- Live demo value: Seeing a model's decision and its confidence helps demystify AI claims.

4) Live demo cues and script (hands-on)
- Show the UI or run a quick call to the backend for an image: upload -> show result JSON.
- Emphasize the confidence number and the metadata (image size, audio length) as hints why the model might be unsure.
- Invite one or two audience members to upload short media and discuss results.

5) Messaging tips for non-technical audience
- Use analogies: "Like a panel of specialists—one looks at visual pixels, another listens for audio oddities." 
- Avoid technical terms; say "patterns" and "signals" instead of "spectral features".
- Reassure: The system helps decide, but a human review is still important for critical decisions.

6) Common non-technical questions & answers
- Q: Is it always right? A: No. It gives probabilities; combine evidence and human judgement.
- Q: What if someone tries to fool it? A: It's an arms race—models must be updated; this demo also shows forensics metadata and signatures.

---

Audience: Technical / Researchers / Engineers

1) Opening (15–30s)
- Quick summary: Multimodal inference stack in `backend/app.py` using PyTorch, OpenCV, librosa, and `timm` backbones. Evidence signatures via HMAC.

2) System architecture (2–3 mins)
- API endpoints: `/detect-image`, `/detect-audio`, `/detect-video`, `/health`, `/verify-evidence`.
- Model loading: `_load_torch_inference_model()` supports TorchScript, `nn.Module`, and checkpoint dicts with `state_dict`; `_build_model_from_state_dict()` can reconstruct `ImageGuardV2` and `RawNet3FSAT` when config is present.
- Concurrency: `model_lock` around inference and swaps; `AdversarialBackgroundService` for background tasks.

3) Model design rationale (3–4 mins)
- Image: `ImageGuardV2` uses dual streams—spatial (RGB) via `timm` backbone + spectral (FFT magnitude) via a single-channel backbone. Rationale: many generative artifacts are spectral or texture-based; combining views increases robustness.
- Audio: `RawNet3FSAT` with learnable Sinc filters captures low-level phase/time-domain information often discarded by spectrograms; attentive-stat pooling provides robust utterance-level features.
- Video: Per-frame RGB + FFT mag features projected and fed into a transformer encoder with CLS token—temporal modeling captures frame inconsistency and temporal artifacts.

4) Preprocessing and outputs (2 mins)
- Image: resize to 224×224, ImageNet normalization, `timm` backbone assumptions.
- Audio: waveform fallback (4s, 16kHz) → mel fallback (3s, 22.05kHz) for compatibility with checkpoints.
- Video: uniform frame sampling with repeat-last padding to ensure fixed `seq_len`.
- Output handling: `_torch_output_to_fake_probability()` handles single-logit or multi-logit outputs and returns consistent fake probability.

5) Calibration & demo features (2 mins)
- `_calibrate_fake_probability()` applies per-modality bias.
- Demo autobalance maintains a target fake detection rate using recent scores (`recent_fake_probs`) and quantile-based thresholding to keep exhibits informative.

6) Forensics & verification (1–2 mins)
- Evidence: SHA-256 of file + HMAC of `"{modality}|{sha256}|{fake_prob:.8f}"` using `EVIDENCE_HMAC_SECRET` allows reproducible verification via `/verify-evidence`.
- Metadata: image dimensions, audio duration/sample rate, video fps/duration—useful to detect suspicious files.

7) Operational considerations (2 mins)
- Model formats and pitfalls: detector supports TorchScript and checkpoint dicts but warns on Git LFS pointers via `_is_lfs_pointer()`.
- Hardware: recommend GPU for real-time video; CPU inference is possible but slower.
- Logging & health: `/health` reports loaded model names, autobalance settings, and `model_load_errors`.

8) Extensions & research directions (2–3 mins)
- Multimodal fusion and joint training for cross-modal consistency checks.
- Larger backbones and spatio-temporal models (Video-Swin, 3D ConvNets) with more GPU resources.
- Explainability: integrate Grad-CAM, spectral attribution, or audio saliency.
- Robustness: adversarial training loops and systematic perturbation evaluations.

9) Demo troubleshooting checklist (quick)
- If `/health` shows `MISSING`, check `backend/models/` and ensure .pt checkpoints are full binaries (not LFS pointers).
- If PyTorch import fails, confirm virtual environment Python and `requirements.txt` are installed.
- If uploads fail, check `UPLOAD_FOLDER` permissions and `MAX_CONTENT_LENGTH`.

10) Ready-to-run notes
- Start server: ensure virtualenv active and run `python backend/app.py`.
- Verify with `GET /health` before demo.

---

Closing (both audiences)
- Invite questions and live trials: "Try uploading a short clip and we'll inspect the result together." 
- For technical visitors: offer to walk them through `backend/app.py` and `adversarial_pipeline.py` after the session.

Contact & Next Steps
- If you want, I can convert these notes into a slide deck, printable handout, or a short one-page exhibit poster.
