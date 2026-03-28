# SignVerse-AI

SignVerse-AI is an end-to-end sign language platform that connects vision-based sign recognition, text translation, and speech generation into real-time, cross-application experiences.

This repository includes:
1. Inference services (FastAPI)
2. Backend user and history services
3. Web and Android clients
4. Training pipelines and dataset tooling

## Current Status

### Working
- Unified inference API server (`api_server/`)
- Backend service with auth, history, and WebSocket rooms (`backend/`)
- Android app (Jetpack Compose) (`ar-vr-app/android/`)
- Web frontend (Vite + React) (`frontend/`)
- NLP text↔gloss training + inference (`training/train_text_gloss.py`, `nlp_translation/inference.py`)

### In Progress
- Full production hardening (CORS, tokens, logging, health checks)
- Model quality improvements (BLEU/WER tracking, curriculum learning)
- Text↔gloss dataset expansion

### Known Limitations
- Text↔gloss translation falls back to rule-based conversion if no trained model exists.
- Avatar retargeting is still a placeholder.
- Large assets (models, datasets, logs) are committed in repo and should be moved to a registry or storage bucket for production.

## Architecture (High Level)

1. **Vision + Pose**: MediaPipe-based feature extraction (225-dim keypoints).
2. **Gesture Recognition**: Token classification and sign-to-text via transformer.
3. **NLP Translation**: Text↔gloss translation (transformer when trained, rule-based fallback otherwise).
4. **Speech**: STT via faster-whisper, TTS via pyttsx3 (baseline offline).
5. **Clients**: Android launcher app + Web UI.

## Services

### Inference API (`api_server`)
Runs the core inference endpoints:
- `POST /vision/extract`
- `POST /gesture/classify`
- `POST /translate/sign-to-text`
- `POST /translate/text-to-sign`
- `POST /translate/sign-to-speech`
- `POST /speech-to-sign/`
- `WS /ws/stream`

Start:
```
uvicorn api_server.server:app --host 0.0.0.0 --port 8000
```

### Backend API (`backend`)
Handles users, auth, translation history, and socket rooms.
Start:
```
uvicorn backend.app:app --host 0.0.0.0 --port 8001
```

### API Gateway (`api_gateway`)
Legacy gateway for speech-to-sign, retained for compatibility.

## Environment Variables

Backend:
- `ENVIRONMENT` = `development` or `production`
- `SECRET_KEY` = required in production
- `ALLOWED_ORIGINS` = comma-separated list (no `*` in production)
- `INFERENCE_API_URL` = inference base URL
- `DATABASE_URL` = DB connection string

Frontend:
- `VITE_BACKEND_URL`
- `VITE_INFERENCE_URL`
- `VITE_ADMIN_TOKEN` (required only for dashboard admin controls)

See `.env.example` for a full template.
See `docs/ENVIRONMENTS.md` for environment profiles.

## Core Model Training

### Build a Larger Text↔Gloss Dataset
```
python training/data_pipeline/build_text_gloss_dataset.py
```

Outputs:
```
datasets/text_sign_pairs/expanded_pairs.csv
datasets/text_sign_pairs/train.csv
datasets/text_sign_pairs/val.csv
datasets/text_sign_pairs/test.csv
reports/text_gloss_dataset_report.json
```

### Train Text↔Gloss Models
```
python training/train_text_gloss.py --train-csv datasets/text_sign_pairs/train.csv --val-csv datasets/text_sign_pairs/val.csv --curriculum --augment
```

Outputs:
- `models/nlp_vocab.json`
- `models/nlp_text2gloss.pt`
- `models/nlp_gloss2text.pt`

Optional:
```
python training/train_text_gloss.py --train-csv datasets/text_sign_pairs/train.csv --val-csv datasets/text_sign_pairs/val.csv --curriculum --augment --register
```
This registers the best checkpoints into the model registry under `deployment/model_registry/`.

### One-Command Pipeline
```
python training/run_text_gloss_pipeline.py --curriculum --augment --register --fail-on-warnings
```

### Admin Trigger (Backend)
To trigger the dataset build + training pipeline from the backend:
- Set `ADMIN_TOKEN` in env.
- Call `POST /admin/pipeline/text-gloss` with header `x-admin-token`.
- Check run progress at `GET /admin/pipeline/text-gloss/status` with header `x-admin-token`.
- Optional dashboard control: set `VITE_ADMIN_TOKEN` in frontend env to run/poll from UI.

Training also writes an evaluation report to:
```
reports/nlp_eval.json
```

## Evaluation Metrics

During training, the script reports:
- `val_loss`
- `val_wer`
- `val_bleu`

## Web Frontend

The frontend is wired to live inference and backend APIs.
Start:
```
cd frontend
npm install
npm run dev
```

## Live Capture (OpenCV)

Run the local webcam capture pipeline:
```
python scripts/run_live_opencv_translate.py --camera 0
```

See `docs/LIVE_CAPTURE_GUIDE.md` for tracking tips.

## Live Stream (WebSocket)

The web UI can stream frames to the inference server using WebSocket:
- `VITE_INFERENCE_WS` (default `ws://localhost:8000/ws/stream`)

## Android App

The Android launcher is in:
```
ar-vr-app/android/
```

Use Android Studio and set the API base URL in Settings.

## Roadmap (Near Term)

1. Replace rule-based text↔gloss with trained transformer by default.
2. Improve sign→text accuracy with better paired datasets.
3. Add on-device pose extraction for mobile latency.
4. Move models/datasets out of git and into artifact storage.
5. Improve avatar retargeting pipeline.

## MVP Definition

See `docs/MVP_SCOPE.md` for the minimum scope and quality gates.

## Release Checklist

See `docs/RELEASE_CHECKLIST.md` before shipping releases.

## Future Plans

- Multi-language and dialect support (ASL/ISL/BSL)
- Model registry with versioned deployment
- Streaming inference over WebSocket
- Real-time avatar animation with Unity rig
- End-to-end evaluation suite (WER/BLEU/MOS)

## Repo Structure

- `ai_engine/` Core AI logic and model integration
- `api_server/` Unified inference API
- `backend/` Auth + history + sockets
- `frontend/` Web UI
- `ar-vr-app/android/` Android client
- `training/` Training scripts
- `nlp_translation/` Text↔gloss models and inference
- `datasets/` Raw datasets and text/gloss pairs
- `models/` Trained checkpoints

## Documentation

More details in `docs/`.
