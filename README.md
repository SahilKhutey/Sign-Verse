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
- `POST /gesture/classify-image-cnn`
- `POST /gesture/classify-image-cnn-fingerspell`
- `POST /gesture/classify-video-lstm`
- `POST /translate/sign-to-text`
- `POST /translate/video-to-text`
- `POST /translate/text-to-sign`
- `POST /translate/text-to-sign-video-plan`
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

Optional synthetic-video + embedding expansion:
```
python training/run_text_gloss_pipeline.py --curriculum --augment --register --generate-synthetic-video --dictionary-manifest datasets/sign_dictionary/manifest.csv --extract-video-embeddings
```

### Train ASL CNN (Optional Keras Path)
Prototype-oriented letter classifier inspired by live interpreter repos:
Install optional dependency:
```
pip install -r requirements-optional-asl-cnn.txt
```
Collect your own alphabet dataset (ASL/ISL style):
```
python scripts/collect_alphabet_dataset.py --camera 0 --flip --labels "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z" --split-mode random --val-ratio 0.1 --image-size 224
```
Then train:
```
python training/train_asl_cnn_keras.py --data-dir datasets/asl_alphabet --arch inceptionv3 --image-size 224 --epochs 15 --fine-tune-epochs 5
```
Outputs:
- `models/asl_cnn.h5`
- `models/asl_cnn_best.h5`
- `models/asl_cnn_labels.json`
- `reports/asl_cnn_eval.json`

### Train Video CNN+LSTM (Optional Keras Path)
Prototype-oriented isolated sign recognizer inspired by video-based SLR repos:
```
python training/data_pipeline/video_to_frames.py --input-dir datasets/isolated_videos --output-dir datasets/video_frames --metadata training-data/video_frames_manifest.csv
python training/extract_cnn_features_keras.py --manifest training-data/video_frames_manifest.csv --output-dir training-data/video_features --output-manifest training-data/video_features_manifest.csv
python training/train_video_lstm_keras.py --manifest training-data/video_features_manifest.csv --epochs 20 --save-model models/video_lstm.h5 --save-best-model models/video_lstm_best.h5 --save-labels models/video_lstm_labels.json --eval-report reports/video_lstm_eval.json
```
Outputs:
- `models/video_lstm.h5`
- `models/video_lstm_best.h5`
- `models/video_lstm_labels.json`
- `reports/video_lstm_eval.json`

### Train Research SLR Baselines (Clean-Room)
For architecture benchmarking (non-production by default):
```
python training/train_research_conv3d.py --manifest-csv training-data/video_frames_manifest.csv --seq-len 24 --image-size 112 --epochs 20
python training/train_research_pose_gcn.py --seq-len 60 --feature-dim 225 --epochs 20
```

### Build Concatenative Text->Sign Video Engine (Optional)
Framework-style sentence synthesis inspired by modular sign-language-translator systems:
```
python training/data_pipeline/build_sign_video_dictionary.py --clips-root datasets/sign_dictionary/clips --output-manifest datasets/sign_dictionary/manifest.csv
python training/data_pipeline/generate_synthetic_sign_video_pairs.py --pairs-csv datasets/text_sign_pairs/expanded_pairs.csv --dictionary-manifest datasets/sign_dictionary/manifest.csv --output-dir datasets/synthetic_sentence_videos --output-manifest training-data/synthetic_sign_video_pairs.csv --min-coverage 0.6 --max-samples 500
python training/data_pipeline/extract_video_embeddings.py --input-dir datasets/isolated_videos --output-dir training-data/video_embeddings --manifest training-data/video_embeddings_manifest.csv
python training/data_pipeline/build_video_sign_text_dataset.py --input-manifest training-data/synthetic_sign_video_pairs.csv --keypoint-dir training-data/video_keypoints --labels-csv training-data/video_sign_text_labels.csv
python training/train_video_sign_to_text.py --labels-csv training-data/video_sign_text_labels.csv --keypoint-dir training-data/video_keypoints --epochs 12
```
Outputs:
- `datasets/sign_dictionary/manifest.csv`
- `training-data/synthetic_sign_video_pairs.csv`
- `reports/synthetic_sign_video_report.json`
- `training-data/video_embeddings_manifest.csv`
- `training-data/video_sign_text_labels.csv`
- `models/video_sign_transformer_*.pt`
- `models/video_sign_transformer_vocab.json`

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

Run optional ASL CNN live interpreter:
```
python scripts/run_asl_cnn_live.py --camera 0 --min-confidence 0.4
python scripts/run_asl_cnn_live.py --camera 0 --spell-mode --min-confidence 0.5
```

Run optional video classifier inference:
```
python scripts/run_video_lstm_inference.py --video path/to/sample.mp4 --min-confidence 0.4
```

Run optional concatenative sentence synthesis:
```
python scripts/run_concatenative_synthesis.py --text "hello how are you" --plan-only
python scripts/run_concatenative_synthesis.py --text "hello how are you" --output datasets/synthetic_sentence_videos/demo.mp4
```

Translate a video clip directly to text (API):
```
curl -X POST "http://localhost:8000/translate/video-to-text?sample_every=2&max_frames=90" -F "file=@sample.mp4"
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
- `docs/ASL_CNN_INTEGRATION.md` for Keras CNN prototype flow.
- `docs/ALPHABET_DATA_COLLECTION.md` for webcam alphabet dataset capture workflow.
- `docs/VIDEO_LSTM_INTEGRATION.md` for Keras video CNN+LSTM prototype flow.
- `docs/SIGN_LANGUAGE_TRANSLATOR_INTEGRATION.md` for concatenative synthesis + video embedding workflow.
- `docs/RESEARCH_SLR_BASELINES.md` for clean-room Conv3D and Pose-GCN baselines.
