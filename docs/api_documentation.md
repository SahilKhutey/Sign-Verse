# API Documentation

This repo contains two FastAPI entrypoints:
- `api_gateway/` (minimal: speech -> sign tokens)
- `api_server/` (unified server: vision + gesture + translation + generation + websocket)

## API Gateway (`api_gateway/main.py`)

### `POST /speech-to-sign/`

Accepts: multipart file upload (audio).

Returns:
```json
{
  "text": "hello how are you",
  "sign_tokens": ["HELLO", "HOW", "YOU"]
}
```

Implementation:
- STT: `ai_engine/modules/speech_to_text.py` (faster-whisper)
- Text->sign tokens: `ai_engine/modules/text_to_sign.py` (rule-based)

## Unified API Server (`api_server/server.py`)

### `GET /health`

Returns:
```json
{ "status": "ok", "models": ["gesture", "diffusion", "..."] }
```

### `POST /vision/extract`

Accepts: multipart file upload (image bytes).

Returns:
```json
{ "features": [0.0, 0.1, "..."], "dim": 225 }
```

Uses:
- `vision_pipeline/feature_extractor.py`

### `POST /gesture/classify`

Accepts:
```json
{ "keypoints": [0.1, 0.2, "..."] }
```

Returns:
```json
{ "gesture_id": 123, "gesture_label": "HELLO" }
```

Note:
- `gesture_label` is returned when a `label_map.json` is available (from training preprocessing).

### `POST /analyze/frame`

Single-shot helper: image frame -> keypoints -> gesture classification.

Accepts: multipart file upload (image bytes).

Returns:
```json
{ "gesture_id": 123, "gesture_label": "HELLO", "sign_tokens": ["HELLO"], "dim": 225 }
```

### `POST /translate/sign-to-text`

Accepts:
```json
{ "sequence": [[0.0, 0.1], [0.1, 0.2]] }
```

Returns:
```json
{ "text": "Sign 0029" }
```

Note:
- Decoding uses `models/sign_transformer_vocab.json` when available; otherwise it falls back to `TOKEN_{id}`.
- Translation quality depends on having real paired sign-to-text data (sentence-level datasets).

### `POST /translate/text-to-sign`

Accepts:
```json
{ "text": "I am going to school tomorrow" }
```

Returns:
```json
{ "tokens": ["I", "GOING", "SCHOOL", "TOMORROW"] }
```

### `POST /generate/motion`

Accepts:
```json
{ "tokens": ["HELLO"], "frames": 30 }
```

Returns:
```json
{ "motion": [[0.0, 0.0], [0.1, 0.2]], "frames": 30 }
```

Note: generated motion vectors are not yet retargeted to an avatar rig.

### `WS /ws/stream`

Input payload (example):
```json
{ "frame_id": 1, "keypoints": [0.0, 0.1], "text": "hello" }
```

Output payload (example):
```json
{ "frame_id": 1, "gesture_id": 42, "gesture_label": "HELLO", "sign_tokens": ["HELLO"] }
```

## Running Locally (after import/layout fixes)

- `uvicorn api_server.server:app --host 0.0.0.0 --port 8000`
- `uvicorn api_gateway.main:app --host 0.0.0.0 --port 8001`
