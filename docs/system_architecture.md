# System Architecture (SignVerse-AI)

This repo implements a 5-layer sign language AI stack and the services around it.

## What Exists Today (Code-Level)

### Services

- `api_gateway/` (FastAPI)
  - Implements `POST /speech-to-sign/` (audio upload -> Whisper STT -> rule-based text-to-sign tokens).
  - Entry: `api_gateway/main.py`.
- `api_server/` (FastAPI, intended as a unified inference server)
  - Exposes endpoints for vision, gesture classification, translation, motion generation, and websocket streaming.
  - Entry: `api_server/server.py`.
- `backend/`
  - Separate backend service (see `deployment/docker/Dockerfile.backend`).
- `avatar_animation/`
  - Intended to serve Unity bridge + animation API, but multiple modules are currently 0-byte stubs.

### Core AI Layers (Model + Pipeline)

1. Vision / Pose extraction
   - `vision_pipeline/` extracts 225-dim frame features using MediaPipe Hands (2x21x3 = 126) + MediaPipe Pose (33x3 = 99).
   - `training/data_pipeline/pose_extractor.py` uses MediaPipe Holistic (hands + body + face) to extract ~1629 dims per frame.
2. Gesture recognition (sign token classification)
   - `ai_models/gesture_recognition/model.py` defines `GestureModel` (BiLSTM -> TransformerEncoder -> classifier).
3. Language / translation
   - `nlp_translation/sign_grammar_converter.py` provides rule-based "English -> gloss-like order".
   - `nlp_translation/models/seq2seq_model.py` exists (learned model), but several glue modules are empty (see "Remaining Work").
4. Motion generation
   - `models/gesture_diffusion.py` (core) and `ai_models/gesture_diffusion/diffusion_model.py` (pipeline wrapper).
5. Multimodal foundation / fusion
   - `models/sign_foundation_transformer.py` (gesture-token GPT-style model).
   - `models/multimodal_decoder.py` (cross-attention sign->text decoder).
   - `ai_models/multimodal_llm/` (gesture encoder + fusion + text decoder + trainer).

## Intended Runtime Flows

### Flow A: Speech -> Sign Tokens (already wired)

Camera not required.

1. Client uploads audio to `api_gateway` endpoint `POST /speech-to-sign/`.
2. `ai_engine/inference_pipeline.py` runs:
   - `SpeechToText` (faster-whisper) -> text
   - `TextToSignConverter` (rule-based stop-word removal) -> sign tokens

### Flow B: Camera/Frame -> Features -> Gesture -> Text (partially wired)

1. Client sends image/frame to `api_server` endpoint `POST /vision/extract`.
2. Server uses `vision_pipeline/feature_extractor.py` to produce 225-dim features.
3. Client sends keypoints to `POST /gesture/classify`.
4. Server uses gesture model to produce `gesture_id`.
5. Translation is currently placeholder or rule-based (varies by path).

### Flow C: Motion Generation (skeleton is present)

1. Client calls `POST /generate/motion` on `api_server`.
2. Server calls diffusion pipeline `ai_models/gesture_diffusion/` to generate motion vectors.
3. Avatar rendering/retargeting is not yet implemented (stubs exist).

## Known Blockers (Must Fix For End-to-End)

- **Python import/package naming mismatch**
  - Python packages in this repo use underscore names (e.g. `ai_engine`, `api_server`, `vision_pipeline`, `ai_models`).
  - Result: many modules will not import as-is without renaming folders, adding compatibility wrappers, or changing imports.
- **Docs are placeholders**
  - All files under `docs/` were 0-byte at the time of writing; this doc fills the gap.
- **Feature-dimension mismatch across pipelines**
  - `vision_pipeline` produces 225 dims (hands + body).
  - `training/data_pipeline/pose_extractor.py` produces ~1629 dims (hands + body + face).
  - Decide which representation is canonical per model (gesture model vs foundation vs multimodal LLM) and align configs.

## Remaining Work (High Priority)

- Standardize Python package structure (remove dash/underscore import breakage).
- Implement 0-byte stubs (see `docs/progress_and_remaining_work.md` for an explicit list).
- Decide and document canonical keypoint schema (225 vs 1629 dims, ordering, normalization).
