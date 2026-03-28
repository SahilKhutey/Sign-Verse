# Model Architecture

This repo is organized around 5 AI model types. The code already contains reference implementations for most of them, but some integration and training pieces are still stubbed.

## 1) Vision Model (Hands/Body/Face Detection)

Purpose: convert raw frames into structured keypoints / feature vectors.

Implemented options:

- Lightweight (225 dims per frame): `vision_pipeline/feature_extractor.py`
  - Hands: 2 hands * 21 landmarks * (x,y,z) = 126 dims
  - Body: 33 landmarks * (x,y,z) = 99 dims
  - Total: 225 dims
- Holistic (1629 dims per frame): `training/data_pipeline/pose_extractor.py`
  - Hands: 126 dims
  - Body: 99 dims
  - Face: 468 landmarks * 3 = 1404 dims
  - Total: 1629 dims

Key choice to make:
- Pick a canonical feature schema for each downstream model and keep it consistent in configs and tokenizers.

Current repo convention:
- Canonical 225-dim layout is `[body(99), left_hand(63), right_hand(63)]` (see `common/keypoint_schema.py`).
- Hands-only models should slice the hands portion (`dims 99:225`) rather than taking the first 126 dims.

## 2) Gesture Recognition Model (Identify Signs / Tokens)

Purpose: map a temporal keypoint sequence to a discrete class or token.

Reference implementation:
- `ai_models/gesture_recognition/model.py` defines `GestureModel`
  - BiLSTM (temporal)
  - TransformerEncoder (global attention)
  - MLP classifier

Inputs/outputs:
- Input shape: `(batch, seq_len, input_size)` where `input_size` is currently `126` in the gesture model code.
- Output: `(batch, num_classes)` logits or a predicted `gesture_id`.

Notes:
- The unified preprocessor stores `(T, 225)` keypoints; the gesture dataset slices hands-only features when training a 126-dim model.
- The API server currently expands a single keypoint vector into a fake sequence for classification (`api_server/realtime_inference.py`). This is fine for wiring but not a real temporal model input.

## 3) Language Model (Grammar + Translation)

Purpose: convert between spoken-language sentences and sign-language gloss / tokens (sign grammar differs).

Current state:
- Rule-based grammar conversion exists: `nlp_translation/sign_grammar_converter.py`.
- Learned translation model scaffolding exists: `nlp_translation/models/seq2seq_model.py`.

Missing pieces:
- Train and integrate the learned translation models (seq2seq/transformer) end-to-end (data, checkpoints, decoding).

## 4) Motion Generation Model (Create Gestures)

Purpose: generate a motion sequence (3D skeleton vectors) from tokens/conditioning.

Reference implementation:
- Diffusion core: `models/gesture_diffusion.py`
- Pipeline wrapper: `ai_models/gesture_diffusion/diffusion_model.py`

Inputs/outputs:
- Output motion: `(seq_len, motion_dim)` where `motion_dim` defaults to `150`.
- Conditioning on gesture tokens is represented in the pipeline interface, but end-to-end token-conditioned training is not fully wired.

Missing pieces:
- Retargeting motion vectors to a Unity rig / avatar is currently stubbed under `avatar_animation/`.

## 5) Multimodal Foundation Model (Gesture + Text + Speech Fusion)

Purpose: unify gesture understanding, translation, and generation across modalities.

Implemented components:
- Gesture-token foundation transformer (GPT-style): `models/sign_foundation_transformer.py` (`SignFoundationModel`)
- Cross-attention sign->text decoder: `models/multimodal_decoder.py` (`SignToTextModel`)
- Multimodal training pipeline:
  - Gesture encoder: `ai_models/multimodal_llm/gesture_encoder.py`
  - Text decoder: `ai_models/multimodal_llm/text_decoder.py`
  - Trainer: `ai_models/multimodal_llm/training_pipeline.py`

Inputs/outputs (multimodal LLM path):
- Gesture input: keypoint sequence (default 225 dims per frame in this trainer)
- Optional text context tokens
- Output: text token logits or decoded text tokens

## Speech Model (Optional but Practical)

Purpose: speech-to-text for "speech -> sign" and for auto-labeling.

Implemented:
- `ai_engine/modules/speech_to_text.py` uses `faster_whisper.WhisperModel`.

## Integration Note (Important)

The repo uses underscore package names (`ai_models`, `vision_pipeline`, `api_server`, `ai_engine`). If you see older references to dashed folders in notes/scripts, update them to the underscore names.
