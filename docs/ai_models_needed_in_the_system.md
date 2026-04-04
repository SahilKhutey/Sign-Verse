# AI Models Needed In The System

This project naturally uses 5 AI model types. The repo already contains starter implementations for most of them; the missing work is mostly integration, training glue, and avatar retargeting.

## 1) Vision Model (Hands, Body, Face)

Purpose:
- Detect hands, body pose, finger positions, facial landmarks.

Repo mapping:
- Hands + body (225 dims): `vision_pipeline/hand_tracking.py`, `vision_pipeline/pose_tracking.py`, `vision_pipeline/feature_extractor.py`
- Hands + body + face (1629 dims): `training/data_pipeline/pose_extractor.py` (MediaPipe Holistic)

Output examples:
- Hand joints: 21 landmarks per hand
- Body joints: 33 landmarks
- Face: 468 landmarks

These become the frame-level features for downstream models.

## 2) Gesture Recognition Model (Identify Signs)

Purpose:
- Convert a keypoint sequence into a sign class/token.

Repo mapping:
- `ai_models/gesture_recognition/model.py` (`GestureModel`: BiLSTM + TransformerEncoder)
- Training scaffolding exists under `ai_models/gesture_recognition/` (train/dataset/inference).

Recommended architectures (as you listed):
- CNN + LSTM (good baseline)
- Temporal CNN (fast inference)
- Transformer (best accuracy, most data hungry)

## 3) Language Model (Sentence Meaning, Sign Grammar)

Purpose:
- Translate between spoken-language text and sign-language gloss/tokens.

Repo mapping:
- Rule-based grammar: `nlp_translation/sign_grammar_converter.py`
- Inference wrapper: `nlp_translation/inference.py`
- Learned seq2seq model: `nlp_translation/models/seq2seq_model.py`

Remaining work:
- Train and integrate a learned translation model (seq2seq/transformer) for higher quality than the rule-based baseline.

## 4) Motion Generation Model (Create Gestures)

Purpose:
- Generate realistic motion trajectories that can drive an avatar.

Repo mapping:
- Diffusion model core: `models/gesture_diffusion.py`
- Pipeline wrapper: `ai_models/gesture_diffusion/diffusion_model.py`
- Training: `training/train_diffusion.py`

Output:
- Motion tensor shaped like `(seq_len, motion_dim)` (default `motion_dim=150`).

Remaining work:
- Retarget motion vectors to the Unity rig in `avatar_animation/` (currently stubbed).

## 5) Multimodal Foundation Model (Connect Gesture + Text + Speech)

Purpose:
- A shared representation that supports recognition, translation, and generation.

Repo mapping:
- Foundation transformer over gesture tokens (GPT-style): `models/sign_foundation_transformer.py`
- Cross-attention sign->text decoder: `models/multimodal_decoder.py`
- Multimodal LLM trainer (gesture encoder + fusion + text decoder): `ai_models/multimodal_llm/training_pipeline.py`

Typical architecture:
- Video/keypoint encoder -> multimodal transformer -> text decoder + gesture/motion heads

## Bonus: Speech Model (Practical Add-On)

Purpose:
- Speech -> text for "speech to sign", and for auto-labeling datasets.

Repo mapping:
- `ai_engine/modules/speech_to_text.py` uses `faster-whisper`.

## Real-Time Inference Target

Recommended production pipeline:

Camera
-> Pose extraction (Vision)
-> Gesture recognition (Tokens/IDs)
-> Multimodal foundation / translation
-> Text output + (optional) diffusion motion generation
-> Avatar animation

Latency target:
- <200 ms is a realistic goal once optimized and running on GPU.

## Training Procedure (Transformers + NN)

Canonical keypoints:
- Preprocessing writes `(T, 225)` arrays in the layout `[body(99), left_hand(63), right_hand(63)]`.
- Hands-only models (126 dims) should use the hands slice (dims `99:225`), not the first 126 dims.

Recommended training order:

1. Preprocess and validate data:
   - `training/run_all_training.py --stage preprocess`
   - `training/run_all_training.py --stage validate`
2. Train gesture recognition (sequence classifier):
   - `training/run_all_training.py --stage gesture`
   - Goal: reliable per-sign classification + a strong temporal encoder for later reuse.
3. Train gesture tokenization + foundation transformer:
   - `training/run_all_training.py --stage tokenize`
   - `training/run_all_training.py --stage foundation`
   - Goal: learn a general-purpose gesture-token model (next-token prediction).
4. Train sign->text transformer (seq2seq):
   - `training/run_all_training.py --stage transformer`
   - Needs real paired sign-to-text data for quality (sentence-level datasets).
5. Train diffusion motion generator:
   - `training/run_all_training.py --stage diffusion`
   - Then retarget to Unity avatar joints in `avatar_animation/`.
6. Train multimodal LLM:
   - `training/run_all_training.py --stage llm`
   - Goal: fuse gesture + text (and later speech) for end-to-end translation and generation.

Execution details (commands, outputs, configs) live in:
- `docs/training_pipeline.md`
