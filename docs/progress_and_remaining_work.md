# Progress And Remaining Work

This document is based on the current code in the repository (not on `docs/*.md`, which were placeholders).

## What Is Implemented (Today)

- Vision feature extraction (MediaPipe):
  - `vision_pipeline/hand_tracking.py` (21 hand landmarks per hand)
  - `vision_pipeline/pose_tracking.py` (33 body landmarks)
  - `vision_pipeline/feature_extractor.py` (225-dim concatenated feature vector)
- Canonical keypoint schema + slices:
  - `common/keypoint_schema.py` (225-dim layout + `HANDS_SLICE_225`)
- Holistic pose extraction (hands + body + face):
  - `training/data_pipeline/pose_extractor.py` (~1629 dims)
- Gesture tokenization (vector quantization):
  - `training/data_pipeline/gesture_tokenizer.py` (KMeans token IDs + label decode)
- Gesture recognition model (sequence classifier):
  - `ai_models/gesture_recognition/model.py` (`GestureModel`)
  - `ai_models/gesture_recognition/train.py` exists (not reviewed line-by-line here), plus dataset + inference modules.
- Foundation gesture-token transformer:
  - `models/sign_foundation_transformer.py` (`SignFoundationModel`)
  - Training: `training/train_foundation_model.py` (uses synthetic tokens if no data)
- Diffusion motion model:
  - `models/gesture_diffusion.py` + wrapper `ai_models/gesture_diffusion/diffusion_model.py`
  - Training: `training/train_diffusion.py` (uses synthetic motion if no data)
- Multimodal fusion model (gesture + text -> text):
  - `ai_models/multimodal_llm/gesture_encoder.py`
  - `ai_models/multimodal_llm/text_decoder.py`
  - `ai_models/multimodal_llm/training_pipeline.py`
- Speech to text:
  - `ai_engine/modules/speech_to_text.py` (faster-whisper)
- Minimal speech -> sign tokens pipeline:
  - `ai_engine/inference_pipeline.py`
  - `api_gateway/main.py` exposes `POST /speech-to-sign/`

## What Is Partially Implemented (Wired But Placeholder)

- Unified inference API server:
  - `api_server/server.py` + `api_server/realtime_inference.py`
  - Sign->text now decodes token IDs via `models/sign_transformer_vocab.json`, but quality still depends on real paired sentence data.
  - Motion generation returns raw vectors; no avatar retargeting.
- Language translation (proper learned text<->gloss):
  - Rule-based grammar converter exists: `nlp_translation/sign_grammar_converter.py`
  - Learned model scaffolding exists: `nlp_translation/models/seq2seq_model.py`

## Major Blockers (Must Fix)

- Python import/package naming mismatch:
  - Resolved by renaming repo folders to underscore package names.
- Feature schema alignment:
  - Canonical saved training keypoints are 225-dim (body + 2 hands).
  - Some models are configured for 126-dim (hands-only). Loaders/inference must slice the hands portion from the 225-dim layout rather than taking the first 126 dims.
  - Holistic extractor can output ~1629 dims (hands+body+face), but face landmarks are not currently included in saved training keypoints.

## Explicit 0-Byte Stubs (Remaining Work)

No remaining 0-byte Python stubs were found (excluding environment/cache directories).

## Recommended Next Steps (Order)

1. Add/ingest real paired data for sentence-level translation (How2Sign / RWTH-PHOENIX), and replace label-id placeholder targets in the sign->text transformer dataset.
2. Train and integrate learned text<->gloss translation (seq2seq/transformer) with real datasets (to beat the rule-based baseline).
3. Integrate sign-to-text decoding end-to-end (gesture/keypoints -> tokens -> text) with a real tokenizer/vocab and evaluation metrics (BLEU/WER).
4. Extend avatar retargeting from "skeleton frames" to a Unity rig (joint mapping, constraints, smoothing).
