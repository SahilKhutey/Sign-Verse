# Dataset Structure

This repo distinguishes between:
- `datasets/` (raw + intermediate assets, multi-dataset layout)
- `training-data/` (standardized outputs used by trainers)

## `datasets/` (Repository-Level)

Observed folders:

- `datasets/raw_videos/`
  - Intended location for raw dataset videos (by dataset name).
- `datasets/extracted_frames/`
  - Frame dumps extracted from videos.
- `datasets/pose_keypoints/`
  - Numpy `.npy` keypoint sequences (used by diffusion training as motion-like vectors).
- `datasets/gesture_sequences/`
  - Sequences prepared for gesture recognition (format depends on preprocessing stage).
- `datasets/gesture_tokens/`
  - Discrete token sequences `.npy` for foundation model pretraining.
- Dataset-specific folders (examples seen):
  - `datasets/WLASL/`, `datasets/AUTSL/`, `datasets/RWTH_PHOENIX/`, `datasets/LSA64/`, `datasets/isl_dataset/`, `datasets/asl_alphabet/`
- `datasets/text_sign_pairs/`
  - Intended location for sentence/gloss pairs.
- `datasets/metadata/`
  - Intended location for dataset metadata and indexes.
- `datasets/processed/`
  - Output of unified preprocess.

## `training-data/` (Training Inputs)

Observed folders:

- `training-data/videos/` (raw or normalized videos used for training)
- `training-data/frames/` (extracted frames)
- `training-data/keypoints/` (canonical keypoint sequences, `.npy`)
- `training-data/annotations/` (aligned text/gloss annotations)
- `training-data/labels.csv` (index file for supervised tasks)

The multimodal LLM trainer expects:
- `training-data/keypoints/`
- `training-data/labels.csv`

## Data Builders

- `dataset-builder/frame_extractor.py`
  - Extract frames from videos.
- `dataset-builder/dataset_formatter.py`
  - Standardize dataset formats.
- `dataset-builder/pose_label_generator.py`
  - Create pose-aligned labels.

## Notes / Decisions Needed

- Canonical keypoint schema:
  - Some code paths use 225-dim (hands + body).
  - Others use ~1629-dim (hands + body + face).
- Normalization and coordinate conventions:
  - Choose and document: pixel coords vs normalized coords, camera-space vs image-space, and per-person centering.
