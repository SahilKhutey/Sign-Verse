# Video LSTM Integration (Keras Optional Path)

This integration adds an end-to-end isolated sign recognition pipeline inspired by:
- `hthuwal/sign-language-gesture-recognition` (MIT)

Reference architecture:
- video -> frames -> InceptionV3 features -> LSTM classifier -> gloss label

## What Was Added

- Frame extraction utility:
  - `training/data_pipeline/video_to_frames.py`
- CNN feature extraction utility (InceptionV3):
  - `training/extract_cnn_features_keras.py`
- LSTM training script:
  - `training/train_video_lstm_keras.py`
- Runtime model wrapper:
  - `ai_models/gesture_recognition/video_lstm_keras.py`
- API endpoint:
  - `POST /gesture/classify-video-lstm`
- Local inference script:
  - `scripts/run_video_lstm_inference.py`

## Dependency Note

This path is optional and requires TensorFlow/Keras.

```bash
pip install -r requirements-optional-asl-cnn.txt
```

## Dataset Layout

Expected isolated video structure:

```text
datasets/isolated_videos/
  HELLO/
    sample_001.mp4
    sample_002.mp4
  THANK_YOU/
    sample_001.mp4
```

## Step 1: Convert Videos to Frames

```bash
python training/data_pipeline/video_to_frames.py --input-dir datasets/isolated_videos --output-dir datasets/video_frames --metadata training-data/video_frames_manifest.csv
```

## Step 2: Extract Inception Features

```bash
python training/extract_cnn_features_keras.py --manifest training-data/video_frames_manifest.csv --output-dir training-data/video_features --output-manifest training-data/video_features_manifest.csv
```

## Step 3: Train the LSTM Classifier

```bash
python training/train_video_lstm_keras.py --manifest training-data/video_features_manifest.csv --epochs 20 --save-model models/video_lstm.h5 --save-best-model models/video_lstm_best.h5 --save-labels models/video_lstm_labels.json --eval-report reports/video_lstm_eval.json
```

Outputs:
- `models/video_lstm.h5`
- `models/video_lstm_best.h5`
- `models/video_lstm_labels.json`
- `reports/video_lstm_eval.json`

## Local Inference

```bash
python scripts/run_video_lstm_inference.py --video path/to/test.mp4 --min-confidence 0.4
```

## API Inference

Endpoint:
- `POST /gesture/classify-video-lstm?min_confidence=0.4`

Form field:
- `file`: video upload

Response shape:

```json
{
  "class_id": 12,
  "label": "HELLO",
  "confidence": 0.86
}
```

## Notes and Limits

- This is an isolated sign pipeline (clip-level classification), not continuous sentence decoding.
- Final quality depends heavily on dataset diversity, class balance, and consistent clip trimming.
- For production, pair this with temporal segmentation and language decoding layers.
