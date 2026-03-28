# ASL CNN Integration (Keras Prototype Path)

This module adds a hackathon-style, image-based ASL classifier inspired by:
- `harshbg/Sign-Language-Interpreter-using-Deep-Learning`

## What Was Added

- Optional Keras model wrapper:
  - `ai_models/gesture_recognition/cnn_asl.py`
- Training script:
  - `training/train_asl_cnn_keras.py`
- Live webcam script:
  - `scripts/run_asl_cnn_live.py`
- API endpoint:
  - `POST /gesture/classify-image-cnn`

## Dataset Layout

Expected structure:

```text
datasets/asl_alphabet/
  train/
    A/*.jpg
    B/*.jpg
    ...
  val/
    A/*.jpg
    B/*.jpg
    ...
```

## Train

```bash
python training/train_asl_cnn_keras.py --data-dir datasets/asl_alphabet --epochs 15
```

Outputs:
- `models/asl_cnn.h5`
- `models/asl_cnn_labels.json`

## Live Demo

```bash
python scripts/run_asl_cnn_live.py --camera 0 --min-confidence 0.4
```

Press `q` to quit.

## API Usage

Endpoint:
- `POST /gesture/classify-image-cnn?min_confidence=0.4`

Form data:
- `file`: image upload

Response:
```json
{
  "class_id": 3,
  "label": "D",
  "confidence": 0.91
}
```

## Dependency Note

This path requires TensorFlow/Keras and is intentionally optional.
If TensorFlow is not installed, core PyTorch pipeline still works.
