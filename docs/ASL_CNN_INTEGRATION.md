# ASL CNN Integration (Keras Prototype Path)

This module adds a hackathon-style, image-based ASL classifier inspired by:
- `harshbg/Sign-Language-Interpreter-using-Deep-Learning`
- `loicmarie/sign-language-alphabet-recognizer`

## What Was Added

- Optional Keras model wrapper:
  - `ai_models/gesture_recognition/cnn_asl.py`
- Training script:
  - `training/train_asl_cnn_keras.py`
- Live webcam script:
  - `scripts/run_asl_cnn_live.py`
- API endpoint:
  - `POST /gesture/classify-image-cnn`
  - `POST /gesture/classify-image-cnn-fingerspell`

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

Simple grayscale CNN:
```bash
python training/train_asl_cnn_keras.py --data-dir datasets/asl_alphabet --arch simple --image-size 64 --epochs 15
```

InceptionV3 baseline (recommended):
```bash
python training/train_asl_cnn_keras.py --data-dir datasets/asl_alphabet --arch inceptionv3 --image-size 224 --epochs 15 --fine-tune-epochs 5
```

Outputs:
- `models/asl_cnn.h5`
- `models/asl_cnn_best.h5`
- `models/asl_cnn_labels.json`
- `reports/asl_cnn_eval.json`

## Live Demo

```bash
python scripts/run_asl_cnn_live.py --camera 0 --min-confidence 0.4
```

Press `q` to quit.
Press `c` to clear typed text in spell mode.

Finger-spelling mode:
```bash
python scripts/run_asl_cnn_live.py --camera 0 --spell-mode --min-confidence 0.5
```

## API Usage

Endpoint:
- `POST /gesture/classify-image-cnn?min_confidence=0.4`
- `POST /gesture/classify-image-cnn-fingerspell?session_id=demo&min_confidence=0.4`

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

Fingerspelling response adds:
```json
{
  "class_id": 3,
  "label": "D",
  "confidence": 0.91,
  "session_id": "demo",
  "stable_label": "D",
  "committed": "D",
  "text": "HELLO"
}
```

## Dependency Note

This path requires TensorFlow/Keras and is intentionally optional.
If TensorFlow is not installed, core PyTorch pipeline still works.
