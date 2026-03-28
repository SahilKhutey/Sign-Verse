# Training Pipeline

This repo supports training the SignVerse model stack end-to-end using the master
orchestrator:

```powershell
Set-Location C:\Users\User\Documents\SignVerse\signverse-ai
.\venv\Scripts\python.exe training\run_all_training.py
```

## Keypoint Schemas (Important)

The preprocessing pipeline saves per-frame keypoints in a *canonical 225-dim*
layout:

- `225 = body(99) + left_hand(63) + right_hand(63)`
- Constants/slices: `common/keypoint_schema.py`

Some models train on *hands-only 126-dim* inputs:

- Gesture recognition: `gesture_model.input_size: 126`
- Sign->text transformer: `sign_transformer.feature_dim: 126`

When training data is stored as 225 dims but a model is configured for 126 dims,
the loaders/inference code slice out the hands portion (dims `99:225`) so we do
not accidentally train on `body + partial hand`.

## Master Orchestrator Stages

The orchestrator is `training/run_all_training.py`. Stages:

1. `download`: dataset presence check (prints where to put raw videos)
2. `preprocess`: raw videos -> `training-data/keypoints/*.npy` + `training-data/labels.csv`
3. `validate`: sanity checks on shapes and missing/corrupt files
4. `tokenize`: keypoints -> discrete gesture tokens (for foundation transformer)
5. `gesture`: train gesture recognition classifier
6. `foundation`: train gesture-token foundation transformer (next-token prediction)
7. `transformer`: train sign->text transformer (seq2seq)
8. `diffusion`: train gesture diffusion motion generator
9. `llm`: train multimodal LLM (gesture + text)
10. `registry`: write `models/model_registry.json`

Run all stages:

```powershell
.\venv\Scripts\python.exe training\run_all_training.py
```

Run a single stage:

```powershell
.\venv\Scripts\python.exe training\run_all_training.py --stage preprocess
.\venv\Scripts\python.exe training\run_all_training.py --stage gesture
```

Run from a stage to the end:

```powershell
.\venv\Scripts\python.exe training\run_all_training.py --from-stage foundation
```

## Direct Trainers

You can also run trainers directly:

```powershell
.\venv\Scripts\python.exe training\train_gesture_model.py
.\venv\Scripts\python.exe training\train_foundation_model.py
.\venv\Scripts\python.exe training\train_sign_transformer.py
.\venv\Scripts\python.exe training\train_diffusion.py
```

## Configuration

Primary config:

- `training/configs/training_config.yaml`

Optional convenience overrides:

- `training/configs/gesture_training.yaml`
- `training/configs/nlp_training.yaml` (currently used for sign_transformer overrides)

## Outputs

Models/checkpoints:

- `models/*.pt`
- `models/model_registry.json`
- `models/label_map.json` (copied from `training-data/label_map.json` when available)
- `models/sign_transformer_vocab.json` (tokenizer vocab for sign->text decoding)

Logs:

- `logs/tensorboard/`
- `logs/training_summary.json`

## Real Data vs Synthetic Fallback

Some trainers fall back to synthetic data if the expected dataset folders are
empty. This is useful for wiring and smoke tests, but not for model quality.

For real training runs, make sure you have:

- `training-data/keypoints/*.npy` with shape `(T, 225)`
- `training-data/labels.csv` with the matching filenames
