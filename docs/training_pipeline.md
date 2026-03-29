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

## Optional Keras Video SLR Pipeline

This optional path adds isolated video sign recognition with a CNN feature extractor
and LSTM classifier.

1. Convert videos to frames:

```powershell
.\venv\Scripts\python.exe training\data_pipeline\video_to_frames.py --input-dir datasets\isolated_videos --output-dir datasets\video_frames --metadata training-data\video_frames_manifest.csv
```

2. Extract InceptionV3 features:

```powershell
.\venv\Scripts\python.exe training\extract_cnn_features_keras.py --manifest training-data\video_frames_manifest.csv --output-dir training-data\video_features --output-manifest training-data\video_features_manifest.csv
```

3. Train LSTM:

```powershell
.\venv\Scripts\python.exe training\train_video_lstm_keras.py --manifest training-data\video_features_manifest.csv --save-model models\video_lstm.h5 --save-best-model models\video_lstm_best.h5 --save-labels models\video_lstm_labels.json --eval-report reports\video_lstm_eval.json
```

## Optional Concatenative Synthetic Video Pipeline

This optional path builds sentence-level synthetic sign videos from a token->clip dictionary.

1. Build dictionary manifest:

```powershell
.\venv\Scripts\python.exe training\data_pipeline\build_sign_video_dictionary.py --clips-root datasets\sign_dictionary\clips --output-manifest datasets\sign_dictionary\manifest.csv
```

2. Generate synthetic text/gloss->video pairs:

```powershell
.\venv\Scripts\python.exe training\data_pipeline\generate_synthetic_sign_video_pairs.py --pairs-csv datasets\text_sign_pairs\expanded_pairs.csv --dictionary-manifest datasets\sign_dictionary\manifest.csv --output-dir datasets\synthetic_sentence_videos --output-manifest training-data\synthetic_sign_video_pairs.csv --min-coverage 0.6 --max-samples 500
```

3. Extract MediaPipe video embeddings:

```powershell
.\venv\Scripts\python.exe training\data_pipeline\extract_video_embeddings.py --input-dir datasets\isolated_videos --output-dir training-data\video_embeddings --manifest training-data\video_embeddings_manifest.csv
```

4. Build video sign->text training set:

```powershell
.\venv\Scripts\python.exe training\data_pipeline\build_video_sign_text_dataset.py --input-manifest training-data\synthetic_sign_video_pairs.csv --keypoint-dir training-data\video_keypoints --labels-csv training-data\video_sign_text_labels.csv
```

5. Train video sign->text transformer:

```powershell
.\venv\Scripts\python.exe training\train_video_sign_to_text.py --labels-csv training-data\video_sign_text_labels.csv --keypoint-dir training-data\video_keypoints --epochs 12
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
