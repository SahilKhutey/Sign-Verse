"""
Train video-level LSTM classifier from CNN feature sequences.

Input:
  training-data/video_features_manifest.csv

Output:
  models/video_lstm.h5
  models/video_lstm_labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np


def _pad_or_truncate(seq: np.ndarray, seq_len: int, feat_dim: int) -> np.ndarray:
    if seq.ndim != 2:
        return np.zeros((seq_len, feat_dim), dtype=np.float32)
    t, d = seq.shape
    if d != feat_dim:
        if d < feat_dim:
            pad = np.zeros((t, feat_dim - d), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)
        else:
            seq = seq[:, :feat_dim]
    if t < seq_len:
        pad_t = np.zeros((seq_len - t, feat_dim), dtype=np.float32)
        seq = np.concatenate([seq, pad_t], axis=0)
    elif t > seq_len:
        seq = seq[:seq_len]
    return seq.astype(np.float32)


def _load_data(manifest: str, seq_len: int, feat_dim: int):
    rows = []
    with open(manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fp = row.get("features_path", "")
            label = row.get("label", "")
            if fp and label and os.path.exists(fp):
                rows.append((fp, label))
    if not rows:
        raise RuntimeError("No feature samples found.")

    labels_sorted = sorted({label for _, label in rows})
    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}

    x, y = [], []
    for fp, label in rows:
        seq = np.load(fp)
        x.append(_pad_or_truncate(seq, seq_len=seq_len, feat_dim=feat_dim))
        y.append(label_to_idx[label])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return x, y, labels_sorted


def _split_train_val(x, y, val_split: float = 0.2, seed: int = 42):
    n = len(x)
    if n == 0:
        raise RuntimeError("No samples available for train/val split.")
    if n == 1:
        # Tiny dataset fallback for smoke tests.
        return x, y, x, y

    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_split))
    n_val = max(1, min(n - 1, n_val))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_lstm(seq_len: int, feat_dim: int, num_classes: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Masking

    model = Sequential(
        [
            Input(shape=(seq_len, feat_dim)),
            Masking(mask_value=0.0),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training-data/video_features_manifest.csv")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--feat-dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--save-model", default="models/video_lstm.h5")
    parser.add_argument("--save-best-model", default="models/video_lstm_best.h5")
    parser.add_argument("--save-labels", default="models/video_lstm_labels.json")
    parser.add_argument("--eval-report", default="reports/video_lstm_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras not installed. Install optional ASL CNN dependencies."
        ) from exc

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    x, y, labels = _load_data(args.manifest, seq_len=args.seq_len, feat_dim=args.feat_dim)
    x_train, y_train, x_val, y_val = _split_train_val(
        x,
        y,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = build_lstm(args.seq_len, args.feat_dim, len(labels))
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6,
        ),
    ]
    if args.save_best_model:
        callbacks.append(
            ModelCheckpoint(
                filepath=args.save_best_model,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    model.save(args.save_model)
    os.makedirs(os.path.dirname(args.save_labels) or ".", exist_ok=True)
    with open(args.save_labels, "w", encoding="utf-8") as f:
        json.dump({i: label for i, label in enumerate(labels)}, f, indent=2)

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    report = {
        "model": "video_lstm_keras",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "num_samples": int(len(x)),
        "num_train_samples": int(len(x_train)),
        "num_val_samples": int(len(x_val)),
        "num_classes": int(len(labels)),
        "seq_len": int(args.seq_len),
        "feature_dim": int(args.feat_dim),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "history": {
            k: [float(vv) for vv in v]
            for k, v in history.history.items()
        },
        "artifacts": {
            "model_path": args.save_model,
            "best_model_path": args.save_best_model,
            "labels_path": args.save_labels,
        },
    }
    os.makedirs(os.path.dirname(args.eval_report) or ".", exist_ok=True)
    with open(args.eval_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model: {args.save_model}")
    print(f"Saved best model: {args.save_best_model}")
    print(f"Saved labels: {args.save_labels}")
    print(f"Saved eval report: {args.eval_report}")


if __name__ == "__main__":
    main()
