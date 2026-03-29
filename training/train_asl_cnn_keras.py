"""
Train Keras CNN model for ASL alphabet / hand gesture classification.

Dataset layout expected:
  datasets/asl_alphabet/train/<label>/*.jpg
  datasets/asl_alphabet/val/<label>/*.jpg

Outputs:
  models/asl_cnn.h5
  models/asl_cnn_labels.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os


def build_simple_model(input_size: int, num_classes: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import SGD

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(input_size, input_size, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model, None


def build_inception_model(input_size: int, num_classes: int, train_base: bool):
    from tensorflow.keras import Model
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.optimizers import Adam

    if input_size < 75:
        raise ValueError("InceptionV3 requires image-size >= 75 (recommend 224).")

    base = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size, input_size, 3),
    )
    base.trainable = bool(train_base)

    inputs = Input(shape=(input_size, input_size, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy"])
    return model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/asl_alphabet")
    parser.add_argument("--arch", choices=["simple", "inceptionv3"], default="inceptionv3")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--save-model", default="models/asl_cnn.h5")
    parser.add_argument("--save-best-model", default="models/asl_cnn_best.h5")
    parser.add_argument("--save-labels", default="models/asl_cnn_labels.json")
    parser.add_argument("--eval-report", default="reports/asl_cnn_eval.json")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras not installed. Install optional ASL CNN dependencies."
        ) from exc

    tf.keras.utils.set_random_seed(42)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing train dir: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    if args.arch == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        color_mode = "rgb"
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        color_mode = "grayscale"
        train_gen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
        )
        val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        color_mode=color_mode,
        class_mode="categorical",
        shuffle=True,
    )
    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        color_mode=color_mode,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = len(train_flow.class_indices)
    if num_classes <= 1:
        raise RuntimeError("Need at least 2 classes for training.")

    if args.arch == "inceptionv3":
        model, base_model = build_inception_model(
            input_size=args.image_size,
            num_classes=num_classes,
            train_base=args.train_base,
        )
    else:
        model, base_model = build_simple_model(args.image_size, num_classes)

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
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=args.save_best_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.arch == "inceptionv3" and args.fine_tune_epochs > 0 and base_model is not None:
        from tensorflow.keras.optimizers import Adam

        base_model.trainable = True
        freeze_until = max(0, len(base_model.layers) - 50)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(args.fine_tune_lr),
            metrics=["accuracy"],
        )
        history_ft = model.fit(
            train_flow,
            validation_data=val_flow,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        for k, vals in history_ft.history.items():
            history.history.setdefault(k, [])
            history.history[k].extend(vals)

    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    model.save(args.save_model)

    idx_to_label = {int(v): k for k, v in train_flow.class_indices.items()}
    os.makedirs(os.path.dirname(args.save_labels) or ".", exist_ok=True)
    with open(args.save_labels, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2)

    val_loss, val_accuracy = model.evaluate(val_flow, verbose=0)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arch": args.arch,
        "image_size": int(args.image_size),
        "num_classes": int(num_classes),
        "train_samples": int(getattr(train_flow, "samples", 0)),
        "val_samples": int(getattr(val_flow, "samples", 0)),
        "epochs_requested": int(args.epochs),
        "fine_tune_epochs_requested": int(args.fine_tune_epochs),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "artifacts": {
            "model_path": args.save_model,
            "best_model_path": args.save_best_model,
            "labels_path": args.save_labels,
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
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
