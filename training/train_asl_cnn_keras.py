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
import json
import os


def build_model(input_size: int, num_classes: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/asl_alphabet")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save-model", default="models/asl_cnn.h5")
    parser.add_argument("--save-labels", default="models/asl_cnn_labels.json")
    args = parser.parse_args()

    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras not installed. Install optional ASL CNN dependencies."
        ) from exc

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing train dir: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    train_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True,
    )
    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = len(train_flow.class_indices)
    if num_classes <= 1:
        raise RuntimeError("Need at least 2 classes for training.")

    model = build_model(args.image_size, num_classes)
    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
    )

    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    model.save(args.save_model)

    idx_to_label = {int(v): k for k, v in train_flow.class_indices.items()}
    os.makedirs(os.path.dirname(args.save_labels), exist_ok=True)
    with open(args.save_labels, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2)

    print(f"Saved model: {args.save_model}")
    print(f"Saved labels: {args.save_labels}")


if __name__ == "__main__":
    main()
