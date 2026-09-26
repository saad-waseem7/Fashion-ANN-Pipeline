"""Train a simple ANN on the processed Fashion-MNIST data.

Loads the train/validation arrays produced by src/preprocess.py, trains a
Flatten -> Dense -> Dropout -> Dense(softmax) network, and saves the
trained model and training history under models/.
"""

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


def build_model(hidden_units: int, dropout_rate: float, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            # Conflict simulation: branch B training configuration
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    fieldnames = ["epoch"] + list(history.history.keys())
    num_epochs = len(history.epoch)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_epochs):
            row = {"epoch": history.epoch[i]}
            for key, values in history.history.items():
                row[key] = values[i]
            writer.writerow(row)


def main() -> None:
    with PARAMS_PATH.open() as f:
        params = yaml.safe_load(f)

    dense_units = params["train"]["dense_units"]
    dropout_rate = params["train"]["dropout_rate"]
    learning_rate = params["train"]["learning_rate"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]

    train_images = np.load(PROCESSED_DIR / "train_images.npy")
    train_labels = np.load(PROCESSED_DIR / "train_labels.npy")
    val_images = np.load(PROCESSED_DIR / "val_images.npy")
    val_labels = np.load(PROCESSED_DIR / "val_labels.npy")

    model = build_model(dense_units, dropout_rate, learning_rate)

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODELS_DIR / "model.h5")
    save_history(history, MODELS_DIR / "history.csv")

    print(f"Saved model to {MODELS_DIR / 'model.h5'}")
    print(f"Saved training history to {MODELS_DIR / 'history.csv'}")


if __name__ == "__main__":
    main()