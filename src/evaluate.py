"""Evaluate the trained ANN on the held-out Fashion-MNIST test set.

Loads the processed test arrays and the trained model, computes test
loss/accuracy and a confusion matrix, and saves both under models/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

NUM_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def compute_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted_labels), 1)
    return matrix


def save_confusion_matrix_plot(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    test_images = np.load(PROCESSED_DIR / "test_images.npy")
    test_labels = np.load(PROCESSED_DIR / "test_labels.npy")

    model = tf.keras.models.load_model(MODELS_DIR / "model.h5")

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    predicted_probs = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predicted_probs, axis=1)

    confusion_matrix = compute_confusion_matrix(test_labels, predicted_labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.savetxt(MODELS_DIR / "confusion_matrix.csv", confusion_matrix, fmt="%d", delimiter=",")
    save_confusion_matrix_plot(confusion_matrix, MODELS_DIR / "confusion_matrix.png")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with (MODELS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved confusion matrix to {MODELS_DIR / 'confusion_matrix.png'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()