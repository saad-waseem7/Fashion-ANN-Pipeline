"""Fetch raw Fashion-MNIST data and save it to data/raw/.

This script performs no preprocessing or normalization. It only obtains
the dataset and persists the raw arrays for downstream pipeline stages.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    np.save(RAW_DIR / "train_images.npy", train_images)
    np.save(RAW_DIR / "train_labels.npy", train_labels)
    np.save(RAW_DIR / "test_images.npy", test_images)
    np.save(RAW_DIR / "test_labels.npy", test_labels)

    print(f"Saved raw Fashion-MNIST data to {RAW_DIR}")


if __name__ == "__main__":
    main()