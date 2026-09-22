"""Normalize raw Fashion-MNIST data and split into train/validation/test.

Reads the raw arrays saved by src/prepare.py, scales pixel values to
[0, 1], carves a deterministic validation split out of the training set,
and writes train/validation/test arrays to data/processed/. The test set
is passed through untouched aside from normalization.
"""

from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

VALIDATION_FRACTION = 0.1
SPLIT_SEED = 42


def main() -> None:
    train_images = np.load(RAW_DIR / "train_images.npy")
    train_labels = np.load(RAW_DIR / "train_labels.npy")
    test_images = np.load(RAW_DIR / "test_images.npy")
    test_labels = np.load(RAW_DIR / "test_labels.npy")

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    num_samples = train_images.shape[0]
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled_indices = rng.permutation(num_samples)

    num_val = int(round(num_samples * VALIDATION_FRACTION))
    val_indices = shuffled_indices[:num_val]
    train_indices = shuffled_indices[num_val:]

    final_train_images = train_images[train_indices]
    final_train_labels = train_labels[train_indices]
    val_images = train_images[val_indices]
    val_labels = train_labels[val_indices]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "train_images.npy", final_train_images)
    np.save(PROCESSED_DIR / "train_labels.npy", final_train_labels)
    np.save(PROCESSED_DIR / "val_images.npy", val_images)
    np.save(PROCESSED_DIR / "val_labels.npy", val_labels)
    np.save(PROCESSED_DIR / "test_images.npy", test_images)
    np.save(PROCESSED_DIR / "test_labels.npy", test_labels)

    print(f"Saved processed data to {PROCESSED_DIR}")
    print(f"train={final_train_images.shape[0]} val={val_images.shape[0]} test={test_images.shape[0]}")


if __name__ == "__main__":
    main()