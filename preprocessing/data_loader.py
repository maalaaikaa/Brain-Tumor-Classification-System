from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, RAW_DATA_DIR, SEED


def _find_split_dir(root: Path, split_name: str) -> Path:
    candidates = [root / split_name, root / split_name.capitalize()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {split_name!r} directory inside {root}.")


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomBrightness(0.12),
        ],
        name="augmentation",
    )


def load_datasets(
    data_dir: Path = RAW_DATA_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train/validation/test datasets from a Kaggle-style folder tree."""
    train_dir = _find_split_dir(data_dir, "Training")
    test_dir = _find_split_dir(data_dir, "Testing")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=0.15,
        subset="training",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=0.15,
        subset="validation",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)
    test_ds = test_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)

    if augment:
        augmentation = build_augmentation()
        train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=autotune)

    return (
        train_ds.cache().prefetch(autotune),
        val_ds.cache().prefetch(autotune),
        test_ds.cache().prefetch(autotune),
    )
