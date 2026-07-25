from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from config import BEST_MODEL_PATH, HISTORY_PATH, MODEL_DIR, REPORTS_DIR
from preprocessing.data_loader import load_datasets
from training.models import get_model


def configure_acceleration() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def train_model(
    model_name: str = "mobilenetv2",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    dropout: float = 0.35,
) -> tf.keras.Model:
    configure_acceleration()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_datasets(batch_size=batch_size)
    model = get_model(model_name, dropout=dropout, learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(HISTORY_PATH)),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(MODEL_DIR / f"{model_name}.keras")

    summary = {
        "model": model_name,
        "epochs": len(history.history["loss"]),
        "best_val_accuracy": max(history.history.get("val_accuracy", [0])),
        "best_val_loss": min(history.history.get("val_loss", [0])),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False)
    return model
