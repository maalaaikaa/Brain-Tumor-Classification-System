from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from config import CLASS_NAMES, METRICS_PATH, REPORTS_DIR
from preprocessing.data_loader import load_datasets
from training.models import ApplicationPreprocess  # noqa: F401 - registers serializable layer


def evaluate_model(model_path: str) -> dict:
    _, _, test_ds = load_datasets()
    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects=custom_objects)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, len(CLASS_NAMES)), y_prob, multi_class="ovr")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(REPORTS_DIR / "classification_report.csv")
    np.savetxt(REPORTS_DIR / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    metrics = {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"], "auc": float(auc)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics
