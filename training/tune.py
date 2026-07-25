from __future__ import annotations

import itertools
import json

import pandas as pd

from config import REPORTS_DIR
from training.trainer import train_model


def run_basic_tuning() -> pd.DataFrame:
    grid = {
        "learning_rate": [1e-3, 1e-4],
        "batch_size": [16, 32],
        "dropout": [0.25, 0.4],
    }
    rows = []
    for lr, batch_size, dropout in itertools.product(*grid.values()):
        model = train_model("mobilenetv2", epochs=8, batch_size=batch_size, learning_rate=lr, dropout=dropout)
        rows.append(
            {
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "val_accuracy": float(max(model.history.history.get("val_accuracy", [0]))),
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    df.to_csv(REPORTS_DIR / "hyperparameter_results.csv", index=False)
    (REPORTS_DIR / "best_hyperparameters.json").write_text(json.dumps(df.iloc[0].to_dict(), indent=2))
    return df


if __name__ == "__main__":
    print(run_basic_tuning())
