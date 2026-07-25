from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from config import RAW_DATA_DIR


KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"


def download_dataset(destination: Path = RAW_DATA_DIR) -> Path:
    """Download the public Brain Tumor MRI dataset from Kaggle Hub."""
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"Dataset ready at: {destination}")
    return destination


if __name__ == "__main__":
    download_dataset()
