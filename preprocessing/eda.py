from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, UnidentifiedImageError

from config import CLASS_NAMES, RAW_DATA_DIR, REPORTS_DIR


def scan_dataset(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    rows = []
    for split in ["Training", "Testing"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for class_name in CLASS_NAMES:
            for path in (split_dir / class_name).glob("*"):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    continue
                pixels = np.asarray(image, dtype="float32")
                w, h = image.size
                rows.append(
                    {
                        "split": split,
                        "class": class_name,
                        "path": str(path),
                        "width": w,
                        "height": h,
                        "channels": 3,
                        "mean_intensity": float(pixels.mean()),
                        "std_intensity": float(pixels.std()),
                    }
                )
    return pd.DataFrame(rows)


def generate_eda_report(data_dir: Path = RAW_DATA_DIR, reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = scan_dataset(data_dir)
    if df.empty:
        raise FileNotFoundError("No images found. Run python -m dataset.download_dataset first.")

    df.to_csv(reports_dir / "dataset_scan.csv", index=False)
    class_fig = px.histogram(df, x="class", color="split", barmode="group", title="Class Distribution")
    class_fig.write_html(reports_dir / "class_distribution.html")
    intensity_fig = px.box(df, x="class", y="mean_intensity", color="split", title="Pixel Intensity by Class")
    intensity_fig.write_html(reports_dir / "pixel_intensity.html")
    size_fig = px.scatter(df, x="width", y="height", color="class", title="Image Dimensions")
    size_fig.write_html(reports_dir / "image_dimensions.html")
    return df


if __name__ == "__main__":
    report = generate_eda_report()
    print(report.groupby(["split", "class"]).size())
