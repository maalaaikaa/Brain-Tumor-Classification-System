from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
STATIC_DIR = PROJECT_ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.json"
