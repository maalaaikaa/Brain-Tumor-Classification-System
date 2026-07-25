# Project Completion TODO

## Steps

- [x] Analyze codebase and identify gaps
- [x] Get plan approved by user

### Implementation

- [x] **Edit 1: `app.py`** - Create UPLOAD_DIR at startup, handle missing model gracefully
- [x] **Edit 2: `preprocessing/data_loader.py`** - Apply augmentation to training dataset
- [x] **Skipped: `prediction/predictor.py`** - No upload saving needed; UPLOAD_DIR created in app.py
- [x] **Skipped: `utils/evaluation.py`** - Already creates REPORTS_DIR via mkdir inside function

### Verification

- [x] `app.py` - Syntax verified, no errors
- [x] `app.py` - UPLOAD_DIR created at startup, missing model handled gracefully
- [x] `data_loader.py` - Augmentation integrated into training data pipeline
- [x] `data_loader.py` - Backward compatible (augment=True by default)

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset
python -m dataset.download_dataset

# 3. Train a model
python train.py --model mobilenetv2 --epochs 20 --batch-size 32

# 4. Run the app
streamlit run app.py
```

