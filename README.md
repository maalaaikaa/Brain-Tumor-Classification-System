# Brain Tumor Classification System

Professional deep learning application for classifying brain MRI scans into four classes: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- MRI upload and real-time prediction
- Confidence score and probability for every class
- Grad-CAM heatmap and highlighted scan regions
- Custom CNN, MobileNetV2, EfficientNetB0, and ResNet50 builders
- Data augmentation, normalization, RGB conversion, train-validation-test loading
- EDA reports for class distribution, image dimensions, and pixel intensity
- Evaluation with accuracy, precision, recall, F1-score, AUC, confusion matrix, and classification report
- Healthcare-style Streamlit interface with dashboard pages
- Medical disclaimer and reliability interpretation
- Docker-ready deployment

## Project Structure

```text
Brain Tumor Classification System/
|-- dataset/
|-- notebooks/
|-- models/
|-- training/
|-- preprocessing/
|-- prediction/
|-- utils/
|-- ui/
|-- static/
|-- screenshots/
|-- reports/
|-- app.py
|-- train.py
|-- predict.py
|-- requirements.txt
|-- README.md
|-- Dockerfile
|-- LICENSE
`-- .gitignore
```

## Dataset

The downloader uses the public Kaggle dataset `masoudnickparvar/brain-tumor-mri-dataset`, which contains:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Download it with:

```bash
python -m dataset.download_dataset
```

The expected layout after download is:

```text
dataset/raw/
|-- Training/
|   |-- glioma/
|   |-- meningioma/
|   |-- notumor/
|   `-- pituitary/
`-- Testing/
    |-- glioma/
    |-- meningioma/
    |-- notumor/
    `-- pituitary/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Download data:

```bash
python -m dataset.download_dataset
```

Generate EDA reports:

```bash
python -m preprocessing.eda
```

Train a model:

```bash
python train.py --model mobilenetv2 --epochs 20 --batch-size 32
```

Predict from the command line:

```bash
python predict.py path\to\mri.png
```

Run the application:

```bash
streamlit run app.py
```

## Model Development

Implemented model options:

- `custom_cnn`: compact baseline CNN with batch normalization, dropout, and L2 regularization
- `mobilenetv2`: fast transfer learning baseline
- `efficientnetb0`: strong accuracy-efficiency transfer model
- `resnet50`: deeper transfer learning baseline

Training includes EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSV logging, mixed precision when a GPU is available, and saved Keras models.

## Evaluation

Run evaluation from Python:

```python
from utils.evaluation import evaluate_model
evaluate_model("models/best_model.keras")
```

Reports are written to `reports/`, including classification metrics and confusion matrix outputs.

## Medical Disclaimer

This project is an educational AI decision support prototype. It is not a medical device and must not be used as a replacement for clinical diagnosis by qualified healthcare professionals.

## Future Improvements

- Ensemble prediction across multiple trained backbones
- PDF prediction report export
- MRI quality assessment
- REST API with FastAPI
- CI/CD with GitHub Actions
- Streamlit Community Cloud or Hugging Face Spaces deployment
