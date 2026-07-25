from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from config import UPLOAD_DIR, METRICS_PATH, TRAINING_SUMMARY_PATH
from database.db_manager import save_record, get_all_records, get_record_by_id
from prediction.predictor import (
    predict_image,
    list_trained_models,
    load_trained_models,
    load_model,
)
from preprocessing.validator import validate_mri_scan
from utils.pdf_generator import generate_pdf_report

app = FastAPI(
    title="Brain Tumor Classification API",
    description="REST API for brain tumor MRI classification, Grad-CAM visualization, and diagnostic reporting.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper to base64 encode images
def get_image_as_base64(path: Path) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", summary="Classify brain tumor MRI scan")
async def api_predict(
    file: UploadFile = File(...),
    patient_id: str = Form("PT-UNKNOWN"),
    patient_age: int = Form(None),
    patient_gender: str = Form(None),
    use_ensemble: bool = Form(False),
    colormap: str = Form("jet"),
    alpha: float = Form(0.42),
    physician_notes: str = Form(None),
):
    """
    Upload an MRI scan image, perform classification (single or ensemble model), 
    generate Grad-CAM, check input validity, and log the diagnostic entry.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file temporarily to run validator
    temp_path = UPLOAD_DIR / f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Validate image format/structure
    validation = validate_mri_scan(temp_path)
    
    try:
        # Load correct models
        if use_ensemble:
            model_paths = list_trained_models()
            if len(model_paths) > 1:
                models = load_trained_models(model_paths)
                prediction_result = predict_image(
                    temp_path,
                    model=models,
                    colormap=colormap,
                    alpha=alpha
                )
            else:
                # Fallback to single default model
                single_model = load_model()
                prediction_result = predict_image(
                    temp_path,
                    model=single_model,
                    colormap=colormap,
                    alpha=alpha
                )
        else:
            single_model = load_model()
            prediction_result = predict_image(
                temp_path,
                model=single_model,
                colormap=colormap,
                alpha=alpha
            )
    except FileNotFoundError as fnf:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Generate permanent filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_filename = f"{patient_id}_{timestamp}_orig.png"
    grad_filename = f"{patient_id}_{timestamp}_grad.png"
    
    orig_path = UPLOAD_DIR / orig_filename
    grad_path = UPLOAD_DIR / grad_filename
    
    # Save the original and highlighted images
    Image.fromarray(prediction_result["original"]).save(orig_path)
    Image.fromarray(prediction_result["highlighted"]).save(grad_path)
    
    # Remove the temp file
    temp_path.unlink(missing_ok=True)
    
    # Save record to SQLite
    record_id = save_record(
        patient_id=patient_id,
        patient_age=patient_age,
        patient_gender=patient_gender,
        prediction_class=prediction_result["prediction"],
        confidence=prediction_result["confidence"],
        probabilities_dict=prediction_result["probabilities"],
        image_path=str(orig_path),
        physician_notes=physician_notes,
    )
    
    # Return structured JSON
    return {
        "record_id": record_id,
        "patient_id": patient_id,
        "prediction": prediction_result["prediction"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "validation": {
            "is_valid_mri": validation["is_valid"],
            "warnings": validation["reasons"]
        },
        "is_ensemble": prediction_result["is_ensemble"],
        "models_used": prediction_result["models_used"],
        "explanation_error": prediction_result["explanation_error"],
        "original_image_base64": get_image_as_base64(orig_path),
        "highlighted_image_base64": get_image_as_base64(grad_path),
    }

@app.get("/history", summary="Get all patient diagnostic records")
def api_history():
    """Retrieve all diagnostic records logged in the database."""
    return get_all_records()

@app.get("/report/{record_id}", summary="Download PDF Diagnostic Report")
def api_report(record_id: int):
    """Generate and return the clinical PDF report for the given record ID."""
    record = get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")
        
    orig_path = Path(record["image_path"])
    # Grad image has _grad.png instead of _orig.png
    grad_path = Path(str(orig_path).replace("_orig.png", "_grad.png"))
    
    pdf_path = UPLOAD_DIR / f"report_{record_id}.pdf"
    
    try:
        generate_pdf_report(
            record=record,
            original_path=orig_path,
            highlighted_path=grad_path,
            output_path=pdf_path
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {exc}")
        
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"Brain_Tumor_Report_{record['patient_id']}.pdf"
    )

@app.get("/metrics", summary="Get model evaluation metrics")
def api_metrics():
    """Retrieve test-set metrics or summary history if available."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    elif TRAINING_SUMMARY_PATH.exists():
        return {
            "note": "Complete test set metrics not found. Showing latest training summary.",
            "data": json.loads(TRAINING_SUMMARY_PATH.read_text())
        }
    else:
        return {"detail": "No model metrics generated yet. Run train.py and evaluate_model first."}
