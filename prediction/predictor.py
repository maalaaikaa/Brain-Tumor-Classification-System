from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from config import BEST_MODEL_PATH, CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE, MODEL_DIR
from prediction.gradcam import make_gradcam_heatmap, overlay_heatmap
from training.models import ApplicationPreprocess  # noqa: F401 - registers serializable layer

def load_model(model_path: Path = BEST_MODEL_PATH) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train one first with: python train.py --model mobilenetv2"
        )
    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    return tf.keras.models.load_model(model_path, safe_mode=False, custom_objects=custom_objects)

def list_trained_models() -> list[Path]:
    """Scan models directory and return paths of all trained models (excluding the best_model checkpoint)."""
    if not MODEL_DIR.exists():
        return []
    return [p for p in MODEL_DIR.glob("*.keras") if p.name != "best_model.keras"]

def load_trained_models(paths: list[Path]) -> dict[str, tf.keras.Model]:
    """Load a dictionary of trained models from a list of paths."""
    loaded = {}
    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    for p in paths:
        try:
            m = tf.keras.models.load_model(p, safe_mode=False, custom_objects=custom_objects)
            loaded[p.stem] = m
        except Exception as e:
            print(f"Error loading model {p.name}: {e}")
    return loaded

def preprocess_image(file_or_path) -> tuple[np.ndarray, np.ndarray]:
    try:
        image = Image.open(file_or_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Please upload a valid MRI image in JPG, JPEG, PNG, or BMP format.") from exc

    original = np.array(image)
    resized = image.resize(IMAGE_SIZE)
    batch = np.expand_dims(np.array(resized).astype("float32") / 255.0, axis=0)
    return original, batch

def predict_image(
    file_or_path,
    model: tf.keras.Model | dict[str, tf.keras.Model] | None = None,
    gradcam_model_name: str | None = None,
    layer_name: str | None = None,
    colormap: str = "jet",
    alpha: float = 0.42,
) -> dict:
    """
    Predict tumor class for a given image. Supports single model or dict of models (ensemble).
    """
    original, batch = preprocess_image(file_or_path)
    
    # Resolve the models
    if model is None:
        model = load_model()
        
    is_ensemble = isinstance(model, dict)
    
    if is_ensemble:
        if not model:
            raise ValueError("Ensemble model dictionary is empty.")
        # Compute probabilities for each model and average them
        all_probs = []
        for name, m in model.items():
            probs = m.predict(batch, verbose=0)[0]
            all_probs.append(probs)
        probabilities = np.mean(all_probs, axis=0)
        
        # Decide which model to use for Grad-CAM
        if gradcam_model_name and gradcam_model_name in model:
            gradcam_model = model[gradcam_model_name]
        else:
            gradcam_model = list(model.values())[0]
            gradcam_model_name = list(model.keys())[0]
    else:
        # Single model prediction
        probabilities = model.predict(batch, verbose=0)[0]
        gradcam_model = model
        gradcam_model_name = getattr(model, "name", "model")

    class_index = int(np.argmax(probabilities))
    class_name = CLASS_NAMES[class_index]
    explanation_error = None
    
    # Compute Grad-CAM heatmap using the chosen model
    try:
        heatmap = make_gradcam_heatmap(batch, gradcam_model, class_index, layer_name=layer_name)
        highlighted = overlay_heatmap(original, heatmap, alpha=alpha, colormap=colormap)
    except Exception as exc:
        heatmap = np.zeros(original.shape[:2], dtype="float32")
        highlighted = original
        explanation_error = f"Model {gradcam_model_name}: {exc}"

    return {
        "class_key": class_name,
        "prediction": DISPLAY_NAMES[class_name],
        "confidence": float(probabilities[class_index]),
        "probabilities": {
            DISPLAY_NAMES[key]: float(probabilities[index])
            for index, key in enumerate(CLASS_NAMES)
        },
        "original": original,
        "heatmap": heatmap,
        "highlighted": highlighted,
        "explanation_error": explanation_error,
        "is_ensemble": is_ensemble,
        "models_used": list(model.keys()) if is_ensemble else [gradcam_model_name],
    }

def confidence_message(confidence: float) -> tuple[str, str]:
    if confidence >= 0.9:
        return "High", "The model is strongly aligned with this class, but clinical review is still required."
    if confidence >= 0.7:
        return "Moderate", "The prediction is useful as a screening signal and should be reviewed carefully."
    return "Low", "The prediction is uncertain. Use another scan or seek specialist review."

