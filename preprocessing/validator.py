from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

def validate_mri_scan(file_or_path) -> dict:
    """
    Validate whether the input image resembles a typical brain MRI scan.
    
    Heuristics checked:
    1. Grayscale representation (low color variance across RGB channels).
    2. Dark background corners (typical for head scans on black background).
    3. Brighter center region (brain structure).
    4. Reasonable aspect ratio and dimensions.
    
    Returns:
        dict: {
            "is_valid": bool,
            "reasons": list[str],
            "metrics": dict
        }
    """
    reasons = []
    
    try:
        if isinstance(file_or_path, (str, Path)):
            img = Image.open(file_or_path)
        else:
            img = Image.open(file_or_path)
    except Exception as exc:
        return {
            "is_valid": False,
            "reasons": [f"Could not open image file: {exc}"],
            "metrics": {}
        }
        
    width, height = img.size
    
    # Check 1: Size constraints
    if width < 128 or height < 128:
        reasons.append(f"Image resolution is too low ({width}x{height}). Minimum recommended is 128x128.")
        
    # Check 2: Aspect ratio constraints
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        reasons.append(f"Unusual image aspect ratio ({aspect_ratio:.2f}). Typical MRI scans are close to square.")
        
    # Convert image to numpy array for pixel checks
    img_rgb = img.convert("RGB")
    pixels = np.array(img_rgb).astype("float32")
    
    # Check 3: Color variance (MRI scans are typically grayscale, meaning R ≈ G ≈ B)
    mean_ch0 = pixels[..., 0].mean()
    mean_ch1 = pixels[..., 1].mean()
    mean_ch2 = pixels[..., 2].mean()
    
    r_g_diff = np.abs(pixels[..., 0] - pixels[..., 1]).mean()
    r_b_diff = np.abs(pixels[..., 0] - pixels[..., 2]).mean()
    color_diff = (r_g_diff + r_b_diff) / 2.0
    
    if color_diff > 15.0:
        reasons.append(
            f"Image appears to contain significant color (avg channel diff: {color_diff:.1f}). "
            "Typical MRI scans are grayscale."
        )
        
    # Check 4: Background corners vs Center brightness
    # Extract 10% corner margins
    h_margin = max(1, int(height * 0.1))
    w_margin = max(1, int(width * 0.1))
    
    top_left = pixels[0:h_margin, 0:w_margin]
    top_right = pixels[0:h_margin, width-w_margin:width]
    bottom_left = pixels[height-h_margin:height, 0:w_margin]
    bottom_right = pixels[height-h_margin:height, width-w_margin:width]
    
    corners_mean = np.mean([top_left.mean(), top_right.mean(), bottom_left.mean(), bottom_right.mean()])
    
    # Extract center 50% area
    cy_start, cy_end = int(height * 0.25), int(height * 0.75)
    cx_start, cx_end = int(width * 0.25), int(width * 0.75)
    center = pixels[cy_start:cy_end, cx_start:cx_end]
    center_mean = center.mean()
    
    # MRI scans have very dark background corners (e.g. < 45)
    if corners_mean > 55.0:
        reasons.append(
            f"Background corners are too bright (mean: {corners_mean:.1f}). "
            "Typical MRI scans are isolated on a dark background."
        )
        
    # Center should be brighter than corners
    if center_mean < corners_mean + 10.0 or center_mean < 25.0:
        reasons.append(
            f"Center region is not sufficiently brighter than corners (center: {center_mean:.1f}, corners: {corners_mean:.1f}). "
            "Typical brain MRI scans contain bright/gray brain tissue inside a dark background outer frame."
        )
        
    metrics = {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "color_diff": float(color_diff),
        "corners_mean": float(corners_mean),
        "center_mean": float(center_mean)
    }
    
    return {
        "is_valid": len(reasons) == 0,
        "reasons": reasons,
        "metrics": metrics
    }
