from __future__ import annotations
import cv2
import numpy as np

def blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def glare_fraction(image_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # very bright, low-saturation areas are likely glare on paper
    mask = (hsv[:, :, 2] >= 248) & (hsv[:, :, 1] <= 35)
    return float(np.mean(mask))

def clipping_risk(marker_points, image_shape) -> bool:
    if marker_points is None:
        return False
    h, w = image_shape[:2]
    margin = 0.015 * min(h, w)
    for x, y in np.asarray(marker_points).reshape(-1, 2):
        if x < margin or y < margin or x > (w - margin) or y > (h - margin):
            return True
    return False

def quality_summary(image_bgr: np.ndarray, marker_points=None) -> dict:
    b = blur_score(image_bgr)
    g = glare_fraction(image_bgr)
    c = clipping_risk(marker_points, image_bgr.shape)
    return {
        "blur_score": b,
        "glare_fraction": g,
        "clipping_risk": c,
        "blur_warning": b < 70.0,
        "glare_warning": g > 0.035,
    }
