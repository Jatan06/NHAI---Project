from __future__ import annotations

import cv2
import numpy as np


def calculate_reflectivity_score(region: np.ndarray) -> int:
    """Estimate retroreflectivity by the mean grayscale brightness."""
    if region is None or region.size == 0:
        return 0

    if len(region.shape) == 3:
        grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = region

    return int(round(float(np.mean(grayscale))))


def classify_reflectivity(score: int) -> str:
    """Map brightness score to the requested condition label."""
    if score > 180:
        return "Good"
    if 100 <= score <= 180:
        return "Moderate"
    return "Poor"
