"""
preprocessing.py — Background & Noise Removal Module
======================================================
RESPONSIBILITY: Clean each ROI image BEFORE any detection occurs.
This module standardizes image quality to improve detection reliability.

SUB-OPERATIONS (in order):
  1. Median blur  → removes salt-and-pepper noise from camera sensor
  2. BGR → HSV    → converts to HSV color space for robust color thresholding

RULES:
  - NO detection logic here
  - NO contour analysis
  - Output is a clean image ready for the detection stages

OUTPUT per ROI:
  {
    "bgr":  cleaned BGR image (blurred),
    "hsv":  HSV version of cleaned image,
    "original": original unmodified ROI (kept for reference)
  }
"""

import cv2
import numpy as np
import config


def preprocess_roi(roi: np.ndarray) -> dict:
    """
    Applies noise reduction and color space conversion to a single ROI.

    Steps:
      1. Median blur to reduce sensor noise
      2. Convert BGR → HSV for color-based thresholding in later stages

    Parameters
    ----------
    roi : np.ndarray
        Raw BGR ROI image slice from frame_divider.

    Returns
    -------
    dict
        {
            "original": np.ndarray  — untouched ROI (for debug overlays),
            "bgr":      np.ndarray  — blurred BGR image,
            "hsv":      np.ndarray  — HSV version of blurred image
        }
    """

    # --- Step 1: Store original ROI for debug/reference ---
    original = roi.copy()

    # --- Step 2: Median Blur ---
    # Median blur is preferred over Gaussian for removing speckling noise
    # because it preserves edges better (it replaces each pixel with the
    # median of its neighborhood, not the mean).
    #
    # BLUR_KERNEL_SIZE from config.py:
    #   Increase (e.g., 7, 9) → stronger smoothing, removes more noise
    #   Decrease (e.g., 3)    → lighter smoothing, keeps finer detail
    blurred = cv2.medianBlur(roi, config.BLUR_KERNEL_SIZE)

    # --- Step 3: BGR → HSV Color Conversion ---
    # HSV (Hue, Saturation, Value) is more robust than BGR for color
    # thresholding under varying lighting conditions.
    # - Hue encodes the actual color regardless of brightness
    # - This lets us detect "green-yellow pear" reliably even if lighting changes
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return {
        "original": original,  # Untouched ROI — use for debug drawings
        "bgr":      blurred,   # Noise-reduced BGR image
        "hsv":      hsv,       # HSV version — used by detection stages
    }


def preprocess_all_rois(roi_dict: dict) -> dict:
    """
    Applies preprocessing to all 6 ROIs.

    Parameters
    ----------
    roi_dict : dict
        Output from frame_divider.frame_divider():
        { "A3": np.ndarray, "B3": np.ndarray, ... }

    Returns
    -------
    dict
        { roi_name: preprocessed_dict } for all 6 ROIs.
    """
    preprocessed = {}

    for roi_name, roi_image in roi_dict.items():
        preprocessed[roi_name] = preprocess_roi(roi_image)

    return preprocessed
