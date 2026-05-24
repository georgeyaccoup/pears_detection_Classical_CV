"""
pear_detection.py — Pear Detection Module
==========================================
RESPONSIBILITY: Determine whether a pear is present in a preprocessed ROI,
and if so, extract its shape mask, area, and contour.

SUB-OPERATIONS (in order):
  1. HSV threshold     → isolate pear-colored pixels (green/yellow range)
  2. Morphology open   → remove small noise blobs
  3. Morphology close  → fill internal holes in pear mask
  4. Find contours     → extract connected regions from mask
  5. Select largest    → largest contour = most likely pear
  6. Validate shape    → check area, aspect ratio, solidity

OUTPUT per ROI:
  {
    "pear_flag":    int   1 = pear present, 0 = no pear,
    "pear_mask":    np.ndarray  binary mask of pear pixels,
    "pear_area":    float  pear area in pixels² (or 0),
    "pear_contour": np.ndarray  pear contour points (or None),
    "pear_bbox":    tuple  (x, y, w, h) bounding box (or None)
  }

RULES:
  - NO infection analysis here
  - NO motor commands
  - ONLY shape-based pear detection
"""

import cv2
import numpy as np
import config


def _apply_pear_threshold(hsv_roi: np.ndarray) -> np.ndarray:
    """
    Applies HSV color thresholding to isolate pear-colored pixels.

    Uses PEAR_HSV_LOWER and PEAR_HSV_UPPER from config.py.
    Pixels inside the range → white (255); outside → black (0).

    Parameters
    ----------
    hsv_roi : np.ndarray
        HSV image of the ROI from preprocessing.

    Returns
    -------
    np.ndarray
        Binary mask: 255 = pear candidate pixels, 0 = background.
    """
    # cv2.inRange returns 255 where all channels are within [lower, upper]
    # Tune PEAR_HSV_LOWER/UPPER in config.py to match your pear variety:
    #   H range controls what color is accepted (green-yellow band)
    #   S min filters out grey/white backgrounds
    #   V min/max filters shadows and overexposed highlights
    mask = cv2.inRange(hsv_roi, config.PEAR_HSV_LOWER, config.PEAR_HSV_UPPER)
    return mask


def _apply_morphology(mask: np.ndarray) -> np.ndarray:
    """
    Cleans the binary pear mask using morphological operations.

    Step 1: Opening (erosion → dilation)
      - Removes small isolated blobs (noise, specks, small reflections)
      - MORPH_OPEN_KERNEL: larger = more aggressive noise removal

    Step 2: Closing (dilation → erosion)
      - Fills small holes inside the pear region (dark spots, shadows)
      - MORPH_CLOSE_KERNEL: larger = fills bigger holes

    Parameters
    ----------
    mask : np.ndarray
        Raw binary mask from HSV threshold.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    # Opening: erode then dilate → eliminates small noise blobs
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, config.MORPH_OPEN_KERNEL)

    # Closing: dilate then erode → fills holes in pear body
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, config.MORPH_CLOSE_KERNEL)

    return closed


def _find_largest_contour(mask: np.ndarray):
    """
    Finds all contours in the mask and returns the largest one.

    The largest contour by area is assumed to be the pear body.
    Smaller contours are considered noise/debris.

    Parameters
    ----------
    mask : np.ndarray
        Binary cleaned mask.

    Returns
    -------
    tuple : (best_contour or None, all_contours list)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, []  # No contours found at all

    # Select contour with maximum area
    best = max(contours, key=cv2.contourArea)
    return best, contours


def _validate_contour(contour) -> bool:
    """
    Validates whether a contour is a plausible pear shape.

    Checks three geometric properties:
      1. Area       — must be within PEAR_MIN_AREA .. PEAR_MAX_AREA
      2. Aspect ratio — width/height must be within reasonable range
      3. Solidity   — contour_area / convex_hull_area (roundness measure)

    Parameters
    ----------
    contour : np.ndarray
        OpenCV contour array.

    Returns
    -------
    bool
        True if contour passes all validation tests.
    """
    area = cv2.contourArea(contour)

    # --- Check 1: Area bounds ---
    # Too small → noise or partial view; too large → merged blobs
    if area < config.PEAR_MIN_AREA or area > config.PEAR_MAX_AREA:
        return False

    # --- Check 2: Aspect ratio ---
    # Bounding box width/height ratio; pears are roughly 0.6–1.4
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < config.PEAR_MIN_ASPECT_RATIO or aspect_ratio > config.PEAR_MAX_ASPECT_RATIO:
        return False

    # --- Check 3: Solidity ---
    # Measures how "convex" the shape is (1.0 = perfect convex hull)
    # Low solidity → highly irregular shape (not a pear)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return False
    solidity = float(area) / float(hull_area)
    if solidity < config.PEAR_MIN_SOLIDITY:
        return False

    return True


def pear_detection(preprocessed_roi: dict) -> dict:
    """
    Main pear detection function for a single preprocessed ROI.

    Runs the full pipeline:
      threshold → morphology → contour extraction → validation

    Parameters
    ----------
    preprocessed_roi : dict
        Output from preprocessing.preprocess_roi():
        { "original", "bgr", "hsv" }

    Returns
    -------
    dict
        {
            "pear_flag":    int,           1 = pear found, 0 = no pear
            "pear_mask":    np.ndarray,    binary mask (0 or 255 per pixel)
            "pear_area":    float,         pear pixel area (0 if no pear)
            "pear_contour": np.ndarray,    contour points (None if no pear)
            "pear_bbox":    tuple,         (x, y, w, h) bounding box
        }
    """

    hsv = preprocessed_roi["hsv"]  # HSV image from preprocessing stage

    # --- Step 1: HSV Threshold ---
    raw_mask = _apply_pear_threshold(hsv)

    # --- Step 2: Morphological cleanup ---
    clean_mask = _apply_morphology(raw_mask)

    # --- Step 3: Find largest contour ---
    best_contour, all_contours = _find_largest_contour(clean_mask)

    # Default output (no pear case)
    result = {
        "pear_flag":    0,
        "pear_mask":    np.zeros(clean_mask.shape, dtype=np.uint8),
        "pear_area":    0.0,
        "pear_contour": None,
        "pear_bbox":    None,
    }

    if best_contour is None:
        return result  # No contour at all → no pear

    # --- Step 4: Validate contour shape ---
    if not _validate_contour(best_contour):
        return result  # Shape doesn't look like a pear

    # --- Step 5: Build pear mask from validated contour ---
    pear_mask = np.zeros(clean_mask.shape, dtype=np.uint8)
    cv2.drawContours(pear_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    area = cv2.contourArea(best_contour)  # Area in pixels²
    bbox = cv2.boundingRect(best_contour)  # (x, y, w, h) bounding rectangle

    result.update({
        "pear_flag":    1,
        "pear_mask":    pear_mask,
        "pear_area":    float(area),
        "pear_contour": best_contour,
        "pear_bbox":    bbox,
    })

    return result


def detect_all_pears(preprocessed_dict: dict) -> dict:
    """
    Runs pear detection on all 6 preprocessed ROIs.

    Parameters
    ----------
    preprocessed_dict : dict
        Output from preprocessing.preprocess_all_rois():
        { roi_name: preprocessed_roi_dict }

    Returns
    -------
    dict
        { roi_name: pear_detection_result_dict }
    """
    results = {}
    for roi_name, prep_roi in preprocessed_dict.items():
        results[roi_name] = pear_detection(prep_roi)
    return results
