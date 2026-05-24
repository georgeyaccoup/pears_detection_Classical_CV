"""
infection_detection.py — Infection Detection Module
=====================================================
RESPONSIBILITY: Detect and quantify surface infections (rot, fungus, bruising,
dark spots) on a pear that has already been confirmed present.

This module operates ONLY inside the pear mask — it never looks at background pixels.

SUB-OPERATIONS (in order):
  1. Extract pear region      → mask out background using pear_mask
  2. Detect dark/brown pixels → threshold for infection-like colors
  3. Morphology cleanup       → remove noise from infection mask
  4. Find infection contours  → extract infection regions
  5. Compute features:
       - infection_area     → total infected pixel count
       - infection_hue      → dominant hue of infected region
       - infection_location → where on the pear (0=top, 1=mid, 2=bottom, 3=none)
  6. Quality decision         → compare against thresholds → quality_flag

OUTPUT per ROI:
  {
    "quality_flag":       int    1 = good pear, 0 = infected,
    "infection_area":     float  total infected area in pixels²,
    "infection_hue":      float  mean hue value of infected pixels,
    "infection_location": int    0=top, 1=middle, 2=bottom, 3=no infection,
    "infection_mask":     np.ndarray  binary mask of infected pixels,
    "infection_ratio":    float  infection_area / pear_area
  }

RULES:
  - NO image capture
  - NO pear contour detection
  - ONLY defect analysis within pear region
"""

import cv2
import numpy as np
import config


def _extract_pear_region(bgr_roi: np.ndarray, pear_mask: np.ndarray) -> np.ndarray:
    """
    Masks out everything outside the pear using bitwise AND.
    Only pear pixels remain; background is set to black (0).

    Parameters
    ----------
    bgr_roi : np.ndarray
        Blurred BGR ROI from preprocessing.
    pear_mask : np.ndarray
        Binary mask from pear_detection (255 = pear, 0 = background).

    Returns
    -------
    np.ndarray
        BGR image with only pear pixels visible, background = black.
    """
    # bitwise_and: pixel = bgr if mask==255, else 0
    pear_only = cv2.bitwise_and(bgr_roi, bgr_roi, mask=pear_mask)
    return pear_only


def _detect_infection_mask(pear_bgr: np.ndarray, pear_mask: np.ndarray) -> np.ndarray:
    """
    Detects infected/discolored pixels within the pear region.

    Two-layer detection:
      Layer 1 — Brown/dark HSV threshold (catches brown rot, light bruising)
      Layer 2 — Very dark pixel threshold (catches black fungal rot)
    Both layers are combined with OR to produce the final infection mask.

    Parameters
    ----------
    pear_bgr : np.ndarray
        BGR pear-only image (background = black).
    pear_mask : np.ndarray
        Binary mask of pear region.

    Returns
    -------
    np.ndarray
        Binary infection mask (255 = infected pixel, 0 = healthy or background).
    """

    # Convert pear region to HSV for color-based thresholding
    pear_hsv = cv2.cvtColor(pear_bgr, cv2.COLOR_BGR2HSV)

    # --- Layer 1: Brown/dark HSV range ---
    # INFECTION_HSV_LOWER/UPPER from config.py:
    #   Hue 0–30 covers red-brown range typical of surface rot
    #   V max (100) excludes healthy bright yellow skin
    infection_mask_hsv = cv2.inRange(
        pear_hsv,
        config.INFECTION_HSV_LOWER,
        config.INFECTION_HSV_UPPER
    )

    # --- Layer 2: Very dark pixels (black rot, severe fungal infection) ---
    # Extract the V (Value/brightness) channel
    v_channel = pear_hsv[:, :, 2]
    # Pixels below INFECTION_DARK_V_MAX are considered "nearly black"
    # np.uint8(255) where V < threshold, 0 elsewhere
    dark_mask = np.where(v_channel < config.INFECTION_DARK_V_MAX, np.uint8(255), np.uint8(0))

    # --- Combine both layers ---
    # A pixel is infected if EITHER layer flags it
    combined_mask = cv2.bitwise_or(infection_mask_hsv, dark_mask)

    # --- Restrict to pear region only ---
    # Remove any false detections that sneak in from background (should be black
    # already, but this is a safety step)
    infection_mask = cv2.bitwise_and(combined_mask, pear_mask)

    return infection_mask


def _clean_infection_mask(mask: np.ndarray) -> np.ndarray:
    """
    Applies morphological cleanup to the infection mask.

    Step 1: Opening → removes tiny noise specks (pin-prick detections)
    Step 2: Closing → merges nearby infected patches into cohesive regions

    Parameters
    ----------
    mask : np.ndarray
        Raw binary infection mask.

    Returns
    -------
    np.ndarray
        Cleaned infection mask.
    """
    # Open: removes isolated single-pixel noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, config.INFECTION_OPEN_KERNEL)

    # Close: fills small gaps between nearby infected areas
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, config.INFECTION_CLOSE_KERNEL)

    return closed


def _compute_infection_features(
    infection_mask: np.ndarray,
    pear_mask: np.ndarray,
    pear_hsv: np.ndarray,
    pear_contour,
    pear_area: float
) -> dict:
    """
    Computes quantitative features from the infection mask.

    Features computed:
      - infection_area     : total infected pixel count (pixels²)
      - infection_ratio    : infected / total pear area (0.0–1.0)
      - infection_hue      : mean hue value across infected pixels
      - infection_location : position code based on centroid vs pear center
          0 = top third of pear
          1 = middle third
          2 = bottom third
          3 = no infection (infection_area == 0)

    Parameters
    ----------
    infection_mask : np.ndarray
        Cleaned binary infection mask.
    pear_mask : np.ndarray
        Binary mask of pear region.
    pear_hsv : np.ndarray
        HSV image of full ROI.
    pear_contour : np.ndarray
        Contour of the pear (from pear_detection).
    pear_area : float
        Area of pear in pixels².

    Returns
    -------
    dict
        { "infection_area", "infection_ratio", "infection_hue", "infection_location" }
    """

    # --- Infection Area ---
    # Count non-zero pixels in the infection mask
    infection_area = float(np.sum(infection_mask > 0))

    # --- Infection Ratio ---
    # Fraction of pear covered by infection
    infection_ratio = (infection_area / pear_area) if pear_area > 0 else 0.0

    # --- Infection Hue ---
    # Mean hue value over all infected pixels (useful for identifying type of defect)
    if infection_area > 0:
        h_channel = pear_hsv[:, :, 0]  # Hue channel from HSV
        # Extract hue values only where infection mask is active
        infected_hues = h_channel[infection_mask > 0]
        infection_hue = float(np.mean(infected_hues))
    else:
        infection_hue = 0.0  # No infection → hue is irrelevant

    # --- Infection Location ---
    infection_location = 3  # Default: no infection

    if infection_area > 0 and pear_contour is not None:
        # Find centroid of infection region using image moments
        moments = cv2.moments(infection_mask)
        if moments["m00"] > 0:
            # Centroid coordinates of infection cluster
            cx = int(moments["m10"] / moments["m00"])  # infection center X
            cy = int(moments["m01"] / moments["m00"])  # infection center Y

            # Find bounding box of pear to determine relative position
            px, py, pw, ph = cv2.boundingRect(pear_contour)
            pear_top = py
            pear_bottom = py + ph

            # Divide pear bounding box into 3 vertical zones
            third_height = ph / 3.0

            if cy < pear_top + third_height:
                infection_location = 0  # Infection in top third of pear
            elif cy < pear_top + 2 * third_height:
                infection_location = 1  # Infection in middle third
            else:
                infection_location = 2  # Infection in bottom third

    return {
        "infection_area":     infection_area,
        "infection_ratio":    infection_ratio,
        "infection_hue":      infection_hue,
        "infection_location": infection_location,
    }


def _decide_quality(infection_area: float, infection_ratio: float) -> int:
    """
    Determines pear quality based on infection measurements.

    Decision logic:
      - If infection_area exceeds INFECTION_AREA_THRESHOLD → REJECT (0)
      - AND if infection_ratio exceeds INFECTION_RATIO_THRESHOLD → REJECT (0)
      - Both conditions must be true to reject (reduces false positives)

    Parameters
    ----------
    infection_area : float
        Total infected area in pixels².
    infection_ratio : float
        Fraction of pear area that is infected.

    Returns
    -------
    int
        1 = good quality (ACCEPT), 0 = infected (REJECT)
    """
    # Both area AND ratio must exceed thresholds to be flagged as infected
    # This double-gate prevents false rejections from tiny noise detections
    if (infection_area > config.INFECTION_AREA_THRESHOLD and
            infection_ratio > config.INFECTION_RATIO_THRESHOLD):
        return 0  # INFECTED → reject
    return 1  # HEALTHY → accept


def infection_detection(pear_data: dict, preprocessed_roi: dict) -> dict:
    """
    Main infection detection function for a single ROI.

    Runs the full infection pipeline:
      extract region → detect abnormal pixels → clean → compute features → decide quality

    Parameters
    ----------
    pear_data : dict
        Output from pear_detection.pear_detection():
        { "pear_flag", "pear_mask", "pear_area", "pear_contour", "pear_bbox" }
    preprocessed_roi : dict
        Output from preprocessing.preprocess_roi():
        { "original", "bgr", "hsv" }

    Returns
    -------
    dict
        {
            "quality_flag":       int    1=good, 0=infected
            "infection_area":     float  total infected area (pixels²)
            "infection_hue":      float  mean hue of infection
            "infection_location": int    0=top,1=mid,2=bot,3=none
            "infection_mask":     np.ndarray
            "infection_ratio":    float
        }
    """

    # Default output for when no pear is present
    default_result = {
        "quality_flag":       1,    # Assume good if no pear (nothing to reject)
        "infection_area":     0.0,
        "infection_hue":      0.0,
        "infection_location": 3,    # 3 = no infection
        "infection_mask":     np.zeros(
            preprocessed_roi["bgr"].shape[:2], dtype=np.uint8
        ),
        "infection_ratio":    0.0,
    }

    # If no pear was detected, skip infection analysis entirely
    if pear_data["pear_flag"] == 0:
        return default_result

    pear_mask    = pear_data["pear_mask"]
    pear_area    = pear_data["pear_area"]
    pear_contour = pear_data["pear_contour"]
    bgr_roi      = preprocessed_roi["bgr"]
    hsv_roi      = preprocessed_roi["hsv"]

    # --- Step 1: Isolate pear pixels ---
    pear_bgr = _extract_pear_region(bgr_roi, pear_mask)

    # --- Step 2: Detect infected pixels ---
    raw_infection_mask = _detect_infection_mask(pear_bgr, pear_mask)

    # --- Step 3: Morphology cleanup ---
    clean_infection_mask = _clean_infection_mask(raw_infection_mask)

    # --- Step 4: Compute infection features ---
    features = _compute_infection_features(
        clean_infection_mask,
        pear_mask,
        hsv_roi,
        pear_contour,
        pear_area
    )

    # --- Step 5: Quality decision ---
    quality_flag = _decide_quality(features["infection_area"], features["infection_ratio"])

    return {
        "quality_flag":       quality_flag,
        "infection_area":     features["infection_area"],
        "infection_hue":      features["infection_hue"],
        "infection_location": features["infection_location"],
        "infection_mask":     clean_infection_mask,
        "infection_ratio":    features["infection_ratio"],
    }


def detect_all_infections(pear_results: dict, preprocessed_dict: dict) -> dict:
    """
    Runs infection detection on all 6 ROIs.

    Parameters
    ----------
    pear_results : dict
        Output from pear_detection.detect_all_pears().
    preprocessed_dict : dict
        Output from preprocessing.preprocess_all_rois().

    Returns
    -------
    dict
        { roi_name: infection_result_dict }
    """
    results = {}
    for roi_name in config.ROI_ORDER:
        results[roi_name] = infection_detection(
            pear_results[roi_name],
            preprocessed_dict[roi_name]
        )
    return results
