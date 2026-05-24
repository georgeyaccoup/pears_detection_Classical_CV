"""
utils.py — Shared Utility Functions
=====================================
General-purpose helpers used across multiple modules.
Includes debug visualization, camera initialization, and performance timing.

RULES:
  - No detection logic
  - No motor commands
  - Only helper / support functions
"""

import time
import cv2
import numpy as np
import config


# =============================================================================
# CAMERA UTILITIES
# =============================================================================

def init_camera() -> cv2.VideoCapture:
    """
    Opens and configures the camera capture device.

    Returns
    -------
    cv2.VideoCapture
        Configured camera object ready for frame capture.

    Raises
    ------
    RuntimeError
        If camera cannot be opened (wrong index, no USB device, etc.)
    """
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"[ERROR] Cannot open camera at index {config.CAMERA_INDEX}. "
            f"Check CAMERA_INDEX in config.py and verify USB connection."
        )

    # Set resolution to match config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # Read back actual resolution (camera may not support exact request)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[CAMERA] Opened camera {config.CAMERA_INDEX}: "
          f"{actual_w}×{actual_h} px")

    return cap


def get_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Captures a single frame from the camera.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Open camera object from init_camera().

    Returns
    -------
    np.ndarray
        BGR frame array, shape (H, W, 3).

    Raises
    ------
    RuntimeError
        If frame capture fails (camera disconnected, stream ended).
    """
    ret, frame = cap.read()

    if not ret or frame is None:
        raise RuntimeError(
            "[ERROR] Failed to read frame from camera. "
            "Camera may have been disconnected."
        )

    return frame


def release_camera(cap: cv2.VideoCapture):
    """
    Releases camera resources cleanly.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Open camera object to release.
    """
    cap.release()
    print("[CAMERA] Camera released.")


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def draw_debug_overlay(
    frame: np.ndarray,
    roi_name: str,
    x1: int, y1: int,
    pear_data: dict,
    infection_data: dict,
    decision: str
) -> np.ndarray:
    """
    Draws a complete debug overlay on the frame for a single ROI.

    Draws:
      - ROI bounding box (green = pear found, grey = empty)
      - Pear contour outline (blue)
      - Decision label: ACCEPT / REJECT / NO_ACTION
      - Infection ratio text

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame to draw on.
    roi_name : str
        ROI label (e.g., "A3").
    x1, y1 : int
        Top-left corner of ROI in full-frame coordinates.
    pear_data : dict
        Output from pear_detection module.
    infection_data : dict
        Output from infection_detection module.
    decision : str
        Motor decision string ("ACCEPT", "REJECT", "NO_ACTION").

    Returns
    -------
    np.ndarray
        Frame with overlays drawn (modifies in place, also returns).
    """
    x2_local, y2_local = config.ROI_DEFINITIONS[roi_name][2], config.ROI_DEFINITIONS[roi_name][3]

    # Color coding for ROI border
    if pear_data["pear_flag"] == 0:
        box_color = (100, 100, 100)  # Grey = no pear
    elif decision == "ACCEPT":
        box_color = (0, 200, 0)      # Green = accept
    else:
        box_color = (0, 0, 200)      # Red = reject

    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2_local, y2_local), box_color, 2)

    # Draw ROI name
    cv2.putText(frame, roi_name, (x1 + 4, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Draw pear contour (if pear found)
    if pear_data["pear_flag"] and pear_data["pear_contour"] is not None:
        # Contour coordinates are relative to ROI — shift by (x1, y1)
        shifted = pear_data["pear_contour"] + np.array([x1, y1])
        cv2.drawContours(frame, [shifted], -1, (255, 128, 0), 2)

    # Draw decision label
    label_color = (0, 200, 0) if decision == "ACCEPT" else (0, 0, 200)
    cv2.putText(frame, decision, (x1 + 4, y2_local - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)

    # Draw infection ratio
    ratio = infection_data.get("infection_ratio", 0.0)
    cv2.putText(
        frame,
        f"Inf:{ratio * 100:.1f}%",
        (x1 + 4, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1
    )

    return frame


# =============================================================================
# PERFORMANCE TIMER
# =============================================================================

class LoopTimer:
    """
    Simple context manager for measuring loop iteration time.

    Usage:
        timer = LoopTimer()
        with timer:
            ... processing ...
        print(f"Frame took {timer.elapsed_ms:.1f} ms")
    """

    def __init__(self):
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# =============================================================================
# FRAME SIMULATION (for testing without camera)
# =============================================================================

def load_test_frame(path: str) -> np.ndarray:
    """
    Loads a static test image instead of a live camera frame.
    Useful for offline testing and HSV threshold tuning.

    Parameters
    ----------
    path : str
        File path to a test image (jpg, png, etc.).

    Returns
    -------
    np.ndarray
        BGR frame resized to configured FRAME_WIDTH × FRAME_HEIGHT.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Test image not found: {path}")

    # Resize to match expected frame dimensions
    resized = cv2.resize(img, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    return resized
