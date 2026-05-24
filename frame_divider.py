"""
frame_divider.py — Frame Division Module
==========================================
RESPONSIBILITY: Split the captured frame into 6 fixed ROI (Region of Interest)
sub-images based on predefined pixel coordinates from config.py.

RULES:
  - NO image processing here (no blur, no color conversion)
  - ONLY slicing the frame array
  - Each ROI is an independent numpy slice of the original frame

OUTPUT:
  Dict mapping ROI name → numpy image slice
  Example: { "A3": np.array(...), "B3": np.array(...), ... }
"""

import cv2
import numpy as np
import config


def frame_divider(frame: np.ndarray) -> dict:
    """
    Splits a full camera frame into 6 fixed ROI sub-images.

    The physical layout matches the conveyor tray positions:
        A3 | B3   ← Top row (first to arrive)
        A2 | B2   ← Middle row
        A1 | B1   ← Bottom row (last to arrive)

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame captured from the camera.
        Expected shape: (FRAME_HEIGHT, FRAME_WIDTH, 3)

    Returns
    -------
    dict
        { roi_name: roi_image_array } for each of the 6 ROIs.
        roi_image_array is a numpy view (not a copy) of the original frame.
    """

    roi_dict = {}  # Will hold { "A3": image, "B3": image, ... }

    for roi_name in config.ROI_ORDER:
        # Retrieve the pixel coordinate tuple for this ROI from config
        x1, y1, x2, y2 = config.ROI_DEFINITIONS[roi_name]

        # Slice the ROI from the frame using NumPy array indexing
        # Frame indexing: frame[y_start:y_end, x_start:x_end]
        roi_image = frame[y1:y2, x1:x2]

        # Validate that the slice is non-empty (sanity check)
        if roi_image.size == 0:
            print(f"[WARN] ROI '{roi_name}' produced an empty slice — "
                  f"check ROI_DEFINITIONS in config.py")
            roi_image = np.zeros((10, 10, 3), dtype=np.uint8)  # Fallback placeholder

        roi_dict[roi_name] = roi_image

    return roi_dict


def visualize_rois(frame: np.ndarray) -> np.ndarray:
    """
    Debug helper: draws ROI boundaries on the frame as colored rectangles.
    Useful during camera alignment and setup.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.

    Returns
    -------
    np.ndarray
        A copy of the frame with ROI rectangles and labels drawn on it.
    """
    debug_frame = frame.copy()

    # Color palette for each ROI (BGR format)
    colors = {
        "A3": (0, 255, 0),    # Green
        "B3": (255, 0, 0),    # Blue
        "A2": (0, 255, 255),  # Yellow
        "B2": (255, 0, 255),  # Magenta
        "A1": (0, 128, 255),  # Orange
        "B1": (255, 128, 0),  # Cyan-ish
    }

    for roi_name in config.ROI_ORDER:
        x1, y1, x2, y2 = config.ROI_DEFINITIONS[roi_name]
        color = colors.get(roi_name, (255, 255, 255))

        # Draw rectangle border around ROI
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

        # Label the ROI at top-left corner of rectangle
        cv2.putText(
            debug_frame,
            roi_name,
            (x1 + 5, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return debug_frame
