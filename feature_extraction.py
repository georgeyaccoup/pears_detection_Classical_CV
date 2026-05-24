"""
feature_extraction.py — Data Formatting / Feature Packing Module
==================================================================
RESPONSIBILITY: Merge the outputs from pear_detection and infection_detection
into the standard 6-value state vector per ROI, and assemble all ROIs into
the final frame data array.

OUTPUT FORMAT per ROI (6-element array):
  Index | Name               | Values / Meaning
  ------|--------------------|------------------------------------------
    0   | pear_flag          | 1 = pear present, 0 = no pear
    1   | quality_flag       | 1 = good/accept, 0 = infected/reject
    2   | infection_hue      | float, mean HSV hue of infected region
    3   | infection_area     | float, infected area in pixels²
    4   | pear_area          | float, total pear area in pixels²
    5   | infection_location | 0=top, 1=middle, 2=bottom, 3=none

FULL FRAME OUTPUT:
  Frame_Data = [ROI_A3, ROI_B3, ROI_A2, ROI_B2, ROI_A1, ROI_B1]
  Each element is a 6-value list as above.

RULES:
  - NO image processing
  - NO detection logic
  - ONLY formatting and packing
"""

import config


def format_roi_output(pear_data: dict, infection_data: dict) -> list:
    """
    Packs pear and infection results into a single 6-element feature vector.

    Parameters
    ----------
    pear_data : dict
        Output from pear_detection.pear_detection():
        { "pear_flag", "pear_mask", "pear_area", "pear_contour", "pear_bbox" }
    infection_data : dict
        Output from infection_detection.infection_detection():
        { "quality_flag", "infection_area", "infection_hue",
          "infection_location", "infection_mask", "infection_ratio" }

    Returns
    -------
    list
        [pear_flag, quality_flag, infection_hue, infection_area,
         pear_area, infection_location]
        All values are rounded to 2 decimal places for clean output.
    """

    pear_flag          = int(pear_data["pear_flag"])
    quality_flag       = int(infection_data["quality_flag"])
    infection_hue      = round(float(infection_data["infection_hue"]), 2)
    infection_area     = round(float(infection_data["infection_area"]) * config.SCALE_FACTOR, 2)
    # infection_area is converted from pixels² to cm² using SCALE_FACTOR
    pear_area          = round(float(pear_data["pear_area"]) * config.SCALE_FACTOR, 2)
    # pear_area is also converted to cm²
    infection_location = int(infection_data["infection_location"])

    return [
        pear_flag,           # Index 0: Is pear present?
        quality_flag,        # Index 1: Is pear healthy?
        infection_hue,       # Index 2: What color is the infection?
        infection_area,      # Index 3: How large is the infection? (cm²)
        pear_area,           # Index 4: How big is the pear? (cm²)
        infection_location,  # Index 5: Where is the infection on pear?
    ]


def format_frame_output(pear_results: dict, infection_results: dict) -> list:
    """
    Assembles the full frame data array from all 6 ROIs.

    Processes ROIs in the order defined by config.ROI_ORDER:
        [A3, B3, A2, B2, A1, B1]

    Parameters
    ----------
    pear_results : dict
        { roi_name: pear_detection_result }
    infection_results : dict
        { roi_name: infection_detection_result }

    Returns
    -------
    list of lists
        [
          [pear_flag, quality_flag, inf_hue, inf_area, pear_area, inf_loc],  # A3
          [pear_flag, quality_flag, inf_hue, inf_area, pear_area, inf_loc],  # B3
          ...
        ]
        Length is always 6 (one entry per ROI).
    """

    frame_data = []  # Will hold 6 feature vectors (one per ROI)

    for roi_name in config.ROI_ORDER:
        roi_vector = format_roi_output(
            pear_results[roi_name],
            infection_results[roi_name]
        )
        frame_data.append(roi_vector)

    return frame_data


def get_roi_summary(roi_name: str, roi_vector: list) -> str:
    """
    Creates a human-readable summary string for a single ROI.
    Useful for logging, debugging, and dashboard display.

    Parameters
    ----------
    roi_name : str
        ROI identifier (e.g., "A3").
    roi_vector : list
        6-element feature vector for this ROI.

    Returns
    -------
    str
        Formatted summary line.
    """
    pear_flag, quality_flag, inf_hue, inf_area, pear_area, inf_loc = roi_vector

    # Readable labels
    pear_status    = "PEAR"    if pear_flag    else "EMPTY"
    quality_status = "GOOD"    if quality_flag else "INFECTED"
    location_map   = {0: "TOP", 1: "MIDDLE", 2: "BOTTOM", 3: "NONE"}
    loc_label      = location_map.get(inf_loc, "?")

    return (
        f"[{roi_name}] {pear_status} | {quality_status} | "
        f"PearArea={pear_area:.1f}cm² | "
        f"InfArea={inf_area:.2f}cm² | "
        f"InfHue={inf_hue:.1f} | "
        f"InfLoc={loc_label}"
    )
