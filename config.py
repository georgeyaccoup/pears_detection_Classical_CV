"""
config.py — Global Configuration File
======================================
All tunable parameters are defined here.
Modify values as needed to adapt to your camera, lighting, and pear variety.

HOW TO TUNE:
  - Increasing a threshold → FEWER detections (more strict)
  - Decreasing a threshold → MORE detections (more lenient)
  - Always test changes against known-good and known-bad sample images.
"""

import numpy as np

# =============================================================================
# CAMERA SETTINGS
# =============================================================================

CAMERA_INDEX = 0
# Which camera to open. 0 = first USB/built-in camera.
# Change to 1, 2, etc. if you have multiple cameras.

FRAME_WIDTH = 1280
# Capture frame width in pixels.
# Increasing → better resolution, but slower processing.
# Decreasing → faster, but less detail for infection detection.

FRAME_HEIGHT = 720
# Capture frame height in pixels.

LOOP_SLEEP_SEC = 1.0
# Delay between frames in seconds (real-time control rate).
# Decrease for faster sorting (e.g., 0.5 = 2 Hz).
# Increase to reduce CPU load.

# =============================================================================
# ROI (REGION OF INTEREST) LAYOUT
# =============================================================================
# The frame is divided into a 2-column × 3-row grid.
# Layout:   A3 | B3
#           A2 | B2
#           A1 | B1
#
# Each ROI is defined as (x_start, y_start, x_end, y_end) in pixels.
# IMPORTANT: These must cover the full tray area visible in your camera feed.
# Adjust after physically aligning camera over the conveyor/tray.

ROI_DEFINITIONS = {
    "A3": (0,    0,   640, 240),   # Top-left ROI (column A, row 3)
    "B3": (640,  0,  1280, 240),   # Top-right ROI (column B, row 3)
    "A2": (0,   240,  640, 480),   # Mid-left ROI (column A, row 2)
    "B2": (640, 240, 1280, 480),   # Mid-right ROI (column B, row 2)
    "A1": (0,   480,  640, 720),   # Bottom-left ROI (column A, row 1)
    "B1": (640, 480, 1280, 720),   # Bottom-right ROI (column B, row 1)
}

ROI_ORDER = ["A3", "B3", "A2", "B2", "A1", "B1"]
# Output order of ROIs in the final frame data array.

# =============================================================================
# PREPROCESSING SETTINGS
# =============================================================================

BLUR_KERNEL_SIZE = 5
# Size of median blur kernel (must be odd: 3, 5, 7, 9...).
# Increasing → smoother image, removes more noise but blurs fine detail.
# Decreasing → sharper edges, but more noise sensitivity.

# =============================================================================
# PEAR DETECTION — HSV COLOR THRESHOLDS
# =============================================================================
# Pears in this system are green-to-yellow varieties.
# HSV range captures the dominant pear skin color.
#
# HSV in OpenCV: H=[0..179], S=[0..255], V=[0..255]
# H for green/yellow: roughly 20–90
# S (saturation): must be somewhat colored (not white/grey background)
# V (brightness): must be bright enough to distinguish from shadow

PEAR_HSV_LOWER = np.array([15, 30, 60])
# Lower HSV bound for pear skin color.
# H↑ → shifts detection toward more yellow (away from green)
# S↑ → requires more saturated (vivid) color; reduces background noise
# V↑ → ignores darker regions (shadows, bruises)

PEAR_HSV_UPPER = np.array([95, 255, 255])
# Upper HSV bound for pear skin color.
# H↓ → reduces upper range; excludes yellower pears if too low
# S↓ → allows less saturated (paler) colors
# V↓ → excludes very bright/overexposed highlights

# =============================================================================
# PEAR DETECTION — MORPHOLOGY
# =============================================================================

MORPH_OPEN_KERNEL = np.ones((7, 7), np.uint8)
# Kernel for morphological opening (removes small noise blobs).
# Increasing size → removes larger noise but may erode real pear edges.
# Decreasing size → preserves more detail, but keeps small noise.

MORPH_CLOSE_KERNEL = np.ones((15, 15), np.uint8)
# Kernel for morphological closing (fills holes in pear mask).
# Increasing size → fills larger gaps (e.g., dark spots from bruises).
# Decreasing size → preserves shape accuracy but may leave holes.

# =============================================================================
# PEAR DETECTION — VALIDATION THRESHOLDS
# =============================================================================

PEAR_MIN_AREA = 3000
# Minimum contour area (pixels²) to be considered a pear.
# Increase → rejects small false detections (reflections, debris).
# Decrease → detects smaller pears or partially visible pears.

PEAR_MAX_AREA = 300000
# Maximum contour area (pixels²) — filters oversized blobs.
# Increase if pears appear very large in frame.
# Decrease to reject merged/double-pear blobs.

PEAR_MIN_ASPECT_RATIO = 0.4
# Minimum width/height ratio. Pears are roughly 0.6–0.9.
# Decrease → accepts more elongated shapes.
# Increase → rejects anything too thin (wires, shadows).

PEAR_MAX_ASPECT_RATIO = 1.6
# Maximum width/height ratio.
# Increase → accepts wider shapes.
# Decrease → rejects wide, flat non-pear objects.

PEAR_MIN_SOLIDITY = 0.6
# Solidity = contour_area / convex_hull_area.
# Measures how "filled" the shape is (1.0 = perfectly convex).
# Increase → requires rounder shapes, rejects C-shapes or fragments.
# Decrease → accepts more irregular shapes.

# =============================================================================
# INFECTION DETECTION — COLOR THRESHOLDS
# =============================================================================
# Infections appear as brown, dark, or discolored patches on pear skin.
# We detect them using both HSV (for brown/dark color) and LAB (for darkness).

# --- Dark/Brown HSV range ---
INFECTION_HSV_LOWER = np.array([0, 20, 0])
# Lower HSV for infection (brown/dark).
# H=0 starts at red-brown; S=20 requires minimal color; V=0 allows very dark.

INFECTION_HSV_UPPER = np.array([30, 255, 100])
# Upper HSV for infection.
# H=30: captures brown/orange tones. Increase H if infections appear more orange.
# V=100: cap brightness to exclude healthy yellow skin.
#   Increase V↑ → captures lighter brown infections, but may include healthy skin.
#   Decrease V↓ → only detects very dark/black infections.

# --- Very dark pixel range (catches black rot/fungal infections) ---
INFECTION_DARK_V_MAX = 60
# Maximum V (brightness) value to flag as "very dark" (near-black infection).
# Increase → captures more dark regions (may include shadows).
# Decrease → only flags truly black spots.

# =============================================================================
# INFECTION DETECTION — MORPHOLOGY
# =============================================================================

INFECTION_OPEN_KERNEL = np.ones((3, 3), np.uint8)
# Opens noise from infection mask (removes tiny specks).
# Increase → filters out more small noise.
# Decrease → preserves very small infection spots.

INFECTION_CLOSE_KERNEL = np.ones((7, 7), np.uint8)
# Closes gaps in infection regions.
# Increase → merges nearby infected patches into one region.
# Decrease → keeps infections as separate regions.

# =============================================================================
# INFECTION DETECTION — QUALITY DECISION THRESHOLDS
# =============================================================================

INFECTION_AREA_THRESHOLD = 500
# Minimum infection area (pixels²) to flag pear as INFECTED.
# Increase → only flags pears with larger infections (fewer rejections).
# Decrease → flags pears with smaller, early-stage infections.

INFECTION_RATIO_THRESHOLD = 0.05
# Maximum infection_area / pear_area ratio allowed (5% of pear = bad).
# Increase → allows more infected surface before rejecting (looser standard).
# Decrease → rejects pears with smaller infection patches (stricter standard).

# =============================================================================
# SCALE FACTOR
# =============================================================================

SCALE_FACTOR = 0.01
# Conversion factor: pixels² → cm².
# Set this by calibrating: place a known-size object in frame and measure.
# Formula: scale = (real_area_cm2) / (pixel_area)
# Increase → reported areas get larger (if camera is zoomed in).
# Decrease → reported areas get smaller (if camera is zoomed out).

# =============================================================================
# MOTOR / ACTUATION SETTINGS
# =============================================================================

MOTOR_ACCEPT_ANGLE = 0
# Servo angle for ACCEPT decision (good pear).
# Set to the physical angle that routes pear to "accept" lane.

MOTOR_REJECT_ANGLE = 180
# Servo angle for REJECT decision (infected/bad pear).
# Set to the physical angle that routes pear to "reject" bin.

MOTOR_CHANNELS = {
    "A3": 0,   # PCA9685 channel for ROI A3 actuator
    "B3": 1,   # PCA9685 channel for ROI B3 actuator
    "A2": 2,   # PCA9685 channel for ROI A2 actuator
    "B2": 3,   # PCA9685 channel for ROI B2 actuator
    "A1": 4,   # PCA9685 channel for ROI A1 actuator
    "B1": 5,   # PCA9685 channel for ROI B1 actuator
}
# Maps each ROI to its corresponding motor/servo channel.
# Adjust channel numbers to match your hardware wiring.

# =============================================================================
# PUBLISHER SETTINGS
# =============================================================================

PUBLISH_MODE = "print"
# Where to send the frame output. Options:
#   "print"  → stdout (for debugging/development)
#   "file"   → write to OUTPUT_FILE_PATH
#   "ros2"   → publish to ROS2 topic (requires ROS2 environment)
#   "serial" → send over serial port (requires pyserial)

OUTPUT_FILE_PATH = "output_log.txt"
# Path to log file (used when PUBLISH_MODE = "file").

ROS2_TOPIC = "/pear_sorter/frame_data"
# ROS2 topic name (used when PUBLISH_MODE = "ros2").

SERIAL_PORT = "/dev/ttyUSB0"
# Serial port for motor controller (used when PUBLISH_MODE = "serial").

SERIAL_BAUD = 115200
# Baud rate for serial communication.
