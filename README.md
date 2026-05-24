# pears_detection_Classical_CV

<div align="center">

<img src="https://img.shields.io/badge/Platform-Jetson%20Nano-green?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-4.x-blue?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/ROS2-Compatible-purple?style=for-the-badge&logo=ros&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>

#  Pear Sorting Vision System

**Real-time, industrial-grade computer vision pipeline for automated pear quality inspection and sorting.**

Detects pear presence · Identifies surface infections · Classifies quality · Drives servo actuators to physically sort pears — all at **1 Hz+ in real time** on edge hardware.

---

[Features](#-features) · [Architecture](#-architecture) · [Pipeline](#-pipeline-stages) · [Installation](#-installation) · [Configuration](#-configuration-guide) · [Running](#-running-the-system) · [Output Format](#-output-format) · [Calibration](#-calibration-workflow) · [Project Structure](#-project-structure)

</div>

---

##  Features

| Feature | Detail |
|---|---|
|  **6-ROI simultaneous inspection** | Full tray frame split into 6 independent positions (A3/B3/A2/B2/A1/B1) |
|  **Two-layer infection detection** | Brown/rot HSV range + very-dark pixel mask combined for robust defect coverage |
|  **Shape validation** | Area · aspect ratio · solidity check to reject shadows and background noise |
|  **Infection localization** | Reports infection position as top / middle / bottom third of the pear |
|  **Single-file configuration** | Every tunable threshold lives in `config.py` with documented tuning guidance |
|  **Hardware + simulation modes** | PCA9685 servo driver with automatic simulation fallback when hardware is absent |
|  **4 output modes** | Terminal · log file · ROS2 topic · serial port |
|  **Debug overlay** | Live `--debug` window draws ROI boxes, contours, and decisions on the frame |
|  **Static image testing** | `--test --image` flag runs the full pipeline on a photo without a camera |
|  **Real-time on edge hardware** | Target: Jetson Nano / Raspberry Pi 4 |

---

## 🏗 Architecture

The system follows a strict **single-responsibility, modular pipeline**. Every file handles exactly one stage. No module performs any logic that belongs to another stage.

```
┌─────────────────────────────────────────────────────────────┐
│                        config.py                            │
│  All tunable parameters — imported by every module          │
└─────────────────────────────────────────────────────────────┘
                            │ (global read)
                            ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│   utils.py   │    │   main.py     │    │   publisher.py   │
│  Camera I/O  │◄───│  Main loop    │───►│  print/file/     │
│  Debug draw  │    │  Orchestrator │    │  ROS2 / serial   │
│  Loop timer  │    └───────┬───────┘    └──────────────────┘
└──────────────┘            │
                            ▼
            ┌───────────────────────────┐
            │     frame_divider.py      │
            │  Splits frame → 6 ROIs    │
            └───────────────────────────┘
                            │  roi_dict {A3..B1: ndarray}
                            ▼
            ┌───────────────────────────┐
            │     preprocessing.py      │
            │  Median blur + BGR→HSV    │
            └───────────────────────────┘
                            │  preprocessed_dict {bgr, hsv, original}
                            ▼
            ┌───────────────────────────┐
            │     pear_detection.py     │
            │  HSV threshold            │
            │  Morphology open/close    │
            │  Contour extraction       │
            │  Shape validation         │
            └───────────────────────────┘
                            │  pear_results {flag, mask, area, contour}
                            ▼
            ┌───────────────────────────┐
            │  infection_detection.py   │
            │  Extract pear region      │
            │  Brown + dark pixel masks │
            │  Area · hue · location    │
            │  Quality decision         │
            └───────────────────────────┘
                            │  infection_results {quality_flag, area, hue, location}
                            ▼
            ┌───────────────────────────┐
            │   feature_extraction.py   │
            │  Pack → 6-value vectors   │
            │  px² → cm² conversion     │
            └───────────────────────────┘
                            │  frame_data [6 × [6 values]]
                            ▼
            ┌───────────────────────────┐
            │       motor_task.py       │
            │  decision_function()      │
            │  action_mapper()          │
            │  motor_command() → servo  │
            └───────────────────────────┘
```

---

##  Pipeline Stages

### Stage 1 — Frame Capture &nbsp;`utils.py`

Opens the camera, sets resolution, and reads one BGR frame per cycle.

```python
cap   = init_camera()          # opens cv2.VideoCapture(CAMERA_INDEX)
frame = get_frame(cap)         # shape: (FRAME_HEIGHT × FRAME_WIDTH × 3)
```

> **Test mode:** `load_test_frame(path)` loads a static JPEG/PNG instead of the live camera — perfect for threshold tuning.

---

### Stage 2 — Frame Division &nbsp;`frame_divider.py`

Slices the full frame into **6 fixed ROIs** matching the physical tray layout. **No processing at all — only NumPy array views.**

```
┌───────────┬───────────┐
│    A3     │    B3     │  ← Top row (first on conveyor)
├───────────┼───────────┤
│    A2     │    B2     │  ← Middle row
├───────────┼───────────┤
│    A1     │    B1     │  ← Bottom row
└───────────┴───────────┘
```

```python
roi_dict = frame_divider(frame)
# { "A3": frame[0:240, 0:640], "B3": frame[0:240, 640:1280], … }
```

---

### Stage 3 — Preprocessing &nbsp;`preprocessing.py`

Two operations per ROI, in order:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | `cv2.medianBlur(roi, BLUR_KERNEL_SIZE)` | Remove sensor noise without blurring edges |
| 2 | `cv2.cvtColor(blurred, COLOR_BGR2HSV)` | Decouple color from brightness for robust thresholding |

```python
preprocessed = preprocess_all_rois(roi_dict)
# each ROI → { "original": ndarray, "bgr": ndarray, "hsv": ndarray }
```

---

### Stage 4 — Pear Detection &nbsp;`pear_detection.py`

Five-step sub-pipeline per ROI:

```
HSV threshold  →  Morph OPEN  →  Morph CLOSE  →  findContours  →  Validate shape
   (color)        (denoise)      (fill holes)    (largest blob)   (area/ratio/solidity)
```

**Validation gates** (all three must pass):

```
area           ∈ [PEAR_MIN_AREA,    PEAR_MAX_AREA]
aspect_ratio   ∈ [PEAR_MIN_ASPECT,  PEAR_MAX_ASPECT]
solidity       ≥  PEAR_MIN_SOLIDITY
```

**Output per ROI:**
```python
{
  "pear_flag":    1,             # 1 = pear found, 0 = empty
  "pear_mask":    ndarray,       # binary mask (255 = pear pixels)
  "pear_area":    14500.0,       # area in pixels²
  "pear_contour": ndarray,       # contour point array
  "pear_bbox":    (x, y, w, h)   # bounding rectangle
}
```

---

### Stage 5 — Infection Detection &nbsp;`infection_detection.py`

Runs **only inside the pear mask** — never touches background pixels.

```
Extract pear region (bitwise_and)
         │
         ▼
  ┌──────────────┐    ┌──────────────────┐
  │  Layer 1:    │    │  Layer 2:         │
  │  Brown/rot   │ OR │  Dark pixels      │
  │  HSV range   │    │  V < DARK_V_MAX   │
  │  hue 0–30    │    │  (black rot)      │
  └──────────────┘    └──────────────────┘
         │ combined mask
         ▼
  Morphology OPEN (remove specks)
  Morphology CLOSE (merge patches)
         │
         ▼
  Compute features:
    infection_area     = np.sum(mask > 0)
    infection_ratio    = area / pear_area
    infection_hue      = mean(H_channel[mask > 0])
    infection_location = centroid zone: 0=top 1=mid 2=bot 3=none
         │
         ▼
  Quality decision (double-gate):
    if area > AREA_THRESHOLD AND ratio > RATIO_THRESHOLD → REJECT (0)
    else → ACCEPT (1)
```

**Infection location map:**
```
┌────────────────┐
│   zone 0: TOP  │
├────────────────┤
│  zone 1: MID   │
├────────────────┤
│  zone 2: BOT   │
└────────────────┘
    zone 3: none
```

---

### Stage 6 — Feature Packing &nbsp;`feature_extraction.py`

Merges pear + infection results into the standard **6-value state vector**:

```
Index │ Field              │ Type    │ Description
──────┼────────────────────┼─────────┼────────────────────────────────────────────
  0   │ pear_flag          │ int     │ 1 = pear present, 0 = empty slot
  1   │ quality_flag       │ int     │ 1 = healthy (ACCEPT), 0 = infected (REJECT)
  2   │ infection_hue      │ float   │ mean HSV hue of infected region (0–179)
  3   │ infection_area     │ float   │ infected surface area in cm²
  4   │ pear_area          │ float   │ total pear area in cm²
  5   │ infection_location │ int     │ 0=top / 1=middle / 2=bottom / 3=no infection
```

Areas are converted from pixels² to cm² using `SCALE_FACTOR`.

---

### Stage 7 — Motor Task &nbsp;`motor_task.py`

**Stateless, deterministic, no image processing.** Pure decision → angle mapping.

```
For each ROI:

  pear_flag == 0  ──────────────────────────► NO_ACTION  (skip motor)

  pear_flag == 1
       │
       ├─ quality_flag == 1 ──────────────► ACCEPT  → motor → 0°
       │
       └─ quality_flag == 0 ──────────────► REJECT  → motor → 180°
```

Internal call chain:
```python
motor_task(frame_data)
  └─ decision_function(roi_vector)   → "ACCEPT" | "REJECT" | "NO_ACTION"
       └─ action_mapper(decision)    → 0 | 180 | None
            └─ motor_command(channel, angle)  → PCA9685.channels[ch].angle
```

> If PCA9685 hardware is not detected at startup, the system automatically falls back to **simulation mode** (prints commands to terminal) with no code changes required.

---

### Stage 8 — Publishing &nbsp;`publisher.py`

Sends the full frame payload to the configured destination:

| Mode | Transport | Format |
|------|-----------|--------|
| `print` | stdout | Formatted table — good for development |
| `file` | disk | JSON lines appended to `OUTPUT_FILE_PATH` |
| `ros2` | ROS2 topic | `std_msgs/String` serialized JSON |
| `serial` | UART | Compact `A,R,N,A,N,R\n` codes per frame |

Payload structure:
```json
{
  "frame_id":   42,
  "timestamp":  "2026-01-05T12:45:23.001",
  "frame_data": [[1,1,0.0,0.12,42.3,3], [1,0,14.3,6.8,38.7,0], ...],
  "decisions":  ["ACCEPT", "REJECT", "NO_ACTION", "ACCEPT", "NO_ACTION", "REJECT"],
  "roi_labels": ["A3","B3","A2","B2","A1","B1"]
}
```

---

##  Installation

### Prerequisites

```bash
# Core vision
pip install opencv-python numpy

# Servo hardware (Jetson / Raspberry Pi)
pip install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor

# Serial output (optional)
pip install pyserial

# ROS2 output (optional — requires a ROS2 environment)
# rclpy is installed as part of your ROS2 distro
```

### Clone the repository

```bash
git clone https://github.com/your-username/pear-sorting-vision.git
cd pear-sorting-vision
```

---

## ▶ Running the System

### Live camera — normal mode
```bash
python main.py
```

### Live camera — with debug overlay window
```bash
python main.py --debug
```
Press **`q`** in the debug window to quit.

### Test with a static image (no camera needed)
```bash
python main.py --test --image samples/pear_infected.jpg --debug
```

### Write output to log file
```python
# In config.py:
PUBLISH_MODE = "file"
OUTPUT_FILE_PATH = "output_log.txt"
```
```bash
python main.py
```

---

## 🖥 Output Format

### Terminal output (`PUBLISH_MODE = "print"`)

```
============================================================
FRAME #7  |  2026-01-05T12:45:23.441
============================================================
ROI    Pear  Quality  InfHue  InfArea   PearArea  InfLoc    Decision
------------------------------------------------------------
A3        1        1     0.0     0.12      42.30    NONE      ACCEPT
B3        1        0    14.3     6.80      38.70     TOP      REJECT
A2        0        1     0.0     0.00       0.00    NONE   NO_ACTION
B2        1        1     0.0     0.32      44.10    NONE      ACCEPT
A1        1        0    22.1     3.40      41.90     MID      REJECT
B1        0        1     0.0     0.00       0.00    NONE   NO_ACTION
============================================================
[MOTOR] Channel 0 →   0°   (A3 ACCEPT)
[MOTOR] Channel 1 → 180°   (B3 REJECT)
[MOTOR] Channel 3 →   0°   (B2 ACCEPT)
[MOTOR] Channel 4 → 180°   (A1 REJECT)
  → Cycle time: 38.4 ms (limit: 1000 ms)
```

---

## ⚙ Configuration Guide

> **All tuning lives in `config.py`.** You should never need to modify any other file during calibration.

### Camera

```python
CAMERA_INDEX   = 0       # USB camera index (try 1, 2 if default doesn't work)
FRAME_WIDTH    = 1280    # ↑ better detail  ↓ faster processing
FRAME_HEIGHT   = 720
LOOP_SLEEP_SEC = 1.0     # ↓ faster sorting rate  ↑ less CPU load
```

### ROI positions

```python
ROI_DEFINITIONS = {
    "A3": (0,    0,   640, 240),   # (x1, y1, x2, y2) in pixels
    "B3": (640,  0,  1280, 240),   # ← adjust these after physical camera alignment
    ...
}
```

### Pear detection — HSV thresholds

> **This is the most critical tuning step.** HSV ranges must match your specific pear variety and lighting.

```python
PEAR_HSV_LOWER = np.array([15, 30, 60])
#                           H   S   V
#   H ↑  →  more yellow, less green  |  H ↓  →  more green
#   S ↑  →  stricter color (less background noise)
#   V ↑  →  ignore shadows

PEAR_HSV_UPPER = np.array([95, 255, 255])
#   H ↓  →  exclude yellower pears   |  H ↑  →  include yellower pears
#   V ↓  →  exclude bright highlights (e.g., ~200 for overexposed)
```

### Infection detection — key thresholds

```python
INFECTION_HSV_UPPER = np.array([30, 255, 100])
#   V (100): ↑ catches lighter brown infections  |  ↓ only very dark spots

INFECTION_DARK_V_MAX    = 60    # ↑ more dark regions flagged  |  ↓ only near-black
INFECTION_AREA_THRESHOLD = 500  # ↑ fewer rejections (looser)  |  ↓ stricter
INFECTION_RATIO_THRESHOLD = 0.05 # 5% of pear area threshold   |  ↓ stricter
```

### Motor angles

```python
MOTOR_ACCEPT_ANGLE = 0    # servo angle for "good pear" gate position
MOTOR_REJECT_ANGLE = 180  # servo angle for "reject bin" gate position

MOTOR_CHANNELS = {        # PCA9685 I²C channel per tray slot
    "A3": 0, "B3": 1, "A2": 2,
    "B2": 3, "A1": 4, "B1": 5,
}
```

---

##  Calibration Workflow

Follow these steps in order when deploying to a new setup:

#### Step 1 — Align the camera
```bash
python main.py --debug
```
Adjust the camera physically until all 6 ROI rectangles align with tray slots. Update `ROI_DEFINITIONS` in `config.py` to match.

#### Step 2 — Tune pear HSV range
Place healthy pears in all slots. Adjust `PEAR_HSV_LOWER` / `PEAR_HSV_UPPER` until the pear bodies appear fully white in the debug mask with no background leaking through.

**Quick HSV sampler:**
```python
import cv2, numpy as np
img = cv2.imread("your_pear.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Click a pixel and read its H, S, V values to find the right range
```

#### Step 3 — Tune infection thresholds
Use infected pear samples. Adjust `INFECTION_HSV_UPPER[2]` (V channel) — if light brown infections are missed, raise it toward 120–130. Adjust `INFECTION_AREA_THRESHOLD` if healthy pears with small natural spots are wrongly rejected.

#### Step 4 — Calibrate scale factor
```
Place a 10×10 cm card in one ROI.
Note the reported pear_area value.
SCALE_FACTOR = 100.0 / reported_pixel_area
```

#### Step 5 — Calibrate motor angles
Set `PUBLISH_MODE = "print"` and observe decisions. Manually test `MOTOR_ACCEPT_ANGLE` and `MOTOR_REJECT_ANGLE` against your physical gate mechanism.

---

## 📁 Project Structure

```
pear-sorting-vision/
│
├── main.py                  # Entry point · main loop · --debug / --test flags
├── config.py                # ← ALL tunable parameters live here
│
├── frame_divider.py         # Task 1: frame → 6 ROI slices (numpy only)
├── preprocessing.py         # Task 2: median blur + BGR→HSV
├── pear_detection.py        # Task 3: HSV threshold → morphology → contour → validate
├── infection_detection.py   # Task 4: brown/dark detection → features → quality flag
├── feature_extraction.py    # Task 5: pack all results → 6-value vectors
├── motor_task.py            # Task 6: decision → angle → PCA9685 servo command
├── publisher.py             # Task 7: print / file / ROS2 / serial output
└── utils.py                 # Camera I/O · debug overlay · loop timer
```

### Data flow at a glance

```
camera frame (ndarray)
       │
frame_divider()         →  roi_dict: { "A3": ndarray, … }
       │
preprocess_all_rois()   →  preprocessed_dict: { roi: { bgr, hsv, original } }
       │
detect_all_pears()      →  pear_results: { roi: { flag, mask, area, contour } }
       │
detect_all_infections() →  infection_results: { roi: { quality, area, hue, loc } }
       │
format_frame_output()   →  frame_data: list[ [6 values] × 6 ROIs ]
       │
motor_task()            →  decisions: list["ACCEPT"|"REJECT"|"NO_ACTION" × 6]
       │                   + servo commands sent to PCA9685
publish()               →  terminal / file / ROS2 / serial
```

---

## 🔌 Hardware Wiring Reference

```
Jetson Nano / Raspberry Pi
         │
         │  I²C  (SDA / SCL)
         ▼
    PCA9685 board
    ├── channel 0  ──►  ROI A3 servo
    ├── channel 1  ──►  ROI B3 servo
    ├── channel 2  ──►  ROI A2 servo
    ├── channel 3  ──►  ROI B2 servo
    ├── channel 4  ──►  ROI A1 servo
    └── channel 5  ──►  ROI B1 servo

Camera  ──► USB / CSI  ──►  Jetson Nano
```

> Channel assignments are configurable via `MOTOR_CHANNELS` in `config.py`.

---

## 🧩 Optional: Class-Based Refactor

The pipeline is designed to be easily refactored into a class-based architecture for larger deployments:

```python
class ROIProcessor:
    def __init__(self, roi_name, config):  ...
    def preprocess(self, roi):             ...
    def detect_pear(self):                 ...
    def detect_infection(self):            ...
    def extract_features(self) -> list:    ...
    def decide(self) -> str:               ...
```

This allows parallel ROI processing (e.g., `ThreadPoolExecutor`) and per-slot state tracking.

---

## 📜 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **One responsibility per module** | `pear_detection.py` never touches motor logic |
| **No cross-task logic** | Infection detection never runs inside pear detection |
| **Structured data passing** | Dicts with named keys — never raw images between stages |
| **Independent ROIs** | Each ROI is a self-contained mini-pipeline |
| **Stateless frames** | No memory of previous cycles — deterministic output |
| **Config-only tuning** | Zero magic numbers in pipeline code |
| **Hardware graceful degradation** | Auto simulation mode if PCA9685 not found |
---

<div align="center">

Built for **real-time edge deployment** on NVIDIA Jetson Nano and Raspberry Pi.  
Designed for industrial pear sorting conveyors with 6-slot tray systems.

</div>
