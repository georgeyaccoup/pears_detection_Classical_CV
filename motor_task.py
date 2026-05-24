"""
motor_task.py — Decision & Actuation Module (Motor Task)
==========================================================
RESPONSIBILITY: Translate the vision pipeline's feature vectors into physical
sorting motor commands (ACCEPT / REJECT / NO_ACTION).

This is the FINAL stage of the processing pipeline. It receives the packed
6-value state vectors and issues motor angles to the servo controller.

DECISION LOGIC per ROI:
  pear_flag == 0   → NO_ACTION    (no pear in slot → skip motor)
  quality_flag == 1 → ACCEPT      → motor → 0°  (good pear route)
  quality_flag == 0 → REJECT      → motor → 180° (reject bin route)

DESIGN RULES (CRITICAL):
  - NO image processing
  - NO detection logic
  - NO state from previous frames (stateless)
  - Deterministic: same input → same motor output, always
  - Lightweight: must execute < 1ms per ROI for real-time use

MOTOR HARDWARE:
  Assumes PCA9685 I²C servo controller (common on Jetson Nano / Raspberry Pi).
  Requires: pip install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
  If running without hardware, set SIMULATE_MOTORS = True in config.py.
"""

import config

# Attempt to import hardware library; fall back to simulation if not available
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as adafruit_servo

    # Initialize I²C bus and PCA9685 board
    _i2c = busio.I2C(board.SCL, board.SDA)
    _pca = PCA9685(_i2c)
    _pca.frequency = 50  # Standard servo frequency: 50 Hz

    # Create servo objects for each channel
    _servos = {
        ch: adafruit_servo.Servo(_pca.channels[ch])
        for ch in config.MOTOR_CHANNELS.values()
    }

    HARDWARE_AVAILABLE = True
    print("[MOTOR] PCA9685 hardware initialized successfully.")

except Exception as e:
    # No hardware connected or library not installed → run in simulation mode
    HARDWARE_AVAILABLE = False
    _servos = {}
    print(f"[MOTOR] Hardware not available ({e}). Running in SIMULATION mode.")


# =============================================================================
# DECISION FUNCTION
# =============================================================================

def decision_function(roi_vector: list) -> str:
    """
    Evaluates a single ROI feature vector and returns the sorting decision.

    Parameters
    ----------
    roi_vector : list
        6-element feature vector:
        [pear_flag, quality_flag, infection_hue, infection_area,
         pear_area, infection_location]

    Returns
    -------
    str
        One of: "NO_ACTION", "ACCEPT", "REJECT"

    Decision rules:
      - No pear present   → "NO_ACTION" (skip motor entirely)
      - Good quality pear → "ACCEPT"    (route to accept lane)
      - Infected pear     → "REJECT"    (route to reject bin)
    """

    pear_flag    = roi_vector[0]  # Index 0: pear present?
    quality_flag = roi_vector[1]  # Index 1: pear quality

    # Rule 1: No pear in this slot → do nothing
    if pear_flag == 0:
        return "NO_ACTION"

    # Rule 2: Pear is healthy
    if quality_flag == 1:
        return "ACCEPT"

    # Rule 3: Pear is infected
    return "REJECT"


# =============================================================================
# ACTION MAPPER
# =============================================================================

def action_mapper(decision: str) -> int | None:
    """
    Maps a sorting decision string to a motor servo angle.

    Parameters
    ----------
    decision : str
        One of: "NO_ACTION", "ACCEPT", "REJECT"

    Returns
    -------
    int or None
        Motor angle in degrees (0–180), or None for NO_ACTION.

    Angle mapping (configured in config.py):
      ACCEPT   → MOTOR_ACCEPT_ANGLE (default: 0°)
      REJECT   → MOTOR_REJECT_ANGLE (default: 180°)
      NO_ACTION → None (motor stays in last position)
    """

    if decision == "ACCEPT":
        return config.MOTOR_ACCEPT_ANGLE   # 0° → accept lane

    elif decision == "REJECT":
        return config.MOTOR_REJECT_ANGLE   # 180° → reject bin

    else:  # "NO_ACTION"
        return None  # Do not move motor


# =============================================================================
# MOTOR COMMAND
# =============================================================================

def motor_command(channel: int, angle: int):
    """
    Sends a servo angle command to a specific motor channel.

    Parameters
    ----------
    channel : int
        PCA9685 channel number (0–15) corresponding to the ROI's actuator.
        Channels are defined in config.MOTOR_CHANNELS.
    angle : int
        Target servo angle in degrees (0–180).

    Hardware behavior:
      - Sets servo to the specified angle via PCA9685 PWM signal
      - If hardware is unavailable → prints simulation message

    Safety:
      - Angle is clamped to [0, 180] to prevent hardware damage
    """

    # Clamp angle to valid servo range
    safe_angle = max(0, min(180, angle))

    if HARDWARE_AVAILABLE and channel in _servos:
        # Send command to real servo hardware
        _servos[channel].angle = safe_angle
        print(f"[MOTOR HW] Channel {channel} → {safe_angle}°")
    else:
        # Simulation mode: print what would have happened
        print(f"[MOTOR SIM] Channel {channel} → {safe_angle}°")


# =============================================================================
# MAIN MOTOR TASK
# =============================================================================

def motor_task(frame_data: list):
    """
    Processes all 6 ROI feature vectors and issues motor commands.

    This is the top-level function called from main.py at the end of each frame.

    Pipeline per ROI:
      roi_vector → decision_function() → action_mapper() → motor_command()

    Parameters
    ----------
    frame_data : list
        List of 6 feature vectors (output from feature_extraction.format_frame_output):
        [
          [pear_flag, quality_flag, inf_hue, inf_area, pear_area, inf_loc],  # A3
          [pear_flag, quality_flag, inf_hue, inf_area, pear_area, inf_loc],  # B3
          ...
        ]

    Returns
    -------
    list of str
        List of decisions for each ROI (for logging/publishing).
        E.g., ["ACCEPT", "REJECT", "NO_ACTION", "ACCEPT", "NO_ACTION", "REJECT"]
    """

    decisions = []  # Track decisions for all ROIs (used by publisher for logging)

    for idx, roi_name in enumerate(config.ROI_ORDER):
        roi_vector = frame_data[idx]  # Get the feature vector for this ROI

        # --- Step 1: Make sorting decision ---
        decision = decision_function(roi_vector)

        # --- Step 2: Map decision to motor angle ---
        angle = action_mapper(decision)

        # --- Step 3: Send motor command (if action needed) ---
        if angle is not None:
            channel = config.MOTOR_CHANNELS[roi_name]
            motor_command(channel, angle)
        else:
            print(f"[MOTOR] ROI {roi_name}: NO_ACTION (no pear)")

        decisions.append(decision)

    return decisions
