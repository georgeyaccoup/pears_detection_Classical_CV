"""
publisher.py — Publishing / Output Module
==========================================
RESPONSIBILITY: Send the final frame data and motor decisions to the
configured output destination.

Supported output modes (set PUBLISH_MODE in config.py):
  "print"  → print to terminal (development/debug)
  "file"   → append to a log file (persistent logging)
  "ros2"   → publish to a ROS2 topic (robot integration)
  "serial" → send over serial port (microcontroller communication)

OUTPUT FORMAT published per frame:
  {
    "frame_id":    int     incrementing frame counter
    "timestamp":   str     ISO timestamp
    "frame_data":  list    6x6 array of feature vectors
    "decisions":   list    6 decision strings
  }

RULES:
  - NO detection logic
  - NO motor commands
  - ONLY data serialization and transmission
"""

import json
import time
import datetime
import config

# Frame counter (increments each published frame)
_frame_counter = 0


def _format_payload(frame_data: list, decisions: list) -> dict:
    """
    Builds the full publishable payload dictionary.

    Parameters
    ----------
    frame_data : list
        6×6 feature vector array from feature_extraction.
    decisions : list
        6 decision strings from motor_task.

    Returns
    -------
    dict
        Complete frame payload with metadata.
    """
    global _frame_counter
    _frame_counter += 1

    return {
        "frame_id":   _frame_counter,
        "timestamp":  datetime.datetime.now().isoformat(),
        "frame_data": frame_data,
        "decisions":  decisions,
        "roi_labels": config.ROI_ORDER,
    }


def _publish_print(payload: dict):
    """
    Prints frame data to terminal in a readable format.
    Best for development and initial testing.
    """
    print("\n" + "=" * 60)
    print(f"FRAME #{payload['frame_id']}  |  {payload['timestamp']}")
    print("=" * 60)
    print(f"{'ROI':<6} {'Pear':>5} {'Quality':>8} {'InfHue':>8} "
          f"{'InfArea':>9} {'PearArea':>10} {'InfLoc':>8} {'Decision':>10}")
    print("-" * 60)

    for i, roi_name in enumerate(payload["roi_labels"]):
        vec = payload["frame_data"][i]
        pear_flag, quality_flag, inf_hue, inf_area, pear_area, inf_loc = vec
        decision = payload["decisions"][i]

        loc_map = {0: "TOP", 1: "MID", 2: "BOT", 3: "NONE"}
        print(
            f"{roi_name:<6} {pear_flag:>5} {quality_flag:>8} "
            f"{inf_hue:>8.1f} {inf_area:>9.2f} {pear_area:>10.2f} "
            f"{loc_map.get(inf_loc,'?'):>8} {decision:>10}"
        )
    print("=" * 60)


def _publish_file(payload: dict):
    """
    Appends frame data as a JSON line to the configured log file.
    Each line is a complete JSON object for easy parsing later.
    """
    with open(config.OUTPUT_FILE_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _publish_ros2(payload: dict):
    """
    Publishes frame data to a ROS2 topic as a JSON string.
    Requires a ROS2 environment and rclpy to be available.
    """
    try:
        import rclpy
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node("pear_sorter_publisher")
        pub  = node.create_publisher(String, config.ROS2_TOPIC, 10)
        msg  = String()
        msg.data = json.dumps(payload)
        pub.publish(msg)
        node.destroy_node()

    except ImportError:
        print("[PUBLISHER] ROS2 not available. Falling back to print.")
        _publish_print(payload)


def _publish_serial(payload: dict):
    """
    Sends the frame decisions over serial port (for microcontroller integration).
    Sends only the compact decisions list to minimize bandwidth.
    """
    try:
        import serial

        with serial.Serial(config.SERIAL_PORT, config.SERIAL_BAUD, timeout=1) as ser:
            # Send compact format: comma-separated decision codes
            # A=ACCEPT, R=REJECT, N=NO_ACTION
            codes = {
                "ACCEPT":    "A",
                "REJECT":    "R",
                "NO_ACTION": "N",
            }
            line = ",".join(codes.get(d, "N") for d in payload["decisions"]) + "\n"
            ser.write(line.encode("ascii"))
            print(f"[PUBLISHER SERIAL] Sent: {line.strip()}")

    except ImportError:
        print("[PUBLISHER] pyserial not installed. Falling back to print.")
        _publish_print(payload)

    except Exception as e:
        print(f"[PUBLISHER SERIAL ERROR] {e}")


def publish(frame_data: list, decisions: list):
    """
    Main publish function. Routes output to the mode set in config.PUBLISH_MODE.

    Parameters
    ----------
    frame_data : list
        6×6 feature vector array from feature_extraction.format_frame_output().
    decisions : list
        6 decision strings from motor_task.motor_task().
    """
    payload = _format_payload(frame_data, decisions)

    mode = config.PUBLISH_MODE.lower()

    if mode == "print":
        _publish_print(payload)

    elif mode == "file":
        _publish_file(payload)
        print(f"[PUBLISHER] Frame #{payload['frame_id']} written to {config.OUTPUT_FILE_PATH}")

    elif mode == "ros2":
        _publish_ros2(payload)

    elif mode == "serial":
        _publish_serial(payload)

    else:
        print(f"[PUBLISHER] Unknown mode '{mode}'. Defaulting to print.")
        _publish_print(payload)
