"""
main.py — System Entry Point & Main Control Loop
==================================================
This is the top-level orchestrator for the Pear Sorting Vision System.

PIPELINE (executed each iteration):
  1. Capture frame from camera
  2. Divide frame into 6 ROIs
  3. Preprocess each ROI (blur + HSV)
  4. Detect pear presence in each ROI
  5. Detect infections in each ROI (only if pear present)
  6. Pack all results into 6-value feature vectors
  7. Execute motor commands (accept/reject)
  8. Publish results to configured output
  9. Sleep until next cycle

ENTRY POINT:
  Run with:   python main.py
  Debug mode: python main.py --debug

FLAGS:
  --debug    Show live camera window with ROI overlays and decisions
  --test     Run with a static test image instead of live camera
  --image    Path to test image (used with --test)
"""

import time
import sys
import argparse
import cv2

# Import all pipeline modules
import config
import utils
from frame_divider       import frame_divider
from preprocessing       import preprocess_all_rois
from pear_detection      import detect_all_pears
from infection_detection import detect_all_infections
from feature_extraction  import format_frame_output, get_roi_summary
from motor_task          import motor_task
from publisher           import publish


def parse_args():
    """
    Parses command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with flags: debug, test, image.
    """
    parser = argparse.ArgumentParser(
        description="Pear Sorting Vision System — Real-Time Pipeline"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show live debug window with ROI overlays"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run with a static test image instead of live camera"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to test image file (use with --test)"
    )
    return parser.parse_args()


def run_pipeline(frame, debug: bool = False):
    """
    Executes one complete frame processing cycle.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame from camera or test image.
    debug : bool
        If True, draws visual overlays on the frame for monitoring.

    Returns
    -------
    tuple : (frame_data, decisions)
        frame_data : list of 6 feature vectors
        decisions  : list of 6 decision strings
    """

    # ------------------------------------------------------------------
    # TASK 1: Frame Division
    # Split the full frame into 6 independent ROI sub-images
    # ------------------------------------------------------------------
    roi_dict = frame_divider(frame)

    # ------------------------------------------------------------------
    # TASK 2: Preprocessing
    # Apply noise reduction (median blur) and convert to HSV
    # ------------------------------------------------------------------
    preprocessed_dict = preprocess_all_rois(roi_dict)

    # ------------------------------------------------------------------
    # TASK 3: Pear Detection
    # For each ROI: threshold → morphology → contour → validation
    # ------------------------------------------------------------------
    pear_results = detect_all_pears(preprocessed_dict)

    # ------------------------------------------------------------------
    # TASK 4: Infection Detection
    # For each ROI (where pear was found): detect dark/brown regions
    # ------------------------------------------------------------------
    infection_results = detect_all_infections(pear_results, preprocessed_dict)

    # ------------------------------------------------------------------
    # TASK 5: Feature Packing
    # Merge pear + infection results into 6-value vectors per ROI
    # ------------------------------------------------------------------
    frame_data = format_frame_output(pear_results, infection_results)

    # ------------------------------------------------------------------
    # TASK 6: Motor Task (Decision + Actuation)
    # Translate feature vectors into ACCEPT/REJECT servo commands
    # ------------------------------------------------------------------
    decisions = motor_task(frame_data)

    # ------------------------------------------------------------------
    # OPTIONAL: Debug visualization
    # Draw ROI boxes, contours, and decision labels on frame
    # ------------------------------------------------------------------
    if debug:
        for idx, roi_name in enumerate(config.ROI_ORDER):
            x1, y1, x2, y2 = config.ROI_DEFINITIONS[roi_name]
            utils.draw_debug_overlay(
                frame, roi_name, x1, y1,
                pear_results[roi_name],
                infection_results[roi_name],
                decisions[idx]
            )
        cv2.imshow("Pear Sorter — Debug View", frame)

    return frame_data, decisions


def main():
    """
    Main entry point. Initializes hardware and runs the real-time loop.

    Loop flow:
      capture → pipeline → publish → sleep(1s) → repeat

    Press Ctrl+C to stop cleanly.
    Press 'q' in the debug window to quit.
    """
    args = parse_args()

    print("=" * 60)
    print("  PEAR SORTING VISION SYSTEM — Starting Up")
    print("=" * 60)
    print(f"  Mode:         {'TEST IMAGE' if args.test else 'LIVE CAMERA'}")
    print(f"  Debug window: {'ON' if args.debug else 'OFF'}")
    print(f"  Publish mode: {config.PUBLISH_MODE}")
    print(f"  Loop rate:    {1.0 / config.LOOP_SLEEP_SEC:.1f} Hz")
    print("=" * 60)

    # --- Camera or test image setup ---
    cap = None  # Camera object (only used in live mode)

    if not args.test:
        # Live camera mode
        cap = utils.init_camera()
    else:
        # Test image mode — validate path
        if args.image is None:
            print("[ERROR] --test requires --image <path>. Exiting.")
            sys.exit(1)
        print(f"[MAIN] Test image: {args.image}")

    timer = utils.LoopTimer()  # Measures per-frame processing time

    frame_count = 0  # Track total frames processed

    try:
        while True:
            with timer:

                # --- Acquire frame ---
                if args.test:
                    # Load static test image each cycle (allows hot-swap of file)
                    frame = utils.load_test_frame(args.image)
                else:
                    frame = utils.get_frame(cap)

                # --- Run full processing pipeline ---
                frame_data, decisions = run_pipeline(frame, debug=args.debug)

                # --- Print per-ROI summaries ---
                frame_count += 1
                print(f"\n[FRAME {frame_count}]")
                for idx, roi_name in enumerate(config.ROI_ORDER):
                    print("  " + get_roi_summary(roi_name, frame_data[idx]))

                # --- Publish results ---
                publish(frame_data, decisions)

            # Report processing time
            print(f"  → Cycle time: {timer.elapsed_ms:.1f} ms "
                  f"(limit: {config.LOOP_SLEEP_SEC * 1000:.0f} ms)")

            # --- Debug window key handler ---
            if args.debug:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[MAIN] 'q' pressed — exiting.")
                    break

            # --- Sleep until next cycle ---
            # Subtract processing time from sleep to maintain target rate
            elapsed_sec = timer.elapsed_ms / 1000.0
            remaining_sleep = max(0.0, config.LOOP_SLEEP_SEC - elapsed_sec)
            time.sleep(remaining_sleep)

    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt received — shutting down.")

    except Exception as e:
        print(f"\n[MAIN ERROR] Unexpected error: {e}")
        raise

    finally:
        # --- Cleanup ---
        if cap is not None:
            utils.release_camera(cap)
        cv2.destroyAllWindows()
        print("[MAIN] System stopped cleanly.")


if __name__ == "__main__":
    main()
