"""Main tester: camera vision pipeline with shared state for RoboDK integration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import time
from typing import Any, Optional, cast

import cv2
import numpy as np

from ARUCO_reader import ArucoReader
from Cam_administrator import prompt_default_camera, create_latest_frame_camera
from homography_Calibrator import HomographyCalibrator
from vision_state import SharedVisionState
from vision_processor import configure_pin_rendering, process_camera_step
from vision_renderer import (
    draw_performance_overlay,
    draw_debug_overlay,
    draw_debug_aruco_markers,
    draw_bolo_overlay,
    draw_ball_position,
)


F9_KEYS = {120, 0x78, 0x780000}
CPU_WORKERS = 2


def toggle_debug(debug_mode: bool) -> bool:
    """Toggle debug mode and print its status."""
    new_state = not debug_mode
    print(f"Debug mode: {'ON' if new_state else 'OFF'}")
    if new_state:
        print("Debug backend info -> CPU multithread=True")
    return new_state


def run_calibration_inline(cam_reader: Any) -> Optional[dict[str, Any]]:
    """Run calibration in the main window (inline, non-blocking)."""
    print("\n=== Calibration Mode ===")
    print("Find and capture chessboard 3 times")
    print("SPACE = capture, ESC/q = cancel, F9 = debug toggle")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    objp = HomographyCalibrator._build_object_points()
    chessboard_size = HomographyCalibrator.CHESSBOARD_SIZE
    frame_times = deque(maxlen=60)
    calibration_debug = False
    cpu_threads = HomographyCalibrator._configure_cpu_backend()

    while len(image_points) < HomographyCalibrator.REQUIRED_CAPTURES:
        loop_start = time.perf_counter()
        frame = cam_reader.read_latest()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard_size)
        display = frame.copy()

        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, chessboard_size, refined, found)
            message = (
                f"Chessboard detected {len(image_points)}/{HomographyCalibrator.REQUIRED_CAPTURES} - press SPACE"
            )
            color = (0, 200, 0)
        else:
            refined = None
            message = f"Find chessboard {len(image_points)}/{HomographyCalibrator.REQUIRED_CAPTURES}"
            color = (0, 0, 255)

        cv2.putText(display, message, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        elapsed = time.perf_counter() - loop_start
        frame_times.append(elapsed)
        avg = sum(frame_times) / len(frame_times) if frame_times else 0.0
        fps = (1.0 / avg) if avg > 0 else 0.0
        cv2.putText(display, f"FPS: {fps:.1f}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if calibration_debug:
            cv2.putText(
                display,
                f"CPU={cpu_threads} threads",
                (12, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        cv2.imshow("main_tester - no RoboDK", display)
        key = cv2.waitKeyEx(1)

        if key in F9_KEYS:
            calibration_debug = not calibration_debug
            status = "ON" if calibration_debug else "OFF"
            print(f"Calibration debug mode: {status}")
            continue

        if key in (27, ord("q"), ord("Q")):
            print("Calibration cancelled.")
            return None

        if key == 32 and found and refined is not None:
            object_points.append(objp.copy())
            image_points.append(refined)
            print(f"Capture saved: {len(image_points)}/{HomographyCalibrator.REQUIRED_CAPTURES}")

    if not image_points:
        return None

    last_gray_shape = gray.shape[::-1]
    calibrate_camera = cast(Any, cv2.calibrateCamera)
    reproj_err, camera_matrix, dist_coeffs, _, _ = calibrate_camera(
        object_points,
        image_points,
        last_gray_shape,
        None,
        None,
    )

    world_xy = objp[:, :2]
    img_all = np.vstack([pts.reshape(-1, 2) for pts in image_points]).astype(np.float32)
    world_all = np.vstack([world_xy for _ in image_points]).astype(np.float32)

    homography_matrix, mask = cv2.findHomography(img_all, world_all, method=cv2.RANSAC)
    if homography_matrix is None:
        print("Unable to compute homography matrix.")
        return None

    data = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "homography_matrix": homography_matrix.tolist(),
        "chessboard_size": list(chessboard_size),
        "square_size_mm": HomographyCalibrator.SQUARE_SIZE_MM,
        "required_captures": HomographyCalibrator.REQUIRED_CAPTURES,
        "reprojection_error": float(reproj_err),
        "inliers": int(mask.sum()) if mask is not None else len(img_all),
    }

    HomographyCalibrator.CALIBRATION_FILE.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(f"Calibration saved to {HomographyCalibrator.CALIBRATION_FILE}")
    print("Returning to main view...\n")
    return HomographyCalibrator.load_positions_file()


def main() -> int:
    """Run full pipeline with camera vision thread and shared state for RoboDK."""
    vision_state = SharedVisionState()
    camera_source = prompt_default_camera()

    calib = HomographyCalibrator.ensure_positions_file(camera_source=camera_source, ask_user=True)
    if calib is None:
        print("Calibration is required. Exiting.")
        return 1

    homography: Any = calib["homography_matrix"]

    cam_reader = create_latest_frame_camera(camera_source)
    aruco_reader = ArucoReader(
        marker_size_m=0.05,
        dictionary_name="original",
        enable_fallback_dictionary=True,
        expected_ids=range(0, 11),
    )
    use_multithread = True
    debug_mode = False
    frame_times = deque(maxlen=60)

    print("Compute backend: CPU multithread")

    use_stl_render = configure_pin_rendering(aruco_reader)

    print("\n=== Main Tester (No RoboDK) ===")
    print("Keys: ESC/q=exit, r=regenerate calibration, F9=toggle debug, SPACE=reset pins")

    executor: Optional[ThreadPoolExecutor] = None
    try:
        if use_multithread:
            executor = ThreadPoolExecutor(max_workers=CPU_WORKERS)

        while True:
            loop_start = time.perf_counter()
            frame = cam_reader.read_latest()
            if frame is None:
                continue

            # Process current frame: detect, extract, build payloads
            step = process_camera_step(
                frame=frame,
                aruco_reader=aruco_reader,
                homography=homography,
                use_stl_render=use_stl_render,
                executor=executor if use_multithread else None,
            )

            # Update shared vision state for RoboDK thread to consume
            vision_state.update_frame(
                ball=step.ball_state,
                markers=step.marker_states,
                bolo_count=step.bolo_count,
            )

            # Render overlays on output frame
            draw_bolo_overlay(step.frame, step.bolo_count, use_stl_render)

            if step.ball_center is not None:
                x_mm, y_mm = HomographyCalibrator.transform_point(step.ball_center, homography)
                draw_ball_position(
                    step.frame,
                    step.ball_center,
                    step.ball_radius,
                    x_mm,
                    y_mm,
                    step.source_label,
                )

            elapsed = time.perf_counter() - loop_start
            frame_times.append(elapsed)
            avg = sum(frame_times) / len(frame_times) if frame_times else 0.0
            fps = (1.0 / avg) if avg > 0 else 0.0
            draw_performance_overlay(step.frame, fps)

            if debug_mode:
                draw_debug_overlay(
                    step.frame,
                    use_multithread=use_multithread,
                    source_label=step.source_label,
                )
                draw_debug_aruco_markers(step.frame, step.detections)

            cv2.imshow("main_tester - no RoboDK", step.frame)
            key = cv2.waitKeyEx(1)

            if key in F9_KEYS:
                debug_mode = toggle_debug(debug_mode)
                continue

            if key in (27, ord("q"), ord("Q"), ord("Q") & 0xFF):
                break

            if key == 32:  # SPACE — reset all pins to standing.
                aruco_reader.reset_all_pins()
                print("Pins reset.")

            if key in (ord("r"), ord("R")):
                print("Starting calibration...")
                new_calib = run_calibration_inline(cam_reader)
                if new_calib is not None:
                    homography = cast(Any, new_calib["homography_matrix"])
                    print("Calibration updated.")
                else:
                    print("Calibration cancelled.")

    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        cam_reader.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
