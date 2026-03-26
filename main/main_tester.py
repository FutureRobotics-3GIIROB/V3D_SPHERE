"""Main tester: camera vision pipeline with shared state for RoboDK integration."""

from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import cv2
import numpy as np
from ARUCO_reader import ArucoReader
from ball_detector import HSVCalibrationUI
from Cam_administrator import create_latest_frame_camera, prompt_default_camera
from homography_Calibrator import HomographyCalibrator
from robodk_worker import RoboDKWorker
from vision_processor import configure_pin_rendering, process_camera_step
from vision_renderer import (
    draw_ball_position,
    draw_bolo_overlay,
    draw_debug_aruco_markers,
    draw_debug_overlay,
    draw_performance_overlay,
)
from vision_state import SharedVisionState

F9_KEYS = {120, 0x780000}
CPU_WORKERS = 2
MAIN_WINDOW_NAME = "main_tester - no RoboDK"


def toggle_debug(debug_mode: bool) -> bool:
    """Toggle debug mode and print its status."""
    new_state = not debug_mode
    print(f"Debug mode: {'ON' if new_state else 'OFF'}")
    if new_state:
        print("Debug backend info -> CPU multithread=True")
    return new_state


def _update_fps(frame_times: deque[float], elapsed_s: float) -> float:
    """Update rolling frame time window and return current FPS."""
    frame_times.append(elapsed_s)
    avg = sum(frame_times) / len(frame_times) if frame_times else 0.0
    return (1.0 / avg) if avg > 0 else 0.0


def _draw_inline_calibration_overlay(
    display: np.ndarray,
    message: str,
    color: tuple[int, int, int],
    fps: float,
    calibration_debug: bool,
    cpu_threads: int,
) -> None:
    """Draw overlay texts for inline calibration mode."""
    cv2.putText(display, message, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

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


def _build_inline_calibration_result(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    objp: np.ndarray,
    chessboard_size: tuple[int, int],
    last_gray_shape: tuple[int, int],
) -> dict[str, Any] | None:
    """Compute calibration matrices and persist positions.json payload."""
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

    HomographyCalibrator.CALIBRATION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Calibration saved to {HomographyCalibrator.CALIBRATION_FILE}")
    print("Returning to main view...\n")
    return HomographyCalibrator.load_positions_file()


def run_calibration_inline(cam_reader: Any) -> dict[str, Any] | None:
    """Run calibration in the main window (inline, non-blocking)."""
    print("\n=== Calibration Mode ===")
    print("Find and capture chessboard 3 times")
    print("SPACE = capture, ESC/q = cancel, F9 = debug toggle")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    objp = HomographyCalibrator._build_object_points()
    chessboard_size = HomographyCalibrator.CHESSBOARD_SIZE
    frame_times: deque[float] = deque(maxlen=60)
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
            message = f"Chessboard detected {len(image_points)}/{HomographyCalibrator.REQUIRED_CAPTURES} - press SPACE"
            color = (0, 200, 0)
        else:
            refined = None
            message = (
                f"Find chessboard {len(image_points)}/{HomographyCalibrator.REQUIRED_CAPTURES}"
            )
            color = (0, 0, 255)

        cv2.putText(display, message, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        elapsed = time.perf_counter() - loop_start
        fps = _update_fps(frame_times, elapsed)
        _draw_inline_calibration_overlay(
            display,
            message=message,
            color=color,
            fps=fps,
            calibration_debug=calibration_debug,
            cpu_threads=cpu_threads,
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
    return _build_inline_calibration_result(
        object_points,
        image_points,
        objp,
        chessboard_size,
        last_gray_shape,
    )


def _render_main_step(
    step: Any,
    homography: Any,
    use_stl_render: bool,
    debug_mode: bool,
    use_multithread: bool,
    frame_times: deque[float],
    loop_start: float,
) -> None:
    """Render overlays and diagnostics for one processed frame."""
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

    fps = _update_fps(frame_times, time.perf_counter() - loop_start)
    draw_performance_overlay(step.frame, fps)

    if debug_mode:
        marker_debug = [
            (m.id, m.xyz_mm, m.estado)
            for m in sorted(step.marker_states, key=lambda marker: int(marker.id))
        ]
        draw_debug_overlay(
            step.frame,
            use_multithread=use_multithread,
            source_label=step.source_label,
            ball_xyz_mm=step.ball_state.xyz_mm,
            marker_debug=marker_debug,
        )
        draw_debug_aruco_markers(step.frame, step.detections)


def _handle_main_key(
    key: int,
    debug_mode: bool,
    aruco_reader: ArucoReader,
    robodk_worker: RoboDKWorker,
    cam_reader: Any,
    homography: Any,
) -> tuple[bool, bool, Any]:
    """Handle keyboard controls and return (should_exit, debug_mode, homography)."""
    if key in F9_KEYS:
        return False, toggle_debug(debug_mode), homography

    if key in (27, ord("q"), ord("Q"), ord("Q") & 0xFF):
        return True, debug_mode, homography

    if key == 32:  # SPACE — reset all pins to standing.
        aruco_reader.reset_all_pins()
        print("Pins reset.")
        return False, debug_mode, homography

    if key in (ord("r"), ord("R")):
        print("Starting calibration...")
        new_calib = run_calibration_inline(cam_reader)
        if new_calib is not None:
            print("Calibration updated.")
            return False, debug_mode, cast(Any, new_calib["homography_matrix"])
        print("Calibration cancelled.")
        return False, debug_mode, homography

    if key in (ord("p"), ord("P")):
        robodk_worker.toggle_ur3e_follow()
        return False, debug_mode, homography

    if key in (ord("g"), ord("G")):
        robodk_worker.request_pick_and_drop()
        return False, debug_mode, homography

    return False, debug_mode, homography


def _handle_hsv_key(key: int, hsv_ui: HSVCalibrationUI) -> None:
    """Toggle HSV calibration mode with keyboard shortcut."""
    if key not in (ord("h"), ord("H")):
        return
    enabled = hsv_ui.toggle()
    if enabled:
        print("HSV calibration: ON (misma ventana, ajusta sliders HSV)")
        return
    print("HSV calibration: OFF")


def _create_runtime(
    camera_source: Any,
) -> tuple[Any, ArucoReader, bool, bool, bool, deque[float]]:
    """Create camera reader, ArUco reader and runtime flags for the main loop."""
    cam_reader = create_latest_frame_camera(camera_source)
    aruco_reader = ArucoReader(
        marker_size_m=0.05,
        dictionary_name="original",
        enable_fallback_dictionary=True,
        expected_ids=range(0, 11),
    )
    use_multithread = True
    debug_mode = False
    frame_times: deque[float] = deque(maxlen=60)
    use_stl_render = configure_pin_rendering(aruco_reader)
    return cam_reader, aruco_reader, use_multithread, debug_mode, use_stl_render, frame_times


def _process_main_frame(
    frame: Any,
    aruco_reader: ArucoReader,
    homography: Any,
    use_stl_render: bool,
    use_multithread: bool,
    executor: ThreadPoolExecutor | None,
    vision_state: SharedVisionState,
    debug_mode: bool,
    frame_times: deque[float],
    loop_start: float,
    hsv_ui: HSVCalibrationUI,
) -> tuple[Any, Any]:
    """Process one frame and render all overlays, returning the processing step result."""
    enable_ball_detection = not vision_state.is_ball_detection_paused()
    ball_config = hsv_ui.get_config()

    step = process_camera_step(
        frame=frame,
        aruco_reader=aruco_reader,
        homography=homography,
        use_stl_render=use_stl_render,
        executor=executor if use_multithread else None,
        enable_ball_detection=enable_ball_detection,
        ball_config=ball_config,
    )

    vision_state.update_frame(
        ball=step.ball_state,
        markers=step.marker_states,
        bolo_count=step.bolo_count,
    )

    _render_main_step(
        step=step,
        homography=homography,
        use_stl_render=use_stl_render,
        debug_mode=debug_mode,
        use_multithread=use_multithread,
        frame_times=frame_times,
        loop_start=loop_start,
    )

    display_frame = hsv_ui.build_calibration_view(
        source_frame=frame,
        fallback_frame=step.frame,
        config=ball_config,
    )
    return step, display_frame


def main() -> int:
    """Run full pipeline with camera vision thread and shared state for RoboDK."""
    vision_state = SharedVisionState()
    robodk_worker = RoboDKWorker(vision_state)
    camera_source = prompt_default_camera()

    calib = HomographyCalibrator.ensure_positions_file(camera_source=camera_source, ask_user=True)
    if calib is None:
        print("Calibration is required. Exiting.")
        return 1

    homography: Any = calib["homography_matrix"]

    (
        cam_reader,
        aruco_reader,
        use_multithread,
        debug_mode,
        use_stl_render,
        frame_times,
    ) = _create_runtime(camera_source)

    print("Compute backend: CPU multithread")
    hsv_ui = HSVCalibrationUI(main_window_name=MAIN_WINDOW_NAME)

    print("\n=== Main Tester (No RoboDK) ===")
    print(
        "Keys: ESC/q=exit, r=regenerate calibration, F9=toggle debug, "
        "SPACE=reset pins, H=HSV calibration, P=toggle UR3e prepick follow, G=UR3e pick+drop"
    )

    executor: ThreadPoolExecutor | None = None
    try:
        robodk_worker.start()

        if use_multithread:
            executor = ThreadPoolExecutor(max_workers=CPU_WORKERS)

        while True:
            loop_start = time.perf_counter()
            frame = cam_reader.read_latest()
            if frame is None:
                continue

            step = _process_main_frame(
                frame=frame,
                aruco_reader=aruco_reader,
                homography=homography,
                use_stl_render=use_stl_render,
                use_multithread=use_multithread,
                executor=executor,
                vision_state=vision_state,
                debug_mode=debug_mode,
                frame_times=frame_times,
                loop_start=loop_start,
                hsv_ui=hsv_ui,
            )

            _, display_frame = step
            cv2.imshow(MAIN_WINDOW_NAME, display_frame)
            key = cv2.waitKeyEx(1)

            should_exit, debug_mode, homography = _handle_main_key(
                key=key,
                debug_mode=debug_mode,
                aruco_reader=aruco_reader,
                robodk_worker=robodk_worker,
                cam_reader=cam_reader,
                homography=homography,
            )
            _handle_hsv_key(key, hsv_ui)
            if should_exit:
                break

    finally:
        hsv_ui.set_enabled(False)
        robodk_worker.stop()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        cam_reader.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
