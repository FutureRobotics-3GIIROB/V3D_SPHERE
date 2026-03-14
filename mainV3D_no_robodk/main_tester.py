from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2

from ARUCO_reader import ArucoReader
from Cam_administrator import prompt_default_camera, create_latest_frame_camera
from ball_detector import detect_ball, detect_ball_gpu, draw_ball
from homography_Calibrator import HomographyCalibrator


BALL_ARUCO_ID = 0
MIN_PIN_ID = 1
MAX_PIN_ID = 10
DEFAULT_BOLO_MODEL = Path(__file__).resolve().parent / "bolo.stl"
OUTPUT_FILE = Path(__file__).resolve().parent / "ball_position_output.json"

F9_KEYS = {120, 0x78, 0x780000}


def _write_output(payload: dict) -> None:
    """Persist current ball position output for external readers."""
    try:
        OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _select_compute_backend() -> tuple[bool, bool]:
    """Return (use_gpu, use_multithread) based on CUDA availability."""
    use_gpu = False
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)
            use_gpu = True
    except Exception:
        use_gpu = False

    # If no CUDA device is available, use CPU multithreading.
    use_multithread = not use_gpu
    return use_gpu, use_multithread


def _draw_performance_overlay(frame, fps: float) -> None:
    """Draw FPS info on every frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def _draw_debug_overlay(
    frame,
    use_gpu: bool,
    use_multithread: bool,
    cuda_devices: int,
    source_label: str,
) -> None:
    """Draw runtime debug details while debug mode is active."""
    debug_lines = [
        "DEBUG MODE: ON",
        f"Backend: {'CUDA GPU' if use_gpu else 'CPU'}",
        f"CUDA devices: {cuda_devices}",
        f"Multithread CPU: {use_multithread}",
        f"Ball source: {source_label}",
    ]
    y = 88
    for line in debug_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
        y += 24


def _draw_debug_aruco_markers(frame, detections) -> None:
    """Draw square and marker id for each ArUco while debug mode is active."""
    for det in detections:
        corners = det.corners.astype(int)
        x_min = int(corners[:, 0].min())
        y_min = int(corners[:, 1].min())
        x_max = int(corners[:, 0].max())
        y_max = int(corners[:, 1].max())

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"ARUCO {det.id}",
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def main() -> int:
    """Run full pipeline without RoboDK integration."""
    camera_source = prompt_default_camera()

    calib = HomographyCalibrator.ensure_positions_file(camera_source=camera_source, ask_user=True)
    if calib is None:
        print("Calibration is required. Exiting.")
        return 1

    homography = calib["homography_matrix"]

    cam_reader = create_latest_frame_camera(camera_source)
    aruco_reader = ArucoReader(
        marker_size_m=0.05,
        dictionary_name="original",
        enable_fallback_dictionary=True,
        expected_ids=range(0, 11),
    )
    use_gpu, use_multithread = _select_compute_backend()
    cuda_devices = 0
    try:
        if hasattr(cv2, "cuda"):
            cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cuda_devices = 0

    debug_mode = False
    frame_times = deque(maxlen=60)

    if use_gpu:
        print("Compute backend: CUDA GPU")
    else:
        print("Compute backend: CPU multithread")

    use_stl_render = DEFAULT_BOLO_MODEL.exists()
    if use_stl_render:
        for marker_id in range(MIN_PIN_ID, MAX_PIN_ID + 1):
            aruco_reader.ensure_object(marker_id).render(
                str(DEFAULT_BOLO_MODEL),
                pose_offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                scale=1.0,
                color=(255, 190, 40),
            )
        print(f"Pin rendering: STL ({DEFAULT_BOLO_MODEL.name})")
    else:
        print("Pin rendering: virtual fallback (bolo.stl not found)")

    print("\n=== Main Tester (No RoboDK) ===")
    print("Keys: ESC/q=exit, r=regenerate calibration, F9=toggle debug")

    try:
        executor: Optional[ThreadPoolExecutor] = None
        if use_multithread:
            executor = ThreadPoolExecutor(max_workers=2)

        while True:
            loop_start = time.perf_counter()
            frame = cam_reader.read_latest()
            if frame is None:
                continue

            if use_multithread and executor is not None:
                # Run ArUco and color detector in parallel on CPU.
                future_aruco = executor.submit(aruco_reader.process_frame, frame, True)
                future_ball = executor.submit(detect_ball, frame)
                aruco_result = future_aruco.result()
                precomputed_ball = future_ball.result()
            else:
                aruco_result = aruco_reader.process_frame(frame, draw=True)
                precomputed_ball = detect_ball_gpu(frame) if use_gpu else detect_ball(frame)

            source_label = "COLOR"
            det = precomputed_ball
            ball_center = None
            if det:
                ball_center = det["center"]
                draw_ball(aruco_result.frame, det, color=(0, 255, 0), label="BALL-COLOR")

            if use_stl_render:
                bolo_count = sum(1 for d in aruco_result.detections if int(d.id) >= MIN_PIN_ID)
            else:
                bolo_count = aruco_reader.draw_virtual_pins(
                    aruco_result.frame,
                    aruco_result.detections,
                    min_pin_id=MIN_PIN_ID,
                )

            cv2.putText(
                aruco_result.frame,
                f"Bolos by ArUco: {bolo_count} [{'STL' if use_stl_render else 'VIRTUAL'}]",
                (12, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )

            if ball_center is not None:
                x_mm, y_mm = HomographyCalibrator.transform_point(ball_center, homography)
                payload = {
                    "pixel": [int(ball_center[0]), int(ball_center[1])],
                    "xyz_mm": [float(x_mm), float(y_mm), 0.0],
                    "source": source_label,
                }
                _write_output(payload)

                cv2.putText(
                    aruco_result.frame,
                    f"X:{x_mm:.1f} Y:{y_mm:.1f} Z:0.0 [{source_label}]",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            elapsed = time.perf_counter() - loop_start
            frame_times.append(elapsed)
            avg = sum(frame_times) / len(frame_times) if frame_times else 0.0
            fps = (1.0 / avg) if avg > 0 else 0.0
            _draw_performance_overlay(aruco_result.frame, fps)

            if debug_mode:
                _draw_debug_overlay(
                    aruco_result.frame,
                    use_gpu=use_gpu,
                    use_multithread=use_multithread,
                    cuda_devices=cuda_devices,
                    source_label=source_label,
                )
                _draw_debug_aruco_markers(aruco_result.frame, aruco_result.detections)

            cv2.imshow("main_tester - no RoboDK", aruco_result.frame)
            key = cv2.waitKeyEx(1)

            if key in F9_KEYS:
                debug_mode = not debug_mode
                status = "ON" if debug_mode else "OFF"
                print(f"Debug mode: {status}")
                if debug_mode:
                    print(
                        "Debug backend info -> "
                        f"CUDA={use_gpu}, CUDA devices={cuda_devices}, "
                        f"CPU multithread={use_multithread}"
                    )
                continue

            if key in (27, ord("q"), ord("Q"), ord("Q") & 0xFF):
                break

            if key in (ord("r"), ord("R")):
                print("Regenerating calibration...")
                new_calib = HomographyCalibrator.generate_positions_file(camera_source)
                if new_calib is not None:
                    homography = new_calib["homography_matrix"]
                    print("Calibration updated.")
                else:
                    print("Calibration unchanged.")

    finally:
        if 'executor' in locals() and executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        cam_reader.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
