"""Vision pipeline processing logic: detection, extraction, and payload building."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from ARUCO_reader import ArucoReader
from ball_detector import detect_ball, draw_ball
from homography_Calibrator import HomographyCalibrator
from vision_state import BallState, FrameStepResult, MarkerState

MIN_PIN_ID = 1
MAX_PIN_ID = 10
DEFAULT_BOLO_MODEL = Path(__file__).resolve().parent / "bolo.stl"
BOLO_STL_SCALE = 0.003
BOLO_STL_ROTATION_RVEC = (3.14159265, 0.0, 0.0)
BOLO_STL_TVEC_OFFSET = (0.0, 0.0, -0.01)


def configure_pin_rendering(aruco_reader: ArucoReader) -> bool:
    """Configure STL rendering for pins when model file is available."""
    use_stl_render = DEFAULT_BOLO_MODEL.exists()
    if use_stl_render:
        for marker_id in range(MIN_PIN_ID, MAX_PIN_ID + 1):
            aruco_reader.ensure_object(marker_id).render(
                str(DEFAULT_BOLO_MODEL),
                pose_offsets=(BOLO_STL_TVEC_OFFSET, BOLO_STL_ROTATION_RVEC),
                scale=BOLO_STL_SCALE,
                color=(255, 190, 40),
            )
        print(f"Pin rendering: STL ({DEFAULT_BOLO_MODEL.name})")
    else:
        print("Pin rendering: virtual fallback (bolo.stl not found)")
    return use_stl_render


def detect_aruco_and_ball(
    frame: Any,
    aruco_reader: ArucoReader,
    executor: ThreadPoolExecutor | None,
) -> tuple[Any, dict[str, Any] | None]:
    """Run ArUco and ball detection for one frame."""
    if executor is not None:
        future_aruco = executor.submit(aruco_reader.process_frame, frame, True)
        future_ball = executor.submit(detect_ball, frame)
        return future_aruco.result(), cast(dict[str, Any] | None, future_ball.result())

    aruco_result = aruco_reader.process_frame(frame, draw=True)
    return aruco_result, cast(dict[str, Any] | None, detect_ball(frame))


def extract_ball_data(det: dict[str, Any] | None) -> tuple[tuple[int, int] | None, float]:
    """Normalize detector output into typed center/radius values."""
    if not det:
        return None, 0.0

    center_raw = det.get("center")
    center: tuple[int, int] | None = None
    if isinstance(center_raw, tuple | list) and len(center_raw) >= 2:
        center = (int(center_raw[0]), int(center_raw[1]))
    radius = float(det.get("radius", 0.0))
    return center, radius


def build_aruco_entries(
    aruco_reader: ArucoReader, detections: list[Any], homography: Any
) -> tuple[list[dict[str, Any]], list[MarkerState]]:
    """Build serializable payload entries and MarkerState objects for all detected markers."""
    entries: list[dict[str, Any]] = []
    marker_states: list[MarkerState] = []
    for det in detections:
        pin_state = aruco_reader._get_pin_state(int(det.id)) if int(det.id) >= MIN_PIN_ID else None
        fallen = (
            pin_state is not None
            and pin_state.current_tilt_deg >= aruco_reader.pin_fall_target_deg - 1.0
        )
        estado = "down" if fallen else "up"
        entry: dict[str, Any] = {
            "id": int(det.id),
            "center_px": [int(det.center_px[0]), int(det.center_px[1])],
            "estado": estado,
        }

        marker_x_mm, marker_y_mm = HomographyCalibrator.transform_point(det.center_px, homography)
        marker_xyz_mm = (float(marker_x_mm), float(marker_y_mm), 0.0)
        entry["xyz_mm"] = list(marker_xyz_mm)

        tvec_m = None
        if det.rvec is not None and det.tvec is not None:
            tvec_flat = det.tvec.flatten()
            tvec_m = (float(tvec_flat[0]), float(tvec_flat[1]), float(tvec_flat[2]))
            entry["tvec_m"] = list(tvec_m)
        entries.append(entry)
        marker_states.append(
            MarkerState(
                id=int(det.id),
                center_px=(int(det.center_px[0]), int(det.center_px[1])),
                estado=estado,
                xyz_mm=marker_xyz_mm,
                tvec_m=tvec_m,
            )
        )
    return entries, marker_states


def build_ball_payload(
    frame: Any,
    ball_center: tuple[int, int] | None,
    ball_radius: float,
    homography: Any,
    source_label: str,
) -> tuple[dict[str, Any] | None, BallState]:
    """Build serializable ball payload and BallState object."""
    if ball_center is None:
        ball_state = BallState(
            pixel=None, xyz_mm=(0.0, 0.0, 0.0), radius_px=0.0, source=source_label
        )
        return None, ball_state

    x_mm, y_mm = HomographyCalibrator.transform_point(ball_center, homography)
    ball_state = BallState(
        pixel=ball_center,
        xyz_mm=(float(x_mm), float(y_mm), 0.0),
        radius_px=float(ball_radius),
        source=source_label,
    )
    payload = {
        "pixel": [int(ball_center[0]), int(ball_center[1])],
        "xyz_mm": [float(x_mm), float(y_mm), 0.0],
        "radius_px": float(ball_radius),
        "source": source_label,
    }
    return payload, ball_state


def process_camera_step(
    frame: Any,
    aruco_reader: ArucoReader,
    homography: Any,
    use_stl_render: bool,
    executor: ThreadPoolExecutor | None,
    enable_ball_detection: bool = True,
) -> FrameStepResult:
    """Process one camera frame and return all output artifacts.

    This is the main vision processing pipeline: detection → extraction → payload building.
    """
    if enable_ball_detection:
        aruco_result, det = detect_aruco_and_ball(frame, aruco_reader, executor)
        source_label = "COLOR"
    else:
        aruco_result = aruco_reader.process_frame(frame, draw=True)
        det = None
        source_label = "PAUSED"

    ball_center, ball_radius = extract_ball_data(det)

    if det:
        draw_ball(aruco_result.frame, det, color=(0, 255, 0), label="BALL-COLOR")

    aruco_reader.update_pin_targets_from_ball(
        aruco_result.detections,
        ball_center_px=ball_center,
        ball_radius_px=ball_radius,
        min_pin_id=MIN_PIN_ID,
    )

    if use_stl_render:
        bolo_count = sum(1 for d in aruco_result.detections if int(d.id) >= MIN_PIN_ID)
    else:
        bolo_count = aruco_reader.draw_virtual_pins(
            aruco_result.frame,
            aruco_result.detections,
            min_pin_id=MIN_PIN_ID,
        )

    aruco_entries, marker_states = build_aruco_entries(
        aruco_reader,
        aruco_result.detections,
        homography,
    )
    _, ball_state = build_ball_payload(
        aruco_result.frame,
        ball_center=ball_center,
        ball_radius=ball_radius,
        homography=homography,
        source_label=source_label,
    )

    return FrameStepResult(
        frame=aruco_result.frame,
        detections=aruco_result.detections,
        ball_center=ball_center,
        ball_radius=ball_radius,
        source_label=source_label,
        bolo_count=bolo_count,
        aruco_entries=aruco_entries,
        ball_payload=None,  # Payload added by renderer
        ball_state=ball_state,
        marker_states=marker_states,
    )
