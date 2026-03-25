from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from Cam_administrator import CameraSource, create_latest_frame_camera

F9_KEYS = {120, 0x780000}


class HomographyCalibrator:
    """Generate and validate chessboard homography calibration."""

    CHESSBOARD_SIZE: tuple[int, int] = (6, 9)
    SQUARE_SIZE_MM: float = 30.0
    REQUIRED_CAPTURES: int = 3
    CALIBRATION_FILE: Path = Path(__file__).resolve().parent / "positions.json"

    @classmethod
    def load_positions_file(cls) -> dict[str, Any] | None:
        """Load calibration file if available and valid."""
        if not cls.CALIBRATION_FILE.exists():
            return None

        try:
            data = cast(
                dict[str, Any], json.loads(cls.CALIBRATION_FILE.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, OSError):
            return None

        if "homography_matrix" not in data or "camera_matrix" not in data:
            return None

        data["homography_matrix"] = np.asarray(data["homography_matrix"], dtype=np.float64)
        data["camera_matrix"] = np.asarray(data["camera_matrix"], dtype=np.float64)
        if "distortion_coefficients" in data:
            data["distortion_coefficients"] = np.asarray(
                data["distortion_coefficients"], dtype=np.float64
            )
        return data

    @classmethod
    def ensure_positions_file(
        cls, camera_source: CameraSource, ask_user: bool = True
    ) -> dict[str, Any] | None:
        """Ensure positions.json exists and return loaded calibration data."""
        existing = cls.load_positions_file()
        if existing is not None:
            return existing

        print(f"Calibration file not found: {cls.CALIBRATION_FILE.name}")
        if ask_user:
            answer = input("Generate calibration now? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                return None

        return cls.generate_positions_file(camera_source)

    @classmethod
    def _build_object_points(cls) -> np.ndarray:
        """Create chessboard 3D points on the Z=0 plane in millimeters."""
        cols, rows = cls.CHESSBOARD_SIZE
        points = np.zeros((rows * cols, 3), np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        points[:, :2] = grid * cls.SQUARE_SIZE_MM
        return points

    @staticmethod
    def _configure_cpu_backend() -> int:
        """Configure CPU multithreading and return the thread count in use."""
        cpu_count = int(cv2.getNumberOfCPUs())
        cpu_threads = max(1, cpu_count - 1)
        try:
            cv2.setNumThreads(cpu_threads)
        except Exception:
            cpu_threads = 1
        return cpu_threads

    @staticmethod
    def _draw_fps(display: np.ndarray, fps: float) -> None:
        """Draw FPS overlay on calibration frame."""
        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    @staticmethod
    def _draw_debug(
        display: np.ndarray,
        cpu_threads: int,
    ) -> None:
        """Draw debug backend details on calibration frame."""
        lines = [
            "DEBUG MODE: ON",
            "Backend: CPU",
            "CPU multithread: True",
            f"CPU threads: {cpu_threads}",
        ]

        y = 92
        for line in lines:
            cv2.putText(
                display,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
            )
            y += 24

    @classmethod
    def generate_positions_file(
        cls, camera_source: CameraSource, cam_reader: Any | None = None
    ) -> dict[str, Any] | None:
        """Interactive chessboard capture and homography generation."""
        cam_reader = create_latest_frame_camera(camera_source)
        cpu_threads = cls._configure_cpu_backend()
        debug_mode = False
        frame_times: deque[float] = deque(maxlen=60)

        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        objp = cls._build_object_points()

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        print("\n=== Homography Calibration ===")
        print(f"Chessboard size: {cls.CHESSBOARD_SIZE[0]}x{cls.CHESSBOARD_SIZE[1]}")
        print(f"Required captures: {cls.REQUIRED_CAPTURES}")
        print("Controls: SPACE = capture, ESC/q = cancel, F9 = toggle debug")
        print(f"Compute backend: CPU multithread ({cpu_threads} threads)")

        last_gray_shape = None

        try:
            while len(image_points) < cls.REQUIRED_CAPTURES:
                loop_start = time.perf_counter()
                frame = cam_reader.read_latest()
                if frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_gray_shape = gray.shape[::-1]

                found, corners = cv2.findChessboardCorners(gray, cls.CHESSBOARD_SIZE)
                display = frame.copy()

                if found:
                    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(display, cls.CHESSBOARD_SIZE, refined, found)
                    message = f"Chessboard detected {len(image_points)}/{cls.REQUIRED_CAPTURES} - press SPACE"
                    color = (0, 200, 0)
                else:
                    refined = None
                    message = f"Find chessboard {len(image_points)}/{cls.REQUIRED_CAPTURES}"
                    color = (0, 0, 255)

                cv2.putText(
                    display,
                    message,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                elapsed = time.perf_counter() - loop_start
                frame_times.append(elapsed)
                avg = sum(frame_times) / len(frame_times) if frame_times else 0.0
                fps = (1.0 / avg) if avg > 0 else 0.0
                cls._draw_fps(display, fps)

                if debug_mode:
                    cls._draw_debug(
                        display,
                        cpu_threads=cpu_threads,
                    )

                cv2.imshow("Homography Calibration", display)
                key = cv2.waitKeyEx(1)

                if key in F9_KEYS:
                    debug_mode = not debug_mode
                    status = "ON" if debug_mode else "OFF"
                    print(f"Calibration debug mode: {status}")
                    if debug_mode:
                        print(
                            "Calibration backend info -> "
                            f"CPU multithread=True, CPU threads={cpu_threads}"
                        )
                    continue

                if key in (27, ord("q"), ord("Q")):
                    print("Calibration cancelled.")
                    return None

                if key == 32 and found and refined is not None:
                    object_points.append(objp.copy())
                    image_points.append(refined)
                    print(f"Capture saved: {len(image_points)}/{cls.REQUIRED_CAPTURES}")

            if not image_points or last_gray_shape is None:
                return None

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
                "chessboard_size": list(cls.CHESSBOARD_SIZE),
                "square_size_mm": cls.SQUARE_SIZE_MM,
                "required_captures": cls.REQUIRED_CAPTURES,
                "reprojection_error": float(reproj_err),
                "inliers": int(mask.sum()) if mask is not None else len(img_all),
            }

            cls.CALIBRATION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Calibration saved to {cls.CALIBRATION_FILE}")
            return cls.load_positions_file()
        finally:
            cam_reader.release()
            cv2.destroyWindow("Homography Calibration")

    @staticmethod
    def transform_point(
        point_px: Sequence[float], homography_matrix: np.ndarray | Sequence[Sequence[float]]
    ) -> tuple[float, float]:
        """Transform one image point into calibrated plane coordinates in millimeters."""
        h = np.asarray(homography_matrix, dtype=np.float64)
        pt = np.array([[[float(point_px[0]), float(point_px[1])]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, h)[0, 0]
        return float(out[0]), float(out[1])
