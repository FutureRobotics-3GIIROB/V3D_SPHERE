from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BallDetectorConfig:
    """HSV and contour settings for color-based ball detection."""

    # Light orange target range in HSV (OpenCV scale: H 0-179).
    hsv_lower: Tuple[int, int, int] = (8, 60, 120)
    hsv_upper: Tuple[int, int, int] = (25, 220, 255)
    min_area: int = 500
    blur_kernel: Tuple[int, int] = (11, 11)
    morph_kernel_size: Tuple[int, int] = (9, 9)
    morph_iterations: int = 2


def _extract_ball_from_mask(mask: np.ndarray, config: BallDetectorConfig) -> Optional[Dict[str, object]]:
    """Extract largest valid blob from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < config.min_area:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    x, y, w, h = cv2.boundingRect(largest)
    return {
        "center": (int(cx), int(cy)),
        "radius": int(radius),
        "bbox": (int(x), int(y), int(w), int(h)),
        "mask": mask,
    }


def detect_ball(frame: np.ndarray, config: BallDetectorConfig = BallDetectorConfig()) -> Optional[Dict[str, object]]:
    """Detect the largest ball-like blob in HSV range."""
    blurred = cv2.GaussianBlur(frame, config.blur_kernel, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array(config.hsv_lower, dtype=np.uint8)
    upper = np.array(config.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iterations)

    return _extract_ball_from_mask(mask, config)


def detect_ball_gpu(
    frame: np.ndarray,
    config: BallDetectorConfig = BallDetectorConfig(),
) -> Optional[Dict[str, object]]:
    """Try CUDA acceleration for ball detection, fallback to CPU if unavailable."""
    try:
        if not hasattr(cv2, "cuda"):
            return detect_ball(frame, config)
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return detect_ball(frame, config)

        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        gaussian = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3,
            cv2.CV_8UC3,
            config.blur_kernel,
            0,
        )
        gpu_blurred = gaussian.apply(gpu_frame)
        gpu_hsv = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2HSV)

        lower = np.array(config.hsv_lower, dtype=np.uint8)
        upper = np.array(config.hsv_upper, dtype=np.uint8)
        gpu_mask = cv2.cuda.inRange(gpu_hsv, lower, upper)
        mask = gpu_mask.download()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iterations)
        return _extract_ball_from_mask(mask, config)
    except Exception:
        return detect_ball(frame, config)


def draw_ball(
    frame: np.ndarray,
    det: Optional[Dict[str, object]],
    color: Tuple[int, int, int] = (0, 255, 0),
    label: str = "BALL",
) -> None:
    """Draw ball center and radius on frame."""
    if not det:
        return

    center = det["center"]
    radius = max(2, int(det["radius"]))
    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, 3, color, -1)
    cv2.putText(
        frame,
        label,
        (center[0] + 8, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
