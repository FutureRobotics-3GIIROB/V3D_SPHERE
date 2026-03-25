"""Ball detection using HSV color range and contour analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallDetectorConfig:
    """HSV and contour settings for color-based ball detection."""

    hsv_lower: tuple[int, int, int] = (8, 60, 120)  # Light orange lower bound
    hsv_upper: tuple[int, int, int] = (25, 220, 255)  # Light orange upper bound
    min_area: int = 500  # Minimum blob area in pixels
    blur_kernel: tuple[int, int] = (11, 11)  # Gaussian blur kernel size
    morph_kernel_size: tuple[int, int] = (9, 9)  # Morphological operations kernel
    morph_iterations: int = 2  # Number of open/close iterations


def _extract_ball_from_mask(
    mask: np.ndarray, config: BallDetectorConfig
) -> dict[str, object] | None:
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


def detect_ball(
    frame: np.ndarray, config: BallDetectorConfig = BallDetectorConfig()
) -> dict[str, object] | None:
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


def draw_ball(
    frame: np.ndarray,
    det: dict[str, object] | None,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "BALL",
) -> None:
    """Draw ball center and radius on frame."""
    if not det:
        return

    center: tuple[int, int] = det["center"]  # type: ignore
    radius = max(2, int(det["radius"]))  # type: ignore
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
