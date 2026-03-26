"""Ball detection using HSV color range and contour analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallDetectorConfig:
    """HSV and contour settings for color-based ball detection."""

    hsv_lower: tuple[int, int, int] = (14, 154, 187)  # Light orange lower bound
    hsv_upper: tuple[int, int, int] = (38, 255, 255)  # Light orange upper bound
    min_area: int = 500  # Minimum blob area in pixels
    blur_kernel: tuple[int, int] = (11, 11)  # Gaussian blur kernel size
    morph_kernel_size: tuple[int, int] = (9, 9)  # Morphological operations kernel
    morph_iterations: int = 2  # Number of open/close iterations


def build_hsv_mask(frame: np.ndarray, config: BallDetectorConfig) -> np.ndarray:
    """Build HSV binary mask for the configured color range."""
    blurred = cv2.GaussianBlur(frame, config.blur_kernel, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array(config.hsv_lower, dtype=np.uint8)
    upper = np.array(config.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iterations)
    return mask


class HSVCalibrationUI:
    """Interactive HSV calibration controls and inline preview composition."""

    def __init__(
        self,
        main_window_name: str,
        initial_config: BallDetectorConfig | None = None,
    ):
        self.window_name = main_window_name
        self._enabled = False
        self._trackbars_created = False
        self._base = initial_config or BallDetectorConfig()

    @property
    def enabled(self) -> bool:
        """Return whether calibration windows are visible."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable calibration mode in the main window."""
        if enabled and not self._enabled:
            self._ensure_trackbars()
            self._enabled = True
            return
        if (not enabled) and self._enabled:
            self._enabled = False
            self._remove_trackbars()

    def toggle(self) -> bool:
        """Toggle calibration UI state and return new state."""
        self.set_enabled(not self._enabled)
        return self._enabled

    def _ensure_trackbars(self) -> None:
        if self._trackbars_created:
            return

        def _noop(_: int) -> None:
            return

        cv2.createTrackbar("H min", self.window_name, int(self._base.hsv_lower[0]), 179, _noop)
        cv2.createTrackbar("S min", self.window_name, int(self._base.hsv_lower[1]), 255, _noop)
        cv2.createTrackbar("V min", self.window_name, int(self._base.hsv_lower[2]), 255, _noop)
        cv2.createTrackbar("H max", self.window_name, int(self._base.hsv_upper[0]), 179, _noop)
        cv2.createTrackbar("S max", self.window_name, int(self._base.hsv_upper[1]), 255, _noop)
        cv2.createTrackbar("V max", self.window_name, int(self._base.hsv_upper[2]), 255, _noop)
        cv2.createTrackbar("Min area", self.window_name, int(self._base.min_area), 20000, _noop)
        self._trackbars_created = True

    def _remove_trackbars(self) -> None:
        """Remove trackbars by recreating the main window."""
        if not self._trackbars_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._trackbars_created = False

    def get_config(self) -> BallDetectorConfig:
        """Return current detector config from UI (or defaults if disabled)."""
        if not self._enabled:
            return BallDetectorConfig(
                hsv_lower=self._base.hsv_lower,
                hsv_upper=self._base.hsv_upper,
                min_area=self._base.min_area,
                blur_kernel=self._base.blur_kernel,
                morph_kernel_size=self._base.morph_kernel_size,
                morph_iterations=self._base.morph_iterations,
            )

        h_min = int(cv2.getTrackbarPos("H min", self.window_name))
        s_min = int(cv2.getTrackbarPos("S min", self.window_name))
        v_min = int(cv2.getTrackbarPos("V min", self.window_name))
        h_max = int(cv2.getTrackbarPos("H max", self.window_name))
        s_max = int(cv2.getTrackbarPos("S max", self.window_name))
        v_max = int(cv2.getTrackbarPos("V max", self.window_name))
        min_area = int(cv2.getTrackbarPos("Min area", self.window_name))

        lower = (
            min(h_min, h_max),
            min(s_min, s_max),
            min(v_min, v_max),
        )
        upper = (
            max(h_min, h_max),
            max(s_min, s_max),
            max(v_min, v_max),
        )

        self._base = BallDetectorConfig(
            hsv_lower=lower,
            hsv_upper=upper,
            min_area=max(1, min_area),
            blur_kernel=self._base.blur_kernel,
            morph_kernel_size=self._base.morph_kernel_size,
            morph_iterations=self._base.morph_iterations,
        )
        return self._base

    def build_calibration_view(
        self,
        source_frame: np.ndarray,
        fallback_frame: np.ndarray,
        config: BallDetectorConfig,
    ) -> np.ndarray:
        """Return composed inline calibration view in a single window."""
        if not self._enabled:
            return fallback_frame

        mask = build_hsv_mask(source_frame, config)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked = cv2.bitwise_and(source_frame, source_frame, mask=mask)

        height, width = source_frame.shape[:2]
        panel_width = max(1, width // 3)

        original_panel = cv2.resize(source_frame, (panel_width, height))
        mask_panel = cv2.resize(mask_bgr, (panel_width, height), interpolation=cv2.INTER_NEAREST)
        masked_panel = cv2.resize(masked, (panel_width, height))

        composed = np.hstack([original_panel, mask_panel, masked_panel])
        cv2.putText(
            composed,
            "Original",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            composed,
            "Mascara",
            (panel_width + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            composed,
            "Mascara aplicada",
            (2 * panel_width + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return composed


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
    mask = build_hsv_mask(frame, config)

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
