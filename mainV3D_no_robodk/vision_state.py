"""Thread-safe shared container for vision pipeline state and detections."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BallState:
    """Current ball position and properties."""

    pixel: tuple[int, int] | None = None
    xyz_mm: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    radius_px: float = 0.0
    source: str = "COLOR"
    timestamp: float = field(default_factory=time.time)


@dataclass
class MarkerState:
    """State of a single ArUco marker (pin)."""

    id: int
    center_px: tuple[int, int]
    estado: str  # "up" or "down"
    tvec_m: tuple[float, float, float] | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class FrameState:
    """Snapshot of one complete vision frame."""

    ball: BallState
    markers: list[MarkerState]
    bolo_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class FrameStepResult:
    """Container for one processed camera step (output of vision pipeline)."""

    frame: Any
    detections: list[Any]
    ball_center: tuple[int, int] | None
    ball_radius: float
    source_label: str
    bolo_count: int
    aruco_entries: list[dict[str, Any]]
    ball_payload: dict[str, Any] | None
    ball_state: BallState
    marker_states: list[MarkerState]


class SharedVisionState:
    """Thread-safe container for camera pipeline output.

    Allows camera thread to write detection results and RoboDK thread to read them
    without file I/O overhead or synchronization issues.
    """

    def __init__(self) -> None:
        """Initialize shared state with locks for thread safety."""
        self._lock = threading.RLock()
        self._current_frame: FrameState | None = None
        self._pause_ball_detection = False

    def update_frame(
        self,
        ball: BallState,
        markers: list[MarkerState],
        bolo_count: int,
    ) -> None:
        """Update current frame state (called by camera thread)."""
        with self._lock:
            self._current_frame = FrameState(
                ball=ball,
                markers=markers,
                bolo_count=bolo_count,
            )

    def get_frame(self) -> FrameState | None:
        """Get current frame state (called by RoboDK thread)."""
        with self._lock:
            return self._current_frame

    def get_ball(self) -> BallState | None:
        """Get current ball state if available."""
        with self._lock:
            return self._current_frame.ball if self._current_frame else None

    def get_markers(self) -> list[MarkerState]:
        """Get list of all detected markers."""
        with self._lock:
            return self._current_frame.markers.copy() if self._current_frame else []

    def get_marker_by_id(self, marker_id: int) -> MarkerState | None:
        """Get specific marker by ID, or None if not detected."""
        with self._lock:
            if not self._current_frame:
                return None
            return next((m for m in self._current_frame.markers if m.id == marker_id), None)

    def set_pause_ball_detection(self, enabled: bool) -> None:
        """Signal camera thread to pause/resume ball detection (for future RoboDK integration)."""
        with self._lock:
            self._pause_ball_detection = enabled

    def is_ball_detection_paused(self) -> bool:
        """Check if ball detection should be paused (e.g., robot is holding the ball)."""
        with self._lock:
            return self._pause_ball_detection

    def to_dict(self) -> dict[str, object]:
        """Export current state as dictionary for logging/debugging."""
        with self._lock:
            if not self._current_frame:
                return {"ball": None, "markers": []}
            return {
                "ball": (
                    {
                        "pixel": self._current_frame.ball.pixel,
                        "xyz_mm": self._current_frame.ball.xyz_mm,
                        "radius_px": self._current_frame.ball.radius_px,
                        "source": self._current_frame.ball.source,
                    }
                    if self._current_frame.ball.pixel
                    else None
                ),
                "markers": [
                    {
                        "id": m.id,
                        "center_px": m.center_px,
                        "estado": m.estado,
                        "tvec_m": m.tvec_m,
                    }
                    for m in self._current_frame.markers
                ],
                "bolo_count": self._current_frame.bolo_count,
            }
