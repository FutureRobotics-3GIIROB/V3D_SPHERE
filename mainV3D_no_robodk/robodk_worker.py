"""RoboDK worker thread for robot control and coordination.

This module is a stub ready for RoboDK integration. It reads from SharedVisionState
to get ball and marker positions without file I/O overhead.

Future enhancements:
- Signal pause_ball_detection when robot is holding the ball
- Send target positions to RoboDK control loop
- Handle collision detection and pin state confirmation
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from vision_state import SharedVisionState


class RoboDKWorker:
    """Coordinates RoboDK robot control with vision system.

    Reads detections from SharedVisionState and controls the robot accordingly.
    """

    def __init__(self, vision_state: SharedVisionState) -> None:
        """Initialize worker with shared vision state reference."""
        self.vision_state = vision_state
        self.thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the RoboDK worker thread."""
        if self.thread is not None and self.thread.is_alive():
            return

        self._running = True
        self.thread = threading.Thread(target=self._run, daemon=False)
        self.thread.start()
        print("RoboDK worker started")

    def stop(self) -> None:
        """Stop the RoboDK worker thread gracefully."""
        self._running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        print("RoboDK worker stopped")

    def _run(self) -> None:
        """Main RoboDK control loop.

        Future implementation will:
        - Move robot to detected ball position
        - Trigger gripper when proximity threshold is met
        - Update pin fall states in simulation
        - Signal pause_ball_detection if robot is holding ball
        """
        print("RoboDK worker loop starting")
        while self._running:
            frame = self.vision_state.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Stub: read frame data without processing
            # (implementation to be added when RoboDK is available)
            time.sleep(0.01)

    def read_ball_position(self) -> Optional[dict[str, object]]:
        """Get current ball position for RoboDK control."""
        ball = self.vision_state.get_ball()
        if ball and ball.pixel:
            return {
                "x_mm": ball.xyz_mm[0],
                "y_mm": ball.xyz_mm[1],
                "z_mm": ball.xyz_mm[2],
                "radius_px": ball.radius_px,
            }
        return None

    def read_pin_positions(self) -> list[dict[str, object]]:
        """Get all detected pin positions and states for RoboDK control."""
        markers = self.vision_state.get_markers()
        return [
            {
                "id": m.id,
                "center_px": m.center_px,
                "estado": m.estado,
                "tvec_m": m.tvec_m,
            }
            for m in markers
        ]

    def pause_ball_detection(self, enabled: bool) -> None:
        """Signal camera to pause/resume ball detection.

        Call with True when robot is holding the ball to avoid false detections.
        Call with False when ball is released to resume monitoring.
        """
        self.vision_state.set_pause_ball_detection(enabled)
