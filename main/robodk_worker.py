"""RoboDK worker thread for robot control and pin animation.

This module reads ball/pin data directly from SharedVisionState and updates RoboDK in
real time using MoveJ for the ball robot and pin fall/recover animations.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from robodk.robomath import transl
from robodk_helpers import create_or_update_target, get_frame, get_robot, move_to, set_speed
from vision_state import SharedVisionState

BALL_ROBOT_NAME = "Bola"
BALL_TARGET_NAME = "BallApproach"
BALL_TARGET_FRAME_NAME = "BallTarget"
PIN_COUNT = 8
PIN_ROBOT_NAME_FMT = "boloCae{index}"
PIN_BASE_NAME_FMT = "boloCaeBase{index}"

PIN_JOINT_UP = [0]
PIN_JOINT_DOWN = [-90]
LOOP_SLEEP_S = 0.02
BALL_MIN_MOVE_MM = 2.0
PIN_MISSING_RECOVER_S = 0.6


class RoboDKWorker:
    """Coordinates RoboDK robot control with vision system.

    Reads detections from SharedVisionState and controls the robot accordingly.
    """

    def __init__(self, vision_state: SharedVisionState) -> None:
        """Initialize worker with shared vision state reference."""
        self.vision_state = vision_state
        self.thread: threading.Thread | None = None
        self._running = False
        self._ball_robot: Any | None = None
        self._ball_target: Any | None = None
        self._ball_target_frame: Any | None = None
        self._pin_robots: dict[int, Any] = {}
        self._pin_bases: dict[int, Any] = {}
        self._last_ball_xyz_mm: tuple[float, float, float] | None = None
        self._last_pin_state: dict[int, str] = {}
        self._last_pin_seen_s: dict[int, float] = {}

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

        Continuously reads shared vision state and updates RoboDK objects:
        - Ball robot "Bola" follows detected ball via MoveJ to cartesian target.
        - Pin bases "boloCaeBaseN" are repositioned from marker tvec data.
        - Pin robots "boloCaeN" animate to 90 (down) / 0 (up) on estado changes.
        """
        print("RoboDK worker loop starting")
        self._initialize_robodk_items()

        while self._running:
            frame = self.vision_state.get_frame()
            if frame is None:
                time.sleep(LOOP_SLEEP_S)
                continue

            ball = self.read_ball_position()
            if ball is not None:
                self._update_ball_robot(ball)

            pins = self.read_pin_positions()
            self._update_pins(pins)

            time.sleep(LOOP_SLEEP_S)

    def _initialize_robodk_items(self) -> None:
        """Resolve RoboDK items used by this worker.

        Missing items are tolerated to avoid stopping the whole pipeline.
        """
        self._ball_robot = get_robot(BALL_ROBOT_NAME)
        self._ball_target_frame = get_frame(BALL_TARGET_FRAME_NAME)

        if self._ball_robot is not None:
            set_speed(self._ball_robot, 250, 500, 45, 120)

        for index in range(1, PIN_COUNT + 1):
            self._pin_robots[index] = get_robot(PIN_ROBOT_NAME_FMT.format(index=index))
            self._pin_bases[index] = get_frame(PIN_BASE_NAME_FMT.format(index=index))
            self._last_pin_state[index] = "up"
            self._last_pin_seen_s[index] = 0.0

        print("RoboDK items initialized")

    @staticmethod
    def _distance_mm(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        """Compute Euclidean distance in mm between two 3D points."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)

    def _update_ball_robot(self, ball: dict[str, float]) -> None:
        """Move ball robot with MoveJ toward latest detected ball position."""
        if self._ball_robot is None:
            return

        x_mm = ball["x_mm"]
        y_mm = ball["y_mm"]
        z_mm = 0.0
        xyz = (x_mm, y_mm, z_mm)

        if self._last_ball_xyz_mm is not None:
            if self._distance_mm(xyz, self._last_ball_xyz_mm) < BALL_MIN_MOVE_MM:
                return

        pose = transl(x_mm, y_mm, z_mm)
        self._ball_target = create_or_update_target(
            BALL_TARGET_NAME,
            self._ball_robot,
            pose,
        )
        move_to(self._ball_robot, self._ball_target, "MoveJ")
        self._last_ball_xyz_mm = xyz

    def _set_pin_base_pose_from_tvec(
        self, pin_index: int, tvec_m: tuple[float, float, float]
    ) -> None:
        """Update pin base frame position from marker translation vector (meters)."""
        base = self._pin_bases.get(pin_index)
        if base is None:
            return

        x_mm = float(tvec_m[0]) * 1000.0
        y_mm = float(tvec_m[1]) * 1000.0
        z_mm = 0.0
        base.setPose(transl(x_mm, y_mm, z_mm))

    def _animate_pin_state(self, pin_index: int, estado: str) -> None:
        """Trigger pin fall/recover animation when state changes.

        - estado=down -> MoveJ to 90
        - estado=up   -> MoveJ to 0
        """
        robot = self._pin_robots.get(pin_index)
        if robot is None:
            return

        last_state = self._last_pin_state.get(pin_index)
        if last_state == estado:
            return

        if estado == "down":
            move_to(robot, PIN_JOINT_DOWN, "MoveJ")
        else:
            move_to(robot, PIN_JOINT_UP, "MoveJ")

        self._last_pin_state[pin_index] = estado

    def _update_pins(self, pins: list[dict[str, object]]) -> None:
        """Update pin bases and trigger pin animations from marker data."""
        now_s = time.perf_counter()
        seen_ids: set[int] = set()

        for marker in pins:
            marker_id_raw = marker.get("id")
            if not isinstance(marker_id_raw, int):
                continue
            marker_id = marker_id_raw
            if marker_id < 1 or marker_id > PIN_COUNT:
                continue

            seen_ids.add(marker_id)
            self._last_pin_seen_s[marker_id] = now_s

            tvec_raw = marker.get("tvec_m")
            if isinstance(tvec_raw, tuple) and len(tvec_raw) >= 3:
                tvec = (float(tvec_raw[0]), float(tvec_raw[1]), float(tvec_raw[2]))
                self._set_pin_base_pose_from_tvec(marker_id, tvec)

            estado = str(marker.get("estado", "up"))
            if estado not in ("up", "down"):
                estado = "up"
            self._animate_pin_state(marker_id, estado)

        # Recovery path: if a pin was down but marker/state gets lost after rotation,
        # automatically return it to "up" after a short timeout.
        for pin_index in range(1, PIN_COUNT + 1):
            if pin_index in seen_ids:
                continue
            if self._last_pin_state.get(pin_index) != "down":
                continue

            last_seen = self._last_pin_seen_s.get(pin_index, 0.0)
            if now_s - last_seen >= PIN_MISSING_RECOVER_S:
                self._animate_pin_state(pin_index, "up")

    def read_ball_position(self) -> dict[str, float] | None:
        """Get current ball position for RoboDK control."""
        ball = self.vision_state.get_ball()
        if ball and ball.pixel:
            return {
                "x_mm": float(ball.xyz_mm[0]),
                "y_mm": float(ball.xyz_mm[1]),
                "z_mm": float(ball.xyz_mm[2]),
                "radius_px": float(ball.radius_px),
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
