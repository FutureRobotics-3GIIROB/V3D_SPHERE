"""RoboDK worker thread for robot control and pin animation.

This module reads ball/pin data directly from SharedVisionState and updates RoboDK in
real time using MoveJ for the ball robot and pin fall/recover animations.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from robodk.robomath import rotx, roty, rotz, transl
from robodk_helpers import (
    create_or_update_target,
    get_frame,
    get_rdk,
    get_robot,
    get_target,
    move_to,
    set_speed,
)
from vision_state import SharedVisionState

BALL_ROBOT_NAME = "Bola"
BALL_TARGET_NAME = "BolaTrackTarget"
BALL_TARGET_FRAME_NAME = "BallTarget"
UR3E_ROBOT_NAME = "UR3e"
UR3E_TARGET_NAME = "BallApproach"
UR3E_DROP_TARGET_NAME = "soltar"
BALL_TOOL_NAME = "BolaBolos"
UR3E_PREPICK_OFFSET_MM = 100.0
# TCP orientation correction for UR3e prepick target (radians).
# 180 deg around Y fixes the "inverse TCP" orientation reported in RoboDK.
UR3E_TCP_RX_RAD = 0.0
UR3E_TCP_RY_RAD = 3.141592653589793
UR3E_TCP_RZ_RAD = 0.0
PIN_COUNT = 8
PIN_ROBOT_NAME_FMT = "boloCae{index}"
PIN_BASE_NAME_FMT = "boloCaeBase{index}"
# Mapping between camera marker IDs and RoboDK pin indices.
# Default assumes markers are labeled 2..9 while RoboDK pins are 1..8.
PIN_MARKER_TO_ROBODK: dict[int, int] = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
}

PIN_JOINT_UP = [0]
PIN_JOINT_DOWN = [-90]
LOOP_SLEEP_S = 0.02
BALL_MIN_MOVE_MM = 2.0
PIN_MISSING_RECOVER_S = 0.6
UR3E_CARRY_STEPS = 8


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
        self._ur3e_robot: Any | None = None
        self._ur3e_follow_enabled = False
        self._last_ur3e_xyz_mm: tuple[float, float, float] | None = None
        self._pin_robots: dict[int, Any] = {}
        self._pin_bases: dict[int, Any] = {}
        self._last_ball_xyz_mm: tuple[float, float, float] | None = None
        self._last_pin_state: dict[int, str] = {}
        self._last_pin_seen_s: dict[int, float] = {}
        self._pick_drop_requested = False
        self._pick_drop_running = False
        self._pick_drop_lock = threading.Lock()

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
            should_run_pick_drop = False
            with self._pick_drop_lock:
                if self._pick_drop_requested and not self._pick_drop_running:
                    self._pick_drop_requested = False
                    self._pick_drop_running = True
                    should_run_pick_drop = True

            if should_run_pick_drop:
                try:
                    self._execute_pick_and_drop_sequence()
                finally:
                    with self._pick_drop_lock:
                        self._pick_drop_running = False

            frame = self.vision_state.get_frame()
            if frame is None:
                time.sleep(LOOP_SLEEP_S)
                continue

            ball = self.read_ball_position()
            if ball is not None:
                self._update_ball_robot(ball)
                if self._ur3e_follow_enabled:
                    self._update_ur3e_prepick(ball)

            pins = self.read_pin_positions()
            self._update_pins(pins)

            time.sleep(LOOP_SLEEP_S)

    def _initialize_robodk_items(self) -> None:
        """Resolve RoboDK items used by this worker.

        Missing items are tolerated to avoid stopping the whole pipeline.
        """
        self._ball_robot = get_robot(BALL_ROBOT_NAME)
        self._ball_target_frame = get_frame(BALL_TARGET_FRAME_NAME)
        self._ur3e_robot = get_robot(UR3E_ROBOT_NAME)

        if self._ball_robot is not None:
            set_speed(self._ball_robot, 250, 500, 45, 120)
        if self._ur3e_robot is not None:
            set_speed(self._ur3e_robot, 200, 400, 35, 90)

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

    @staticmethod
    def _ur3e_target_pose(x_mm: float, y_mm: float, z_mm: float) -> Any:
        """Build UR3e target pose with corrected TCP orientation."""
        return (
            transl(x_mm, y_mm, z_mm)
            * rotx(UR3E_TCP_RX_RAD)
            * roty(UR3E_TCP_RY_RAD)
            * rotz(UR3E_TCP_RZ_RAD)
        )

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
            frame=self._ball_target_frame,
        )
        move_to(self._ball_robot, self._ball_target, "MoveJ")
        self._last_ball_xyz_mm = xyz

    def _move_ur3e_to_best_effort(self, desired_xyz_mm: tuple[float, float, float]) -> None:
        """Try to move UR3e to target, then fallback to closest interpolated points."""
        if self._ur3e_robot is None:
            return

        desired_pose = self._ur3e_target_pose(
            desired_xyz_mm[0], desired_xyz_mm[1], desired_xyz_mm[2]
        )
        target = create_or_update_target(
            UR3E_TARGET_NAME,
            self._ur3e_robot,
            desired_pose,
            frame=self._ball_target_frame,
        )
        if move_to(self._ur3e_robot, target, "MoveJ"):
            self._last_ur3e_xyz_mm = desired_xyz_mm
            return

        # Fallback: interpolate from current robot absolute position toward desired point.
        try:
            current_pose = self._ur3e_robot.PoseAbs()
            current_pos = current_pose.Pos()
            cx = float(current_pos[0])
            cy = float(current_pos[1])
            cz = float(current_pos[2])
        except Exception:
            return

        for alpha in (0.8, 0.6, 0.4, 0.25, 0.1):
            ix = cx + (desired_xyz_mm[0] - cx) * alpha
            iy = cy + (desired_xyz_mm[1] - cy) * alpha
            iz = cz + (desired_xyz_mm[2] - cz) * alpha
            fallback_pose = self._ur3e_target_pose(ix, iy, iz)
            target = create_or_update_target(
                UR3E_TARGET_NAME,
                self._ur3e_robot,
                fallback_pose,
                frame=self._ball_target_frame,
            )
            if move_to(self._ur3e_robot, target, "MoveJ"):
                self._last_ur3e_xyz_mm = (ix, iy, iz)
                return

    def _move_ur3e_and_ball_to_xyz(self, desired_xyz_mm: tuple[float, float, float]) -> None:
        """Move UR3e toward destination in short interpolated steps."""
        if self._ur3e_robot is None:
            return

        try:
            current_pose = self._ur3e_robot.PoseAbs()
            current_pos = current_pose.Pos()
            cx = float(current_pos[0])
            cy = float(current_pos[1])
            cz = float(current_pos[2])
        except Exception:
            self._move_ur3e_to_best_effort(desired_xyz_mm)
            self._move_ball_visual_to_xyz(desired_xyz_mm[0], desired_xyz_mm[1], 0.0)
            return

        dx = float(desired_xyz_mm[0]) - cx
        dy = float(desired_xyz_mm[1]) - cy
        dz = float(desired_xyz_mm[2]) - cz

        for step_idx in range(1, UR3E_CARRY_STEPS + 1):
            alpha = step_idx / UR3E_CARRY_STEPS
            ix = cx + dx * alpha
            iy = cy + dy * alpha
            iz = cz + dz * alpha
            self._move_ur3e_to_best_effort((ix, iy, iz))
            # Keep Bola robot synchronized with UR3e transport for visual continuity.
            self._move_ball_visual_to_xyz(ix, iy, 0.0)

    @staticmethod
    def _reparent_item_keep_pose(item: Any, new_parent: Any) -> bool:
        """Reparent an item while preserving world pose when possible."""
        if item is None or new_parent is None:
            return False

        for method_name in ("setParentStatic", "setParent"):
            method = getattr(item, method_name, None)
            if method is None:
                continue
            try:
                method(new_parent)
                return True
            except Exception:
                continue
        return False

    def _get_ball_tool(self) -> Any | None:
        """Resolve the shared ball tool item by name."""
        rdk = get_rdk()
        tool = rdk.Item(BALL_TOOL_NAME)
        try:
            if not tool.Valid():
                return None
        except Exception:
            return None
        return tool

    @staticmethod
    def _set_robot_tool(robot: Any, tool: Any) -> bool:
        """Assign tool as active TCP for robot when API supports it."""
        if robot is None or tool is None:
            return False
        set_tool = getattr(robot, "setTool", None)
        if set_tool is None:
            return False
        try:
            set_tool(tool)
            return True
        except Exception:
            return False

    def _attach_ball_tool_to_robot(self, robot: Any) -> bool:
        """Attach BolaBolos tool to the given robot."""
        tool = self._get_ball_tool()
        if tool is None or robot is None:
            return False

        attached = self._reparent_item_keep_pose(tool, robot)
        self._set_robot_tool(robot, tool)
        return attached

    def _update_ur3e_prepick(self, ball: dict[str, float]) -> None:
        """Move UR3e to a prepick point 10 cm above the ball plane."""
        if self._ur3e_robot is None:
            return

        target_xyz = (
            float(ball["x_mm"]),
            float(ball["y_mm"]),
            float(UR3E_PREPICK_OFFSET_MM),
        )
        if self._last_ur3e_xyz_mm is not None:
            if self._distance_mm(target_xyz, self._last_ur3e_xyz_mm) < BALL_MIN_MOVE_MM:
                return
        self._move_ur3e_to_best_effort(target_xyz)

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

    def _set_pin_base_pose_from_xy_mm(self, pin_index: int, x_mm: float, y_mm: float) -> None:
        """Update pin base frame position from shared homography-plane coordinates."""
        base = self._pin_bases.get(pin_index)
        if base is None:
            return
        base.setPose(transl(float(x_mm), float(y_mm), 0.0))

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
            pin_index = PIN_MARKER_TO_ROBODK.get(marker_id)
            if pin_index is None:
                continue

            seen_ids.add(pin_index)
            self._last_pin_seen_s[pin_index] = now_s

            xy_mm_raw = marker.get("xyz_mm")
            if isinstance(xy_mm_raw, tuple | list) and len(xy_mm_raw) >= 2:
                self._set_pin_base_pose_from_xy_mm(
                    pin_index,
                    float(xy_mm_raw[0]),
                    float(xy_mm_raw[1]),
                )

            tvec_raw = marker.get("tvec_m")
            if (
                not isinstance(xy_mm_raw, tuple | list)
                and isinstance(tvec_raw, tuple)
                and len(tvec_raw) >= 3
            ):
                tvec = (float(tvec_raw[0]), float(tvec_raw[1]), float(tvec_raw[2]))
                self._set_pin_base_pose_from_tvec(pin_index, tvec)

            estado = str(marker.get("estado", "up"))
            if estado not in ("up", "down"):
                estado = "up"
            self._animate_pin_state(pin_index, estado)

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

    def _move_ball_visual_to_xyz(self, x_mm: float, y_mm: float, z_mm: float = 0.0) -> None:
        """Move visual ball robot to explicit coordinates for pick/drop visualization."""
        if self._ball_robot is None:
            return
        pose = transl(float(x_mm), float(y_mm), float(z_mm))
        ball_target = create_or_update_target(
            BALL_TARGET_NAME,
            self._ball_robot,
            pose,
            frame=self._ball_target_frame,
        )
        move_to(self._ball_robot, ball_target, "MoveJ")

    def _execute_pick_and_drop_sequence(self) -> None:
        """Run UR3e pick-and-drop sequence while pausing ball tracking."""
        if self._ur3e_robot is None:
            print("UR3e pick/drop aborted: robot UR3e no disponible")
            return

        ball = self.read_ball_position()
        if ball is None:
            print("UR3e pick/drop aborted: pelota no detectada")
            return

        x_mm = float(ball["x_mm"])
        y_mm = float(ball["y_mm"])
        prepick_xyz = (x_mm, y_mm, float(UR3E_PREPICK_OFFSET_MM))
        pick_xyz = (x_mm, y_mm, 0.0)

        previous_follow_state = self._ur3e_follow_enabled
        self._ur3e_follow_enabled = False
        self.pause_ball_detection(True)

        try:
            print("UR3e pick/drop: acercando a prepick")
            self._move_ur3e_to_best_effort(prepick_xyz)

            print("UR3e pick/drop: bajando a pick")
            self._move_ur3e_to_best_effort(pick_xyz)

            print("UR3e pick/drop: acoplando herramienta BolaBolos a UR3e")
            if not self._attach_ball_tool_to_robot(self._ur3e_robot):
                print("UR3e pick/drop: no se pudo acoplar BolaBolos a UR3e")

            # Ensure Bola starts at the grasp point before smooth carry updates.
            self._move_ball_visual_to_xyz(pick_xyz[0], pick_xyz[1], 0.0)

            drop_target = get_target(UR3E_DROP_TARGET_NAME)
            if drop_target is None:
                print("UR3e pick/drop: target 'soltar' no encontrado")
                return

            print("UR3e pick/drop: moviendo suave a 'soltar'")
            drop_pose = drop_target.Pose()
            drop_pos = drop_pose.Pos()
            drop_xyz = (float(drop_pos[0]), float(drop_pos[1]), float(drop_pos[2]))
            self._move_ur3e_and_ball_to_xyz(drop_xyz)
            self._last_ur3e_xyz_mm = None

            print("UR3e pick/drop: posicionando robot Bola en soltar")
            self._move_ball_visual_to_xyz(drop_xyz[0], drop_xyz[1], 0.0)

            print("UR3e pick/drop: devolviendo BolaBolos al robot Bola")
            if not self._attach_ball_tool_to_robot(self._ball_robot):
                print("UR3e pick/drop: no se pudo devolver BolaBolos al robot Bola")
        finally:
            print("UR3e pick/drop: reactivar tracking")
            self.pause_ball_detection(False)
            self._ur3e_follow_enabled = previous_follow_state

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
                "xyz_mm": m.xyz_mm,
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

    def toggle_ur3e_follow(self) -> bool:
        """Toggle UR3e prepick follow mode and return its new state."""
        self._ur3e_follow_enabled = not self._ur3e_follow_enabled
        status = "ON" if self._ur3e_follow_enabled else "OFF"
        print(f"UR3e prepick follow: {status}")
        return self._ur3e_follow_enabled

    def request_pick_and_drop(self) -> bool:
        """Request asynchronous UR3e pick/drop sequence execution.

        Returns False if a sequence is already running.
        """
        with self._pick_drop_lock:
            if self._pick_drop_running:
                print("UR3e pick/drop: ya en ejecucion")
                return False
            self._pick_drop_requested = True
        print("UR3e pick/drop: solicitud recibida")
        return True
