"""Local RoboDK helper functions used by the main package.

These helpers were copied/adapted from legacy utilities so this package can run
independently without external RoboDK helper modules.
"""

from __future__ import annotations

import builtins
from typing import Any

from robodk.robolink import ITEM_TYPE_FRAME, ITEM_TYPE_ROBOT, ITEM_TYPE_TARGET, Robolink
from robodk.robomath import transl

PRINT_CONSOLE = False
SHOW_ERROR = True


def get_rdk() -> Robolink:
    """Create a RoboDK API connection object."""
    return Robolink()


def show_message(message: str, popup: bool = True) -> None:
    """Display a message in RoboDK or console, depending on configuration."""
    if PRINT_CONSOLE:
        builtins.print(message)
    else:
        get_rdk().ShowMessage(message, popup)


def add_frame(name: str, pose_xyz: tuple[float, float, float] | None = None) -> Any:
    """Create a reference frame, optionally setting XYZ position in mm."""
    rdk = get_rdk()
    frame = rdk.AddFrame(name)
    if pose_xyz is None:
        pose = transl(0, 0, 0)
    else:
        pose = transl(float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2]))
    frame.setPose(pose)
    return frame


def get_robot(name: str) -> Any | None:
    """Get robot item by name, returning None if missing."""
    rdk = get_rdk()
    robot = rdk.Item(name, ITEM_TYPE_ROBOT)
    if not robot.Valid():
        show_message(f"Error: Robot {name} no encontrado.", SHOW_ERROR)
        return None
    return robot


def get_frame(name: str) -> Any | None:
    """Get frame by name, creating it if not found."""
    rdk = get_rdk()
    frame = rdk.Item(name, ITEM_TYPE_FRAME)
    if not frame.Valid():
        show_message(f"Error: Sistema de referencia {name} no encontrado, creandolo", SHOW_ERROR)
        return add_frame(name)
    return frame


def create_or_update_target(name: str, robot: Any, pose: Any) -> Any:
    """Create/reuse a target associated to robot and update its pose."""
    rdk = get_rdk()
    target = rdk.Item(name, ITEM_TYPE_TARGET)
    if not target.Valid():
        parent: Any = robot.Parent() if robot is not None else None
        target = rdk.AddTarget(name, parent, robot)
    target.setAsCartesianTarget()
    target.setPose(pose)
    return target


def set_speed(
    robot: Any,
    linear_speed: float,
    linear_accel: float,
    angular_speed: float,
    angular_accel: float,
) -> bool:
    """Set robot speed/acceleration parameters."""
    if robot is None:
        return False
    robot.setSpeed(linear_speed, angular_speed, linear_accel, angular_accel)
    return True


def move_to(robot: Any, obj: Any, move_type: str = "MoveJ") -> bool:
    """Execute MoveJ/MoveL to target object, pose, or joint list."""
    if robot is None or obj is None:
        return False

    if move_type == "MoveJ":
        robot.MoveJ(obj)
    elif move_type == "MoveL":
        robot.MoveL(obj)
    else:
        return False
    return True
