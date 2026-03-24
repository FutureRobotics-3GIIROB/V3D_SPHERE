"""Vision pipeline rendering: overlay drawing and display management."""

from __future__ import annotations

from typing import Any

import cv2


def draw_performance_overlay(frame: Any, fps: float) -> None:
    """Draw FPS info on every frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def draw_debug_overlay(
    frame: Any,
    use_multithread: bool,
    source_label: str,
) -> None:
    """Draw runtime debug details while debug mode is active."""
    debug_lines = [
        "DEBUG MODE: ON",
        "Backend: CPU",
        f"Multithread CPU: {use_multithread}",
        f"Ball source: {source_label}",
    ]
    y = 88
    for line in debug_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
        y += 24


def draw_debug_aruco_markers(frame: Any, detections: list[Any]) -> None:
    """Draw square and marker id for each ArUco while debug mode is active."""
    for det in detections:
        corners = det.corners.astype(int)
        x_min = int(corners[:, 0].min())
        y_min = int(corners[:, 1].min())
        x_max = int(corners[:, 0].max())
        y_max = int(corners[:, 1].max())

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"ARUCO {det.id}",
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def draw_bolo_overlay(frame: Any, bolo_count: int, use_stl_render: bool) -> None:
    """Draw pin count summary overlay."""
    cv2.putText(
        frame,
        f"Bolos by ArUco: {bolo_count} [{'STL' if use_stl_render else 'VIRTUAL'}]",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 255),
        2,
    )


def draw_ball_position(
    frame: Any,
    ball_center: tuple[int, int],
    ball_radius: float,
    x_mm: float,
    y_mm: float,
    source_label: str,
) -> None:
    """Draw ball position and coordinates on frame."""
    cv2.putText(
        frame,
        f"X:{x_mm:.1f} Y:{y_mm:.1f} Z:0.0 [{source_label}]",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
