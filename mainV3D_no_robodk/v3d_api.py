"""
v3d_api.py — Simple API to query live ball and ArUco positions.

The main pipeline (main_tester.py) writes ball_position_output.json every
frame.  These helpers read that file so any external script can poll the
latest state without touching OpenCV.

Usage example
-------------
    from v3d_api import get_posicion_bola, get_posicion_aruco

    ball = get_posicion_bola()
    # {'pixel': [320, 240], 'xyz_mm': [150.3, 80.1, 0.0], 'radius_px': 22.5, 'source': 'COLOR'}
    # or None if no ball detected

    marker = get_posicion_aruco(3)
    # {'id': 3, 'center_px': [520, 200], 'estado': 'up', 'tvec_m': [...]}
    # or None if that id is not currently visible
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_OUTPUT_FILE = Path(__file__).resolve().parent / "ball_position_output.json"


def _read_snapshot() -> dict:
    """Read the latest JSON snapshot written by the pipeline."""
    try:
        return json.loads(_OUTPUT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_posicion_bola() -> Optional[dict]:
    """Return the latest ball position, or None if no ball is detected.

    Returned dict keys:
        pixel     — [x, y] in image pixels
        xyz_mm    — [x, y, z] in millimetres (z is always 0.0)
        radius_px — detected radius in pixels
        source    — detection method (always 'COLOR')
    """
    return _read_snapshot().get("ball")


def get_posicion_aruco(marker_id: int) -> Optional[dict]:
    """Return the latest data for a specific ArUco marker, or None if not found.

    Parameters
    ----------
    marker_id : int
        The ArUco marker ID to look up (0–10).

    Returned dict keys:
        id        — marker id
        center_px — [x, y] centre in image pixels
        estado    — 'up' (standing) or 'down' (knocked over)
        tvec_m    — [x, y, z] translation in metres (present when pose is known)
    """
    markers = _read_snapshot().get("aruco_markers", [])
    for entry in markers:
        if entry.get("id") == int(marker_id):
            return entry
    return None


def get_todos_los_aruco() -> list:
    """Return a list of all ArUco markers visible in the last frame."""
    return _read_snapshot().get("aruco_markers", [])
