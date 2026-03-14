from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from cv2 import aruco


@dataclass
class ArucoDetection:
    """Single ArUco detection data."""

    id: int
    center_px: Tuple[int, int]
    corners: np.ndarray
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


@dataclass
class ArucoFrameResult:
    """Detection output for one frame."""

    frame: np.ndarray
    detections: List[ArucoDetection] = field(default_factory=list)


class STLMesh:
    """Simple STL mesh container with OpenCV projection render."""

    def __init__(self, triangles: np.ndarray):
        self.triangles = triangles.astype(np.float32)

    @classmethod
    def from_file(cls, stl_path: str) -> "STLMesh":
        path = Path(stl_path)
        if not path.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        raw = path.read_bytes()
        triangles = _parse_stl(raw)

        # Recenter model to keep transformations intuitive.
        center = triangles.reshape(-1, 3).mean(axis=0)
        triangles = triangles - center
        return cls(triangles)


class ArucoObject:
    """Configurable model renderer associated to one marker id."""

    def __init__(self):
        self.stl_mesh: Optional[STLMesh] = None
        self.scale: float = 1.0
        self.offset_tvec = np.zeros((3,), dtype=np.float32)
        self.offset_rvec = np.zeros((3,), dtype=np.float32)
        self.color: Tuple[int, int, int] = (255, 180, 40)

    def render(
        self,
        stl_path: str,
        pose_offsets: Optional[Sequence[Sequence[float]]] = None,
        scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 180, 40),
    ) -> "ArucoObject":
        """Load a model and configure how it should be rendered for this marker."""
        self.stl_mesh = STLMesh.from_file(stl_path)
        self.scale = float(scale)
        self.color = color

        if pose_offsets and len(pose_offsets) == 2:
            self.offset_tvec = np.asarray(pose_offsets[0], dtype=np.float32).reshape(3)
            self.offset_rvec = np.asarray(pose_offsets[1], dtype=np.float32).reshape(3)
        return self

    def draw(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        base_rvec: np.ndarray,
        base_tvec: np.ndarray,
    ) -> None:
        """Project model triangles and draw wireframe over image."""
        if self.stl_mesh is None:
            return

        rvec = np.asarray(base_rvec, dtype=np.float32).reshape(3, 1) + self.offset_rvec.reshape(3, 1)
        tvec = np.asarray(base_tvec, dtype=np.float32).reshape(3, 1) + self.offset_tvec.reshape(3, 1)

        triangles = self.stl_mesh.triangles * self.scale
        for tri in triangles:
            pts_2d, _ = cv2.projectPoints(
                tri,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            poly = pts_2d.reshape(-1, 2).astype(np.int32)
            cv2.polylines(frame, [poly], True, self.color, 1, cv2.LINE_AA)


class ArucoReader:
    """ArUco reader with configurable dictionary and per-id model rendering."""

    DICTIONARY_MAP = {
        "original": aruco.DICT_ARUCO_ORIGINAL,
        "4x4_50": aruco.DICT_4X4_50,
    }

    def __init__(
        self,
        marker_size_m: float = 0.05,
        dictionary_name: str = "original",
        enable_fallback_dictionary: bool = True,
        expected_ids: Optional[Iterable[int]] = None,
    ):
        self.marker_size_m = marker_size_m
        self.dictionary_name = dictionary_name
        self.enable_fallback_dictionary = enable_fallback_dictionary
        self.expected_ids = set(expected_ids) if expected_ids is not None else None

        dict_id = self.DICTIONARY_MAP.get(dictionary_name.lower(), aruco.DICT_ARUCO_ORIGINAL)
        self.dictionary = aruco.getPredefinedDictionary(dict_id)
        self.fallback_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        params = aruco.DetectorParameters()
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.08
        params.minDistanceToBorder = 8
        params.minMarkerDistanceRate = 0.08
        params.maxErroneousBitsInBorderRate = 0.2
        params.errorCorrectionRate = 0.45
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        self.detector = aruco.ArucoDetector(self.dictionary, params)
        self.fallback_detector = aruco.ArucoDetector(self.fallback_dictionary, params)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.objects: Dict[int, ArucoObject] = {}

    def _filter_candidates(
        self,
        corners: List[np.ndarray],
        ids: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Filter out weak/false detections and deduplicate by marker id."""
        if ids is None or len(ids) == 0:
            return [], None

        frame_h, frame_w = frame_shape
        kept: Dict[int, Tuple[float, np.ndarray]] = {}

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if self.expected_ids is not None and marker_id not in self.expected_ids:
                continue

            pts = marker_corners.reshape(4, 2).astype(np.float32)
            area = float(cv2.contourArea(pts))
            if area < 350.0:
                continue

            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            if x <= 4 or y <= 4 or (x + w) >= (frame_w - 4) or (y + h) >= (frame_h - 4):
                continue

            edges = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[3] - pts[2]),
                np.linalg.norm(pts[0] - pts[3]),
            ]
            min_edge = min(edges)
            max_edge = max(edges)
            if min_edge < 14.0:
                continue
            if max_edge / max(min_edge, 1e-6) > 2.2:
                continue

            # Keep strongest area when duplicate IDs appear in same frame.
            current = kept.get(marker_id)
            if current is None or area > current[0]:
                kept[marker_id] = (area, marker_corners)

        if not kept:
            return [], None

        ordered_ids = sorted(kept.keys())
        filtered_corners = [kept[mid][1] for mid in ordered_ids]
        filtered_ids = np.asarray(ordered_ids, dtype=np.int32).reshape(-1, 1)
        return filtered_corners, filtered_ids

    def ensure_object(self, marker_id: int) -> ArucoObject:
        """Get or create editable object configuration for a marker id."""
        if marker_id not in self.objects:
            self.objects[marker_id] = ArucoObject()
        return self.objects[marker_id]

    def set_camera_params(
        self, frame_shape: Tuple[int, int], fx: float = 900.0, fy: float = 900.0
    ) -> None:
        """Set simple camera intrinsics from frame size when no calibration is provided."""
        height, width = frame_shape
        cx = width / 2.0
        cy = height / 2.0
        self.camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float32,
        )

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> ArucoFrameResult:
        """Detect markers, estimate pose and render optional models."""
        output = frame.copy() if draw else frame
        if self.camera_matrix is None:
            self.set_camera_params(frame.shape[:2])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        # Optional fallback for compatibility when marker set is mixed.
        if (ids is None or len(ids) == 0) and self.enable_fallback_dictionary:
            corners, ids, _ = self.fallback_detector.detectMarkers(gray)

        corners, ids = self._filter_candidates(corners, ids, frame.shape[:2])

        result = ArucoFrameResult(frame=output, detections=[])

        if ids is None or len(ids) == 0:
            return result

        if draw:
            aruco.drawDetectedMarkers(output, corners, ids)

        obj_points = np.array(
            [
                [-self.marker_size_m / 2, -self.marker_size_m / 2, 0],
                [self.marker_size_m / 2, -self.marker_size_m / 2, 0],
                [self.marker_size_m / 2, self.marker_size_m / 2, 0],
                [-self.marker_size_m / 2, self.marker_size_m / 2, 0],
            ],
            dtype=np.float32,
        )

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            corner_2d = marker_corners.reshape(4, 2)
            center = corner_2d.mean(axis=0).astype(int)

            det = ArucoDetection(
                id=int(marker_id),
                center_px=(int(center[0]), int(center[1])),
                corners=corner_2d,
            )

            ok, rvec, tvec = cv2.solvePnP(
                obj_points,
                corner_2d.astype(np.float32),
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                det.rvec = rvec
                det.tvec = tvec

                if draw and int(marker_id) in self.objects:
                    self.objects[int(marker_id)].draw(
                        output,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                    )

                if draw:
                    cv2.putText(
                        output,
                        f"ID:{int(marker_id)}",
                        (det.center_px[0] - 20, det.center_px[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            result.detections.append(det)

        return result

    @staticmethod
    def draw_virtual_pins(
        frame: np.ndarray,
        detections: Sequence[ArucoDetection],
        min_pin_id: int = 2,
    ) -> int:
        """Draw virtual bowling pins on top of detected ArUco markers."""
        count = 0
        for det in detections:
            if int(det.id) < int(min_pin_id):
                continue

            corners = det.corners.astype(np.int32)
            center = (int(det.center_px[0]), int(det.center_px[1]))

            # Marker outline as the pin base reference.
            cv2.polylines(frame, [corners], True, (0, 220, 255), 2, cv2.LINE_AA)

            # Lightweight pin icon.
            pin_top = (center[0], center[1] - 26)
            pin_bottom_left = (center[0] - 8, center[1] + 8)
            pin_bottom_right = (center[0] + 8, center[1] + 8)
            triangle = np.array([pin_top, pin_bottom_left, pin_bottom_right], dtype=np.int32)
            cv2.fillConvexPoly(frame, triangle, (255, 120, 0))
            cv2.circle(frame, (center[0], center[1] + 10), 10, (255, 120, 0), 2)

            cv2.putText(
                frame,
                f"BOLO {det.id}",
                (center[0] - 34, center[1] - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
            )
            count += 1

        return count


def _parse_stl(raw_bytes: bytes) -> np.ndarray:
    """Parse STL data (binary or ASCII) and return triangles Nx3x3."""
    if len(raw_bytes) < 84:
        raise ValueError("Invalid STL: file too short")

    if _looks_ascii_stl(raw_bytes):
        return _parse_ascii_stl(raw_bytes.decode("utf-8", errors="ignore"))
    return _parse_binary_stl(raw_bytes)


def _looks_ascii_stl(raw_bytes: bytes) -> bool:
    """Heuristic for ASCII STL detection."""
    head = raw_bytes[:80].decode("utf-8", errors="ignore").strip().lower()
    if not head.startswith("solid"):
        return False
    return b"facet" in raw_bytes[:512]


def _parse_binary_stl(raw_bytes: bytes) -> np.ndarray:
    """Parse binary STL triangles."""
    tri_count = struct.unpack("<I", raw_bytes[80:84])[0]
    expected_size = 84 + tri_count * 50
    if len(raw_bytes) < expected_size:
        raise ValueError("Invalid binary STL: inconsistent triangle count")

    triangles = np.zeros((tri_count, 3, 3), dtype=np.float32)
    offset = 84
    for i in range(tri_count):
        # Skip normal vector (12 bytes), keep only 3 vertices (36 bytes).
        v_start = offset + 12
        chunk = raw_bytes[v_start : v_start + 36]
        values = struct.unpack("<9f", chunk)
        triangles[i] = np.array(values, dtype=np.float32).reshape(3, 3)
        offset += 50
    return triangles


def _parse_ascii_stl(text: str) -> np.ndarray:
    """Parse ASCII STL triangles."""
    verts: List[List[float]] = []
    triangles: List[np.ndarray] = []

    for line in text.splitlines():
        line = line.strip().lower()
        if not line.startswith("vertex"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(verts) == 3:
            triangles.append(np.asarray(verts, dtype=np.float32))
            verts = []

    if not triangles:
        raise ValueError("Invalid ASCII STL: no triangles found")
    return np.stack(triangles, axis=0)
