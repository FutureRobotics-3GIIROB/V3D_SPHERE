from __future__ import annotations

import os
import sys
from pathlib import Path
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Union

# Keep OpenCV native logger quiet before importing cv2.
# Force quiet mode to suppress backend probing spam on Windows.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_VIDEOCAPTURE_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
import cv2


CAMERA_CONFIG_FILE = Path(__file__).resolve().parent / ".camera_source.txt"
DEFAULT_REMOTE_URL = "http://127.0.0.1:8080/video"
AUTO_PROBE_CAMERAS = True
SHOW_PROBED_CAMERAS = False


CameraSource = Union[int, str]


@contextmanager
def _suppress_stderr():
    """Temporarily suppress native stderr output (OpenCV backend warnings)."""
    devnull = open(os.devnull, "w", encoding="utf-8")
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 2)
        try:
            old_stream = sys.stderr
            sys.stderr = devnull
        except Exception:
            old_stream = None
        yield
    finally:
        if 'old_stream' in locals() and old_stream is not None:
            sys.stderr = old_stream
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        devnull.close()


def _camera_backends_for_os() -> List[int]:
    """Return preferred backends for local camera indexes."""
    if os.name == "nt":
        candidates = []
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"):
            candidates.append(cv2.CAP_DSHOW)
        return candidates if candidates else [cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _open_local_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Open local camera index with controlled backend fallback."""
    for backend in _camera_backends_for_os():
        with _suppress_stderr():
            cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def _set_opencv_quiet_mode() -> None:
    """Reduce noisy OpenCV logs in console output."""
    try:
        if hasattr(cv2, "LOG_LEVEL_SILENT"):
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
        else:
            cv2.setLogLevel(0)
    except Exception:
        pass


_set_opencv_quiet_mode()


class LatestFrameCamera:
    """Background camera reader that always keeps only the most recent frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame = None

    def start(self) -> None:
        """Start reader thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Continuously read camera and discard old queued frames."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame

    def read_latest(self):
        """Return newest frame only, or None if no frame is available yet."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def release(self) -> None:
        """Stop thread and release camera resource."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._cap.release()


def list_local_cameras(max_devices: int = 8) -> List[int]:
    """Probe local camera indexes and return available devices."""
    available: List[int] = []
    for index in range(max_devices):
        cap = _open_local_camera(index)
        if cap is None:
            continue
        ok, _ = cap.read()
        if ok:
            available.append(index)
        cap.release()
    return available


def load_last_camera_source() -> Optional[str]:
    """Load the last user camera source from disk."""
    if not CAMERA_CONFIG_FILE.exists():
        return None
    try:
        value = CAMERA_CONFIG_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if value else None


def save_last_camera_source(source: CameraSource) -> None:
    """Persist camera source so it can be reused later."""
    try:
        CAMERA_CONFIG_FILE.write_text(str(source), encoding="utf-8")
    except OSError:
        pass


def normalize_camera_source(value: str) -> CameraSource:
    """Normalize input camera source to local index or full URL."""
    clean = value.strip()
    if clean.isdigit():
        return int(clean)
    if clean.startswith("http://") or clean.startswith("https://"):
        return clean
    return f"http://{clean}/video"


def prompt_default_camera() -> CameraSource:
    """Ask user for default camera source via console."""
    local_cameras: List[int] = []
    if AUTO_PROBE_CAMERAS:
        local_cameras = list_local_cameras()
    last_source = load_last_camera_source() or DEFAULT_REMOTE_URL

    print("\n=== Camera Selection ===")
    if AUTO_PROBE_CAMERAS:
        if SHOW_PROBED_CAMERAS:
            print("Local cameras detected:")
            if local_cameras:
                for cam_index in local_cameras:
                    print(f"  - {cam_index}")
            else:
                print("  - none")
        else:
            print("Local camera scan: enabled (silent)")
    else:
        print("Local camera scan: disabled (type 'scan' to run manual scan)")

    print(f"Last remote source: {last_source}")
    print("Type camera index (e.g. 0) or IP/URL (e.g. 192.168.0.10:8080 or http://...)")

    user_input = input("Camera source [Enter to reuse last]: ").strip()
    if user_input.lower() == "scan":
        local_cameras = list_local_cameras()
        print("Local cameras detected:")
        if local_cameras:
            for cam_index in local_cameras:
                print(f"  - {cam_index}")
        else:
            print("  - none")
        user_input = input("Camera source [Enter to reuse last]: ").strip()

    if not user_input:
        source = normalize_camera_source(last_source)
    else:
        source = normalize_camera_source(user_input)

    save_last_camera_source(source)
    return source


def create_camera(
    source: CameraSource,
    width: Optional[int] = 1280,
    height: Optional[int] = 720,
    buffer_size: int = 1,
) -> cv2.VideoCapture:
    """Create and configure a cv2.VideoCapture object."""
    if isinstance(source, int):
        cap = _open_local_camera(source)
        if cap is None:
            raise RuntimeError(f"Unable to open local camera index: {source}")
    else:
        cap = cv2.VideoCapture(source)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {source}")
    return cap


def create_latest_frame_camera(
    source: CameraSource,
    width: Optional[int] = 1280,
    height: Optional[int] = 720,
    buffer_size: int = 1,
) -> LatestFrameCamera:
    """Create a latest-frame-only camera reader to minimize input lag."""
    cap = create_camera(source, width=width, height=height, buffer_size=buffer_size)
    reader = LatestFrameCamera(cap)
    reader.start()
    return reader


def read_frame(cap: cv2.VideoCapture):
    """Read one frame or return None if read fails."""
    ok, frame = cap.read()
    if not ok:
        return None
    return frame
