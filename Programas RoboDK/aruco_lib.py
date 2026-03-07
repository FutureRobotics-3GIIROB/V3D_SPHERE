"""
aruco_lib.py - Librería para detección de ArUcos y gestión de bolos

Este módulo proporciona:
- Detección de ArUcos con multihilo para evitar lag
- Sistema de coordenadas con QR como referencia del mundo
- Gestión de bolos (ArUco ID >= 2) que se pueden tumbar
- Renderizado opcional de ejes y objetos STL

Uso:
    from aruco_lib import ArucoTracker, BolosManager
    
    tracker = ArucoTracker(camera_source="http://ip:port/video")
    tracker.start()
    
    # En tu loop principal:
    frame, detections = tracker.get_latest_frame()
"""

import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from collections import deque
import warnings

# Suprimir warnings de OpenCV
warnings.filterwarnings("ignore", category=UserWarning)
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

CONFIG_FILE = Path(__file__).parent / ".aruco_config.txt"
DEFAULT_CAMERA_URL = "http://10.161.249.237:8080/video"

# IDs especiales
QR_WORLD_REFERENCE_ID = 0  # QR/ArUco 0 = referencia del mundo
BALL_ARUCO_ID = 1          # ArUco 1 = bola
MIN_BOLO_ID = 2            # ArUco >= 2 = bolos


# ============================================================================
# DATACLASSES PARA DATOS
# ============================================================================

@dataclass
class ArucoDetection:
    """Representa un ArUco detectado."""
    id: int
    corners: np.ndarray
    center_px: Tuple[int, int]
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    world_position: Optional[Tuple[float, float, float]] = None


@dataclass
class Bolo:
    """Representa un bolo en el juego."""
    aruco_id: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_standing: bool = True
    last_seen: float = 0.0
    hit_count: int = 0


@dataclass
class FrameData:
    """Datos de un frame procesado."""
    frame: np.ndarray
    timestamp: float
    aruco_detections: List[ArucoDetection] = field(default_factory=list)
    qr_detections: List[dict] = field(default_factory=list)
    world_reference_found: bool = False


# ============================================================================
# GESTIÓN DE CONFIGURACIÓN
# ============================================================================

def load_camera_config() -> str:
    """Carga la última URL de cámara usada."""
    if CONFIG_FILE.exists():
        try:
            content = CONFIG_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
        except OSError:
            pass
    return DEFAULT_CAMERA_URL


def save_camera_config(camera_url: str):
    """Guarda la URL de cámara."""
    try:
        CONFIG_FILE.write_text(camera_url, encoding="utf-8")
    except OSError:
        pass


def normalize_camera_url(ip_or_url: str) -> str:
    """Normaliza una IP o URL a formato completo."""
    value = ip_or_url.strip()
    if value.isdigit():
        return int(value)  # Cámara local por índice
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"http://{value}/video"


def list_available_cameras(max_devices: int = 6) -> List[int]:
    """Lista cámaras locales disponibles."""
    available = []
    for index in range(max_devices):
        try:
            # En Windows usamos CAP_DSHOW para evitar warnings
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(index)
            cap.release()
        except Exception:
            pass
    return available


# ============================================================================
# GESTOR DE BOLOS
# ============================================================================

class BolosManager:
    """Gestiona el estado de los bolos en el juego."""
    
    def __init__(self, collision_threshold: float = 0.05):
        """
        Args:
            collision_threshold: Distancia mínima para detectar colisión (metros)
        """
        self.bolos: Dict[int, Bolo] = {}
        self.collision_threshold = collision_threshold
        self._lock = threading.Lock()
        
    def update_bolo(self, aruco_id: int, position: Tuple[float, float, float]):
        """Actualiza o crea un bolo con la posición detectada."""
        if aruco_id < MIN_BOLO_ID:
            return  # No es un bolo
            
        with self._lock:
            if aruco_id not in self.bolos:
                self.bolos[aruco_id] = Bolo(
                    aruco_id=aruco_id,
                    position=position,
                    is_standing=True,
                    last_seen=time.time()
                )
            else:
                bolo = self.bolos[aruco_id]
                bolo.position = position
                bolo.last_seen = time.time()
    
    def check_ball_collision(self, ball_position: Tuple[float, float, float]) -> List[int]:
        """
        Verifica si la bola colisiona con algún bolo.
        
        Returns:
            Lista de IDs de bolos tumbados
        """
        knocked_down = []
        
        with self._lock:
            for bolo_id, bolo in self.bolos.items():
                if not bolo.is_standing:
                    continue
                    
                # Calcular distancia 2D (X, Y) - ignoramos Z para colisión cenital
                dist = np.sqrt(
                    (ball_position[0] - bolo.position[0]) ** 2 +
                    (ball_position[1] - bolo.position[1]) ** 2
                )
                
                if dist < self.collision_threshold:
                    bolo.is_standing = False
                    bolo.hit_count += 1
                    knocked_down.append(bolo_id)
        
        return knocked_down
    
    def reset_all_bolos(self):
        """Resetea todos los bolos a posición de pie."""
        with self._lock:
            for bolo in self.bolos.values():
                bolo.is_standing = True
    
    def reset_bolo(self, aruco_id: int):
        """Resetea un bolo específico."""
        with self._lock:
            if aruco_id in self.bolos:
                self.bolos[aruco_id].is_standing = True
    
    def get_bolos_status(self) -> Dict[int, dict]:
        """Retorna el estado de todos los bolos."""
        with self._lock:
            return {
                bolo_id: {
                    "position": bolo.position,
                    "is_standing": bolo.is_standing,
                    "hit_count": bolo.hit_count
                }
                for bolo_id, bolo in self.bolos.items()
            }
    
    def get_standing_count(self) -> int:
        """Retorna el número de bolos de pie."""
        with self._lock:
            return sum(1 for b in self.bolos.values() if b.is_standing)
    
    def get_knocked_count(self) -> int:
        """Retorna el número de bolos tumbados."""
        with self._lock:
            return sum(1 for b in self.bolos.values() if not b.is_standing)


# ============================================================================
# TRACKER DE ARUCOS CON MULTIHILO
# ============================================================================

class ArucoTracker:
    """
    Tracker de ArUcos con captura en hilo separado.
    
    Usa un sistema de doble buffer para mostrar siempre el último frame
    disponible, evitando acumulación de lag.
    """
    
    def __init__(
        self,
        camera_source: Union[str, int] = None,
        marker_size_m: float = 0.05,
        show_axes: bool = False,
        stl_path: Optional[str] = None,
        debug_mode: bool = False
    ):
        """
        Args:
            camera_source: URL de cámara IP, índice de cámara local, o None para usar config guardada
            marker_size_m: Tamaño del marcador en metros
            show_axes: Si mostrar ejes 3D sobre los marcadores
            stl_path: Ruta opcional a archivo STL para renderizar
            debug_mode: Si activar modo debug con info extra
        """
        # Configuración de cámara
        if camera_source is None:
            camera_source = load_camera_config()
        self.camera_source = camera_source
        
        # Parámetros
        self.marker_size_m = marker_size_m
        self.show_axes = show_axes
        self.stl_path = stl_path
        self.debug_mode = debug_mode
        
        # Estado del tracker
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        
        # Buffers con locks
        self._raw_frame_lock = threading.Lock()
        self._processed_lock = threading.Lock()
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_processed: Optional[FrameData] = None
        
        # Detección de ArUco - DICT_ARUCO_ORIGINAL (5x5 con borde)
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self._aruco_params = self._create_detector_params()
        self._detector = aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        
        # Detector QR
        self._qr_detector = cv2.QRCodeDetector()
        
        # Calibración de cámara (genérica, se actualiza al abrir)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Gestor de bolos
        self.bolos_manager = BolosManager()
        
        # STL mesh si se especifica
        self._mesh_data = None
        if stl_path:
            self._load_stl(stl_path)
        
        # Estadísticas
        self._frame_count = 0
        self._detection_count = 0
        self._fps_counter = deque(maxlen=30)
    
    def _create_detector_params(self) -> aruco.DetectorParameters:
        """Crea parámetros optimizados para detección."""
        params = aruco.DetectorParameters()
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.05
        # Sin refinamiento para evitar errores de OpenCV
        params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
        params.detectInvertedMarker = False
        return params
    
    def _load_stl(self, stl_path: str):
        """Carga un modelo STL para renderizado."""
        try:
            import trimesh
            mesh = trimesh.load(stl_path)
            mesh.apply_scale(0.001)  # Escalar a metros
            mesh.vertices -= mesh.centroid
            self._mesh_data = {
                "vertices": mesh.vertices,
                "faces": mesh.faces
            }
            if self.debug_mode:
                print(f"[DEBUG] STL cargado: {len(mesh.vertices)} vértices")
        except Exception as e:
            print(f"[WARN] No se pudo cargar STL: {e}")
            self._mesh_data = None
    
    def start(self) -> bool:
        """
        Inicia la captura y procesamiento.
        
        Returns:
            True si se inició correctamente
        """
        if self._running:
            return True
        
        # Abrir cámara
        self._cap = cv2.VideoCapture(self.camera_source)
        if not self._cap.isOpened():
            print(f"[ERROR] No se pudo abrir: {self.camera_source}")
            return False
        
        # Configurar cámara
        if isinstance(self.camera_source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Obtener dimensiones reales
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Crear matriz de cámara
        fx, fy = 800, 800
        cx, cy = width / 2, height / 2
        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"[INFO] Cámara abierta: {width}x{height} @ {fps:.1f} FPS")
        
        # Guardar config
        if isinstance(self.camera_source, str):
            save_camera_config(self.camera_source)
        
        # Iniciar hilos
        self._running = True
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        return True
    
    def stop(self):
        """Detiene la captura y procesamiento."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        print("[INFO] Tracker detenido")
    
    def _capture_loop(self):
        """Hilo de captura - solo lee frames lo más rápido posible."""
        while self._running:
            if self._cap is None:
                break
            
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Guardar el frame más reciente
            with self._raw_frame_lock:
                self._latest_raw_frame = frame
            
            self._frame_count += 1
    
    def _process_loop(self):
        """Hilo de procesamiento - procesa el último frame disponible."""
        while self._running:
            # Obtener el frame más reciente
            with self._raw_frame_lock:
                frame = self._latest_raw_frame
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Procesar frame
            start_time = time.time()
            frame_data = self._process_frame(frame.copy())
            process_time = time.time() - start_time
            self._fps_counter.append(process_time)
            
            # Guardar resultado procesado
            with self._processed_lock:
                self._latest_processed = frame_data
    
    def _process_frame(self, frame: np.ndarray) -> FrameData:
        """Procesa un frame completo."""
        timestamp = time.time()
        display_frame = frame.copy()  # Trabajar sobre una copia para dibujar
        
        frame_data = FrameData(
            frame=display_frame,
            timestamp=timestamp,
            aruco_detections=[],
            qr_detections=[]
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        
        # Detectar ArUcos
        corners = None
        ids = None
        try:
            corners, ids, rejected = self._detector.detectMarkers(gray)
        except cv2.error as e:
            if self.debug_mode:
                print(f"[DEBUG] Error detectMarkers: {e}")
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error inesperado: {e}")
        
        # Dibujar marcadores rechazados en debug (ayuda a ver si hay candidatos)
        if self.debug_mode and rejected is not None and len(rejected) > 0:
            cv2.putText(display_frame, f"Candidatos rechazados: {len(rejected)}",
                       (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Procesar ArUcos detectados
        if ids is not None and len(ids) > 0:
            self._detection_count += len(ids)
            
            # Dibujar marcadores detectados con drawDetectedMarkers
            try:
                aruco.drawDetectedMarkers(display_frame, corners, ids)
            except Exception:
                pass
            
            # Además dibujar manualmente para asegurar visualización
            for corner_set, marker_id in zip(corners, ids.flatten()):
                pts = corner_set.reshape(4, 2).astype(np.int32)
                # Dibujar polígono verde grueso
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)
                # Centro del marcador
                center = pts.mean(axis=0).astype(int)
                # Círculo en el centro
                cv2.circle(display_frame, tuple(center), 8, (0, 0, 255), -1)
                # ID del marcador
                cv2.putText(display_frame, f"ID:{marker_id}",
                           (center[0] - 20, center[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Procesar cada marcador
            ball_position = None
            
            for i, (marker_corners, marker_id) in enumerate(zip(corners, ids.flatten())):
                detection = self._process_marker(
                    display_frame, marker_corners, int(marker_id), width, height
                )
                frame_data.aruco_detections.append(detection)
                
                # Verificar si es referencia del mundo
                if marker_id == QR_WORLD_REFERENCE_ID:
                    frame_data.world_reference_found = True
                
                # Verificar si es la bola
                if marker_id == BALL_ARUCO_ID and detection.world_position:
                    ball_position = detection.world_position
                
                # Actualizar bolos
                if marker_id >= MIN_BOLO_ID and detection.world_position:
                    self.bolos_manager.update_bolo(marker_id, detection.world_position)
            
            # Verificar colisiones de bola
            if ball_position:
                knocked = self.bolos_manager.check_ball_collision(ball_position)
                if knocked and self.debug_mode:
                    print(f"[DEBUG] ¡Bolos tumbados: {knocked}!")
        
        # Detectar QR (opcional)
        try:
            retval, decoded_info, qr_points, _ = self._qr_detector.detectAndDecodeMulti(display_frame)
            if retval and qr_points is not None:
                for i, pts in enumerate(qr_points):
                    pts_int = np.int32(pts).reshape(-1, 2)
                    cv2.polylines(display_frame, [pts_int], True, (0, 128, 255), 2)
                    
                    text = decoded_info[i] if decoded_info and i < len(decoded_info) else ""
                    frame_data.qr_detections.append({
                        "points": pts,
                        "text": text
                    })
        except Exception:
            pass
        
        # Dibujar info de debug
        self._draw_overlay(display_frame, frame_data)
        
        frame_data.frame = display_frame
        return frame_data
    
    def _process_marker(
        self,
        frame: np.ndarray,
        marker_corners: np.ndarray,
        marker_id: int,
        width: int,
        height: int
    ) -> ArucoDetection:
        """Procesa un marcador individual."""
        corners_2d = marker_corners.reshape(4, 2)
        center = np.mean(corners_2d, axis=0).astype(int)
        
        detection = ArucoDetection(
            id=marker_id,
            corners=corners_2d,
            center_px=(int(center[0]), int(center[1]))
        )
        
        # Estimar pose 3D
        try:
            obj_points = np.array([
                [-self.marker_size_m / 2, -self.marker_size_m / 2, 0],
                [self.marker_size_m / 2, -self.marker_size_m / 2, 0],
                [self.marker_size_m / 2, self.marker_size_m / 2, 0],
                [-self.marker_size_m / 2, self.marker_size_m / 2, 0]
            ], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corners_2d.astype(np.float32),
                self._camera_matrix,
                self._dist_coeffs
            )
            
            if success:
                detection.rvec = rvec
                detection.tvec = tvec
                detection.world_position = (
                    float(tvec[0][0]),
                    float(tvec[1][0]),
                    float(tvec[2][0])
                )
                
                # Dibujar ejes si está habilitado
                if self.show_axes:
                    cv2.drawFrameAxes(
                        frame, self._camera_matrix, self._dist_coeffs,
                        rvec, tvec, self.marker_size_m * 0.5
                    )
                
                # Renderizar STL si hay
                if self._mesh_data is not None:
                    self._render_stl(frame, rvec, tvec, width, height)
        
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error pose ID {marker_id}: {e}")
        
        # Dibujar info del marcador
        self._draw_marker_info(frame, detection)
        
        return detection
    
    def _render_stl(self, frame: np.ndarray, rvec, tvec, width: int, height: int):
        """Renderiza el modelo STL sobre el frame."""
        if self._mesh_data is None:
            return
        
        try:
            for face in self._mesh_data["faces"]:
                pts_3d = self._mesh_data["vertices"][face]
                pts_2d, _ = cv2.projectPoints(
                    pts_3d.astype(np.float32),
                    rvec, tvec,
                    self._camera_matrix, self._dist_coeffs
                )
                pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)
                cv2.polylines(frame, [pts_2d], True, (200, 150, 0), 1)
        except Exception:
            pass
    
    def _draw_marker_info(self, frame: np.ndarray, detection: ArucoDetection):
        """Dibuja información sobre un marcador."""
        cx, cy = detection.center_px
        marker_id = detection.id
        
        # Determinar tipo y color
        if marker_id == QR_WORLD_REFERENCE_ID:
            label = "WORLD"
            color = (0, 255, 255)  # Amarillo
        elif marker_id == BALL_ARUCO_ID:
            label = "BALL"
            color = (255, 0, 255)  # Magenta
        else:
            # Es un bolo
            bolo = self.bolos_manager.bolos.get(marker_id)
            if bolo and not bolo.is_standing:
                label = f"B{marker_id} [X]"
                color = (0, 0, 255)  # Rojo = tumbado
            else:
                label = f"B{marker_id}"
                color = (0, 255, 0)  # Verde = de pie
        
        cv2.putText(
            frame, label,
            (cx - 20, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        # Mostrar coordenadas en debug
        if self.debug_mode and detection.world_position:
            x, y, z = detection.world_position
            coord_text = f"({x:.2f}, {y:.2f}, {z:.2f})"
            cv2.putText(
                frame, coord_text,
                (cx - 50, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )
    
    def _draw_overlay(self, frame: np.ndarray, frame_data: FrameData):
        """Dibuja información overlay sobre el frame."""
        height = frame.shape[0]
        
        # FPS
        if self._fps_counter:
            avg_time = sum(self._fps_counter) / len(self._fps_counter)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Contadores
        aruco_count = len(frame_data.aruco_detections)
        standing = self.bolos_manager.get_standing_count()
        knocked = self.bolos_manager.get_knocked_count()
        
        cv2.putText(
            frame, f"ArUcos: {aruco_count}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        cv2.putText(
            frame, f"Bolos: {standing} de pie, {knocked} tumbados",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2
        )
        
        # Referencia del mundo
        if frame_data.world_reference_found:
            cv2.putText(
                frame, "WORLD REF OK",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
        
        # Modo debug
        if self.debug_mode:
            cv2.putText(
                frame, "[DEBUG MODE]",
                (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """
        Obtiene el último frame procesado.
        
        Returns:
            FrameData con el frame y detecciones, o None si no hay frame
        """
        with self._processed_lock:
            return self._latest_processed
    
    def get_detections(self) -> List[ArucoDetection]:
        """Obtiene las últimas detecciones de ArUco."""
        with self._processed_lock:
            if self._latest_processed:
                return self._latest_processed.aruco_detections
        return []
    
    def is_running(self) -> bool:
        """Retorna si el tracker está activo."""
        return self._running
    
    @property
    def frame_count(self) -> int:
        """Número total de frames capturados."""
        return self._frame_count
    
    @property
    def detection_count(self) -> int:
        """Número total de detecciones realizadas."""
        return self._detection_count


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def run_viewer(
    camera_source: Union[str, int] = None,
    show_axes: bool = False,
    stl_path: Optional[str] = None,
    debug_mode: bool = False
):
    """
    Ejecuta el visor de ArUcos interactivo.
    
    Args:
        camera_source: Fuente de video
        show_axes: Mostrar ejes 3D
        stl_path: Ruta a archivo STL
        debug_mode: Modo debug activado
    """
    tracker = ArucoTracker(
        camera_source=camera_source,
        show_axes=show_axes,
        stl_path=stl_path,
        debug_mode=debug_mode
    )
    
    if not tracker.start():
        print("[ERROR] No se pudo iniciar el tracker")
        return
    
    print("\n=== Controles ===")
    print("  q: Salir")
    print("  r: Resetear bolos")
    print("  d: Toggle debug")
    print("  a: Toggle ejes")
    print("  s: Guardar frame")
    print("==================\n")
    
    save_count = 0
    
    try:
        while True:
            frame_data = tracker.get_latest_frame()
            
            if frame_data is not None:
                cv2.imshow("ArUco Tracker - q=salir", frame_data.frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.bolos_manager.reset_all_bolos()
                print("[INFO] Bolos reseteados")
            elif key == ord('d'):
                tracker.debug_mode = not tracker.debug_mode
                print(f"[INFO] Debug: {'ON' if tracker.debug_mode else 'OFF'}")
            elif key == ord('a'):
                tracker.show_axes = not tracker.show_axes
                print(f"[INFO] Ejes: {'ON' if tracker.show_axes else 'OFF'}")
            elif key == ord('s'):
                save_count += 1
                filename = f"aruco_capture_{save_count}.png"
                if frame_data:
                    cv2.imwrite(filename, frame_data.frame)
                    print(f"[INFO] Guardado: {filename}")
    
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
    
    # Estadísticas finales
    print(f"\n=== Estadísticas ===")
    print(f"Frames capturados: {tracker.frame_count}")
    print(f"Detecciones totales: {tracker.detection_count}")
    print(f"Bolos registrados: {len(tracker.bolos_manager.bolos)}")


# ============================================================================
# EJECUCIÓN DIRECTA
# ============================================================================

if __name__ == "__main__":
    print("=== ArUco Library ===")
    print("Este módulo debe ser importado o usado con test_aruco.py")
    print()
    print("Ejemplo de uso:")
    print("  from aruco_lib import ArucoTracker, run_viewer")
    print("  run_viewer(debug_mode=True)")
