#!/usr/bin/env python
"""
test_camara.py - Script de pruebas completo para cámara, ArUcos, QR y pelota

Este script permite:
- Seleccionar cámara (local o IP)
- Recordar la última cámara usada
- Detectar ArUcos, códigos QR y la pelota simultáneamente
- Toggle de visualización con teclas 1, 2, 3

Controles:
    1        → Toggle ArUcos
    2        → Toggle QR
    3        → Toggle Pelota
    D        → Toggle modo debug
    ESC / Q  → Salir

Uso:
    uv run test_camara.py                    # Modo interactivo
    uv run test_camara.py --debug            # Modo debug
    uv run test_camara.py --camera 0         # Usar cámara local 0
    uv run test_camara.py --camera http://ip:port/video
"""

import argparse
import sys
import time
import threading
from pathlib import Path

# Suprimir warnings antes de importar OpenCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

try:
    import cv2
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

import numpy as np
from cv2 import aruco

# Importar módulos del proyecto
from aruco_lib import (
    list_available_cameras,
    load_camera_config,
    save_camera_config,
    normalize_camera_url,
    DEFAULT_CAMERA_URL,
    QR_WORLD_REFERENCE_ID,
    BALL_ARUCO_ID,
    MIN_BOLO_ID
)
from ball_detector import detect_ball, draw_ball
from qr_depth import QRDepth, focal_length


def print_header():
    """Imprime cabecera del programa."""
    print()
    print("=" * 55)
    print("  📷 Test de Cámara - ArUco + QR + Pelota")
    print("=" * 55)
    print()


def select_camera_interactive() -> str:
    """Selección interactiva de cámara."""
    print("Buscando cámaras disponibles...")
    local_cameras = list_available_cameras(max_devices=6)
    last_url = load_camera_config()
    
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║           FUENTES DE VIDEO DISPONIBLES             ║")
    print("╠════════════════════════════════════════════════════╣")
    
    # Mostrar cámaras locales
    if local_cameras:
        for i, cam_idx in enumerate(local_cameras, start=1):
            print(f"║  {i}. Cámara local (índice {cam_idx})")
    else:
        print("║  (No se detectaron cámaras locales)")
    
    print("║")
    print("║  i. Ingresar IP de cámara manualmente")
    print(f"║  Enter. Usar última: {last_url[:40]}...")
    print("╚════════════════════════════════════════════════════╝")
    print()
    
    while True:
        choice = input("Selecciona opción (o pega URL directamente): ").strip()
        
        # Enter = usar última
        if not choice:
            print(f"→ Usando: {last_url}")
            return last_url
        
        # URL directa
        if choice.startswith("http://") or choice.startswith("https://"):
            url = choice
            save_camera_config(url)
            print(f"→ Usando URL: {url}")
            return url
        
        # IP sin protocolo
        if "." in choice or ":" in choice:
            url = normalize_camera_url(choice)
            save_camera_config(url)
            print(f"→ Usando URL: {url}")
            return url
        
        # Opción 'i' para ingresar IP
        if choice.lower() == "i":
            ip_input = input(f"IP/URL (Enter = {last_url}): ").strip()
            if ip_input:
                url = normalize_camera_url(ip_input)
                save_camera_config(url)
                print(f"→ Usando URL: {url}")
                return url
            return last_url
        
        # Número de cámara local
        if choice.isdigit() and local_cameras:
            idx = int(choice)
            if 1 <= idx <= len(local_cameras):
                cam = local_cameras[idx - 1]
                print(f"→ Usando cámara local: {cam}")
                return cam
        
        print("❌ Opción inválida. Intenta de nuevo.")


class CameraTester:
    """
    Clase para probar la detección de ArUcos, QR y pelota.
    Con captura en hilo separado para reducir latencia.
    """
    
    def __init__(self, camera_source, debug_mode: bool = False):
        """
        Args:
            camera_source: URL de cámara IP o índice de cámara local
            debug_mode: Si activar información de debug extra
        """
        self.camera_source = camera_source
        self.debug_mode = debug_mode
        
        # Toggles de visualización (todo ON por defecto)
        self.show_aruco = True
        self.show_qr = True
        self.show_ball = True
        
        # VideoCapture y threading
        self._cap = None
        self._running = False
        self._capture_thread = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        
        # Detector de ArUco - DICT_ARUCO_ORIGINAL (5x5 con borde)
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self._aruco_params = self._create_detector_params()
        self._detector = aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        
        # Detector QR con profundidad
        self._qr_depth = QRDepth()
        
        # Calibración de cámara (genérica)
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Tamaño del marcador en metros
        self.marker_size_m = 0.05
        
        # Estadísticas
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0
        
        # Últimos datos de detección (para mostrar aunque no se dibujen)
        self._last_aruco_count = 0
        self._last_qr_detected = False
        self._last_qr_depth = None
        self._last_ball_detected = False
        self._last_ball_coords = None  # (X, Y, Z) en mm respecto al QR
        
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
    
    def _capture_loop(self):
        """Hilo de captura - lee frames continuamente y guarda solo el último."""
        while self._running:
            if self._cap is None:
                break
            
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Guardar el frame más reciente (descarta los anteriores)
            with self._frame_lock:
                self._latest_frame = frame
    
    def _get_latest_frame(self):
        """Obtiene el último frame disponible (skip buffer)."""
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None  # Marcar como consumido
        return frame
    
    def start(self) -> bool:
        """Abre la cámara e inicia el hilo de captura."""
        self._cap = cv2.VideoCapture(self.camera_source)
        if not self._cap.isOpened():
            print(f"[ERROR] No se pudo abrir: {self.camera_source}")
            return False
        
        # Configurar cámara
        if isinstance(self.camera_source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Buffer mínimo para reducir latencia
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Obtener dimensiones
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
        
        # Focal length para QR depth
        self._focal_px = focal_length(width)
        
        print(f"[INFO] Cámara abierta: {width}x{height} @ {fps:.1f} FPS")
        
        # Guardar config
        if isinstance(self.camera_source, str):
            save_camera_config(self.camera_source)
        
        # Iniciar hilo de captura
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        return True
    
    def stop(self):
        """Cierra la cámara y detiene el hilo."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
        print("[INFO] Cámara cerrada")
    
    def run(self):
        """Loop principal de detección."""
        print()
        print("╔════════════════════════════════════════════════════╗")
        print("║                    CONTROLES                       ║")
        print("╠════════════════════════════════════════════════════╣")
        print("║  1        → Toggle ArUcos (ON/OFF)                 ║")
        print("║  2        → Toggle QR (ON/OFF)                     ║")
        print("║  3        → Toggle Pelota (ON/OFF)                 ║")
        print("║  D        → Toggle modo debug                      ║")
        print("║  ESC / Q  → Salir                                  ║")
        print("╚════════════════════════════════════════════════════╝")
        print()
        
        # Esperar a que llegue el primer frame
        while self._running:
            frame = self._get_latest_frame()
            if frame is not None:
                break
            time.sleep(0.01)
        
        while self._running:
            # Obtener el último frame (skip buffer)
            frame = self._get_latest_frame()
            if frame is None:
                # No hay frame nuevo, reusar el anterior o esperar
                time.sleep(0.001)
                continue
            
            # Calcular FPS
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self._fps_time
                self._fps = 30 / elapsed if elapsed > 0 else 0
                self._fps_time = now
            
            # Procesar frame (siempre procesa todo, pero dibuja según toggles)
            display_frame = self._process_frame(frame)
            
            # Mostrar
            cv2.imshow("Test Camara - ArUco + QR + Pelota", display_frame)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
            elif key == ord('1'):
                self.show_aruco = not self.show_aruco
                print(f"[TOGGLE] ArUcos: {'ON' if self.show_aruco else 'OFF'}")
            elif key == ord('2'):
                self.show_qr = not self.show_qr
                print(f"[TOGGLE] QR: {'ON' if self.show_qr else 'OFF'}")
            elif key == ord('3'):
                self.show_ball = not self.show_ball
                print(f"[TOGGLE] Pelota: {'ON' if self.show_ball else 'OFF'}")
            elif key == ord('d') or key == ord('D'):
                self.debug_mode = not self.debug_mode
                print(f"[TOGGLE] Debug: {'ON' if self.debug_mode else 'OFF'}")
        
        self.stop()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame detectando ArUcos, QR y pelota."""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ═══════════════════════════════════════════════════════
        # 1. DETECCIÓN DE ARUCOS (siempre procesa)
        # ═══════════════════════════════════════════════════════
        aruco_count = 0
        corners = None
        ids = None
        rejected = None
        
        try:
            corners, ids, rejected = self._detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                aruco_count = len(ids)
                self._last_aruco_count = aruco_count
                
                # Solo dibujar si está activo
                if self.show_aruco:
                    # Dibujar marcadores con función oficial
                    try:
                        aruco.drawDetectedMarkers(display_frame, corners, ids)
                    except Exception:
                        pass
                    
                    # Dibujar manualmente para asegurar visualización
                    for corner_set, marker_id in zip(corners, ids.flatten()):
                        pts = corner_set.reshape(4, 2).astype(np.int32)
                        
                        # Polígono verde
                        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)
                        
                        # Centro del marcador
                        center = pts.mean(axis=0).astype(int)
                        cv2.circle(display_frame, tuple(center), 8, (0, 0, 255), -1)
                        
                        # Etiqueta según tipo
                        if marker_id == QR_WORLD_REFERENCE_ID:
                            label = "WORLD"
                            color = (0, 255, 255)  # Amarillo
                        else:
                            # Todos los demás son bolos
                            label = f"B{marker_id}"
                            color = (0, 255, 0)  # Verde
                        
                        cv2.putText(display_frame, label,
                                   (center[0] - 30, center[1] - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Debug: mostrar ID numérico
                        if self.debug_mode:
                            cv2.putText(display_frame, f"ID:{marker_id}",
                                       (center[0] - 20, center[1] + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                self._last_aruco_count = 0
            
            # Debug: candidatos rechazados
            if self.debug_mode and self.show_aruco and rejected is not None and len(rejected) > 0:
                cv2.putText(display_frame, f"Candidatos rechazados: {len(rejected)}",
                           (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error ArUco: {e}")
        
        # ═══════════════════════════════════════════════════════
        # 2. DETECCIÓN DE QR (siempre procesa)
        # ═══════════════════════════════════════════════════════
        qr_detected = False
        try:
            self._qr_depth.update(frame, self._focal_px)
            if self._qr_depth.corners is not None:
                qr_detected = True
                self._last_qr_detected = True
                self._last_qr_depth = self._qr_depth.depth_z
                
                # Solo dibujar si está activo
                if self.show_qr:
                    self._qr_depth.draw(display_frame)
            else:
                self._last_qr_detected = False
                self._last_qr_depth = None
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error QR: {e}")
        
        # ═══════════════════════════════════════════════════════
        # 3. DETECCIÓN DE PELOTA (siempre procesa)
        # ═══════════════════════════════════════════════════════
        ball_detected = False
        self._last_ball_coords = None
        try:
            ball_det = detect_ball(frame)
            if ball_det:
                ball_detected = True
                self._last_ball_detected = True
                
                cx, cy = ball_det['center']
                radius = ball_det['radius']
                
                # Calcular coordenadas respecto al QR (si está detectado)
                if self._last_qr_detected and self._last_qr_depth is not None:
                    # Usar QR como referencia del mundo
                    # Calcular posición relativa en mm
                    qr_center = self._qr_depth.corners.mean(axis=0) if self._qr_depth.corners is not None else (width/2, height/2)
                    
                    # Convertir pixel a mm usando la profundidad del QR
                    z_mm = self._last_qr_depth
                    # X e Y en mm respecto al centro del QR
                    x_mm = (cx - qr_center[0]) * z_mm / self._focal_px
                    y_mm = (cy - qr_center[1]) * z_mm / self._focal_px
                    
                    self._last_ball_coords = (round(x_mm, 1), round(y_mm, 1), round(z_mm, 1))
                    
                    # Imprimir coordenadas en consola
                    print(f"[PELOTA] X:{x_mm:7.1f}  Y:{y_mm:7.1f}  Z:{z_mm:7.1f} mm  |  Pixel: ({cx}, {cy})")
                
                # Solo dibujar si está activo
                if self.show_ball:
                    draw_ball(display_frame, ball_det, color=(0, 255, 0))
                    
                    # Añadir etiqueta
                    cv2.putText(display_frame, "PELOTA",
                               (cx - 30, cy - radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Mostrar coordenadas si están disponibles
                    if self._last_ball_coords:
                        x, y, z = self._last_ball_coords
                        coord_text = f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f} mm"
                        cv2.putText(display_frame, coord_text,
                                   (cx - 60, cy + radius + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    if self.debug_mode:
                        cv2.putText(display_frame, f"R:{radius}px",
                                   (cx - 20, cy + radius + 45),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                self._last_ball_detected = False
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error pelota: {e}")
        
        # ═══════════════════════════════════════════════════════
        # 4. OVERLAY DE INFORMACIÓN
        # ═══════════════════════════════════════════════════════
        y_offset = 30
        
        # FPS
        cv2.putText(display_frame, f"FPS: {self._fps:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Estado de detecciones (muestra estado real aunque esté oculto)
        # ArUcos
        aruco_on = "[1] ON " if self.show_aruco else "[1] OFF"
        status_aruco = f"ArUcos {aruco_on}: {self._last_aruco_count}"
        color_aruco = (0, 255, 0) if self._last_aruco_count > 0 else (0, 0, 255)
        if not self.show_aruco:
            color_aruco = (128, 128, 128)  # Gris si está oculto
        cv2.putText(display_frame, status_aruco,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_aruco, 2)
        y_offset += 25
        
        # QR
        qr_on = "[2] ON " if self.show_qr else "[2] OFF"
        status_qr = f"QR {qr_on}: "
        if self._last_qr_detected:
            status_qr += f"Z:{self._last_qr_depth:.0f}mm" if self._last_qr_depth else "detectado"
        else:
            status_qr += "no detectado"
        color_qr = (0, 255, 0) if self._last_qr_detected else (0, 0, 255)
        if not self.show_qr:
            color_qr = (128, 128, 128)
        cv2.putText(display_frame, status_qr,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_qr, 2)
        y_offset += 25
        
        # Pelota
        ball_on = "[3] ON " if self.show_ball else "[3] OFF"
        if self._last_ball_detected and self._last_ball_coords:
            x, y, z = self._last_ball_coords
            status_ball = f"Pelota {ball_on}: X:{x:.0f} Y:{y:.0f} Z:{z:.0f}mm"
        elif self._last_ball_detected:
            status_ball = f"Pelota {ball_on}: detectada (sin QR)"
        else:
            status_ball = f"Pelota {ball_on}: no detectada"
        color_ball = (0, 255, 0) if self._last_ball_detected else (0, 0, 255)
        if not self.show_ball:
            color_ball = (128, 128, 128)
        cv2.putText(display_frame, status_ball,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ball, 2)
        
        # Debug mode indicator
        if self.debug_mode:
            cv2.putText(display_frame, "[DEBUG MODE]",
                       (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Instrucciones
        cv2.putText(display_frame, "1/2/3: Toggles | D: Debug | ESC: Salir",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Test de cámara con detección de ArUco, QR y pelota"
    )
    parser.add_argument(
        "--camera", "-c",
        help="Cámara: índice local (0,1...) o URL (http://ip:port/video)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Activar modo debug"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Determinar fuente de cámara
    if args.camera:
        camera_source = args.camera
        if camera_source.isdigit():
            camera_source = int(camera_source)
        else:
            camera_source = normalize_camera_url(camera_source)
        print(f"Usando cámara: {camera_source}")
    else:
        camera_source = select_camera_interactive()
    
    # Crear y ejecutar tester
    tester = CameraTester(
        camera_source=camera_source,
        debug_mode=args.debug
    )
    
    if tester.start():
        tester.run()
    else:
        print("[ERROR] No se pudo iniciar la cámara")
        sys.exit(1)


if __name__ == "__main__":
    main()
