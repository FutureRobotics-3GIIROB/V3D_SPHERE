import json, time, cv2
import numpy as np

from ball_detector import detect_ball, draw_ball
from qr_depth import QRDepth, focal_length, pixel_to_xyz
from tracker import BallTracker
from aruco_lib import (
    ArucoTracker,
    list_available_cameras,
    load_camera_config,
    save_camera_config,
    QR_WORLD_REFERENCE_ID,
    BALL_ARUCO_ID,
    MIN_BOLO_ID
)

OUTPUT = 'positions.json'
REDETECT_EVERY = 5   # frames before re-scanning when lost

# Modos de detección
USE_ARUCO_TRACKER = True  # Si True, usa ArucoTracker; si False, usa detección original


def main_with_aruco():
    """
    Modo principal usando ArucoTracker para detección de ArUcos, QR y pelota.
    """
    print("[INFO] Iniciando modo con ArucoTracker...")
    
    # Crear tracker de ArUcos
    tracker = ArucoTracker(
        camera_source=None,  # Usa config guardada o cámara 0
        marker_size_m=0.05,
        show_axes=False,
        debug_mode=True
    )
    
    if not tracker.start():
        print("[ERROR] No se pudo iniciar el tracker de ArUcos")
        return
    
    # Tracker de pelota por color (backup)
    ball_tracker = BallTracker()
    lost_count = 0
    
    print("[INFO] Presiona ESC para salir")
    
    try:
        while True:
            # Obtener frame procesado del tracker
            frame_data = tracker.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            frame = frame_data.frame
            h, w = frame.shape[:2]
            
            # Buscar pelota por ArUco (ID=1) o por color
            ball_position = None
            ball_center_px = None
            
            # 1. Primero buscar ArUco de la pelota
            for det in frame_data.aruco_detections:
                if det.id == BALL_ARUCO_ID:
                    ball_center_px = det.center_px
                    if det.world_position:
                        ball_position = {
                            'pixel': list(ball_center_px),
                            'xyz_mm': [det.world_position[0] * 1000,
                                       det.world_position[1] * 1000,
                                       det.world_position[2] * 1000]
                        }
                    break
            
            # 2. Si no hay ArUco de pelota, buscar por color
            if ball_center_px is None:
                ball_det = detect_ball(frame)
                if ball_det:
                    ball_center_px = ball_det['center']
                    draw_ball(frame, ball_det, color=(0, 255, 0))
                    
                    # Intentar obtener profundidad del QR
                    if frame_data.qr_detections:
                        # Usar el primer QR detectado para profundidad
                        qr_text = frame_data.qr_detections[0].get('text', '')
                        try:
                            qr_size_mm = float(qr_text.strip()) if qr_text else None
                            if qr_size_mm:
                                f_px = focal_length(w)
                                cx, cy = w / 2.0, h / 2.0
                                # Calcular Z aproximada
                                xyz = pixel_to_xyz(ball_center_px[0], ball_center_px[1],
                                                   100, f_px, cx, cy)  # Z placeholder
                                ball_position = {
                                    'pixel': list(ball_center_px),
                                    'xyz_mm': list(xyz)
                                }
                        except (ValueError, TypeError):
                            pass
            
            # Guardar posición
            if ball_position:
                print(json.dumps(ball_position))
                try:
                    with open(OUTPUT, 'w') as f:
                        json.dump(ball_position, f)
                except OSError:
                    pass
            
            # Mostrar
            cv2.imshow('V3D Tracking (ArUco Mode)', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


def main_original():
    """
    Modo original sin ArucoTracker (solo QR + pelota por color).
    """
    print("[INFO] Iniciando modo original...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    # Warm up
    for _ in range(30):
        cap.read()
    time.sleep(0.1)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('No frame from webcam')

    h, w = frame.shape[:2]
    f_px = focal_length(w)
    cx, cy = w / 2.0, h / 2.0
    print(f'Frame {w}x{h}  |  focal ~{f_px:.0f}px')

    # Auto-detect ball on first frame
    tracker = BallTracker()
    det = detect_ball(frame)
    if det:
        tracker.init(frame, det['bbox'])
        print(f'Ball found at {det["center"]}')
    else:
        print('No ball yet — will keep searching.')

    qr = QRDepth()
    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # QR depth
        qr.update(frame, f_px)
        qr.draw(frame)

        # Track
        if tracker.ok or tracker.bbox is not None:
            ok, _ = tracker.update(frame)
        else:
            ok = False

        # Re-detect if lost
        last_det = None
        if not ok:
            lost_count += 1
            if lost_count >= REDETECT_EVERY or tracker.bbox is None:
                last_det = detect_ball(frame)
                if last_det:
                    tracker.init(frame, last_det['bbox'])
                    lost_count = 0
                    print(f'Ball re-detected at {last_det["center"]}')
                else:
                    cv2.putText(frame, 'Searching...', (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            lost_count = 0

        # Visualise detection circle (reuse cached result, no extra detection call)
        draw_ball(frame, last_det)

        # XYZ output
        pos = None
        c = tracker.center
        if c and tracker.ok:
            if qr.depth_z is not None:
                xyz = pixel_to_xyz(c[0], c[1], qr.depth_z, f_px, cx, cy)
                label = f'X:{xyz[0]} Y:{xyz[1]} Z:{xyz[2]}'
                pos = {'pixel': list(c), 'xyz_mm': list(xyz)}
            else:
                label = 'No QR depth'
                pos = {'pixel': list(c), 'xyz_mm': None}
            tracker.draw(frame, label)
        elif tracker.bbox is not None:
            tracker.draw(frame, 'LOST')

        if pos:
            print(json.dumps(pos))
            try:
                with open(OUTPUT, 'w') as f:
                    json.dump(pos, f)
            except OSError:
                pass

        cv2.imshow('V3D Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Punto de entrada principal."""
    if USE_ARUCO_TRACKER:
        main_with_aruco()
    else:
        main_original()


if __name__ == '__main__':
    main()
