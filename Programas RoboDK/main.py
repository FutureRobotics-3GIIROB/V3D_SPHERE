# pyright: reportMissingImports=false

import json
import time

import cv2
from aruco_lib import BALL_ARUCO_ID, ArucoTracker
from ball_detector import detect_ball, draw_ball
from calibracion import HomographyCalibrator
from tracker import BallTracker

OUTPUT = "positions.json"
REDETECT_EVERY = 5  # frames before re-scanning when lost

# Modos de detección
USE_ARUCO_TRACKER = True  # Si True, usa ArucoTracker; si False, usa detección original


def main_with_aruco():
    """
    Modo principal usando ArucoTracker para detección de ArUcos y pelota.
    Usa calibración por homografía.
    """
    print("[INFO] Iniciando modo con ArucoTracker...")

    # Cargar calibración
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("[ERROR] No se encontró archivo de calibración")
        print("Ejecuta: python calibracion.py --capture")
        return

    homography = calib_data["homography_matrix"]

    # Crear tracker de ArUcos
    tracker = ArucoTracker(camera_source=None, marker_size_m=0.05, show_axes=False, debug_mode=True)

    if not tracker.start():
        print("[ERROR] No se pudo iniciar el tracker de ArUcos")
        return

    print("[INFO] Presiona ESC para salir")

    try:
        while True:
            frame_data = tracker.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue

            frame = frame_data.frame
            h, w = frame.shape[:2]

            ball_position = None
            ball_center_px = None

            # Buscar pelota por ArUco (ID=1)
            for det in frame_data.aruco_detections:
                if det.id == BALL_ARUCO_ID:
                    ball_center_px = det.center_px
                    if det.world_position:
                        ball_position = {
                            "pixel": list(ball_center_px),
                            "xyz_mm": [
                                det.world_position[0] * 1000,
                                det.world_position[1] * 1000,
                                det.world_position[2] * 1000,
                            ],
                        }
                    break

            # Si no hay ArUco, buscar por color
            if ball_center_px is None:
                ball_det = detect_ball(frame)
                if ball_det:
                    ball_center_px = ball_det["center"]
                    draw_ball(frame, ball_det, color=(0, 255, 0))

                    # Transformar usando homografía
                    point_transformed = HomographyCalibrator.transform_point(
                        ball_center_px, homography
                    )

                    ball_position = {
                        "pixel": list(ball_center_px),
                        "xyz_mm": list(point_transformed) + [0],
                    }

            if ball_position:
                print(json.dumps(ball_position))
                try:
                    with open(OUTPUT, "w") as f:
                        json.dump(ball_position, f)
                except OSError:
                    pass

            cv2.imshow("V3D Tracking (ArUco Mode)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        tracker.stop()
        cv2.destroyAllWindows()


def main_original():
    """
    Modo original con detección por color y homografía.
    """
    print("[INFO] Iniciando modo original...")

    # Cargar calibración
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("[ERROR] No se encontró archivo de calibración")
        print("Ejecuta: python calibracion.py --capture")
        return

    homography = calib_data["homography_matrix"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir webcam")

    for _ in range(30):
        cap.read()
    time.sleep(0.1)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Sin frame de webcam")

    h, w = frame.shape[:2]
    print(f"Frame {w}x{h}")

    # Auto-detectar pelota
    tracker = BallTracker()
    det = detect_ball(frame)
    if det:
        tracker.init(frame, det["bbox"])
        print(f'Pelota encontrada en {det["center"]}')
    else:
        print("Buscando pelota...")

    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rastrear
        if tracker.ok or tracker.bbox is not None:
            ok, _ = tracker.update(frame)
        else:
            ok = False

        # Re-detectar si se pierde
        last_det = None
        if not ok:
            lost_count += 1
            if lost_count >= REDETECT_EVERY or tracker.bbox is None:
                last_det = detect_ball(frame)
                if last_det:
                    tracker.init(frame, last_det["bbox"])
                    lost_count = 0
                    print(f'Pelota re-detectada en {last_det["center"]}')
                else:
                    cv2.putText(
                        frame,
                        "Buscando...",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
        else:
            lost_count = 0

        draw_ball(frame, last_det)

        # Salida XYZ
        pos = None
        c = tracker.center
        if c and tracker.ok:
            point_transformed = HomographyCalibrator.transform_point(c, homography)
            xyz = list(point_transformed) + [0]
            label = f"X:{xyz[0]:.1f} Y:{xyz[1]:.1f} Z:{xyz[2]:.1f}"
            pos = {"pixel": list(c), "xyz_mm": xyz}
            tracker.draw(frame, label)
        elif tracker.bbox is not None:
            tracker.draw(frame, "PERDIDA")

        if pos:
            print(json.dumps(pos))
            try:
                with open(OUTPUT, "w") as f:
                    json.dump(pos, f)
            except OSError:
                pass

        cv2.imshow("V3D Tracking", frame)
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


if __name__ == "__main__":
    main()
