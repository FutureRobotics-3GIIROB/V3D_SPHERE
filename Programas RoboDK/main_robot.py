"""
main_robot.py
─────────────
Combina el tracking de la pelota con el control del robot en RoboDK.

  • Hilo de cámara  → detecta la pelota y actualiza la variable global
                      `ball_position` con los datos pixel + XYZ en mm.
  • Hilo de robot   → lee `ball_position` y mueve el robot "bola" de RoboDK
                      a la posición detectada.
"""

import time
import threading
import cv2

from ball_detector import detect_ball, draw_ball
from qr_depth import QRDepth, focal_length, pixel_to_xyz
from tracker import BallTracker
from FuncionesBase import getRobot, getFrame
from FuncionesRobot import moveTo, setSpeed

# ── Configuración ─────────────────────────────────────────────────────────────
ROBOT_NAME     = "bola"       # nombre del robot en RoboDK
TARGET_FRAME   = "BallTarget" # nombre del frame objetivo en RoboDK
REDETECT_EVERY = 5            # frames sin tracking antes de re-detectar
ROBOT_SPEED_LIN  = 200        # mm/s   velocidad lineal
ROBOT_SPEED_ANG  = 30         # deg/s  velocidad angular

# ── Variable global compartida ────────────────────────────────────────────────
# Formato: {'pixel': [px, py], 'xyz_mm': [X, Y, Z]}  ó  None
ball_position: dict | None = None
_lock = threading.Lock()      # acceso seguro entre hilos


# ── Hilo de cámara ────────────────────────────────────────────────────────────
def camera_loop() -> None:
    """Captura video, detecta la pelota y actualiza ball_position."""
    global ball_position

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la webcam")

    # Calentamiento de la cámara
    for _ in range(30):
        cap.read()
    time.sleep(0.1)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Sin frame de la webcam")

    h, w = frame.shape[:2]
    f_px = focal_length(w)
    cx, cy = w / 2.0, h / 2.0
    print(f"[Cámara] Frame {w}x{h}  |  focal ~{f_px:.0f} px")

    # Inicialización del tracker
    tracker = BallTracker()
    det = detect_ball(frame)
    if det:
        tracker.init(frame, det["bbox"])
        print(f"[Cámara] Pelota encontrada en {det['center']}")
    else:
        print("[Cámara] Pelota no encontrada — buscando...")

    qr = QRDepth()
    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Actualizar profundidad por QR
        qr.update(frame, f_px)
        qr.draw(frame)

        # Actualizar tracker
        if tracker.ok or tracker.bbox is not None:
            ok, _ = tracker.update(frame)
        else:
            ok = False

        # Re-detectar si se pierde la pelota
        if not ok:
            lost_count += 1
            if lost_count >= REDETECT_EVERY or tracker.bbox is None:
                det = detect_ball(frame)
                if det:
                    tracker.init(frame, det["bbox"])
                    lost_count = 0
                    print(f"[Cámara] Pelota re-detectada en {det['center']}")
                else:
                    cv2.putText(frame, "Buscando...", (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            lost_count = 0

        # Dibujar círculo de detección
        draw_ball(frame, detect_ball(frame))

        # Calcular posición XYZ y actualizar variable global
        c = tracker.center
        if c and tracker.ok:
            if qr.depth_z is not None:
                xyz = pixel_to_xyz(c[0], c[1], qr.depth_z, f_px, cx, cy)
                pos = {"pixel": list(c), "xyz_mm": list(xyz)}
                label = f"X:{xyz[0]}  Y:{xyz[1]}  Z:{xyz[2]}"
            else:
                pos = {"pixel": list(c), "xyz_mm": None}
                label = "Sin profundidad QR"

            # ── Actualizar variable global de forma segura ──
            with _lock:
                ball_position = pos

            tracker.draw(frame, label)

        elif tracker.bbox is not None:
            tracker.draw(frame, "PERDIDA")

        cv2.imshow("V3D Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:   # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Hilo de robot ─────────────────────────────────────────────────────────────
def robot_loop() -> None:
    """Lee ball_position y mueve el robot 'bola' de RoboDK a esa posición."""
    robot = getRobot(ROBOT_NAME)
    if robot is None:
        print(f"[Robot] No se encontró el robot '{ROBOT_NAME}' en RoboDK.")
        return

    setSpeed(robot, ROBOT_SPEED_LIN, ROBOT_SPEED_LIN, ROBOT_SPEED_ANG, ROBOT_SPEED_ANG)
    print(f"[Robot] Robot '{ROBOT_NAME}' listo.")

    last_xyz = None   # evita mover el robot si la posición no ha cambiado

    while True:
        # ── Leer variable global de forma segura ──
        with _lock:
            pos = ball_position

        if pos and pos["xyz_mm"] is not None:
            xyz = pos["xyz_mm"]

            if xyz != last_xyz:
                # Mover el robot al frame objetivo
                moveTo(robot, xyz, "MoveJ")

                last_xyz = xyz
                print(f"[Robot] Moviendo a  X:{xyz[0]}  Y:{xyz[1]}  Z:{xyz[2]} mm")

        time.sleep(0.05)   # 20 Hz


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    # El robot corre en un hilo secundario (daemon: muere al cerrar la app)
    robot_thread = threading.Thread(target=robot_loop, name="RobotThread", daemon=True)
    robot_thread.start()

    # La cámara corre en el hilo principal (OpenCV en Windows lo requiere)
    camera_loop()


if __name__ == "__main__":
    main()
