# pyright: reportMissingImports=false

"""
main_robot.py
─────────────
Combina el tracking de la pelota con el control del robot en RoboDK.

  • Hilo de cámara  → detecta la pelota (por ArUco o color) y actualiza la variable global
                      `ball_position` con los datos pixel + XYZ en mm.
  • Hilo de robot   → lee `ball_position` y mueve el robot "bola" de RoboDK
                      a la posición detectada.
"""

import threading
import time

import cv2
from aruco_lib import (
    BALL_ARUCO_ID,
    ArucoTracker,
)
from ball_detector import detect_ball, draw_ball
from calibracion import HomographyCalibrator
from FuncionesBase import createOrUpdateTarget, getRobot
from FuncionesRobot import moveTo, setSpeed
from robodk.robolink import TargetReachError
from robodk.robomath import roty, transl
from tracker import BallTracker

# ── Configuración ─────────────────────────────────────────────────────────────
ROBOT_NAME = "Bola"  # nombre del objeto marcador en RoboDK
UR3E_NAME = "UR3e"  # nombre del robot UR3e en RoboDK  ← ajusta si es distinto
TARGET_NAME = "BallApproach"  # nombre del target que se crea/reutiliza en RoboDK
TARGET_FRAME = "BallTarget"  # nombre del frame objetivo en RoboDK
REDETECT_EVERY = 5  # frames sin tracking antes de re-detectar
ROBOT_SPEED_LIN = 200  # mm/s   velocidad lineal
ROBOT_SPEED_ANG = 30  # deg/s  velocidad angular
# Modo de detección: True = usar ArucoTracker, False = solo detección por color
USE_ARUCO_TRACKER = True

# ── Variable global compartida ────────────────────────────────────────────────
# Formato: {'pixel': [px, py], 'xyz_mm': [X, Y, Z]}  ó  None
ball_position: dict | None = None
_lock = threading.Lock()  # acceso seguro entre hilos


def camera_loop_aruco() -> None:
    """Captura video usando ArucoTracker con homografía."""
    global ball_position

    # Cargar calibración
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("[Cámara] ERROR: No se encontró calibración")
        print("[Cámara] Ejecuta: python calibracion.py --capture")
        return

    homography = calib_data["homography_matrix"]

    tracker = ArucoTracker(camera_source=0, marker_size_m=0.05, show_axes=False, debug_mode=False)

    if not tracker.start():
        print("[Cámara] ERROR: No se pudo iniciar ArucoTracker")
        return

    print("[Cámara] ArucoTracker iniciado")

    try:
        while True:
            frame_data = tracker.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue

            frame = frame_data.frame
            h, w = frame.shape[:2]

            ball_found = False
            for det in frame_data.aruco_detections:
                if det.id == BALL_ARUCO_ID:
                    ball_center_px = det.center_px
                    if det.world_position:
                        xyz = [
                            det.world_position[0] * 1000,
                            det.world_position[1] * 1000,
                            det.world_position[2] * 1000,
                        ]
                        pos = {"pixel": list(ball_center_px), "xyz_mm": xyz}

                        with _lock:
                            ball_position = pos

                        ball_found = True
                        break

            if not ball_found:
                ball_det = detect_ball(frame)
                if ball_det:
                    ball_center_px = ball_det["center"]
                    draw_ball(frame, ball_det, color=(0, 255, 0))

                    point_transformed = HomographyCalibrator.transform_point(
                        ball_center_px, homography
                    )
                    xyz_mm = list(point_transformed) + [0]

                    pos = {"pixel": list(ball_center_px), "xyz_mm": xyz_mm}
                    with _lock:
                        ball_position = pos

            cv2.imshow("V3D Tracking (Robot)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        tracker.stop()
        cv2.destroyAllWindows()


# ── Hilo de cámara (modo original) ───────────────────────────────────────────
def camera_loop() -> None:
    """Captura video con homografía."""
    global ball_position

    # Cargar calibración
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("[Cámara] ERROR: No se encontró calibración")
        print("[Cámara] Ejecuta: python calibracion.py --capture")
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
    print(f"[Cámara] Frame {w}x{h}")

    tracker = BallTracker()
    det = detect_ball(frame)
    if det:
        tracker.init(frame, det["bbox"])
        print(f"[Cámara] Pelota encontrada en {det['center']}")
    else:
        print("[Cámara] Buscando pelota...")

    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker.ok or tracker.bbox is not None:
            ok, _ = tracker.update(frame)
        else:
            ok = False

        last_det = None
        if not ok:
            lost_count += 1
            if lost_count >= REDETECT_EVERY or tracker.bbox is None:
                last_det = detect_ball(frame)
                if last_det:
                    tracker.init(frame, last_det["bbox"])
                    lost_count = 0
                    print(f"[Cámara] Pelota re-detectada en {last_det['center']}")
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

        c = tracker.center
        if c and tracker.ok:
            point_transformed = HomographyCalibrator.transform_point(c, homography)
            xyz = list(point_transformed) + [0]
            pos = {"pixel": list(c), "xyz_mm": xyz}
            label = f"X:{xyz[0]:.1f}  Y:{xyz[1]:.1f}  Z:{xyz[2]:.1f}"

            with _lock:
                ball_position = pos

            tracker.draw(frame, label)

        elif tracker.bbox is not None:
            tracker.draw(frame, "PERDIDA")

        cv2.imshow("V3D Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Helpers de acercamiento máximo ────────────────────────────────────────
def _pose_xyz(pose):
    """Extrae (x, y, z) de un Mat de RoboDK usando .rows para evitar IndexError."""
    return pose.rows[0][3], pose.rows[1][3], pose.rows[2][3]


def _is_reachable(robot, pose) -> bool:
    """Comprueba sin mover el robot si una pose es alcanzable (IK)."""
    try:
        joints = robot.SolveIK(pose)
        # Mat.rows es una lista de listas; si está vacía, no hay solución
        if hasattr(joints, "rows"):
            return len(joints.rows) > 0 and len(joints.rows[0]) > 0
        return hasattr(joints, "__len__") and len(joints) > 0
    except Exception:
        return False


def _move_closest(robot, target_pose, steps: int = 10) -> bool:
    """
    Intenta mover el robot al objetivo.
    Si no es alcanzable, se acerca lo máximo posible.
    Nunca lanza excepción.
    """

    try:
        robot.MoveJ(target_pose)
        return True
    except TargetReachError:
        pass

    current = robot.Pose()

    tx, ty, tz = _pose_xyz(current)
    gx, gy, gz = _pose_xyz(target_pose)

    lo, hi = 0.0, 1.0
    best_pose = current

    # Búsqueda binaria
    for _ in range(steps):
        mid = (lo + hi) / 2.0

        interp = transl(
            tx + mid * (gx - tx),
            ty + mid * (gy - ty),
            tz + mid * (gz - tz),
        )

        try:
            robot.MoveJ(interp, blocking=False)
            best_pose = interp
            lo = mid
        except TargetReachError:
            hi = mid

    # Intentar mover al mejor punto encontrado
    try:
        robot.MoveJ(best_pose)
    except TargetReachError:
        pass

    return False


# ── Hilo de robot ─────────────────────────────────────────────────────────────
def robot_loop() -> None:
    """
    Lee ball_position:
      - Mueve el robot 'Bola' a la posición detectada (como antes).
      - Crea/actualiza un target en esa misma posición y mueve el UR3e hacia él.
    """
    bola = getRobot(ROBOT_NAME)
    if bola is None:
        print(f"[Robot] No se encontró el robot '{ROBOT_NAME}' en RoboDK.")
        return

    ur3e = getRobot(UR3E_NAME)
    if ur3e is None:
        print(f"[Robot] No se encontró el robot '{UR3E_NAME}' en RoboDK.")
        return

    setSpeed(bola, ROBOT_SPEED_LIN, ROBOT_SPEED_LIN, ROBOT_SPEED_ANG, ROBOT_SPEED_ANG)
    setSpeed(ur3e, ROBOT_SPEED_LIN, ROBOT_SPEED_LIN, ROBOT_SPEED_ANG, ROBOT_SPEED_ANG)
    print(f"[Robot] '{ROBOT_NAME}' y '{UR3E_NAME}' listos.")

    last_xyz = None  # evita mover si la posición no ha cambiado

    while True:
        # ── Leer variable global de forma segura ──
        with _lock:
            pos = ball_position

        if pos and pos["xyz_mm"] is not None:
            xyz = pos["xyz_mm"]
            xyz_flat = [xyz[1], xyz[0], 0]

            if xyz_flat != last_xyz:
                bx, by, bz = xyz[1] - 265.000, xyz[0], 50

                # Mover robot Bola (como antes)
                moveTo(bola, xyz_flat, "MoveJ")

                # Crear/actualizar target en la posición exacta de la pelota (visual)
                ball_pose = transl(bx, by, bz) * roty(3.1416)
                createOrUpdateTarget(TARGET_NAME, ur3e, ball_pose)

                # Mover UR3e lo más cerca posible del objetivo
                reached = _move_closest(ur3e, ball_pose)
                if not reached:
                    print(
                        f"[Robot] UR3e: máximo acercamiento a  X:{bx:.1f}  Y:{by:.1f}  Z:{bz:.1f} mm"
                    )

                last_xyz = xyz_flat
                print(
                    f"[Robot] Bola → X:{bx:.1f}  Y:{by:.1f}  Z:0  |  UR3e → X:{bx:.1f}  Y:{by:.1f}  Z:{bz:.1f} mm"
                )

        time.sleep(0.05)  # 20 Hz


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    # El robot corre en un hilo secundario (daemon: muere al cerrar la app)
    robot_thread = threading.Thread(target=robot_loop, name="RobotThread", daemon=True)
    robot_thread.start()

    # La cámara corre en el hilo principal (OpenCV en Windows lo requiere)
    if USE_ARUCO_TRACKER:
        print("[Main] Usando ArucoTracker para detección")
        camera_loop_aruco()
    else:
        print("[Main] Usando detección por color (modo original)")
        camera_loop()


if __name__ == "__main__":
    main()
