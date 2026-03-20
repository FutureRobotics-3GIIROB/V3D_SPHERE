import json, time, cv2  # Importamos las librerías necesarias

from ball_detector import detect_ball, draw_ball  # Funciones para detectar y dibujar la bola
from qr_depth import QRDepth, focal_length, pixel_to_xyz  # Funciones para calcular profundidad usando QR
from tracker import BallTracker  # Clase para seguir la bola

OUTPUT = 'positions.json'  # Archivo donde se guardan las posiciones, para que RoboDK pueda leerlas
REDETECT_EVERY = 5   # Número de frames antes de volver a buscar la bola si se pierde


def main():
    cap = cv2.VideoCapture(0)  # Abrimos la cámara
    if not cap.isOpened():
        raise RuntimeError('No se puede abrir la webcam')

    # Calentamos la cámara leyendo algunos frames
    for _ in range(30):
        cap.read()
    time.sleep(0.1)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('No se obtuvo frame de la webcam')

    h, w = frame.shape[:2]  # Obtenemos alto y ancho del frame
    f_px = focal_length(w)  # Calculamos la distancia focal en píxeles
    cx, cy = w / 2.0, h / 2.0  # Centro del frame
    print(f'Frame {w}x{h}  |  focal ~{f_px:.0f}px')

    # Detectamos la bola automáticamente en el primer frame
    tracker = BallTracker()  # Creamos el tracker
    det = detect_ball(frame)  # Detectamos la bola
    if det:
        tracker.init(frame, det['bbox'])  # Inicializamos el tracker con la bola
        print(f'Bola encontrada en {det["center"]}')
    else:
        print('No hay bola aún — seguimos buscando.')

    qr = QRDepth()  # Inicializamos el lector de QR para profundidad
    lost_count = 0  # Contador de frames perdidos

    while True:
        ret, frame = cap.read()  # Leemos un nuevo frame
        if not ret:
            break

        # Actualizamos la profundidad usando QR
        qr.update(frame, f_px)
        qr.draw(frame)

        # Seguimos la bola
        if tracker.ok or tracker.bbox is not None:
            ok, _ = tracker.update(frame)
        else:
            ok = False

        # Si se pierde la bola, intentamos redetectar
        if not ok:
            lost_count += 1
            if lost_count >= REDETECT_EVERY or tracker.bbox is None:
                det = detect_ball(frame)  # Volvemos a buscar la bola
                if det:
                    tracker.init(frame, det['bbox'])  # Reinicializamos el tracker
                    lost_count = 0
                    print(f'Bola redetectada en {det["center"]}')
                else:
                    # Mostramos mensaje de búsqueda en pantalla
                    cv2.putText(frame, 'Buscando...', (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            lost_count = 0  # Si la bola está siendo seguida, reiniciamos el contador

        # Visualizamos el círculo de detección de la bola
        draw_ball(frame, detect_ball(frame))

        # Calculamos y mostramos la posición XYZ
        pos = None
        c = tracker.center  # Centro de la bola
        if c and tracker.ok:
            if qr.depth_z is not None:
                xyz = pixel_to_xyz(c[0], c[1], qr.depth_z, f_px, cx, cy)  # Convertimos a coordenadas reales
                label = f'X:{xyz[0]} Y:{xyz[1]} Z:{xyz[2]}'
                pos = {'pixel': list(c), 'xyz_mm': list(xyz)}
            else:
                label = 'Sin profundidad QR'
                pos = {'pixel': list(c), 'xyz_mm': None}
            tracker.draw(frame, label)  # Dibujamos la información en el frame
        elif tracker.bbox is not None:
            tracker.draw(frame, 'PERDIDA')  # Indicamos que la bola está perdida

        # Guardamos la posición en el archivo si existe
        if pos:
            print(json.dumps(pos))
            try:
                with open(OUTPUT, 'w') as f:
                    json.dump(pos, f)
            except OSError:
                pass

        # Mostramos el frame en pantalla
        cv2.imshow('V3D Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Salimos si se presiona ESC
            break

    cap.release()  # Liberamos la cámara
    cv2.destroyAllWindows()  # Cerramos todas las ventanas


# Punto de entrada del script
if __name__ == '__main__':
    main()
