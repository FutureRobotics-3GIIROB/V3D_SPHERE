import cv2  # Librería para procesamiento de imágenes
import numpy as np  # Librería para operaciones numéricas


class QRDepth:
    """
    Clase para estimar la profundidad usando un código QR.
    """
    def __init__(self):
        self._det = cv2.QRCodeDetector()  # Detector de QR
        self.side_mm = None       # Lado del QR en mm (extraído del contenido)
        self.depth_z = None       # Última estimación de profundidad Z (mm)
        self.corners = None       # Últimos 4 puntos de esquina detectados

    def update(self, frame, focal_px):
        """
        Detecta, decodifica y estima la profundidad. Devuelve las esquinas o None.
        """
        data, pts, _ = self._det.detectAndDecode(frame)
        if pts is None or len(pts) == 0:
            self.corners = None
            return None

        self.corners = pts[0]
        if data:
            try:
                v = float(data.strip())  # Extrae el valor del lado del QR
                if v > 0:
                    self.side_mm = v
            except ValueError:
                pass

        # Si tenemos el tamaño del QR, estimamos la profundidad
        if self.side_mm is not None:
            side_px = float(np.mean([np.linalg.norm(self.corners[(i+1) % 4] - self.corners[i]) for i in range(4)]))
            if side_px > 0:
                self.depth_z = (self.side_mm * focal_px) / side_px
        return self.corners

    def draw(self, frame):
        """
        Dibuja el QR y la información de profundidad sobre el frame.
        """
        if self.corners is None:
            return
        pts = self.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1) % 4]), (255, 0, 255), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        txt = ''
        if self.side_mm is not None:
            txt += f'QR:{self.side_mm:.0f}mm'
        if self.depth_z is not None:
            txt += f' Z:{self.depth_z:.0f}mm'
        if txt:
            cv2.putText(frame, txt, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


def focal_length(frame_w, hfov=60.0):
    """
    Estima la distancia focal (en píxeles) a partir del campo de visión horizontal.
    """
    return frame_w / (2 * np.tan(np.radians(hfov / 2)))


def pixel_to_xyz(px, py, z, f, cx, cy):
    """
    Proyecta un píxel y profundidad a coordenadas XYZ en mm.
    """
    return round((px - cx) * z / f, 1), round((py - cy) * z / f, 1), round(z, 1)
