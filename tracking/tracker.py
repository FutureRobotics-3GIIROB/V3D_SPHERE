import cv2  # Librería para procesamiento de imágenes


def _make_csrt():
    """
    Intenta crear un tracker CSRT.
    Devuelve el tracker o None si no se puede crear.
    """
    for fn in [lambda: cv2.legacy.TrackerCSRT_create(),
               lambda: cv2.TrackerCSRT_create(),
               lambda: cv2.TrackerCSRT.create()]:
        try:
            t = fn()
            print(f'Tracker: {type(t).__name__}')
            return t
        except Exception:
            continue
    return None


class BallTracker:
    """
    Clase para seguir la bola usando un tracker CSRT de OpenCV.
    """
    def __init__(self):
        self._tr = None  # Tracker CSRT
        self.bbox = None      # Bounding box: (x, y, w, h)
        self.ok = False      # Estado del tracker

    def init(self, frame, bbox):
        """
        Inicializa el tracker con el frame y la bounding box de la bola.
        """
        self._tr = _make_csrt()
        if self._tr is None:
            raise RuntimeError('No se puede crear el tracker CSRT')
        self._tr.init(frame, bbox)
        self.bbox = tuple(bbox)
        self.ok = True

    def update(self, frame):
        """
        Actualiza el tracker con el nuevo frame.
        Devuelve (ok, bbox): ok indica si el tracking es exitoso.
        """
        if self._tr is None:
            self.ok = False
            return False, self.bbox
        ok, box = self._tr.update(frame)
        self.ok = bool(ok)
        if self.ok:
            self.bbox = tuple(int(v) for v in box)
        return self.ok, self.bbox

    @property
    def center(self):
        """
        Devuelve el centro de la bola (x, y) si existe.
        """
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2

    def draw(self, frame, label='', color=(0, 255, 0)):
        """
        Dibuja la bounding box y el centro de la bola en el frame.
        Si se pasa un label, lo muestra encima.
        """
        if self.bbox is None:
            return
        x, y, w, h = self.bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.circle(frame, self.center, 4, color, -1)
        if label:
            cv2.putText(frame, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
