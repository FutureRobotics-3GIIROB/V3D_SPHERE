import cv2  # Librería para procesamiento de imágenes
import numpy as np  # Librería para operaciones numéricas

MIN_RADIUS = 10  # Radio mínimo para considerar un objeto como bola
MIN_CIRCULARITY = 0.70   # Circularidad mínima, una bola real se acerca a 1.0
MIN_FILL_RATIO = 0.65    # Proporción mínima de relleno: área del contorno / área del círculo envolvente


def _score_candidate(contour):
    """
    Calcula la puntuación de un contorno para determinar si es una bola.
    Devuelve (score, center, radius, bbox) o None si el contorno no es válido.
    """
    area = cv2.contourArea(contour)  # Área del contorno
    if area < np.pi * MIN_RADIUS ** 2:
        return None  # Descartar si el área es muy pequeña

    perim = cv2.arcLength(contour, True)  # Perímetro del contorno
    if perim == 0:
        return None

    circularity = 4 * np.pi * area / (perim ** 2)  # Medida de circularidad
    if circularity < MIN_CIRCULARITY:
        return None  # Descartar si no es suficientemente circular

    (cx, cy), r = cv2.minEnclosingCircle(contour)  # Círculo envolvente
    if r < MIN_RADIUS:
        return None  # Descartar si el radio es muy pequeño

    # Proporción de relleno: cuánto del círculo está realmente ocupado
    fill = area / (np.pi * r * r)
    if fill < MIN_FILL_RATIO:
        return None

    # Puntuación combinada: preferir blobs grandes, circulares y bien llenos
    score = area * circularity * fill
    return score, (int(cx), int(cy)), int(r), cv2.boundingRect(contour)


def _candidates_from_contours(mask):
    """
    Evalúa todos los contornos externos en la máscara y devuelve los candidatos puntuados.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        res = _score_candidate(c)
        if res is not None:
            results.append(res)
    return results


def _candidates_from_hough(gray):
    """
    Usa HoughCircles como detector complementario de círculos.
    """
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=40,
        minRadius=MIN_RADIUS, maxRadius=0,
    )
    results = []
    if circles is not None:
        for (cx, cy, r) in np.round(circles[0]).astype(int):
            r = int(r)
            if r < MIN_RADIUS:
                continue  # Descartar círculos pequeños
            x, y = int(cx) - r, int(cy) - r
            bbox = (max(x, 0), max(y, 0), 2 * r, 2 * r)
            score = np.pi * r * r          # Usar el área como puntuación base
            results.append((score, (int(cx), int(cy)), r, bbox))
    return results


def detect_ball(frame):
        """
        Devuelve {'center', 'radius', 'bbox'} para el mejor blob circular, o None.

        Usa dos métodos:
            1. Umbral Otsu + limpieza morfológica ➜ análisis de contornos
            2. HoughCircles en escala de grises difuminada
        El candidato con mayor puntuación entre ambos métodos es el elegido.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        blur = cv2.GaussianBlur(gray, (11, 11), 0)  # Difuminar para reducir ruido

        # --- Método 1: máscara binaria adaptativa ---
        # Otsu elige el umbral automáticamente según el histograma
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Abrir (eliminar ruido pequeño) y cerrar (rellenar huecos pequeños)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidates = _candidates_from_contours(bw)

        # --- Método 2: detección de círculos Hough ---
        candidates += _candidates_from_hough(blur)

        if not candidates:
                return None

        # Elegir el mejor candidato por puntuación combinada
        best = max(candidates, key=lambda c: c[0])
        _, center, radius, bbox = best
        return {'center': center, 'radius': radius, 'bbox': bbox}


def draw_ball(frame, det, color=(0, 255, 0)):
    """
    Dibuja la bola detectada sobre el frame.
    """
    if det:
        cv2.circle(frame, det['center'], det['radius'], color, 2)  # Dibuja el círculo
        cv2.circle(frame, det['center'], 3, color, -1)  # Dibuja el centro
