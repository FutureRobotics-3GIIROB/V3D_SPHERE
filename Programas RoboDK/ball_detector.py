import cv2
import numpy as np

HSV_LOWER = np.array([0, 51, 0])
HSV_UPPER = np.array([121, 255, 217])

MIN_AREA = 500  # minimum blob area in pixels


def detect_ball(frame):
    """Threshold → largest connected blob → center.

    Returns {'center', 'radius', 'bbox'} or None.
    """
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Remove noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Largest blob by area
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    x, y, w, h = cv2.boundingRect(largest)
    return {'center': (int(cx), int(cy)), 'radius': int(radius), 'bbox': (x, y, w, h)}


def draw_ball(frame, det, color=(0, 255, 0)):
    if det:
        cv2.circle(frame, det['center'], det['radius'], color, 2)
        cv2.circle(frame, det['center'], 3, color, -1)
