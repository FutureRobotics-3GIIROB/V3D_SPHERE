import cv2
import numpy as np

MIN_RADIUS = 10
MIN_CIRCULARITY = 0.70   # stricter: a real ball is close to 1.0
MIN_FILL_RATIO = 0.65    # contour area / enclosing-circle area


def _score_candidate(contour):
    """Return (score, center, radius, bbox) or None if the contour is not ball-like."""
    area = cv2.contourArea(contour)
    if area < np.pi * MIN_RADIUS ** 2:
        return None

    perim = cv2.arcLength(contour, True)
    if perim == 0:
        return None

    circularity = 4 * np.pi * area / (perim ** 2)
    if circularity < MIN_CIRCULARITY:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(contour)
    if r < MIN_RADIUS:
        return None

    # Fill ratio: how much of the enclosing circle is actually filled
    fill = area / (np.pi * r * r)
    if fill < MIN_FILL_RATIO:
        return None

    # Combined score: prefer large, circular, well-filled blobs
    score = area * circularity * fill
    return score, (int(cx), int(cy)), int(r), cv2.boundingRect(contour)


def _candidates_from_contours(mask):
    """Evaluate all external contours in *mask* and return scored candidates."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        res = _score_candidate(c)
        if res is not None:
            results.append(res)
    return results


def _candidates_from_hough(gray):
    """Use HoughCircles as a complementary circle detector."""
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
                continue
            x, y = int(cx) - r, int(cy) - r
            bbox = (max(x, 0), max(y, 0), 2 * r, 2 * r)
            score = np.pi * r * r          # use area as score baseline
            results.append((score, (int(cx), int(cy)), r, bbox))
    return results


def detect_ball(frame):
    """Return {'center', 'radius', 'bbox'} for the best circular blob, or None.

    Uses a two-pronged approach:
      1. Otsu threshold + morphological cleanup ➜ contour analysis
      2. HoughCircles on blurred grayscale
    The highest-scoring candidate across both methods wins.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # --- Method 1: adaptive binary mask ---
    # Otsu picks the threshold automatically based on the image histogram
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological open (remove small noise) then close (fill small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates = _candidates_from_contours(bw)

    # --- Method 2: Hough circle detection ---
    candidates += _candidates_from_hough(blur)

    if not candidates:
        return None

    # Pick the best candidate by combined score
    best = max(candidates, key=lambda c: c[0])
    _, center, radius, bbox = best
    return {'center': center, 'radius': radius, 'bbox': bbox}


def draw_ball(frame, det, color=(0, 255, 0)):
    if det:
        cv2.circle(frame, det['center'], det['radius'], color, 2)
        cv2.circle(frame, det['center'], 3, color, -1)
