"""
HSV Tuner — run this script to find the correct HSV thresholds for your ball.

Usage:
    python hsv_tuner.py          # uses webcam (index 0)
    python hsv_tuner.py 1        # uses webcam index 1
    python hsv_tuner.py img.jpg  # uses a still image (loops)

The trackbars let you adjust H/S/V min-max in real time.
When the mask shows only the ball (white blob, black background), note the
six values and paste them into ball_detector.py as HSV_LOWER / HSV_UPPER.
Press 'q' or Esc to quit.
"""

import sys
import cv2
import numpy as np

# ── default starting values (current ball_detector.py values) ─────────────
DEFAULTS = dict(H_min=48, H_max=93, S_min=26, S_max=210, V_min=0, V_max=132)

WINDOW = "HSV Tuner"


def nothing(_):
    pass


def build_trackbars():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 900, 600)
    for name, val in DEFAULTS.items():
        upper = 180 if name.startswith("H") else 255
        cv2.createTrackbar(name, WINDOW, val, upper, nothing)


def get_range():
    h_min = cv2.getTrackbarPos("H_min", WINDOW)
    h_max = cv2.getTrackbarPos("H_max", WINDOW)
    s_min = cv2.getTrackbarPos("S_min", WINDOW)
    s_max = cv2.getTrackbarPos("S_max", WINDOW)
    v_min = cv2.getTrackbarPos("V_min", WINDOW)
    v_max = cv2.getTrackbarPos("V_max", WINDOW)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return lower, upper


def process(frame, lower, upper):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, result


def overlay_text(frame, lower, upper):
    text = (
        f"HSV_LOWER = np.array([{lower[0]}, {lower[1]}, {lower[2]}])  "
        f"HSV_UPPER = np.array([{upper[0]}, {upper[1]}, {upper[2]}])"
    )
    cv2.putText(frame, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    try:
        source = int(source)          # webcam index
    except ValueError:
        pass                          # image/video path

    is_image = isinstance(source, str) and not source.endswith((".mp4", ".avi", ".mov"))

    if is_image:
        static = cv2.imread(source)
        if static is None:
            print(f"Could not open image: {source}")
            return
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open video source: {source}")
            return

    build_trackbars()
    print("Adjust sliders until only the ball is white in the mask view.")
    print("Press 'q' or Esc to quit and print the final values.")

    while True:
        if is_image:
            frame = static.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        lower, upper = get_range()
        mask, result = process(frame, lower, upper)

        # Stack: original | masked result | binary mask
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack([
            cv2.resize(frame,   (400, 300)),
            cv2.resize(result,  (400, 300)),
            cv2.resize(mask_bgr,(400, 300)),
        ])
        overlay_text(display, lower, upper)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break

    lower, upper = get_range()
    print("\n── Copy these into ball_detector.py ──")
    print(f"HSV_LOWER = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
    print(f"HSV_UPPER = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")

    if not is_image:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
