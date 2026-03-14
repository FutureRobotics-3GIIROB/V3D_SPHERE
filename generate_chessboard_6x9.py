from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Internal corners used by OpenCV findChessboardCorners.
INNER_CORNERS = (6, 9)
# Number of drawn squares is corners + 1 per axis.
SQUARES_X = INNER_CORNERS[0] + 1
SQUARES_Y = INNER_CORNERS[1] + 1

# Print settings.
DPI = 300
SQUARE_SIZE_MM = 30.0
MARGIN_MM = 15.0


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    """Convert millimeters to pixels at selected DPI."""
    return int(round(mm * dpi / 25.4))


def build_chessboard_image() -> np.ndarray:
    """Build a white/black chessboard image in grayscale."""
    square_px = mm_to_px(SQUARE_SIZE_MM)
    margin_px = mm_to_px(MARGIN_MM)

    board_w = SQUARES_X * square_px
    board_h = SQUARES_Y * square_px

    image_w = board_w + 2 * margin_px
    image_h = board_h + 2 * margin_px

    # White background to preserve print margins.
    img = np.full((image_h, image_w), 255, dtype=np.uint8)

    for y in range(SQUARES_Y):
        for x in range(SQUARES_X):
            if (x + y) % 2 == 0:
                continue
            x0 = margin_px + x * square_px
            y0 = margin_px + y * square_px
            x1 = x0 + square_px
            y1 = y0 + square_px
            img[y0:y1, x0:x1] = 0

    return img


def add_footer_text(img_gray: np.ndarray) -> np.ndarray:
    """Add calibration metadata text at bottom area."""
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    text = (
        f"Chessboard {INNER_CORNERS[0]}x{INNER_CORNERS[1]} inner corners | "
        f"Square: {SQUARE_SIZE_MM:.1f} mm | DPI: {DPI}"
    )
    cv2.putText(
        img,
        text,
        (20, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def main() -> int:
    """Generate and save a printable 6x9 chessboard pattern."""
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "chessboard_6x9_30mm_300dpi.png"

    board = build_chessboard_image()
    board_with_text = add_footer_text(board)

    ok = cv2.imwrite(str(out_png), board_with_text)
    if not ok:
        print("Failed to write chessboard image.")
        return 1

    print(f"Saved: {out_png}")
    print("Print at 100% scale (no fit-to-page) for accurate calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
