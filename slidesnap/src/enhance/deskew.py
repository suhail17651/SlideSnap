"""Hough-based deskew: dominant text-baseline angle -> rotate upright."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def estimate_skew_angle(gray_or_bw):
    """Return skew angle in degrees (positive = rotate CCW to fix)."""
    if len(gray_or_bw.shape) == 3:
        gray = cv2.cvtColor(gray_or_bw, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bw
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=150)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0, :]:
        deg = theta * 180.0 / np.pi - 90.0  # 0 => horizontal
        if abs(deg) <= C.DESKEW_MAX_ANGLE + 10:
            angles.append(deg)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(frame, angle=None):
    """Rotate `frame` (BGR color OR single-channel gray) by `angle` degrees.

    Accepts both: with angle=None the skew is estimated from a gray
    conversion (color input) or directly (gray input). Returns (fixed, angle).
    """
    if angle is None:
        if len(frame.shape) == 3:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            g = frame
        angle = estimate_skew_angle(g)
    # clamp to search range
    angle = float(np.clip(angle, -C.DESKEW_MAX_ANGLE, C.DESKEW_MAX_ANGLE))
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    fixed = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return fixed, angle
