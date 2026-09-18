"""Screen/board quadrilateral localisation: Canny -> contours -> approxPolyDP."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def order_corners(pts):
    """Consistent order: tl, tr, br, bl via sum/diff trick."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_screen_quad(frame_bgr):
    """Return ordered 4x2 corners or None (fallback to full frame)."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, C.CANNY_LOW, C.CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(w * h)
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * C.MIN_QUAD_AREA_RATIO:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, C.APPROX_EPSILON_RATIO * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2)
        # aspect check on bounding rect
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0:
            continue
        aspect = bw / float(bh)
        if not (C.ASPECT_MIN <= aspect <= C.ASPECT_MAX):
            continue
        if area > best_area:
            best_area = area
            best = quad
    if best is None:
        return None
    return order_corners(best)


def full_frame_quad(frame_bgr):
    h, w = frame_bgr.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
