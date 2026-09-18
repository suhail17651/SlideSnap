"""Homography rectification: getPerspectiveTransform + warpPerspective."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C
from .screen_detect import find_screen_quad, full_frame_quad


def rectify(frame_bgr, quad=None):
    """Warp quad to WARP_WIDTH x WARP_HEIGHT top-down canvas.
    Returns (warped, quad_used, fallback_bool)."""
    if quad is None:
        quad = find_screen_quad(frame_bgr)
    fallback = False
    if quad is None:
        quad = full_frame_quad(frame_bgr)
        fallback = True
    dst = np.array([[0, 0], [C.WARP_WIDTH - 1, 0],
                    [C.WARP_WIDTH - 1, C.WARP_HEIGHT - 1],
                    [0, C.WARP_HEIGHT - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst)
    warped = cv2.warpPerspective(frame_bgr, M, (C.WARP_WIDTH, C.WARP_HEIGHT))
    return warped, np.asarray(quad, dtype=np.float32), fallback


def rectify_keyframes(keyframes):
    out = []
    for k in keyframes:
        warped, quad, fb = rectify(k["frame_full"])
        k2 = dict(k)
        k2["rectified"] = warped
        k2["quad"] = quad
        k2["rectify_fallback"] = fb
        out.append(k2)
    return out
