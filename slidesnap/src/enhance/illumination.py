"""Illumination flattening: morphological closing estimates background field."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def estimate_background(gray):
    k = max(21, gray.shape[1] // C.BG_KERNEL_RATIO)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def flatten_illumination(frame_bgr):
    """Divide out background field; returns (flat_gray, background)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg = estimate_background(gray)
    bg_f = bg.astype(np.float32) + 1.0
    flat = (gray.astype(np.float32) / bg_f) * float(np.mean(bg_f))
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    # also normalize color version mildly (keep diagrams intact)
    return flat, bg


def naive_global_equalize(frame_bgr):
    """Baseline for the report figure: global equalisation (amplifies noise)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)
