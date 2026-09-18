"""Sharpness scoring: variance of Laplacian + Tenengrad cross-check."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def variance_of_laplacian(frame_bgr):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=C.LAP_KERNEL)
    return float(lap.var())


def tenengrad(frame_bgr):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=C.TENENGRAD_KERNEL)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=C.TENENGRAD_KERNEL)
    return float(np.mean(gx ** 2 + gy ** 2))


def score_frames(frames):
    """Return list of (laplacian, tenengrad) per frame."""
    return [(variance_of_laplacian(f), tenengrad(f)) for f in frames]


def best_index(frames):
    """Best frame by Laplacian; also report Tenengrad agreement."""
    scores = score_frames(frames)
    laps = [s[0] for s in scores]
    tens = [s[1] for s in scores]
    i_lap = int(np.argmax(laps))
    i_ten = int(np.argmax(tens))
    return {"index": i_lap, "laplacian": laps[i_lap],
            "tenengrad": tens[i_lap], "agree": bool(i_lap == i_ten),
            "ten_best": i_ten, "scores": scores}
