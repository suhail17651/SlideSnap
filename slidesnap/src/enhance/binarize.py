"""Adaptive binarization (beats Otsu on uneven projector lighting)."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def adaptive_binarize(gray):
    block = C.ADAPTIVE_BLOCK
    if block % 2 == 0:
        block += 1
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, C.ADAPTIVE_C)
    k = np.ones(C.MORPH_OPEN_KERNEL, np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    return bw


def otsu_binarize(gray):
    """Baseline for report comparison figure."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw
