"""Unit tests: keyframe sharpness ranking (LoG variance + Tenengrad agreement)."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.keyframe import sharpness as SH  # noqa: E402


def _slide():
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (640, 60), (20, 40, 90), -1)
    cv2.putText(img, "Slide One Title", (150, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "Convolution layers pool", (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(img, (180, 230), (460, 320), (20, 40, 90), -1)
    return img


def test_sharpness_prefers_sharp():
    sharp = _slide()
    blur = cv2.GaussianBlur(sharp, (15, 15), 0)
    assert SH.variance_of_laplacian(sharp) > SH.variance_of_laplacian(blur)
    assert SH.tenengrad(sharp) > SH.tenengrad(blur)


def test_best_index_agrees():
    sharp = _slide()
    blur = cv2.GaussianBlur(sharp, (15, 15), 0)
    r = SH.best_index([blur, sharp])
    assert r["index"] == 1
    assert r["agree"] is True
    assert r["laplacian"] > 0
