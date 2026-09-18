"""Unit tests: difference scoring primitives (abs-diff, SSIM, histogram)."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src.segmentation import difference as D  # noqa: E402


def _slide(color, text="Slide One Title", bg=255):
    # Content fills the CENTER (signals score a center ROI): banner, title,
    # bullets and footer all overlap the middle 64% x 64% of the frame.
    # Banner spans full width so its COLOR counts as a slide-level change.
    img = np.full((360, 640, 3), bg, np.uint8)
    cv2.rectangle(img, (0, 0), (640, 60), color, -1)
    cv2.putText(img, text, (150, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "Convolution layers pool", (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Pooling and stride here", (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(img, (180, 230), (460, 320), color, -1)  # filled diagram
    return img


def test_difference_triggers_on_change():
    a = _slide((20, 40, 90), "Slide One Title")
    b = _slide((90, 30, 30), "Slide Two Other")
    sig = D.score_pair(a, b)
    # banner colour + title + filled diagram change: abs-diff must fire.
    assert sig["diff"] > C.DIFF_HIGH, sig
    assert D.high_votes(sig) >= 1, sig
    # full-frame swap (background + content change, like a real cut)
    c = _slide((90, 30, 30), "Slide Two Other", bg=30)
    sig2 = D.score_pair(a, c)
    assert D.high_votes(sig2) >= C.MIN_VOTES, sig2


def test_difference_quiet_on_identical():
    a = _slide((20, 40, 90))
    sig = D.score_pair(a, a.copy())
    assert D.is_stable(sig), sig
    assert D.high_votes(sig) == 0


def test_histogram_primitive_runs():
    a = _slide((20, 40, 90))
    b = _slide((90, 30, 30))
    d_same = D.hist_distance(a, a.copy())
    d_diff = D.hist_distance(a, b)
    assert d_same < 0.05
    assert 0.0 <= d_diff <= 1.0
