"""Unit tests: homography rectification (identity + angled-quad lock-on)."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src.geometry import screen_detect as SD  # noqa: E402
from src.geometry import rectify as RC  # noqa: E402


def _slide1280():
    img = np.full((720, 1280, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (1280, 120), (20, 40, 90), -1)
    cv2.putText(img, "Slide One Title", (300, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    cv2.putText(img, "Convolution layers pool", (300, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.rectangle(img, (360, 460), (920, 640), (20, 40, 90), -1)
    return img


def test_order_corners_consistent():
    pts = np.array([[100, 200], [500, 100], [520, 400], [80, 420]])
    q = SD.order_corners(pts)
    assert q.shape == (4, 2)
    # tl has min sum, br max sum
    assert q[0].sum() <= q[1].sum() and q[0].sum() <= q[3].sum()
    assert q[2].sum() >= q[1].sum() and q[2].sum() >= q[3].sum()


def test_rectify_output_size():
    warped, quad, fb = RC.rectify(_slide1280())
    assert warped.shape[1] == C.WARP_WIDTH and warped.shape[0] == C.WARP_HEIGHT


def test_rectify_angled_quad():
    img = _slide1280()
    src = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])
    dst = np.float32([[120, 40], [1220, 90], [1140, 690], [60, 660]])
    M = cv2.getPerspectiveTransform(src, dst)
    angled = cv2.warpPerspective(img, M, (1280, 720), borderValue=(40, 40, 40))
    warped, quad, fb = RC.rectify(angled)
    assert fb is False, "should lock onto the projected screen quad"
    assert warped.shape[:2] == (C.WARP_HEIGHT, C.WARP_WIDTH)
