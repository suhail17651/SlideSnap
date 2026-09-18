"""Unit tests: build deduplication (superset text + similar image collapse)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.assemble import dedup as DD  # noqa: E402


def _slide(text, seed=0):
    rng = np.random.RandomState(seed)
    img = np.full((90, 160, 3), 255, np.uint8)
    img[10:80, 10:150] = rng.randint(0, 255, (70, 140, 3), dtype=np.uint8)
    return {"text": text, "rectified": img, "timestamp": 0.0,
            "timestamp_start": 0.0}


def test_build_superset_collapses():
    a = _slide("convolution layers pooling", seed=1)
    b = _slide("convolution layers pooling stride receptive fields", seed=1)
    kept, removed = DD.deduplicate([a, b])
    assert removed == 1 and len(kept) == 1
    # final (most complete) text wins, earliest timestamp preserved
    assert "receptive" in kept[0]["text"]


def test_distinct_slides_kept():
    a = _slide("convolution layers pooling", seed=1)
    b = _slide("kalman filter tracking sort", seed=2)
    kept, removed = DD.deduplicate([a, b])
    assert removed == 0 and len(kept) == 2


def test_empty():
    assert DD.deduplicate([]) == ([], 0)
