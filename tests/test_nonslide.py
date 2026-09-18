"""Unit tests: non-slide (speaker) filter — sparse bottom-hugging text drops."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.assemble import nonslide as NS  # noqa: E402

H = 450


def _words(n, y0=400, y1=430):
    # n fake word boxes in a narrow bottom band (watermark-like)
    return [{"text": "w%d" % i, "conf": 0.9,
             "box": (10 + i * 40, y0, 30, y1 - y0)} for i in range(n)]


def _slide(words):
    return {"text": " ".join(w["text"] for w in words), "words": words,
            "rectified": np.zeros((H, 800, 3), np.uint8), "timestamp": 0.0,
            "timestamp_start": 0.0}


def test_speaker_frame_drops():
    # measured speaker case: 4 words, yspread ~0.55 -> non-slide
    assert NS.is_nonslide(_words(4), H) is True


def test_sparse_but_spread_survives():
    # few words spread over the body = real sparse slide -> keep
    ws = [{"text": "a", "conf": 0.9, "box": (10, 20, 30, 20)},
          {"text": "b", "conf": 0.9, "box": (10, 200, 30, 20)},
          {"text": "c", "conf": 0.9, "box": (10, 400, 30, 20)}]
    assert NS.is_nonslide(ws, H) is False


def test_dense_slide_survives():
    # measured true slides: 5-26 words -> keep regardless of spread
    assert NS.is_nonslide(_words(5), H) is False
    assert NS.is_nonslide(_words(26, y0=20, y1=430), H) is False


def test_filter_splits_with_reasons():
    kept, dropped = NS.filter_slides([_slide(_words(4)), _slide(_words(26, 20, 430))])
    assert len(kept) == 1 and len(dropped) == 1
    assert "non-slide" in dropped[0]["reason"]
