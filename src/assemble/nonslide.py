"""Non-slide filter: drop keyframes showing the presenter, not a slide.

A speaker frame has almost no OCR text and what little there is hugs the
frame bottom (watermark/footer) instead of spreading over the body. Both
conditions must hold (AND), so a sparse-but-real slide (few words spread
wide) and a dense footer (many words, narrow band) both survive.

Thresholds in config.py (NONSLIDE_MIN_WORDS / NONSLIDE_MAX_YSPREAD), measured
on a real CS231n lecture clip — see the tuning comment there. Pure function
over (words, image height); pipeline keeps the dropped ones in the result as
`filtered` with reasons, so nothing vanishes silently.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def word_spread(words, height):
    """Vertical spread of word boxes as a fraction of image height (0..1)."""
    if not words or height <= 0:
        return 0.0
    tops = [w["box"][1] for w in words]
    bots = [w["box"][1] + w["box"][3] for w in words]
    return (max(bots) - min(tops)) / float(height)


def is_nonslide(words, height):
    """True if this keyframe looks like a presenter shot, not a slide."""
    if not C.NONSLIDE_ENABLE:
        return False
    n = len(words or [])
    if n >= C.NONSLIDE_MIN_WORDS:
        return False
    return word_spread(words or [], height) < C.NONSLIDE_MAX_YSPREAD


def filter_slides(slides):
    """Split slides into (kept, filtered). Filtered entries carry `reason`."""
    kept, dropped = [], []
    for s in slides:
        img = s.get("rectified")
        h = img.shape[0] if img is not None else 0
        words = s.get("words", []) or []
        if is_nonslide(words, h):
            d = dict(s)
            d["reason"] = (f"non-slide: {len(words)} words, "
                           f"yspread={word_spread(words, h):.2f}")
            dropped.append(d)
        else:
            kept.append(s)
    return kept, dropped
