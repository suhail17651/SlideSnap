"""Deduplication: collapse slide builds (superset text + similar image)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C

try:
    from src.segmentation.difference import ssim_score
except ImportError:
    from segmentation.difference import ssim_score


def _words(text):
    return set(w.lower() for w in str(text).split() if w.strip())


def is_build(prev_slide, next_slide):
    """True if next is a build-extension of prev (keep only final)."""
    wp = _words(prev_slide.get("text", ""))
    wn = _words(next_slide.get("text", ""))
    if not wp:
        # fall back to image-only: very similar images collapse
        try:
            s = ssim_score(prev_slide["rectified"], next_slide["rectified"])
            return s > 0.95
        except Exception:
            return False
    overlap = len(wp & wn) / max(1, len(wp))
    if overlap < C.DEDUP_TEXT_SUPERSET_RATIO:
        return False
    # next must contain strictly more (or equal with image similar)
    if len(wn) <= len(wp):
        try:
            s = ssim_score(prev_slide["rectified"], next_slide["rectified"])
            return s > C.DEDUP_SSIM
        except Exception:
            return False
    try:
        s = ssim_score(prev_slide["rectified"], next_slide["rectified"])
    except Exception:
        return True  # text says superset; trust it
    return s > C.DEDUP_SSIM or len(wn) > len(wp)


def deduplicate(slides):
    """Collapse consecutive builds. Returns (kept, removed_count)."""
    if not slides:
        return [], 0
    kept = [slides[0]]
    removed = 0
    for s in slides[1:]:
        if is_build(kept[-1], s):
            # replace with the more complete (final) one, keep earliest timestamp?
            # Keep final image/text but preserve first timestamp for bookmark range.
            s2 = dict(s)
            s2["timestamp_start"] = kept[-1].get("timestamp_start", kept[-1].get("timestamp"))
            kept[-1] = s2
            removed += 1
        else:
            kept.append(s)
    return kept, removed
