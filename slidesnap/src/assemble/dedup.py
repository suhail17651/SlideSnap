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
    """True if next is a build-extension of prev (keep only final).

    Guard: identical OCR text (common on real footage when the OCR backend
    under-reads two different photo slides) must NOT collapse — the image
    check below would pass on any two static-camera frames. Empty/identical
    texts return False unless images are near-duplicates (exact or
    compression-level duplicates).
    """
    wp = _words(prev_slide.get("text", ""))
    wn = _words(next_slide.get("text", ""))
    try:
        s = ssim_score(prev_slide["rectified"], next_slide["rectified"])
    except Exception:
        s = None
    near_dup = s is not None and s > C.DEDUP_NEAR_DUP_SSIM
    if not wp or not wn:
        # fall back to image-only: collapse only near-duplicates
        return near_dup
    if wp == wn:
        # identical OCR text proves nothing (backend may have under-read
        # both); collapse only if the images are ALSO near-duplicates.
        return near_dup
    overlap = len(wp & wn) / max(1, len(wp))
    if overlap < C.DEDUP_TEXT_SUPERSET_RATIO:
        return False
    if len(wn) <= len(wp):
        # equal-or-shorter text with high overlap: same words reshuffled or a
        # subset — not a build. Collapse only true near-duplicates.
        return near_dup
    # next is a strict superset by word count: a build step (bullets added).
    # Image similarity is NOT required here — builds legitimately change many
    # pixels — but identical-image pairs were already handled above.
    return True


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
