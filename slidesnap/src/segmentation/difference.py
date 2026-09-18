"""Three-signal difference scoring from primitives (no scene-detection libs)."""
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False


def _gray_blur(frame_small):
    g = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, C.BLUR_KERNEL, C.BLUR_SIGMA)


def centered_roi(frame_small, border_frac=0.18):
    """Center crop excluding frame borders (presenter walks the edges).

    Returns the ROI view used by ALL scoring signals. Cropping is the
    cheapest form of presenter-robustness: the dark silhouette spends most
    of its time near the left/right borders, while slide content fills the
    center. Applied identically to every video so no per-video tuning.
    """
    h, w = frame_small.shape[:2]
    bx, by = int(w * border_frac), int(h * border_frac)
    return frame_small[by:h - by, bx:w - bx]


def abs_diff_score(prev_small, curr_small):
    """Mean absolute frame difference after Gaussian blur, 0..1 (center ROI)."""
    a = _gray_blur(centered_roi(prev_small)).astype(np.float32) / 255.0
    b = _gray_blur(centered_roi(curr_small)).astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))


def hist_distance(prev_small, curr_small):
    """Bhattacharyya distance between HSV histograms (0=identical, center ROI).

    NOTE (measured): on MPEG-4/mp4v footage the encoder's keyframe pumping
    makes static-pair distances jitter up to ~0.13, while true slide changes
    on flat vector-graphics slides score only 0.015-0.05. The signal is
    therefore INVERTED on this codec (static > change) and is excluded from
    fusion by default. Kept as a documented, tested primitive for real H.264
    lecture captures where compression jitter is far lower; the report's
    design-decisions section explains the failure mode.
    """
    hsv_a = cv2.cvtColor(centered_roi(prev_small), cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(centered_roi(curr_small), cv2.COLOR_BGR2HSV)
    h_a = cv2.calcHist([hsv_a], [0, 1], None, [C.HIST_BINS_H, C.HIST_BINS_S], [0, 180, 0, 256])
    h_b = cv2.calcHist([hsv_b], [0, 1], None, [C.HIST_BINS_H, C.HIST_BINS_S], [0, 180, 0, 256])
    cv2.normalize(h_a, h_a)
    cv2.normalize(h_b, h_b)
    return float(cv2.compareHist(h_a, h_b, cv2.HISTCMP_BHATTACHARYYA))


def ssim_score(prev_small, curr_small):
    """SSIM (1=identical) on center ROI. Falls back to NCC-based proxy if skimage missing."""
    a = cv2.cvtColor(centered_roi(prev_small), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(centered_roi(curr_small), cv2.COLOR_BGR2GRAY)
    if _HAS_SSIM:
        try:
            return float(_ssim(a, b))
        except Exception:
            pass
    # fallback: normalized cross-correlation mapped to [-1,1]-ish range
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-8
    return float(np.dot(af, bf) / denom)


def score_pair(prev_small, curr_small):
    """Return dict with the raw signals (hist recorded, fusion uses diff+SSIM)."""
    return {
        "diff": abs_diff_score(prev_small, curr_small),
        "hist": hist_distance(prev_small, curr_small),
        "ssim": ssim_score(prev_small, curr_small),
    }


def high_votes(sig):
    """Votes counted against HIGH thresholds (diff + SSIM; hist excluded, see above)."""
    v = 0
    if sig["diff"] > C.DIFF_HIGH:
        v += 1
    if sig["ssim"] < C.SSIM_HIGH_CHANGE:
        v += 1
    return v


def is_stable(sig):
    """Stable iff below ALL low thresholds (diff + SSIM)."""
    return (sig["diff"] < C.DIFF_LOW
            and sig["ssim"] > C.SSIM_LOW_STABLE)


def score_sequence(small_frames):
    """Score consecutive pairs. Returns list of dicts (index i => pair i-1->i)."""
    out = []
    for i in range(1, len(small_frames)):
        sig = score_pair(small_frames[i - 1], small_frames[i])
        sig["index"] = i
        sig["votes"] = high_votes(sig)
        sig["spike"] = sig["votes"] >= C.MIN_VOTES
        sig["stable"] = is_stable(sig)
        out.append(sig)
    return out
