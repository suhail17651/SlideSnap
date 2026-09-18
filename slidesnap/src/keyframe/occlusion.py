"""Occlusion detection: presenter/cursor overlap via deviation from segment median."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def segment_median(frames_small):
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_small], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def occlusion_ratio(frame_small, median_gray):
    g = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    if g.shape != median_gray.shape:
        median_gray = cv2.resize(median_gray, (g.shape[1], g.shape[0]))
    diff = cv2.absdiff(g, median_gray)
    mask = (diff > C.OCCLUSION_PIXEL_DIFF).astype(np.uint8)
    return float(mask.mean()), diff


def is_occluded(frame_small, median_gray):
    ratio, _ = occlusion_ratio(frame_small, median_gray)
    return ratio > C.OCCLUSION_AREA_RATIO, ratio


def pick_unoccluded(candidates_full, candidates_small, sharp_scores):
    """Prefer unoccluded frame even if slightly less sharp.

    candidates_full/small: aligned lists for one segment.
    sharp_scores: list of laplacian variances (same order).
    Returns chosen position + diagnostics.
    """
    median = segment_median(candidates_small)
    flags = []
    for s in candidates_small:
        occ, ratio = is_occluded(s, median)
        flags.append((occ, ratio))
    order = sorted(range(len(candidates_full)),
                   key=lambda i: sharp_scores[i], reverse=True)
    best = order[0]
    best_score = sharp_scores[best]
    for i in order:
        occ, ratio = flags[i]
        if not occ or sharp_scores[i] >= best_score * C.OCCLUSION_SHARPNESS_TRADEOFF:
            if not occ:
                return {"pos": i, "occluded": False, "ratio": ratio,
                        "median": median, "flags": flags}
    # all occluded: return least-occluded among sharp ones
    least = min(order, key=lambda i: flags[i][1])
    return {"pos": least, "occluded": True, "ratio": flags[least][1],
            "median": median, "flags": flags}
