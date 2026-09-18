"""Hysteresis thresholding + stability windows for slide-change detection."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def detect_changes(timestamps, scores, sample_fps=None):
    """Hysteresis detector.

    A change fires on a high-threshold spike (votes>=MIN_VOTES), then the
    STABILITY_SECONDS window after the spike must contain NO further
    unmasked spike: one calm window confirms a new stable slide.
    Within-slide edits (build steps, writing reveals) are single-pair blips
    with calm neighbours, so they can neither open a false change nor break
    a real window; a real neighbouring slide change inside the window vetoes
    confirmation. A presenter walk-by fires motion spikes on consecutive
    scored pairs for ~1s, so any spike with more than BURST_MAX_SPIKES
    spikes in its +-BURST_RADIUS neighbourhood is masked as motion before
    confirmation (clean cuts are isolated single spikes and always survive
    the mask). Covered by unit tests with synthetic transient spikes.

    Returns:
      changes: list of sample indices (index into timestamps/frames) where a
               new stable slide begins.
      segments: list of (start_idx, end_idx) inclusive stable segments.
    """
    sample_fps = sample_fps or C.SAMPLE_FPS
    need = max(1, int(round(C.STABILITY_SECONDS * sample_fps)))
    min_len = max(1, int(round(C.MIN_SEGMENT_SECONDS * sample_fps)))
    n = len(timestamps)
    if n == 0:
        return [], []
    # scores[i] corresponds to pair (i -> i+1) in sample indexing where
    # scores list entry with "index" == i+1. Build lookup by right index.
    by_right = {s["index"]: s for s in scores}
    # A presenter walk-by is a run of consecutive SPIKES (motion every
    # scored pair for ~1s). Mask those runs first: any spike whose
    # +-BURST_RADIUS neighbourhood contains more than BURST_MAX_SPIKES
    # spikes is motion, not a cut. Clean cuts are isolated single spikes
    # (need=2 calm window), so they always survive the mask.
    radius = C.BURST_RADIUS
    spike_set = {k for k, s in by_right.items() if s["spike"]}
    masked = set()
    for k in spike_set:
        cnt = sum(1 for j in range(k - radius, k + radius + 1) if j in spike_set)
        if cnt > C.BURST_MAX_SPIKES:
            masked.add(k)
    changes = []
    seg_start = 0
    segments = []
    i = 1
    while i < n:
        s = by_right.get(i)
        if s is not None and s["spike"] and i not in masked:
            # confirmation window: next `need` pairs must contain no
            # UNMASKED spike (masked motion spikes don't veto real cuts)
            ok = True
            end = min(n, i + 1 + need)
            for k in range(i + 1, end):
                sk = by_right.get(k)
                if sk is None:
                    ok = False
                    break
                if sk["spike"] and k not in masked:
                    ok = False
                    break
            # need full window available (except trailing spike: confirm if
            # all remaining pairs are spike-free)
            if i + need >= n:
                for k in range(i + 1, n):
                    sk = by_right.get(k)
                    if sk is None or (sk["spike"] and k not in masked):
                        ok = False
                        break
            if ok:
                seg_end = i - 1
                if seg_end - seg_start + 1 >= min_len:
                    segments.append((seg_start, seg_end))
                else:
                    # merge tiny flicker: extend previous if exists
                    if segments:
                        pass
                    else:
                        segments.append((seg_start, seg_end))
                changes.append(i)
                seg_start = i
                i += need  # skip stability window
                continue
            # transient: ignore spike, keep going
        i += 1
    segments.append((seg_start, n - 1))
    # filter segments shorter than min_len by merging into previous
    filtered = []
    for seg in segments:
        if filtered and (seg[1] - seg[0] + 1) < min_len:
            ps, pe = filtered.pop()
            filtered.append((ps, seg[1]))
        else:
            filtered.append(seg)
    return changes, filtered


def change_timestamps(timestamps, changes):
    return [float(timestamps[i]) for i in changes]


def evaluate(detected, truth, tol=None):
    """Precision/recall/F1 with ±tol window. Times in seconds."""
    tol = C.EVAL_TOLERANCE_SECONDS if tol is None else tol
    truth = sorted(truth)
    detected = sorted(detected)
    matched_t = set()
    tp = 0
    for d in detected:
        best, bestj = None, None
        for j, t in enumerate(truth):
            if j in matched_t:
                continue
            if abs(d - t) <= tol and (best is None or abs(d - t) < best):
                best, bestj = abs(d - t), j
        if bestj is not None:
            matched_t.add(bestj)
            tp += 1
    fp = len(detected) - tp
    fn = len(truth) - tp
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-9, (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec,
            "recall": rec, "f1": f1}
