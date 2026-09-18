"""Debug overlays: dump intermediates to debug/ ."""
import os
import cv2
import numpy as np


def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


def save_grid(images, out_path, cols=4, thumb_w=320):
    if not images:
        return None
    thumbs = []
    for im in images:
        h, w = im.shape[:2]
        th = int(thumb_w * h / max(1, w))
        thumbs.append(cv2.resize(im, (thumb_w, th)))
    rows = []
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i:i + cols]
        mh = max(t.shape[0] for t in chunk)
        padded = []
        for t in chunk:
            pad = mh - t.shape[0]
            if pad > 0:
                t = cv2.copyMakeBorder(t, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded.append(t)
        rows.append(np.hstack(padded))
    mw = max(r.shape[1] for r in rows)
    full = []
    for r in rows:
        if r.shape[1] < mw:
            r = cv2.copyMakeBorder(r, 0, 0, 0, mw - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        full.append(r)
    grid = np.vstack(full)
    cv2.imwrite(out_path, grid)
    return out_path


def draw_quad(frame_bgr, quad):
    vis = frame_bgr.copy()
    if quad is not None:
        pts = quad.astype(int).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    return vis


def plot_scores(timestamps, scores, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
    import config as _C
    diff = [s["diff"] for s in scores]
    hist = [s["hist"] for s in scores]
    ssim = [s["ssim"] for s in scores]
    ts = timestamps[1:1 + len(scores)]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(ts, diff)
    ax[0].axhline(_C.DIFF_HIGH, color="r", ls="--", label="DIFF_HIGH")
    ax[0].axhline(_C.DIFF_LOW, color="g", ls="--", label="DIFF_LOW")
    ax[0].set_ylabel("abs-diff")
    ax[0].legend(fontsize=8)
    ax[1].plot(ts, hist, color="orange")
    ax[1].axhline(_C.HIST_HIGH, color="r", ls="--")
    ax[1].axhline(_C.HIST_LOW, color="g", ls="--")
    ax[1].set_ylabel("hist dist")
    ax[2].plot(ts, ssim, color="purple")
    ax[2].axhline(_C.SSIM_HIGH_CHANGE, color="r", ls="--")
    ax[2].axhline(_C.SSIM_LOW_STABLE, color="g", ls="--")
    ax[2].set_ylabel("SSIM")
    ax[2].set_xlabel("time (s)")
    fig.suptitle("Per-frame difference signals (observability log)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    try:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    except Exception:
        pass
    return out_path
