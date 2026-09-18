"""Frame sampling / decimation reader. Streams video without loading it fully into RAM."""
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C


def sample_frames(video_path, sample_fps=None, scoring_size=None):
    """Yield (timestamp_sec, full_frame_bgr, small_frame_bgr).

    Decimates to sample_fps and downscales scoring copy to 360p.
    Streaming: constant memory regardless of video length.
    """
    sample_fps = sample_fps or C.SAMPLE_FPS
    scoring_size = scoring_size or (C.SCORING_WIDTH, C.SCORING_HEIGHT)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 0:
        src_fps = 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % step == 0:
            t = float(idx) / float(src_fps)
            small = cv2.resize(frame, scoring_size, interpolation=cv2.INTER_AREA)
            yield (t, frame, small)
            kept += 1
        idx += 1
    cap.release()


def load_samples(video_path, **kw):
    """Convenience: return lists (timestamps, full_frames, small_frames)."""
    ts, fulls, smalls = [], [], []
    for t, f, s in sample_frames(video_path, **kw):
        ts.append(t)
        fulls.append(f)
        smalls.append(s)
    return ts, fulls, smalls


def video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False}
    info = {
        "ok": True,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info
