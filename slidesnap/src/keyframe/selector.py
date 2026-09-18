"""Select one keyframe per stable segment: sharpest + unoccluded."""
from .sharpness import score_frames
from .occlusion import pick_unoccluded


def select_keyframes(segments, full_frames, small_frames):
    """segments: list of (start,end) inclusive sample indices.
    Returns list of dicts: {segment, sample_idx, frame_full, ...}.
    """
    out = []
    for si, (s, e) in enumerate(segments):
        cand_full = full_frames[s:e + 1]
        cand_small = small_frames[s:e + 1]
        if not cand_full:
            continue
        scores = score_frames(cand_full)
        laps = [x[0] for x in scores]
        pick = pick_unoccluded(cand_full, cand_small, laps)
        pos = pick["pos"]
        out.append({
            "segment": si,
            "seg_start": s, "seg_end": e,
            "pos_in_seg": pos,
            "sample_idx": s + pos,
            "frame_full": cand_full[pos],
            "laplacian": laps[pos],
            "occluded": pick["occluded"],
            "occlusion_ratio": pick["ratio"],
            "segment_mean_sharp": sum(laps) / len(laps),
        })
    return out
