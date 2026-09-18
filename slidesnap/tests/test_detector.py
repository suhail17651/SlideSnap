"""Unit tests: hysteresis detector (isolated cut / motion burst / build blip)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.segmentation import detector as DET  # noqa: E402


def _scores_from_flags(spikes, stables):
    out = []
    for i, (sp, st) in enumerate(zip(spikes, stables)):
        out.append({"index": i + 1, "votes": 2 if sp else 0,
                    "spike": sp, "stable": st,
                    "diff": 0.01 if sp else 0.0, "hist": 0.0,
                    "ssim": 0.9 if sp else 1.0})
    return out


def test_hysteresis_confirms_isolated_spike():
    # one spike, calm windows around -> confirmed at index 4
    spikes = [0, 0, 0, 1, 0, 0, 0, 0]
    stables = [1, 1, 1, 0, 1, 1, 1, 1]
    ts = [i * 0.5 for i in range(9)]
    changes, segs = DET.detect_changes(ts, _scores_from_flags(spikes, stables))
    assert changes == [4], changes
    assert segs == [(0, 3), (4, 8)], segs


def test_hysteresis_rejects_motion_burst():
    # sustained motion burst (walk-by: 3 consecutive spikes, each within
    # +-2 of the others, so the burst mask kills all of them) -> nothing.
    spikes = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    stables = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
    ts = [i * 0.5 for i in range(11)]
    changes, segs = DET.detect_changes(ts, _scores_from_flags(spikes, stables))
    assert changes == [], changes


def test_hysteresis_ignores_build_blip_without_opening():
    # single non-spike blip (build step: only 1 vote) never opens a change
    spikes = [0, 0, 0, 0, 0, 0, 0, 0]
    stables = [1, 1, 1, 0, 1, 1, 1, 1]
    ts = [i * 0.5 for i in range(9)]
    changes, _ = DET.detect_changes(ts, _scores_from_flags(spikes, stables))
    assert changes == []


def test_evaluate_counts_tp_fp_fn():
    m = DET.evaluate([4.0, 9.9], [4.0, 8.0], tol=1.0)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9
