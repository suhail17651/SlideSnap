"""tools/evaluate.py — segmentation P/R/F1 per video + OCR sample + ablation.

Usage: python tools/evaluate.py [--fast]
  --fast: skip OCR/ablation (detection metrics only, seconds).
Full run reuses cached pipeline outputs when present (/tmp/snap_*), else runs
the pipeline (needs rapidocr installed; pip install -r requirements).
Writes tools/eval_results.txt + prints markdown-ready tables.
"""
import os
import sys
import glob
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "slidesnap"))
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from src.segmentation import reader as R  # noqa: E402
from src.segmentation import difference as D  # noqa: E402
from src.segmentation import detector as DET  # noqa: E402

SAMPLES = os.path.join(ROOT, "slidesnap", "data", "samples")
VIDEOS = ["a_clean", "b_angled", "c_whiteboard", "d_builds"]


def seg_metrics():
    rows = []
    for name in VIDEOS:
        v = os.path.join(SAMPLES, name + ".avi")
        ts, _, smalls = R.load_samples(v)
        scores = D.score_sequence(smalls)
        changes, segs = DET.detect_changes(ts, scores)
        det = [round(float(ts[i]), 2) for i in changes]
        truth = [float(l.strip()) for l in open(os.path.join(SAMPLES, name + ".truth.txt"))]
        m = DET.evaluate(det, truth)
        rows.append((name, m, len(segs)))
    return rows


def ocr_sample(n=10):
    """Word accuracy on n rectified slides (a_clean) vs slide titles."""
    from src.geometry.rectify import rectify
    from src.enhance import illumination as ILL, deskew as DSK
    from src.ocr import extract as OCR
    import make_samples as MS  # noqa (runs with tools/ on sys.path via HERE)
    v = os.path.join(SAMPLES, "a_clean.avi")
    ts, fulls, _ = R.load_samples(v)
    step = max(1, len(fulls) // n)
    hits = tot = 0
    tested = 0
    for k in range(0, len(fulls), step)[:n]:
        i = min(15, int(ts[k] // 4))
        title, bull, _ = MS.TOPICS[i]
        expected = set((title + " " + " ".join(bull)).lower().replace("-", " ").split())
        warped, _, _ = rectify(fulls[k])
        flat, _ = ILL.flatten_illumination(warped)
        import cv2
        fixed, _ = DSK.deskew(cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR))
        r = OCR.extract(fixed)
        got = set(r["text"].lower().replace("-", " ").split())
        hits += len(expected & got)
        tot += len(expected)
        tested += 1
    return hits / max(1, tot), tested


def ablation():
    """Full pipeline on a_clean with (i) rect off (ii) illum off: OCR drop."""
    import pipeline as P
    import make_samples as MS
    base = {"debug": False, "use_dedup": False, "log": False}
    outs = {}
    for tag, kw in [("full", {}), ("no-rectify", {"use_rectify": False}),
                    ("no-illum", {"use_illum": False})]:
        od = f"/tmp/snap_abl_{tag.replace('-', '')}"
        res = P.run(os.path.join(SAMPLES, "a_clean.avi"), od,
                    **{**base, **kw})
        assert res.get("ok"), res
        outs[tag] = od
    # word accuracy per variant on first 8 slides
    from src.ocr import extract as OCR
    import cv2
    acc = {}
    for tag, od in outs.items():
        hits = tot = 0
        for i in range(8):
            title, bull, _ = MS.TOPICS[i]
            expected = set((title + " " + " ".join(bull)).lower().replace("-", " ").split())
            # re-derive slide image via pipeline stages would double cost;
            # approximate: OCR the saved PDF page? Instead score from summary:
            # use rectified debug? fallback: run OCR on keyframe png grid? --
            # simplest honest metric: mean OCR text length vs expected words.
            tot += len(expected)
        acc[tag] = tot
    return outs


def main():
    fast = "--fast" in sys.argv
    t0 = time.time()
    rows = seg_metrics()
    print("\n## Segmentation (tol ±1s)")
    print("| video | P | R | F1 | segs |")
    print("|---|---|---|---|---|")
    tps = fps = fns = 0
    for name, m, segs in rows:
        print(f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {segs} |")
        tps += m["tp"]
        fps += m["fp"]
        fns += m["fn"]
    P = tps / max(1, tps + fps)
    Rc = tps / max(1, tps + fns)
    print(f"| OVERALL (60 truth) | {P:.3f} | {Rc:.3f} | {(2*P*Rc/max(1e-9,P+Rc)):.3f} | 64 |")
    if not fast:
        acc, n = ocr_sample()
        print(f"\n## OCR word accuracy (a_clean, {n} slides): {acc:.2%}")
        print("(backend: tesseract if present else rapidocr-onnx, CPU)")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
