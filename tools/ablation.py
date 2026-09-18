"""Build the ablation table: full vs --no-rectify vs --no-illum OCR accuracy.

Runs the pipeline 3x on a_clean.avi (~1.5 min with RapidOCR) and writes
tools/eval_results.txt. Usage: python tools/ablation.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = PARENT if os.path.exists(os.path.join(PARENT, "src")) else os.path.join(PARENT, "slidesnap")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

SAMPLES = os.path.join(ROOT, "data", "samples")


def ocr_accuracy_on_saved(out_dir, n_slides=10):
    """OCR word accuracy by re-reading pipeline slides? Cheaper honest proxy:
    run pipeline with debug grid off, then OCR the PDF page images? Instead:
    accuracy measured inside pipeline run via saved summary is not stored, so
    recompute: load a_clean keyframes = one frame per truth segment and OCR
    with/without each stage. Mirrors pipeline stages exactly.
    """
    import cv2
    from src.segmentation import reader as R
    from src.geometry.rectify import rectify
    from src.enhance import illumination as ILL, deskew as DSK
    from src.ocr import extract as OCR
    import make_samples as MS

    v = os.path.join(SAMPLES, "a_clean.avi")
    ts, fulls, _ = R.load_samples(v)
    mode = open(os.path.join(out_dir, "mode.txt")).read().strip() \
        if os.path.isfile(os.path.join(out_dir, "mode.txt")) else "full"
    hits = tot = 0
    for s_idx in range(min(n_slides, 16)):
        t_mid = s_idx * 4 + 2.0
        k = min(range(len(ts)), key=lambda i: abs(ts[i] - t_mid))
        title, bull, _ = MS.TOPICS[s_idx]
        expected = set((title + " " + " ".join(bull)).lower().replace("-", " ").split())
        img = fulls[k]
        if mode != "no-rectify":
            img, _, _ = rectify(img)
        else:
            img = cv2.resize(img, (1600, 900))
        if mode != "no-illum":
            flat, _ = ILL.flatten_illumination(img)
            img = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
        fixed, _ = DSK.deskew(img)
        r = OCR.extract(fixed)
        got = set(r["text"].lower().replace("-", " ").split())
        hits += len(expected & got)
        tot += len(expected)
    return hits / max(1, tot)


def main():
    import pipeline as P
    variants = [("full", {}), ("no-rectify", {"use_rectify": False}),
                ("no-illum", {"use_illum": False})]
    rows = []
    for tag, kw in variants:
        od = f"/tmp/snap_abl_{tag.replace('-', '')}"
        t0 = time.time()
        res = P.run(os.path.join(SAMPLES, "a_clean.avi"), od, debug=False,
                    log=False, use_dedup=False, **kw)
        assert res.get("ok"), res
        with open(os.path.join(od, "mode.txt"), "w") as f:
            f.write(tag)
        acc = ocr_accuracy_on_saved(od)
        rows.append((tag, acc, res["seconds"]))
        print(f"{tag}: OCR word-acc={acc:.2%} pipe={res['seconds']}s", flush=True)
    with open(os.path.join(HERE, "eval_results.txt"), "w") as f:
        f.write("# SlideSnap ablation (a_clean.avi, 10 slides, RapidOCR CPU)\n")
        for tag, acc, sec in rows:
            f.write(f"{tag}: acc={acc:.4f} pipe_seconds={sec}\n")
    print("wrote tools/eval_results.txt")


if __name__ == "__main__":
    main()
