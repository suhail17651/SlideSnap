"""SlideSnap end-to-end CLI pipeline (command-line executable, no GUI needed).

Usage:
  python -m slidesnap.pipeline INPUT.mp4 --out out/ [--debug] [--keep-all]
  python slidesnap/pipeline.py INPUT.mp4 --out out/
Ablation flags (for report table):
  --no-rectify   skip homography rectification
  --no-illum     skip illumination flattening
  --no-dedup     alias for --keep-all
"""
import argparse
import csv
import os
import sys
import time
import traceback

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
from src.segmentation import reader as R  # noqa: E402
from src.segmentation import difference as D  # noqa: E402
from src.segmentation import detector as DET  # noqa: E402
from src.keyframe.selector import select_keyframes  # noqa: E402
from src.geometry.rectify import rectify_keyframes  # noqa: E402
from src.enhance import illumination as ILL  # noqa: E402
from src.enhance import binarize as BIN  # noqa: E402
from src.enhance import deskew as DSK  # noqa: E402
from src.ocr import extract as OCR  # noqa: E402
from src.assemble import dedup as DD  # noqa: E402
from src.assemble import pdf_builder as PDF  # noqa: E402
from src.debug import overlay as OV  # noqa: E402


def fmt_ts(s):
    s = float(s)
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def run(video_path, out_dir, debug=True, use_rectify=True, use_illum=True,
        use_dedup=True, log=True):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    dbg = os.path.join(out_dir, "debug") if debug else None
    if dbg:
        os.makedirs(dbg, exist_ok=True)
    notes = []  # soft-failure log

    # ---- Stage 1: sampling + scoring ----
    try:
        ts, fulls, smalls = R.load_samples(video_path)
    except Exception as e:
        return {"ok": False, "error": f"cannot read video: {e}"}
    if not ts:
        return {"ok": False, "error": "no frames decoded (corrupt/empty video?)"}
    scores = D.score_sequence(smalls)
    if log:
        try:
            with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "diff", "hist", "ssim", "votes", "spike", "stable"])
                for i, s in enumerate(scores):
                    w.writerow([f"{ts[s['index']]:.2f}", f"{s['diff']:.4f}",
                                f"{s['hist']:.4f}", f"{s['ssim']:.4f}",
                                s["votes"], int(s["spike"]), int(s["stable"])])
        except Exception as e:
            notes.append(f"score CSV failed: {e}")
    if dbg:
        try:
            OV.plot_scores(ts, scores, os.path.join(dbg, "scores.png"))
        except Exception as e:
            notes.append(f"score plot failed: {e}")

    # ---- Stage 2: detection ----
    changes, segments = DET.detect_changes(ts, scores)
    change_ts = DET.change_timestamps(ts, changes)

    # ---- Stage 3: keyframes ----
    try:
        keyframes = select_keyframes(segments, fulls, smalls)
    except Exception as e:
        notes.append(f"keyframe selection failed: {e}; using segment mid-frames")
        keyframes = []
        for si, (s, e) in enumerate(segments):
            mid = (s + e) // 2
            keyframes.append({"segment": si, "seg_start": s, "seg_end": e,
                              "pos_in_seg": mid - s, "sample_idx": mid,
                              "frame_full": fulls[mid], "laplacian": 0,
                              "occluded": False, "occlusion_ratio": 0,
                              "segment_mean_sharp": 0})

    # ---- Stage 4: rectification ----
    try:
        if use_rectify:
            keyframes = rectify_keyframes(keyframes)
        else:
            for k in keyframes:
                k["rectified"] = cv2.resize(k["frame_full"], (C.WARP_WIDTH, C.WARP_HEIGHT))
                k["quad"] = None
                k["rectify_fallback"] = True
    except Exception as e:
        notes.append(f"rectification failed: {e}; using resized raw frames")
        for k in keyframes:
            if "rectified" not in k:
                k["rectified"] = cv2.resize(k["frame_full"], (C.WARP_WIDTH, C.WARP_HEIGHT))
                k["rectify_fallback"] = True

    # ---- Stage 5: enhancement + deskew + OCR ----
    slides = []
    skipped = []
    for i, k in enumerate(keyframes):
        try:
            img = k["rectified"]
            if use_illum:
                flat, _bg = ILL.flatten_illumination(img)
                # keep colour for the PDF page; use flat gray for OCR pre-clean
                work = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
            else:
                work = img
            fixed, angle = DSK.deskew(work)
            try:
                gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
                _bw = BIN.adaptive_binarize(gray)  # artefact for report/debug
            except Exception:
                pass
            ocr = OCR.extract(fixed)
            t = ts[k["sample_idx"]]
            slides.append({"rectified": fixed, "text": ocr.get("text", ""),
                           "words": ocr.get("words", []), "timestamp": t,
                           "timestamp_start": t, "angle": angle,
                           "ocr_available": ocr.get("available", False)})
        except Exception as e:
            skipped.append({"slide": i, "reason": str(e)[:200]})
            notes.append(f"slide {i} failed: {e}")
    if dbg:
        try:
            OV.save_grid([s["rectified"] for s in slides[:24]],
                         os.path.join(dbg, "keyframes.png"))
        except Exception as e:
            notes.append(f"keyframe grid failed: {e}")

    # ---- Stage 6: dedup + PDF ----
    pages_before = len(slides)
    dedup_removed = 0
    if use_dedup and len(slides) > 1:
        try:
            slides, dedup_removed = DD.deduplicate(slides)
        except Exception as e:
            notes.append(f"dedup failed: {e}")
    pdf_path = os.path.join(out_dir, "slides.pdf")
    try:
        PDF.build_pdf(slides, pdf_path)
    except Exception as e:
        return {"ok": False, "error": f"PDF build failed: {e}",
                "notes": notes}

    dt = time.time() - t0
    result = {
        "ok": True,
        "video": os.path.abspath(video_path),
        "slides": len(slides),
        "pages_before_dedup": pages_before,
        "dedup_removed": dedup_removed,
        "changes": [round(float(x), 2) for x in change_ts],
        "segments": len(segments),
        "skipped": skipped,
        "notes": notes,
        "seconds": round(dt, 1),
        "pdf": os.path.abspath(pdf_path),
        "ocr_available": any(s.get("ocr_available") for s in slides),
        "rectify_fallback_rate": sum(1 for k in keyframes if k.get("rectify_fallback")) / max(1, len(keyframes)),
    }
    # summary file
    try:
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(f"SlideSnap result for {video_path}\n")
            for kk, vv in result.items():
                if kk != "notes":
                    f.write(f"{kk}: {vv}\n")
            f.write("notes:\n")
            for n in notes:
                f.write(f" - {n}\n")
    except Exception:
        pass
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="SlideSnap: lecture video -> searchable slide PDF")
    ap.add_argument("video", help="input lecture video file")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--no-debug", action="store_true", help="skip debug artefacts")
    ap.add_argument("--no-rectify", action="store_true", help="ablation: skip homography")
    ap.add_argument("--no-illum", action="store_true", help="ablation: skip illumination flattening")
    ap.add_argument("--keep-all", "--no-dedup", dest="keep_all", action="store_true",
                    help="disable build-deduplication")
    ap.add_argument("--no-log", action="store_true", help="skip scores.csv")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.video):
        print(f"ERROR: input not found: {a.video}", file=sys.stderr)
        return 2
    try:
        res = run(a.video, a.out, debug=not a.no_debug,
                  use_rectify=not a.no_rectify, use_illum=not a.no_illum,
                  use_dedup=not a.keep_all, log=not a.no_log)
    except Exception:
        traceback.print_exc()
        return 1
    if not res.get("ok"):
        print(f"FAILED: {res.get('error')}", file=sys.stderr)
        for n in res.get("notes", []):
            print(f" note: {n}", file=sys.stderr)
        return 1
    print(f"OK: {res['slides']} slides ({res['pages_before_dedup']} before dedup) -> {res['pdf']}")
    print(f"  changes @ {res['changes']}")
    print(f"  time {res['seconds']}s | skipped {len(res['skipped'])} | "
          f"rectify_fallback {res['rectify_fallback_rate']:.0%} | ocr {'yes' if res['ocr_available'] else 'NO (tesseract missing?)'}")
    for n in res.get("notes", []):
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
