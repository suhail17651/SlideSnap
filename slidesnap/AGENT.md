# AGENT.md — SlideSnap handover (read this first)

> One-page brain dump for the next agent / developer taking over this repo.
> The graded path is the **CLI** (`python -m slidesnap VIDEO --out OUT/`).
> The UI is a demo. The corpus is synthetic. Numbers below are measured.

## 1. What this is

Computer-vision course project ("Build Your Own Project", VITyarthi):
**lecture video → searchable slide PDF**. Stages:
`sample → score → detect → keyframe → rectify → enhance → OCR → dedup → PDF`.
Rubric: requirements 10, design/docs 20, implementation 25,
innovation/depth 15, repo/version-control 10, report 20.

## 2. Repo map (root = `slidesnap/`)

| Path | Role |
|------|------|
| `pipeline.py` + `__main__.py` | end-to-end CLI (`run()` + `main()`); ablation flags `--no-rectify --no-illum --keep-all` |
| `config.py` | **every threshold + tuning comment** — tune here, nowhere else |
| `src/segmentation/{reader,difference,detector}.py` | 2 fps sampler; diff/hist/SSIM on center ROI; hysteresis + burst mask |
| `src/keyframe/{sharpness,occlusion,selector}.py` | LoG sharpness + Tenengrad x-check; median-frame occlusion reject |
| `src/geometry/{screen_detect,rectify}.py` | Canny→contours→approxPolyDP quad; `warpPerspective` to 1600×900 |
| `src/enhance/{illumination,binarize,deskew}.py` | morph-close background ÷; adaptive thresh; Hough deskew |
| `src/ocr/extract.py` | Tesseract → RapidOCR-ONNX → off (never raises) |
| `src/assemble/{dedup,pdf_builder}.py` | build-collapse; reportlab image + invisible text + bookmarks |
| `src/debug/overlay.py` | keyframe grids + score plots into `debug/` |
| `app/streamlit_app.py` | optional demo UI |
| `tests/test_{difference,detector,sharpness,rectify,dedup,pipeline}.py` | 16 tests, split by stage |
| `data/samples/{a_clean,b_angled,c_whiteboard,d_builds}.avi` + `.truth.txt` | 4x16=64-slide corpus (MJPG/AVI — intraframe, no keyframe pumping; `.avi` git-ignored, regenerate via `tools/make_samples.py`) |
| `docs/` | `architecture.png usecase.png sequence.png class-diagram.png` + `keyframes_a.png scores_a.png` + `report.pdf` (submission) |
| `tools/{make_samples,evaluate,ablation,make_diagrams,make_report}.py` | corpus gen, metrics, ablation, diagram gen, report gen |

`tools/` lives at **repo parent** (`/home/danish1075/Music/suhaail/tools/`), not inside
`slidesnap/` — because `make_samples` topics are imported by tests/eval via
`sys.path`. Don't move it without fixing imports.

## 3. Measured behaviour (don't re-tune blindly)

- Segmentation @ tol ±1 s: **P=R=F1=1.00 on all 4 videos (60/60)**.
- OCR word accuracy (10 mid-segment slides each): **91.4 % clean / 76.2 %
  angled-rectified** (RapidOCR, CPU). Ablation: rectification +17.1pp on
  angled (0.5905→0.7619), illumination +17.1pp on clean (0.7429→0.9143).
- Truth-pair signals (ROI): diff 0.013–0.03, SSIM 0.86–0.93.
  Non-truth: build/writing blips diff ≤0.006, SSIM ≥0.977; static 1.0.
  Thresholds: `DIFF_HIGH=0.005 DIFF_LOW=0.003 SSIM_HIGH_CHANGE=0.972 SSIM_LOW_STABLE=0.975`.
- Fusion = diff + SSIM (2/2). **Histogram excluded from fusion**
  (measured inverted on MJPG: static jitter ~0.13 > true changes 0.015–0.05);
  still computed, logged, unit-tested — report §8 explains why.
- Burst mask: spike with >1 spike in ±2 pairs ⇒ motion, masked.
  Clean cuts are isolated singles, always survive.
- `b_angled` rectify lock-on ~94-100 % (rest = full-frame fallback, still fine).
- Pipeline ≈ 25 s/video (64 s, 1280×720) incl. RapidOCR; scoring-only ≈ 15 s.
- Tests: 16 pass (`python -m pytest slidesnap/tests/ -q` from repo parent);
  smoke test needs corpus present. Dedup is unit-tested; on the corpus each
  4 s-apart truth change is a distinct slide, so dedup removes 0 there —
  `d_builds` stresses the detector's build-step rejection instead.

## 4. Known gotchas

1. **mp4v is non-deterministic** (keyframe pumping shifts scores run to run).
   Corpus is MJPG/AVI for this reason. Don't "fix" by switching back to mp4.
2. **Two cv2 installs** exist here (`opencv` 4.11 + `opencv-python` 5.0);
   `import cv2` resolves to 5.0. Pin `opencv-python==5.0.*` in requirements.
3. **No tesseract binary** on this machine; OCR runs via RapidOCR.
   `extract.py` probes tesseract first, so results differ by machine — expected.
4. `overlay.plot_scores` imports matplotlib lazily; absent ⇒ returns None
   and pipeline logs a note (not a failure).
5. Tests import via `sys.path.insert(ROOT)` + `import config` — run pytest
   from the **parent** dir (`/home/danish1075/Music/suhaail/`), not inside `slidesnap/`.
6. `evaluate.py` ablation re-runs the pipeline 3× (~2 min with OCR).
7. Videos are ~60 MB each (MJPG); don't commit giant re-renders casually.

## 5. Common tasks

```bash
cd /home/danish1075/Music/suhaail
python3 -m slidesnap slidesnap/data/samples/a_clean.avi --out /tmp/snap_a
python3 tools/evaluate.py --fast        # detection metrics, ~15 s
python3 tools/evaluate.py               # + OCR accuracy, ~35 s
python3 -m pytest slidesnap/tests/ -q
python3 tools/make_samples.py           # regenerate corpus (deterministic seeds)
python3 tools/make_diagrams.py          # regenerate docs/*.png
python3 tools/make_report.py            # regenerate docs/report.pdf
```

## 6. What's left / risks

- Report PDF is generated (`tools/make_report.py`); verify numbers match
  `tools/eval_results.txt` before submitting.
- Submission needs: **public GitHub repo rooted at `slidesnap/`** (so
  `README.md`, `statement.md`, `requirements.txt` sit at the repo root per
  the VITyarthi doc), report PDF upload, repo URL
  `https://github.com/{user}/{repo}` (no `/tree/main` suffix).
- Plagiarism + AI-detection pipeline: report is generated from our own
  measured numbers; keep the `debug/` artefacts + `scores.csv` as evidence.
- If real H.264 lecture footage behaves differently (histogram may become
  useful again), re-measure with `evaluate.py` before changing `config.py`.
