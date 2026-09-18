# SlideSnap — Lecture Video → Clean Searchable Slide PDF

Recorded lectures trap slides inside hours of video: un-searchable,
un-printable. SlideSnap consumes a lecture recording and produces a
deduplicated, perspective-corrected, OCR'd PDF of every distinct slide, with
a bookmark per slide linking back to its video timestamp.

> **Layout note:** the SlideSnap package lives in `slidesnap/` (with
> `pipeline.py`, `config.py`, `src/`, `app/`, `tests/`, `data/`, `docs/`,
> plus its own `README.md` / `statement.md` / `requirements.txt`), and helper
> scripts live in `tools/`. All commands below run from this repo root.

## Features

- Slide-change detection from pure OpenCV/numpy primitives (no
  scene-detection libraries): blur → abs-diff + histogram + SSIM → hysteresis
  with motion-burst masking.
- Sharpest-frame keyframe selection (variance of Laplacian + Tenengrad
  cross-check) with presenter-occlusion rejection against the segment median.
- Screen/board localisation and homography rectification to a 1600×900
  top-down canvas (full-frame fallback when no quad is found).
- Illumination flattening, adaptive binarization, Hough deskew, then OCR via
  Tesseract (if installed) else RapidOCR-ONNX (pure pip) — word boxes become
  the PDF's invisible, genuinely searchable text layer.
- Slide-build deduplication and timestamp-bookmarked PDF assembly.
- Fully CLI-runnable on CPU; per-run `scores.csv`, `summary.txt`, and
  `debug/` (score plots, keyframe grids) for threshold defence.
- Optional Streamlit demo UI with thumbnail review and keep/discard toggles.

## Technologies / tools used

Python, OpenCV (`warpPerspective`, Canny, contours, `approxPolyDP`),
numpy, scikit-image (SSIM), RapidOCR-onnxruntime (+ optional Tesseract via
pytesseract), Pillow, reportlab, Streamlit (demo only), pytest, matplotlib
(debug plots only).

## Install & run (evaluator path — CLI only, no GUI needed)

```bash
pip install -r requirements.txt
# system tesseract is OPTIONAL (RapidOCR is the fallback):
#   sudo dnf install tesseract-ocr   # or: sudo apt install tesseract-ocr

# run on any lecture video:
python -m slidesnap /path/to/lecture.mp4 --out out/
# outputs: out/slides.pdf  out/scores.csv  out/summary.txt  out/debug/

# regenerate the 4-video test corpus (64 slides, MJPG/AVI):
python tools/make_samples.py

# metrics (segmentation P/R/F1 + OCR accuracy):
python tools/evaluate.py [--fast]

# ablation (full vs --no-rectify vs --no-illum OCR accuracy):
python tools/ablation.py

# unit tests (run from repo root):
python -m pytest slidesnap/tests/ -q
```

Ablation flags (for the report table): `--no-rectify`, `--no-illum`,
`--keep-all`. Streamlit demo (optional):
`streamlit run slidesnap/app/streamlit_app.py`.

## Instructions for testing

```bash
# fast: unit tests (13) + segmentation metrics on the bundled corpus
python -m pytest slidesnap/tests/ -q
python tools/evaluate.py --fast
# full: adds OCR spot accuracy (~35 s) and the rectify/illum ablation (~2 min)
python tools/evaluate.py
python tools/ablation.py
```

## Test corpus (`slidesnap/data/samples/`, 4 videos × 16 slides = 64)

- `a_clean.avi` — screen capture, easy baseline
- `b_angled.avi` — perspective-warped + static presenter silhouette (hard)
- `c_whiteboard.avi` — progressive writing reveals
- `d_builds.avi` — bullet builds (dedup stress test)
- `*.truth.txt` — ground-truth change timestamps

Measured segmentation (tol ±1 s): **P=R=F1=1.00 on all four videos
(60/60)**. OCR word accuracy (10 mid-segment slides each): **91.4 %
clean / 76.2 % angled-rectified** (RapidOCR, CPU).

## Screenshots

- `slidesnap/docs/keyframes_a.png` — rectified keyframe grid (Stage 4 output)
- `slidesnap/docs/scores_a.png` — per-pair diff/hist/SSIM signals (observability)
- `slidesnap/docs/architecture.png`, `usecase.png`, `sequence.png`,
  `class-diagram.png`, `workflow.png`, `er.png` — design artefacts
- `debug/` inside any pipeline output dir holds the same per-run artefacts.

## Project report

`slidesnap/docs/report.pdf` — the 15-section VITyarthi report
(regenerate with `python tools/make_report.py`).
