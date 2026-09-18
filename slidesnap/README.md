# SlideSnap — Lecture Video → Clean Searchable Slide PDF

Recorded lectures trap slides inside hours of video: un-searchable,
un-printable. SlideSnap consumes a lecture recording and produces a
deduplicated, perspective-corrected, OCR'd PDF of every distinct slide, with
a bookmark per slide linking back to its video timestamp.

> **A note on paths:** this repository is rooted at `slidesnap/` — i.e. this
> README, `statement.md`, `requirements.txt`, `pipeline.py`, `config.py`,
> `src/`, `app/`, `tests/`, `data/`, and `docs/` all sit at the repo root.
> Helper scripts (`tools/make_samples.py`, `tools/evaluate.py`,
> `tools/ablation.py`, `tools/make_diagrams.py`, `tools/make_report.py`) live
> one level up in `tools/` and import the package via `sys.path`; run them
> from the repo root as shown below.

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
pytesseract), Pillow, reportlab, python-pptx, Streamlit (demo only), pytest,
matplotlib (debug plots only).

## Functional modules (rubric: >=3)

| # | Module | Input | Output |
|---|--------|-------|--------|
| 1 | Temporal Segmentation (`src/segmentation/`) | video file | slide-change timestamps + stable segments |
| 2 | Keyframe Selection & Rectification (`src/keyframe/`, `src/geometry/`) | candidate frames | one sharp, deskewed, top-down image per slide |
| 3 | Enhancement & OCR (`src/enhance/`, `src/ocr/`) | rectified images | cleaned images + extracted text + word boxes |
| 4 | Document Assembly (`src/assemble/`) | images + text + timestamps | searchable PDF with bookmarks |

## How the CV works (read this before tuning)

1. **Sampling** — video is decimated to 2 fps and downscaled to 360p for
   scoring (`reader.py`). Full-res frames are kept for export. Constant
   memory: the video is streamed, never loaded fully.
2. **Three difference signals** (`difference.py`, all on an 18%-border
   center ROI so a presenter at the frame edges can't vote):
   mean abs-diff after Gaussian blur, HSV-histogram Bhattacharyya distance,
   SSIM. Fusion: diff + SSIM vote (histogram is recorded + tested but
   excluded from fusion — measured inverted on MJPG/mp4v: static-pair
   jitter ~0.13 > true-change values 0.015–0.05; see report §8).
3. **Hysteresis detector** (`detector.py`): a spike (2/2 votes) opens a
   candidate; the next 1.0 s window must contain no further unmasked spike.
   Runs of ≥3 consecutive spikes are masked as motion first
   (`BURST_MIN_RUN=3`); lone cuts always survive.
   Single-pair blips (build steps, writing reveals) can neither open a
   change nor break a window.
4. **Keyframe** (`keyframe/`): sharpest frame per segment by variance of
   Laplacian (Tenengrad cross-check), preferring frames that match the
   segment median (occlusion rejection).
5. **Geometry** (`geometry/`, the CV core): Canny → contours → approxPolyDP
   → largest convex quadrilateral → `getPerspectiveTransform` +
   `warpPerspective` to a 1600×900 canvas. Falls back to full frame.
6. **Enhancement** (`enhance/`): morphological-closing background estimate
   divided out (kills projector hotspots), adaptive binarization,
   Hough deskew.
7. **OCR** (`ocr/`): Tesseract if installed, else RapidOCR-ONNX (pure pip),
   else text-less PDF with a note. Word boxes become the PDF's invisible
   text layer, so the PDF is genuinely searchable.
8. **Dedup + PDF** (`assemble/`): consecutive build-slides collapse when
   text is a superset and images are SSIM-similar; `reportlab` writes
   image + invisible text + timestamp bookmarks.

## Install & run (evaluator path — CLI only, no GUI needed)

```bash
pip install -r requirements.txt
# system tesseract is OPTIONAL (RapidOCR is the fallback):
#   sudo dnf install tesseract-ocr   # or: sudo apt install tesseract-ocr

# run on any lecture video (PDF + editable PPTX by default):
python -m slidesnap /path/to/lecture.mp4 --out out/
# outputs: out/slides.pdf  out/slides.pptx  out/scores.csv  out/summary.txt  out/debug/
# --formats pdf   # PDF only;  --keep-all disables build-dedup

# regenerate the 4-video test corpus (64 slides, MJPG/AVI):
python tools/make_samples.py

# metrics (segmentation P/R/F1 + OCR accuracy):
python tools/evaluate.py [--fast]

# ablation (full vs --no-rectify vs --no-illum OCR accuracy):
python tools/ablation.py

# unit tests (run from repo root):
python -m pytest tests/ -q
```

Ablation flags (for the report table): `--no-rectify`, `--no-illum`,
`--keep-all`. Streamlit demo (optional): `streamlit run app/streamlit_app.py`.

## Test corpus (`data/samples/`, 4 videos × 16 slides = 64)

- `a_clean.avi` — screen capture, easy baseline
- `b_angled.avi` — perspective-warped + static presenter silhouette (hard)
- `c_whiteboard.avi` — progressive writing reveals
- `d_builds.avi` — bullet builds (dedup stress test)
- `*.truth.txt` — ground-truth change timestamps

## Demo on real footage (`demo/`)

- `demo/cs231n_clip.mp4` — 100 s of Stanford CS231n Lecture 1 (speaker intro
  → "Welcome" photo slide), shipped so anyone can reproduce the real-world run.
- `demo/slides.pdf` + `demo/slides.pptx` — the generated deck (color pages,
  OCR text layer / speaker notes), `demo/keyframes.png`, `demo/scores.csv`.
- See `demo/DEMO.md` for measured numbers, what was fixed because of this
  clip (burst-mask run rule, color pages, dedup/OCR guards), and honest limits.

Measured segmentation (tol ±1 s): **P=R=F1=1.00 on all four videos
(60/60)**. OCR word accuracy (10 mid-segment slides each): **91.4 %
clean / 76.2 % angled-rectified** (RapidOCR, CPU).

## Instructions for testing

```bash
# fast: unit tests (21) + segmentation metrics on the bundled corpus
python -m pytest tests/ -q
python tools/evaluate.py --fast
# full: adds OCR spot accuracy (~35 s) and the rectify/illum ablation (~2 min)
python tools/evaluate.py
python tools/ablation.py
```

## Repo layout

```
./
  __main__.py  pipeline.py  config.py   # every threshold + tuning comment
  README.md  statement.md  AGENT.md  requirements.txt
  src/segmentation/  src/keyframe/  src/geometry/
  src/enhance/  src/ocr/  src/assemble/  src/debug/
  app/streamlit_app.py  tests/  data/samples/  docs/
../tools/
  make_samples.py  evaluate.py  ablation.py  make_diagrams.py  make_report.py
```

## Screenshots

- `docs/keyframes_a.png` — rectified keyframe grid (Stage 4 output)
- `docs/scores_a.png` — per-pair diff/hist/SSIM signals (observability)
- `debug/` inside any pipeline output dir holds the same per-run artefacts.
