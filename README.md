# SlideSnap — Lecture Video → Clean Searchable Slide PDF

Recorded lectures trap slides inside hours of video: un-searchable,
un-printable. SlideSnap consumes a lecture recording and produces a
deduplicated, perspective-corrected, OCR'd PDF of every distinct slide, with
a bookmark per slide linking back to its video timestamp.

## Features

- Slide-change detection using OpenCV/numpy primitives (no scene-detection
  libraries): blur → abs-diff + histogram + SSIM → hysteresis with
  motion-burst masking.
- Sharpest-frame keyframe selection (variance of Laplacian + Tenengrad
  cross-check) with presenter-occlusion rejection against the segment median.
- Screen/board localisation and homography rectification to a 1600×900
  top-down canvas (full-frame fallback when no quad is found).
- Illumination flattening, adaptive binarization, Hough deskew, then OCR via
  Tesseract (if installed) else RapidOCR-ONNX (pure pip) — word boxes become
  the PDF's invisible, genuinely searchable text layer.
- Slide-build deduplication and timestamp-bookmarked PDF assembly.
- Fully CLI-runnable on CPU; per-run scores.csv, summary.txt, and
  debug/ (score plots, keyframe grids) for threshold tuning.
- Optional Streamlit demo UI with thumbnail review and keep/discard toggles.

## Technologies / Tools Used

Python, OpenCV (warpPerspective, Canny, contours, approxPolyDP),
numpy, scikit-image (SSIM), RapidOCR-onnxruntime (+ optional Tesseract via
pytesseract), Pillow, reportlab, python-pptx, Streamlit (demo only), pytest,
matplotlib (debug plots only).

## Functional Modules (>=3 required)

| # | Module | Input | Output |
|---|--------|-------|--------|
| 1 | Temporal Segmentation (src/segmentation/) | video file | slide-change timestamps + stable segments |
| 2 | Keyframe Selection & Rectification (src/keyframe/, src/geometry/) | candidate frames | one sharp, deskewed, top-down image per slide |
| 3 | Enhancement & OCR (src/enhance/, src/ocr/) | rectified images | cleaned images + extracted text + word boxes |
| 4 | Document Assembly (src/assemble/) | images + text + timestamps | searchable PDF with bookmarks |

## Non-Functional Requirements

- **Performance** — video is streamed frame-by-frame; constant memory regardless of duration.
- **Reliability** — full fallback chain: geometry → full-frame; OCR → Tesseract → RapidOCR → text-less PDF.
- **Usability** — single CLI command produces a ready-to-use PDF/PPTX with no manual steps.
- **Maintainability** — every threshold is centralised in config.py with inline comments.
- **Error handling** — each pipeline stage catches and logs exceptions without crashing the run.
- **Scalability** — modular stage design; any stage can be swapped or extended independently.

## How It Works

1. **Sampling** — video is decimated to 2 fps and downscaled to 360p for
   scoring. Full-res frames are kept for export. The video is streamed, never
   fully loaded into memory.
2. **Difference signals** — mean abs-diff after Gaussian blur, HSV-histogram
   Bhattacharyya distance, and SSIM — all computed on an 18%-border centre ROI
   so a presenter at the frame edges cannot influence the score.
3. **Hysteresis detector** — a spike (2/2 votes) opens a candidate; the next
   1.0 s window must contain no further unmasked spike. Runs of >=3 consecutive
   spikes are masked as motion first; lone cuts always survive.
4. **Keyframe selection** — sharpest frame per segment by variance of
   Laplacian (Tenengrad cross-check), preferring frames that match the segment
   median to reject presenter occlusion.
5. **Geometry** — Canny → contours → approxPolyDP → largest convex
   quadrilateral → getPerspectiveTransform + warpPerspective to a
   1600x900 canvas. Falls back to full frame if no quad found.
6. **Enhancement** — morphological-closing background estimate divided out
   (removes projector hotspots), adaptive binarization, Hough deskew.
7. **OCR** — Tesseract if installed, else RapidOCR-ONNX (pure pip), else
   text-less PDF with a note. Word boxes form the PDF invisible text layer.
8. **Dedup + PDF** — consecutive build-slides collapse when text is a superset
   and images are SSIM-similar; reportlab writes image + invisible text +
   timestamp bookmarks.

## Install & Run

```bash
pip install -r requirements.txt
# Tesseract is OPTIONAL (RapidOCR is the fallback):
#   sudo dnf install tesseract-ocr   # or: sudo apt install tesseract-ocr

# Run on any lecture video (PDF + editable PPTX by default):
python -m slidesnap /path/to/lecture.mp4 --out out/
# outputs: out/slides.pdf  out/slides.pptx  out/scores.csv  out/summary.txt  out/debug/

# PDF only:
python -m slidesnap /path/to/lecture.mp4 --out out/ --formats pdf

# Regenerate the 4-video test corpus (64 slides, MJPG/AVI):
python tools/make_samples.py

# Evaluation metrics (segmentation P/R/F1 + OCR accuracy):
python tools/evaluate.py [--fast]

# Ablation study (full vs --no-rectify vs --no-illum):
python tools/ablation.py
```

Ablation flags: --no-rectify, --no-illum, --keep-all.
Streamlit demo (optional): streamlit run app/streamlit_app.py

## Instructions for Testing

```bash
# Fast: unit tests (21 tests) + segmentation metrics on bundled corpus
python -m pytest tests/ -q
python tools/evaluate.py --fast

# Full: adds OCR spot accuracy (~35 s) and rectify/illum ablation (~2 min)
python tools/evaluate.py
python tools/ablation.py
```

## Test Corpus (data/samples/, 4 videos x 16 slides = 64)

- a_clean.avi — screen capture, easy baseline
- b_angled.avi — perspective-warped + static presenter silhouette (hard)
- c_whiteboard.avi — progressive writing reveals
- d_builds.avi — bullet builds (dedup stress test)
- *.truth.txt — ground-truth change timestamps

## Demo on Real Footage (demo/)

- demo/cs231n_clip.mp4 — 100 s of Stanford CS231n Lecture 1, shipped so
  anyone can reproduce the real-world run.
- demo/slides.pdf + demo/slides.pptx — the generated deck (colour pages,
  OCR text layer / speaker notes).
- demo/keyframes.png, demo/scores.csv — per-run observability artefacts.

Measured segmentation (tol +-1 s): P=R=F1=1.00 on all four synthetic videos
(60/60). OCR word accuracy (10 mid-segment slides each): 91.4% clean /
76.2% angled-rectified (RapidOCR, CPU).

## Repo Layout

```
./
  __main__.py    pipeline.py    config.py        # entry point + full pipeline + all thresholds
  README.md      statement.md   requirements.txt
  src/
    segmentation/   keyframe/   geometry/
    enhance/        ocr/        assemble/        debug/
  app/streamlit_app.py
  tests/
  data/samples/
  docs/            # diagrams + report PDF
  demo/            # real-footage clip + generated outputs
tools/             # helper scripts (import package via sys.path)
  make_samples.py  evaluate.py  ablation.py  make_diagrams.py  make_report.py
```

## Screenshots / Diagrams

- docs/architecture.png — system architecture diagram
- docs/usecase.png — use case diagram
- docs/class-diagram.png — class/component diagram
- docs/sequence.png — sequence diagram
- docs/workflow.png — process workflow diagram
- docs/er.png — ER diagram
- docs/keyframes_a.png — rectified keyframe grid (pipeline output)
- docs/scores_a.png — per-pair diff/hist/SSIM signals (observability)
- docs/report.pdf — full project report
