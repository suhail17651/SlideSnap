# SlideSnap — Lecture Video to Searchable Slide PDF & PPTX

**Author:** Shaik Mohammed Suhail  
**Registration No.:** 24BAI10942  
**Course:** Computer Vision — Build Your Own Project (VITyarthi)  
**Repository:** [https://github.com/suhail17651/slidesnap](https://github.com/suhail17651/slidesnap)  

---

Recorded lecture videos trap slide material inside hours of non-searchable, linear video streams. Scrubbing timelines to find specific formulas, code blocks, or diagrams is slow and frustrating for exam revision. 

**SlideSnap** automates slide extraction directly from video recordings using classical computer vision primitives (OpenCV, NumPy, scikit-image). It extracts every distinct slide, corrects camera tilt via projective homography, removes projector hotspots, performs OCR, deduplicates bullet builds, and compiles a searchable, bookmarked PDF alongside an editable PowerPoint (PPTX) deck.

Entirely CPU-based, deterministic, and self-contained — built from scratch without black-box scene detection packages (e.g., PySceneDetect).

---

## Key Features

- **Robust Transition Detection:** Gaussian-blurred frame differencing combined with Structural Similarity (SSIM). Employs temporal hysteresis and consecutive-run burst masking to ignore walking presenters while catching authentic slide cuts.
- **Keyframe Selection & Occlusion Gate:** Ranks candidate frames within each stable segment by Variance of Laplacian (focus measure, cross-validated via Tenengrad). Rejects frames where the presenter blocks the screen by comparing against the segment's temporal median frame.
- **Projective Rectification:** Automatic screen corner detection (Canny edge $\to$ contour extraction $\to$ `approxPolyDP` polygon approximation) and homography transformation (`warpPerspective`) to a standard 1600×900 canvas. Falls back to a clean full frame if no reliable quad is detected.
- **Lighting Flattening & Deskew:** Large-kernel morphological closing division to eliminate projector hotspots and ambient shadows, Sauvola adaptive binarization, and Hough line baseline deskewing.
- **Dual-Engine OCR:** Word-level bounding boxes via Tesseract if locally installed, with automatic pure-Python fallback to RapidOCR-ONNX (never raises on missing binaries). Extracted text is overlaid as an invisible, selectable text layer on full-color PDF pages.
- **Build Deduplication:** Compares successive slides for text-superset and image similarity (SSIM > 0.94) to collapse progressive bullet-point reveals into single comprehensive slides.
- **Dual Deliverable Export:** Emits timestamp-bookmarked, searchable PDFs via ReportLab and native 16:9 PowerPoint presentation decks via `python-pptx` (with OCR text stored in speaker notes).
- **Auditability & Observability:** Logs per-frame difference metrics to `scores.csv` and outputs visual score plots and keyframe diagnostic grids in `debug/`.

---

## Core Pipeline Architecture

The system is organized into four modular stages:

| # | Stage | Directory | Responsibilities |
|---|---|---|---|
| **1** | **Temporal Segmentation** | `src/segmentation/` | 2 fps video decimation, 640×360 downscaled scoring proxy, center 18% ROI difference & SSIM calculation, burst masking, hysteresis cut confirmation. |
| **2** | **Keyframe & Geometry** | `src/keyframe/`, `src/geometry/` | Laplacian variance focus scoring, temporal median occlusion gate, screen quad detection, 1600×900 perspective warp. |
| **3** | **Enhancement & OCR** | `src/enhance/`, `src/ocr/` | Morphological illumination division, Sauvola adaptive binarization, Hough deskew, word-box text extraction. |
| **4** | **Deduplication & Assembly**| `src/assemble/` | Bullet-build text superset collapse, presenter shot filtering, searchable PDF assembly, PPTX deck generation. |

All operational thresholds, filter kernel sizes, and hysteresis parameters are strictly centralized in `config.py` with rationale documented inline.

---

## Measured Performance & Benchmark Results

Evaluated on a 4-video synthetic benchmark corpus (64 slides, 60 ground-truth transitions) and validated on a 100-second live classroom recording from Stanford CS231n:

- **Temporal Segmentation:** **Precision = 1.000, Recall = 1.000, F1 = 1.000** across all 4 benchmark videos (60 out of 60 ground-truth transitions correctly identified at $\pm$1.0 s tolerance, zero false cuts).
- **OCR Word Accuracy (CPU):** Reaches **91.43%** on clean digital captures and **76.19%** on angled camera captures.
- **Empirical Ablation Study:**
  - Enabling homography rectification on angled footage increases word accuracy from **59.05% to 76.19% (+17.14 pp)**.
  - Enabling morphological illumination flattening on clean footage increases word accuracy from **74.29% to 91.43% (+17.14 pp)**.
- **Real-World Validation (Stanford CS231n Lecture 1):** Successfully detects the 12.0 s speaker-to-slide cut through presenter pacing using the consecutive-run rule, and filters out 2 presenter-only frames while retaining true content slides.
- **Processing Throughput:** Processes a 64 s 720p video in ~25.4 s on a standard laptop CPU (~2.5× faster than real-time).

---

## Repository Layout & Project Structure

```text
SlideSnap/
├── pipeline.py                 # End-to-end CLI pipeline orchestrator
├── config.py                   # Centralized hyperparameter & threshold configurations
├── statement.md                # Formal problem statement & scope document (Rubric §5.2)
├── requirements.txt            # Pinned CPU dependencies (OpenCV, RapidOCR, ReportLab, etc.)
├── __main__.py                 # Module execution entry point (python3 -m slidesnap)
│
├── src/                        # Core modular engine (16 functional components)
│   ├── segmentation/           # Phase 1: Temporal boundary & transition detection
│   │   ├── reader.py           # Subsampled video reader & frame generator
│   │   ├── difference.py       # Gaussian abs-diff, color histogram & SSIM metrics
│   │   └── detector.py         # Hysteresis state machine & run-rule burst filter
│   ├── keyframe/               # Phase 2: Focus scoring & occlusion filtration
│   │   ├── sharpness.py        # Variance of Laplacian & Tenengrad focus operators
│   │   ├── occlusion.py        # Segment temporal median presenter filter
│   │   └── selector.py         # Multi-candidate sharpness & stability ranker
│   ├── geometry/               # Phase 3: Projective rectification & deskew
│   │   ├── screen_detect.py    # Canny edge, contour & approxPolyDP quad detector
│   │   └── rectify.py          # Homography perspective warp (1600x900 standard canvas)
│   ├── enhance/                # Phase 4: Illumination flattening & binarization
│   │   ├── illumination.py     # Morphological closing background division
│   │   ├── binarize.py         # Sauvola local adaptive thresholding
│   │   └── deskew.py           # Hough transform baseline rotation corrector
│   ├── ocr/                    # Phase 5: Optical character recognition
│   │   └── extract.py          # RapidOCR-ONNX / Tesseract word & box extractor
│   ├── assemble/               # Phase 6: Deduplication & dual-format delivery
│   │   ├── dedup.py            # SSIM & text superset slide-build collapse
│   │   ├── nonslide.py         # Edge density & text presence validation gate
│   │   ├── pdf_builder.py      # Searchable PDF with invisible text layer
│   │   └── pptx_builder.py     # Native 16:9 PowerPoint presentation export
│   └── debug/                  # Phase 7: Diagnostic tracing & visualization
│       └── overlay.py          # Metric traces, score plots & keyframe grids
│
├── tests/                      # Automated test suite (24 passing unit & integration tests)
│   ├── test_detector.py        # State machine & transition edge test cases
│   ├── test_difference.py      # Pixel diff & SSIM signal unit tests
│   ├── test_sharpness.py       # Focus measure & blur detection unit tests
│   ├── test_rectify.py         # Homography & corner ordering unit tests
│   ├── test_dedup.py           # Slide-build superset collapsing unit tests
│   ├── test_nonslide.py        # Non-slide content rejection tests
│   ├── test_pptx.py            # PPTX export generation & formatting tests
│   └── test_pipeline.py        # End-to-end synthetic video integration tests
│
├── data/samples/               # Bundled evaluation corpus (4 distinct recording regimes)
│   ├── a_clean.avi             # Direct screen-capture baseline
│   ├── b_angled.avi            # Perspective-distorted projector capture
│   ├── c_whiteboard.avi        # Uneven lighting with walking presenter
│   └── d_builds.avi            # Progressive bullet-point slide builds
│
├── demo/                       # Real-world lecture proof (Stanford CS231n)
│   ├── cs231n_clip.mp4         # 100-second real-world lecture video clip
│   ├── slides.pdf              # Generated searchable PDF deliverable
│   ├── slides.pptx             # Generated PowerPoint deck deliverable
│   ├── keyframes.png           # Extracted slide keyframe matrix
│   ├── scores.csv              # Per-frame difference metric trace log
│   └── DEMO.md                 # Real-footage evaluation notes & reproduction
│
├── docs/                       # Academic report, design models & visual artifacts
│   ├── report.pdf              # 13-page formal academic report (LaTeX pdflatex)
│   ├── report.tex              # Complete LaTeX report source code
│   ├── architecture.png        # System architecture diagram
│   ├── workflow.png            # Stage-by-stage process flow diagram
│   ├── usecase.png             # UML Use Case diagram
│   ├── sequence.png            # UML Sequence execution diagram
│   ├── class-diagram.png       # UML Class and component diagram
│   ├── er.png                  # Entity-Relationship diagram
│   ├── scores_a.png            # Empirical metric trace visualization
│   └── keyframes_a.png         # Pipeline output keyframe matrix
│
├── app/                        # Optional interactive review dashboard
│   └── streamlit_app.py        # Streamlit web UI with keep/discard toggles
│
└── tools/                      # Benchmark, ablation & report generation scripts
    ├── evaluate.py             # Segmentation P/R/F1 & OCR evaluation runner
    ├── ablation.py             # Feature ablation experiment runner
    ├── make_samples.py         # Synthetic benchmark video generator
    ├── make_diagrams.py        # Diagram generator script
    └── make_report.py          # Automated LaTeX compilation & audit script
```

## Installation & Setup

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/suhail17651/slidesnap.git
cd slidesnap
pip install -r requirements.txt
```

*Note: RapidOCR-ONNX is installed via pip and runs out of the box. System Tesseract (`tesseract-ocr`) is optional; if present, SlideSnap uses it automatically.*

---

## How to Run

### Run the CLI Pipeline on a Video
```bash
# Process video and export both PDF and editable PPTX:
python3 pipeline.py /path/to/lecture.mp4 --out out/

# Or run as a module:
python3 -m slidesnap /path/to/lecture.mp4 --out out/

# Export PDF only:
python3 pipeline.py /path/to/lecture.mp4 --out out/ --formats pdf
```

### Generated Outputs
```text
out/
├── slides.pdf         # Clean full-color slides with invisible, searchable OCR text & bookmarks
├── slides.pptx        # 16:9 presentation deck with extracted text in speaker notes
├── scores.csv         # Per-frame diff, hist, and SSIM metrics log
├── summary.txt        # Runtime statistics, detected cuts, and soft-failure notes
└── debug/             # Diagnostic score plots and keyframe inspection grids
```

### Ablation Flags (Benchmarking & Analysis)
```bash
python3 pipeline.py video.mp4 --no-rectify   # Disable perspective homography
python3 pipeline.py video.mp4 --no-illum     # Disable illumination flattening
python3 pipeline.py video.mp4 --keep-all     # Disable build deduplication
```

### Interactive Streamlit Web UI (Optional Demo)
```bash
streamlit run app/streamlit_app.py
```
Provides an interactive drag-and-drop dashboard with stage progress bars, thumbnail review grids, and per-slide keep/discard toggles before exporting.

---

## Running Tests & Evaluation

SlideSnap includes **24 automated unit and integration tests** verifying each stage in isolation as well as end-to-end execution:

```bash
# Run the complete test suite:
python3 -m pytest tests/ -q

# Run segmentation metrics evaluation on the bundled corpus:
python3 tools/evaluate.py --fast

# Run full evaluation including OCR spot accuracy:
python3 tools/evaluate.py

# Run ablation evaluation across pipeline configurations:
python3 tools/ablation.py
```

---

## Project Artefacts & Report

- **Comprehensive 15-Section Academic Report:** [`docs/report.pdf`](docs/report.pdf)  
  Authored in LaTeX ([`docs/report.tex`](docs/report.tex)) and compiled via `pdflatex`. Covers full problem analysis, functional/non-functional requirements, UML diagrams, design justifications, implementation details, live segmentation results, ablation studies, and testing methodology.
- **Design Diagrams:**
  - `docs/architecture.png` — System architecture diagram
  - `docs/workflow.png` — Process flow diagram
  - `docs/usecase.png` — UML use case model
  - `docs/sequence.png` — UML execution sequence model
  - `docs/class-diagram.png` — Component & class structure
  - `docs/er.png` — Data entity-relationship diagram
- **Empirical Visualizations:**
  - `docs/scores_a.png` — Per-frame difference/SSIM metric trace with threshold boundaries
  - `docs/keyframes_a.png` — Extracted rectified slide keyframe grid
- **Real-Footage Demo:** `demo/` contains a 100 s clip from Stanford CS231n along with committed, reproducible `slides.pdf` and `slides.pptx` outputs (see [`demo/DEMO.md`](demo/DEMO.md)).
