"""
SlideSnap global configuration.
Every threshold lives here as a named constant with a tuning comment.
Tuned on the synthetic 4-video corpus (see tools/make_samples.py).
"""
import os

# ---- Frame sampling ----
SAMPLE_FPS = 2.0
# Decimation rationale: lecture slides are static for seconds; 2 fps cuts
# decode+score cost ~15x vs 30fps with no recall loss on static content.
# Higher rates only add compression-noise comparisons. Measured ~8x faster.
SCORING_HEIGHT = 360  # downscale to 360p for scoring; full-res kept for export
SCORING_WIDTH = 640

# All three difference signals are computed on a center ROI that excludes
# the outer 18% border on every side (see difference.centered_roi). This is
# the cheapest presenter-robustness: the silhouette walks the frame edges
# while slide content fills the center. Thresholds below are measured with
# the ROI in place.
ROI_BORDER_FRAC = 0.18
# Gaussian blur before differencing suppresses compression noise + cursor jitter.
# Kernel 5x5 / sigma 1.0: removes single-pixel noise, preserves slide-change edges.
BLUR_KERNEL = (5, 5)
BLUR_SIGMA = 1.0

# Mean absolute frame difference (0..1 on normalized grayscale).
# True changes on the corpus (measured, ROI in place): clean 0.006-0.008,
# angled 0.008-0.10 (silhouette adds to slide-change magnitude);
# within-slide edits (build steps, writing reveals) 0.001-0.004.
# HIGH=0.005 splits them: every truth pair exceeds it, every within-slide
# edit stays below it (measured max ~0.004). LOW=0.003 admits compression
# jitter while rejecting genuine edits, so confirmation windows only pass
# on real stability.
DIFF_HIGH = 0.005  # candidate change fires above this
DIFF_LOW = 0.003   # scene considered stable below this

# HSV histogram Bhattacharyya distance (0=identical, 1=totally different).
# MEASURED: mp4v keyframe pumping makes static-pair histograms jitter up to
# ~0.11, while true changes score only 0.015-0.05 on the synthetic corpus
# (flat vector-graphics slides). Histogram is therefore a weak signal here
# (kept for real H.264 footage where compression jitter is far lower);
# HIGH=0.15 means it only votes on violent colour changes, LOW=0.12 keeps
# jittery static pairs inside the stability window.
HIST_HIGH = 0.15
HIST_LOW = 0.12
HIST_BINS_H = 50
HIST_BINS_S = 60

# SSIM (1=identical). MEASURED: static 1.0, build steps 0.978-0.991,
# progressive-writing reveals 0.978-0.991, true changes 0.93-0.966 (clean
# ROI) / ~0.81-0.95 (angled). 0.972 splits truth from builds while keeping
# a margin from the weakest truth pair (0.966); stability floor 0.975
# tolerates within-slide edits so they don't break confirmation windows.
SSIM_HIGH_CHANGE = 0.972  # below this => candidate change
SSIM_LOW_STABLE = 0.975   # above this => stable

# Fusion: a change candidate requires at least 2 of 3 signals to fire.
# Rationale: abs-diff fails on projector dimming (global gain, low abs-diff
# but histogram shifts); histogram fails on colour-preserving text edits
# (small region change, histogram barely moves but SSIM drops); SSIM fails
# on heavy compression artefacts (false drop while others stay stable).
MIN_VOTES = 2

# ---- Hysteresis / stability (detector.py) ----
# After a high-threshold spike, scene must stay calm for STABILITY_SECONDS
# to confirm a new slide: the window must contain NO further change spike
# (votes>=MIN_VOTES). Within-slide edits (build steps, writing reveals:
# single-pair blips with calm neighbours) never break a window, but a real
# neighbouring slide change does. Combined with hysteresis (spike to open,
# calm window to confirm), this kills false positives from presenter
# walk-bys (transient spike with no calm window after it) while true
# slide changes, 4s apart on the corpus, confirm cleanly.
STABILITY_SECONDS = 1.0
# Hysteresis burst mask: presenter walk-bys / camera pans fire spikes on
# CONSECUTIVE scored pairs (measured 5-14 long on a real CS231n lecture
# clip). Mask only runs of >= BURST_MIN_RUN consecutive spikes as motion.
# Clean cuts are lone spikes (run length 1) and always survive; 2-runs are
# kept too, so a cut sitting next to motion is never vetoed by its
# neighbour (a neighbourhood-count would kill it — measured failure).
BURST_MIN_RUN = 3
# Minimum segment length: ignore flicker segments shorter than this.
MIN_SEGMENT_SECONDS = 1.0
# Tolerance window for evaluation (±1s around ground truth).
EVAL_TOLERANCE_SECONDS = 1.0

# ---- Keyframe sharpness (sharpness.py) ----
# Variance of Laplacian: higher = sharper. Beats gradient-sum on low-texture
# slides because Laplacian penalises smooth blur more steeply.
LAP_KERNEL = 3
# Tenengrad cross-check threshold: both measures must agree on best frame
# in >80% of segments or thresholds need retuning.
TENENGRAD_KERNEL = 3

# ---- Occlusion (occlusion.py) ----
# Candidate vs segment median: pixels differing by >40 gray levels over
# >2% of slide area => presenter/occlusion blob => reject frame.
OCCLUSION_PIXEL_DIFF = 40
OCCLUSION_AREA_RATIO = 0.02
# Prefer unoccluded frame even if its sharpness is up to 25% lower.
OCCLUSION_SHARPNESS_TRADEOFF = 0.75

# ---- Screen detection / rectification (geometry/) ----
CANNY_LOW = 50
CANNY_HIGH = 150
# approxPolyDP epsilon as fraction of contour perimeter.
APPROX_EPSILON_RATIO = 0.02
# Plausible slide aspect ratios (4:3 through 16:9 plus board tolerance).
MIN_QUAD_AREA_RATIO = 0.15  # quad must cover >=15% of frame
ASPECT_MIN = 1.1
ASPECT_MAX = 2.2
# Rectified canvas size.
WARP_WIDTH = 1600
WARP_HEIGHT = 900

# ---- Enhancement (enhance/) ----
# Illumination background: large-kernel morphological closing estimates the
# projector hotspot / board shadow field, then we divide it out. Kernel ~1/8
# of image width. Beats global hist-equalisation which amplifies noise.
BG_KERNEL_RATIO = 8
# Sauvola/adaptive binarization: Otsu fails on uneven projector lighting
# (single global threshold), adaptive uses local windows.
ADAPTIVE_BLOCK = 51  # must be odd
ADAPTIVE_C = 10
MORPH_OPEN_KERNEL = (3, 3)

# ---- Deskew ----
# Hough transform on text mask: accept baselines within ±5 deg search.
DESKEW_MAX_ANGLE = 5.0

# ---- OCR ----
TESSERACT_PSM = 6  # assume uniform block of text
TESSERACT_OEM = 1
MIN_OCR_CONF = 30  # tesseract: drop words below this confidence (0-100 scale)
RAPID_MIN_CONF = 0.3  # rapidocr: drop words below this (0-1 scale)

# ---- Deduplication (assemble/dedup.py) ----
# Near-duplicate collapse (exact/compression-level duplicates, eyeball-graphics
# builds): text empty-or-identical AND image SSIM > threshold. MEASURED on a
# real CS231n lecture clip (RapidOCR, rectified pages): exact photo-slide
# re-shows 0.9918, eyeball-graphics build 0.951, speaker-vs-speaker 0.9317.
# 0.94 splits the near-dups from the speaker pair with margin on both sides.
DEDUP_NEAR_DUP_SSIM = 0.94
DEDUP_TEXT_SUPERSET_RATIO = 0.9  # >=90% of words of slide N appear in N+1

# ---- Non-slide (speaker) filter (assemble/nonslide.py) ----
# A keyframe showing the presenter (not a slide) has almost no OCR text and
# what little there is hugs the frame bottom (watermark/footer), never the
# body. MEASURED on a real CS231n clip (RapidOCR, 450px-high OCR input):
# speaker frames 3-5 words / yspread 0.46-0.62, true slides 5-26 words /
# yspread 0.93-0.96. Conservative cut: < NONSLIDE_MIN_WORDS words AND
# yspread < NONSLIDE_MAX_YSPREAD. Margin: nearest true slide has 5 words
# (above the count cut) and 0.93 spread (far above the spread cut) — both
# conditions must hold, so no true slide is within one condition of the cut.
NONSLIDE_MIN_WORDS = 5
NONSLIDE_MAX_YSPREAD = 0.75
# A slide is meaningful only if it has at least this many words; shorter
# texts (watermarks) still count when spread wide — the AND above decides.
NONSLIDE_ENABLE = True

# ---- Performance ----
# Target: 60-min 1080p lecture in <5 min on CPU via decimation + 360p scoring.
FFMPEG_FALLBACK = True

# ---- Paths ----
def repo_root():
    return os.path.dirname(os.path.abspath(__file__))

DEBUG_DIR = os.path.join(repo_root(), "debug")
SAMPLES_DIR = os.path.join(repo_root(), "data", "samples")
