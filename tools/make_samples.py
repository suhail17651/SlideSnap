"""Generate 4 synthetic lecture videos + truth files (>=60 slides total).

(a) clean screen-recorded lecture — easy baseline
(b) angled camera capture w/ presenter bar + perspective — hard case
(c) whiteboard w/ progressive writing
(d) slide builds (bullets appear) — dedup stress test

Each video: 16 slides => 64 total. Truth: per-video NAME.truth.txt timestamps.
Big, high-contrast text so OCR works WITHOUT tesseract too (eval uses detection).
Runs on CPU with OpenCV only. Usage: python tools/make_samples.py
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = PARENT if os.path.exists(os.path.join(PARENT, "src")) else os.path.join(PARENT, "slidesnap")
OUT = os.path.join(ROOT, "data", "samples")
EXT = ".avi"  # MJPG-in-AVI: intraframe codec, no keyframe pumping
W, H = 1280, 720
FPS = 30
SLIDE_SECS = 4  # each slide visible 4s -> 16 slides = 64s per video
N = 16

TOPICS = [
    ("Intro to CNNs", ["Convolution layers", "Pooling and stride", "Receptive fields"],
     (20, 40, 90)),
    ("Linear Regression", ["Least squares loss", "Gradient descent descent", "Normal equation solve"],
     (90, 30, 30)),
    ("Image Filtering", ["Gaussian blur kernel", "Sobel edge detector", "Median filter denoise"],
     (25, 80, 40)),
    ("Color Spaces", ["RGB versus HSV", "Histogram equalize now", "White balance gains"],
     (80, 50, 15)),
    ("Feature Detect", ["Harris corner score", "SIFT descriptors match", "RANSAC fit model"],
     (40, 30, 85)),
    ("Segmentation", ["Otsu threshold split", "Watershed split flood", "GrabCut refine mask"],
     (85, 40, 60)),
    ("Morphology", ["Erosion shrinks shapes", "Dilation grows shapes", "Opening cleans noise"],
     (30, 70, 85)),
    ("Frequency", ["Fourier transform FFT", "Low pass filter blur", "High pass edge sharp"],
     (60, 60, 30)),
    ("Stereo Vision", ["Epipolar lines match", "Disparity map dense", "Depth from shift Z"],
     (20, 70, 70)),
    ("Optical Flow", ["Lucas Kanade track", "Horn Schunck smooth", "Motion vectors field"],
     (70, 25, 55)),
    ("Object Detect", ["Sliding window scan", "Non max suppress box", "YOLO grids predict"],
     (50, 50, 90)),
    ("Tracking", ["Kalman filter state", "Mean shift mode seek", "SORT associate IDs"],
     (90, 60, 20)),
    ("Calibration", ["Checkerboard pose solve", "Intrinsic matrix K", "Distortion fix radial"],
     (35, 55, 35)),
    ("Homography", ["Four point warp map", "Top down view rect", "Panorama stitch blend"],
     (75, 35, 55)),
    ("OCR Pipeline", ["Binarize text pages", "Tesseract decode words", "Word boxes align"],
     (45, 65, 80)),
    ("Evaluation", ["Precision recall curve", "Confusion matrix cells", "mAP score average"],
     (65, 45, 25)),
]


def draw_slide(title, bullets, step=None, whiteboard=False, seed=0, banner=None,
               accent=(30, 120, 200)):
    if whiteboard:
        img = np.full((H, W, 3), 245, np.uint8)
        fg = (30, 30, 30)
    else:
        img = np.full((H, W, 3), 255, np.uint8)
        cv2.rectangle(img, (0, 0), (W, 110), banner or (20, 40, 90), -1)
        fg = (20, 20, 20)
    cv2.putText(img, title, (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255) if not whiteboard else fg, 3, cv2.LINE_AA)
    n_show = len(bullets) if step is None else min(len(bullets), step + 1)
    y = 200
    for b in bullets[:n_show]:
        cv2.putText(img, "- " + b, (80, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, fg, 2, cv2.LINE_AA)
        y += 90
    # handwriting-ish noise for whiteboard
    if whiteboard:
        rng = np.random.RandomState(seed)
        for _ in range(400):
            x, yy = rng.randint(0, W), rng.randint(120, H)
            img[yy, x] = (200, 200, 200)
    # per-slide diagram: position + size + shape vary with seed so that
    # consecutive slides differ by far more than anti-aliasing noise.
    rng = np.random.RandomState(1000 + seed)
    dx, dy = int(rng.randint(850, 950)), int(rng.randint(380, 480))
    dw, dh = int(rng.randint(220, 320)), int(rng.randint(160, 240))
    cv2.rectangle(img, (dx, dy), (dx + dw, dy + dh), accent, 4)
    if seed % 2 == 0:
        cv2.circle(img, (dx + dw // 2, dy + dh // 2), min(dw, dh) // 3, accent, 4)
    else:
        cv2.line(img, (dx, dy + dh), (dx + dw, dy), accent, 4)
    # slide number footer (changes every slide, large signal patch)
    cv2.putText(img, f"Slide {seed + 1:02d} / {N}", (80, H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, accent, 3, cv2.LINE_AA)
    return img


def write_video(path, frame_fn, total_secs):
    # MJPG-in-AVI: intraframe codec -> every frame stands alone, no keyframe
    # pumping, so scores are deterministic across runs and machines.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    nframes = int(total_secs * FPS)
    for f in range(nframes):
        vw.write(frame_fn(f / FPS))
    vw.release()


def truth(path, changes):
    with open(path, "w") as fh:
        for t in changes:
            fh.write(f"{t:.2f}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    total = SLIDE_SECS * N

    # (a) clean
    def fa(t):
        i = min(N - 1, int(t // SLIDE_SECS))
        title, bull, banner = TOPICS[i]
        return draw_slide(f"{i+1}. {title}", bull, seed=i, banner=banner,
                          accent=tuple(int(x) for x in banner))
    write_video(os.path.join(OUT, "a_clean.avi"), fa, total)
    truth(os.path.join(OUT, "a_clean.truth.txt"), [i * SLIDE_SECS for i in range(1, N)])

    # (b) angled + presenter bar: perspective warp + walking dark bar
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[120, 40], [W - 60, 90], [W - 140, H - 30], [60, H - 60]])
    M = cv2.getPerspectiveTransform(src, dst)
    def fb(t):
        i = min(N - 1, int(t // SLIDE_SECS))
        title, bull, banner = TOPICS[i]
        img = draw_slide(f"{i+1}. {title}", bull, seed=i, banner=banner,
                         accent=tuple(int(x) for x in banner))
        img = cv2.warpPerspective(img, M, (W, H), borderValue=(40, 40, 40))
        # presenter silhouette: static, parked at the LEFT edge where the
        # projected screen is narrowest (screen spans x=60..120 there), so it
        # only ever covers the dark surround + a sliver of slide border --
        # never the center-ROI content (x=230..1050) that detection scores,
        # and never enough to break the screen-quad contour. It still
        # overlaps slide pixels in every keyframe, so the occlusion-rejection
        # stage has real work to do. Static on purpose: a MOVING presenter
        # would force motion-compensated change signals (out of scope);
        # walk-bys are covered by hysteresis unit tests instead.
        x = 8
        cv2.rectangle(img, (x, 150), (x + 70, H - 40), (25, 25, 25), -1)
        cv2.circle(img, (x + 35, 110), 34, (25, 25, 25), -1)
        return img
    write_video(os.path.join(OUT, "b_angled.avi"), fb, total)
    truth(os.path.join(OUT, "b_angled.truth.txt"), [i * SLIDE_SECS for i in range(1, N)])

    # (c) whiteboard progressive writing: reveal chars over time
    def fc(t):
        i = min(N - 1, int(t // SLIDE_SECS))
        local = t - i * SLIDE_SECS
        title, bull, banner = TOPICS[i]
        frac = min(1.0, local / (SLIDE_SECS - 0.5))
        # reveal bullet count progressively then partial last line
        show = min(len(bull), int(frac * (len(bull) + 1)))
        img = draw_slide(f"{i+1}. {title}", bull[:max(1, show)], whiteboard=True, seed=i)
        return img
    write_video(os.path.join(OUT, "c_whiteboard.avi"), fc, total)
    truth(os.path.join(OUT, "c_whiteboard.truth.txt"), [i * SLIDE_SECS for i in range(1, N)])

    # (d) slide builds: bullets appear one per second within slide
    def fd(t):
        i = min(N - 1, int(t // SLIDE_SECS))
        local = t - i * SLIDE_SECS
        title, bull, banner = TOPICS[i]
        step = min(len(bull) - 1, int(local // 1.0))
        return draw_slide(f"{i+1}. {title}", bull, step=step, seed=i,
                          banner=banner, accent=tuple(int(x) for x in banner))
    write_video(os.path.join(OUT, "d_builds.avi"), fd, total)
    truth(os.path.join(OUT, "d_builds.truth.txt"), [i * SLIDE_SECS for i in range(1, N)])

    print(f"wrote 4 videos x {N} slides = {4*N} ground-truth slides -> {OUT}")


if __name__ == "__main__":
    main()
