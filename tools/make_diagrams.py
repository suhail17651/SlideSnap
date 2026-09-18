"""Generate docs/*.png diagrams with matplotlib (no graphviz needed).

Outputs: architecture.png, usecase.png, sequence.png, class-diagram.png,
         workflow.png (process flow), er.png (storage design).
Usage: python tools/make_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "slidesnap", "docs")
os.makedirs(DOCS, exist_ok=True)


def _box(ax, xy, w, h, text, fs=9, fc="#EAF2FF", ec="#2456A6"):
    r = mp.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                          fc=fc, ec=ec, lw=1.5)
    ax.add_patch(r)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def _arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("SlideSnap — System Architecture", fontsize=13, weight="bold")
    stages = [
        (0.3, 2.5, 1.9, 1.6, "1. Temporal\nSegmentation\nreader/difference/\ndetector"),
        (2.6, 2.5, 1.9, 1.6, "2. Keyframe\nsharpness +\nocclusion reject"),
        (4.9, 2.5, 1.9, 1.6, "3. Rectification\nscreen quad +\nhomography warp"),
        (7.2, 2.5, 1.6, 1.6, "4. Enhance\n+ OCR"),
        (9.2, 2.5, 1.5, 1.6, "5. Dedup\n+ PDF"),
    ]
    for s in stages:
        _box(ax, s[:2], s[2], s[3], s[4])
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2]
        x2 = stages[i + 1][0]
        _arrow(ax, (x1, 3.3), (x2, 3.3))
    _box(ax, (0.3, 0.5), 3.2, 1.1, "lecture video (.mp4/.avi)\n+ config.py thresholds", fs=8, fc="#FFF6E5", ec="#A66A24")
    _box(ax, (4.0, 0.5), 2.8, 1.1, "scores.csv + debug/\nobservability log", fs=8, fc="#F1F1F1", ec="#666")
    _box(ax, (7.3, 0.5), 3.4, 1.1, "slides.pdf (image + invisible\ntext layer + bookmarks)", fs=8, fc="#E7F7E7", ec="#2A7A2A")
    _arrow(ax, (1.9, 1.2), (1.2, 2.5))
    _arrow(ax, (5.4, 2.5), (5.4, 1.6))
    _arrow(ax, (9.9, 2.5), (9.0, 1.6))
    _box(ax, (0.3, 4.6), 10.4, 1.3,
         "Cross-cutting: config.py (all thresholds)  •  CLI pipeline.py  •  pytest suite  •  Streamlit demo (optional)",
         fs=8, fc="#F5F0FF", ec="#5A3FA6")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "architecture.png"), dpi=150)
    plt.close(fig)


def usecase():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("SlideSnap — Use Case Diagram", fontsize=13, weight="bold")
    ax.add_patch(mp.Circle((1.3, 2.7), 0.35, fc="#EAF2FF", ec="#2456A6", lw=1.5))
    ax.text(1.3, 1.8, "Student", ha="center", fontsize=10)
    ax.add_patch(mp.Circle((7.7, 2.7), 0.35, fc="#EAF2FF", ec="#2456A6", lw=1.5))
    ax.text(7.7, 1.8, "Evaluator", ha="center", fontsize=10)
    ax.add_patch(mp.Ellipse((4.5, 2.7), 3.4, 3.6, fc="none", ec="#333", lw=1.5))
    for i, uc in enumerate(["Upload lecture\nvideo", "Review slide\nthumbnails", "Download\nsearchable PDF",
                            "Run CLI +\nread scores.csv", "Run pytest\nsuite"]):
        y = 4.1 - i * 0.72
        ax.add_patch(mp.Ellipse((4.5, y), 2.4, 0.55, fc="#FFF6E5", ec="#A66A24", lw=1.2))
        ax.text(4.5, y, uc, ha="center", va="center", fontsize=7.5)
        if i < 3:
            ax.plot([1.65, 3.3], [2.9, y], color="#888", lw=0.9)
        else:
            ax.plot([7.35, 5.7], [2.9, y], color="#888", lw=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "usecase.png"), dpi=150)
    plt.close(fig)


def sequence():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("SlideSnap — Sequence Diagram (CLI run)", fontsize=13, weight="bold")
    actors = ["User", "pipeline", "segmentation", "keyframe+geometry", "enhance+ocr", "assemble"]
    xs = [0.7 + i * 1.72 for i in range(len(actors))]
    for x, a in zip(xs, actors):
        _box(ax, (x - 0.65, 4.4), 1.3, 0.5, a, fs=8)
        ax.plot([x, x], [0.4, 4.4], color="#888", lw=0.9, ls="--")
    msgs = [
        (0, 1, 4.0, "run(video, out/)"),
        (1, 2, 3.6, "load_samples → score → detect"),
        (1, 3, 3.2, "select keyframes → rectify"),
        (1, 4, 2.8, "flatten → deskew → OCR"),
        (1, 5, 2.4, "dedup → build_pdf"),
        (5, 1, 2.0, "slides.pdf + summary"),
        (1, 0, 1.4, "OK: N slides → pdf"),
    ]
    for a, b, y, t in msgs:
        xa, xb = (xs[a], xs[b]) if a < b else (xs[b], xs[a])
        _arrow(ax, (xa, y), (xb, y))
        ax.text((xa + xb) / 2, y + 0.12, t, ha="center", fontsize=7,
                bbox=dict(fc="white", ec="none", pad=1))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "sequence.png"), dpi=150)
    plt.close(fig)


def class_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("SlideSnap — Component / Class Diagram", fontsize=13, weight="bold")
    comps = [
        (0.2, 3.4, 3.1, 2.2, "segmentation\n+sample_frames()\n+score_pair()\n+detect_changes()"),
        (3.7, 3.4, 3.1, 2.2, "keyframe + geometry\n+best_index()\n+find_screen_quad()\n+rectify()"),
        (7.2, 3.4, 3.1, 2.2, "enhance + ocr\n+flatten_illumination()\n+deskew() / +extract()"),
        (0.2, 0.6, 3.1, 2.0, "assemble\n+Deduplicator\n+build_pdf()"),
        (3.7, 0.6, 3.1, 2.0, "debug.overlay\n+save_grid()\n+plot_scores()"),
        (7.2, 0.6, 3.1, 2.0, "config\nthresholds\n(no logic)"),
    ]
    for c in comps:
        _box(ax, c[:2], c[2], c[3], c[4], fs=8)
    for (x1, x2) in [((1.75, 3.4), (1.75, 2.6)), ((5.25, 3.4), (5.25, 2.6)),
                     ((8.75, 3.4), (5.25, 2.6)), ((3.3, 4.5), (3.7, 4.5))]:
        _arrow(ax, x1, x2)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "class-diagram.png"), dpi=150)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("SlideSnap — Process Flow", fontsize=13, weight="bold")
    steps = ["video in", "2 fps\nsample", "score pairs\ndiff/SSIM", "hysteresis\ndetect", "sharpest\nkeyframe",
             "homography\nrectify", "flatten +\nOCR", "dedup", "PDF out"]
    for i, s in enumerate(steps):
        x = 0.2 + i * 1.18
        fc = "#E7F7E7" if i in (0, 8) else "#EAF2FF"
        ec = "#2A7A2A" if i in (0, 8) else "#2456A6"
        _box(ax, (x, 1.0), 1.0, 1.0, s, fs=7.5, fc=fc, ec=ec)
        if i < len(steps) - 1:
            _arrow(ax, (x + 1.0, 1.5), (x + 1.18, 1.5))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "workflow.png"), dpi=150)
    plt.close(fig)


def er():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("SlideSnap — Storage Design (artefacts, no DB)", fontsize=13, weight="bold")
    tables = [
        (0.3, 0.7, 2.4, 1.8, "scores.csv\nt | diff | hist\nssim | spike"),
        (3.3, 0.7, 2.4, 1.8, "slide\nimage | text\nwords[] | ts"),
        (6.3, 0.7, 2.4, 1.8, "slides.pdf\npages + text\nlayer + bookmarks"),
    ]
    for t in tables:
        _box(ax, t[:2], t[2], t[3], t[4], fs=8, fc="#FFF6E5", ec="#A66A24")
    _arrow(ax, (2.7, 1.6), (3.3, 1.6))
    _arrow(ax, (5.7, 1.6), (6.3, 1.6))
    ax.text(3.0, 1.75, "1:N", fontsize=7)
    ax.text(6.0, 1.75, "N:1", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "er.png"), dpi=150)
    plt.close(fig)


def main():
    architecture()
    usecase()
    sequence()
    class_diagram()
    workflow()
    er()
    print("wrote", sorted(f for f in os.listdir(DOCS) if f.endswith(".png")))


if __name__ == "__main__":
    main()
