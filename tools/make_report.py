"""Generate docs/report.pdf (all 15 VITyarthi sections) using LaTeX (pdflatex).

Uses measured numbers from tools/eval_results.txt + live segmentation eval.
Embeds docs/*.png (diagrams + debug artefacts) via graphicx.
Professional LaTeX academic typography with pdflatex.

Usage:
    cd /home/danish1075/Music/suhaail
    python3 tools/make_report.py
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = PARENT if os.path.exists(os.path.join(PARENT, "src")) else os.path.join(PARENT, "slidesnap")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

DOCS = os.path.join(ROOT, "docs")
SAMPLES = os.path.join(ROOT, "data", "samples")


def seg_table():
    """Live segmentation evaluation on synthetic corpus."""
    from src.segmentation import reader as R, difference as D, detector as DET
    rows = []
    tps = fps = fns = 0
    for name in ["a_clean", "b_angled", "c_whiteboard", "d_builds"]:
        v = os.path.join(SAMPLES, name + ".avi")
        ts, _, smalls = R.load_samples(v)
        scores = D.score_sequence(smalls)
        changes, segs = DET.detect_changes(ts, scores)
        det = [round(float(ts[i]), 2) for i in changes]
        truth = [float(x.strip()) for x in open(os.path.join(SAMPLES, name + ".truth.txt"))]
        m = DET.evaluate(det, truth)
        tps += m["tp"]
        fps += m["fp"]
        fns += m["fn"]
        rows.append((name, m["precision"], m["recall"], m["f1"], len(segs)))
    P = tps / max(1, tps + fps)
    Rc = tps / max(1, tps + fns)
    F1 = 2 * P * Rc / max(1e-9, P + Rc)
    return rows, (P, Rc, F1)


def main():
    print("Evaluating live segmentation on corpus...")
    rows, overall = seg_table()
    print(f"Segmentation verified: P={overall[0]:.3f}, R={overall[1]:.3f}, F1={overall[2]:.3f}")

    tex_file = os.path.join(DOCS, "report.tex")
    if not os.path.exists(tex_file):
        raise FileNotFoundError(f"LaTeX template not found: {tex_file}")

    print("Compiling LaTeX report with pdflatex (Pass 1)...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=DOCS, capture_output=True, text=True
    )
    if res1.returncode != 0:
        print("LaTeX compilation error (Pass 1):")
        print("\n".join(res1.stdout.splitlines()[-35:]))
        sys.exit(res1.returncode)

    print("Compiling LaTeX report with pdflatex (Pass 2 for TOC & references)...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=DOCS, capture_output=True, text=True
    )
    if res2.returncode != 0:
        print("LaTeX compilation error (Pass 2):")
        print("\n".join(res2.stdout.splitlines()[-35:]))
        sys.exit(res2.returncode)

    out_pdf = os.path.join(DOCS, "report.pdf")
    print(f"Successfully generated LaTeX report -> {out_pdf}")


if __name__ == "__main__":
    main()
