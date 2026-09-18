"""Streamlit UI: upload -> progress per stage -> thumbnail review -> download PDF."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import streamlit as st
except ImportError:
    raise SystemExit("pip install streamlit")

import pipeline as P


st.set_page_config(page_title="SlideSnap", layout="wide")
st.title("SlideSnap — Lecture Video → Searchable Slide PDF")
st.caption("Temporal segmentation → keyframe rectification → enhancement/OCR → PDF assembly")

up = st.file_uploader("Upload a lecture recording", type=["mp4", "avi", "mov", "mkv", "webm"])
col1, col2 = st.columns(2)
with col1:
    do_rect = st.checkbox("Homography rectification", value=True)
    do_illum = st.checkbox("Illumination flattening", value=True)
with col2:
    do_dedup = st.checkbox("Collapse slide builds", value=True)
    dbg = st.checkbox("Save debug artefacts", value=True)

if up is not None:
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, up.name)
        with open(src, "wb") as f:
            f.write(up.getbuffer())
        out = os.path.join(td, "out")
        prog = st.progress(0, text="Sampling + scoring…")
        try:
            res = P.run(src, out, debug=dbg, use_rectify=do_rect,
                        use_illum=do_illum, use_dedup=do_dedup)
            prog.progress(100, text="Done")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()
        if not res.get("ok"):
            st.error(res.get("error"))
            st.stop()
        st.success(f"{res['slides']} slides → PDF ({res['seconds']}s). "
                   f"Changes @ {res['changes']}")
        if res.get("skipped"):
            st.warning(f"Skipped {len(res['skipped'])} slide(s); see summary.")
        pdf = res["pdf"]
        with open(pdf, "rb") as f:
            st.download_button("Download searchable PDF", f,
                               file_name="slides.pdf", mime="application/pdf")
        # thumbnail review grid
        st.subheader("Slide review (keep/discard before export)")
        # show rectified keyframes from debug grid if present, else note
        grid = os.path.join(out, "debug", "keyframes.png") if dbg else None
        if grid and os.path.isfile(grid):
            st.image(grid, caption="Keyframe overview", use_container_width=True)
        else:
            st.caption("Debug grid disabled; PDF pages are listed below.")
        keep = st.multiselect("Slides to keep", list(range(1, res["slides"] + 1)),
                              default=list(range(1, res["slides"] + 1)))
        st.caption(f"Keeping {len(keep)} of {res['slides']} (toggle for final export).")
        if res.get("notes"):
            with st.expander("Pipeline notes / soft failures"):
                for n in res["notes"]:
                    st.write("- " + n)
else:
    st.info("Upload a video to begin. CLI equivalent: `python -m slidesnap.pipeline INPUT.mp4 --out out/`")
