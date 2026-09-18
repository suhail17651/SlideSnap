"""Editable PPTX export: one slide per keyframe + speaker notes with OCR text.

Each PPTX slide = full-bleed keyframe image (16:9) + timestamp title +
OCR text in the notes (searchable in PowerPoint via Outline view).
python-pptx is a hard dep (pure pip, no system libs).
"""
import io
import cv2
from pptx import Presentation
from pptx.util import Inches
from PIL import Image


def _fmt_ts(sec):
    sec = float(sec)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_pptx(slides, out_path, title="SlideSnap Lecture Slides"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]  # blank
    for i, s in enumerate(slides):
        img = s["rectified"]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        buf.seek(0)
        slide = prs.slides.add_slide(blank)
        # full-bleed image
        slide.shapes.add_picture(buf, Inches(0), Inches(0),
                                 width=prs.slide_width, height=prs.slide_height)
        # timestamp title bar (small, bottom-left, readable on dark footer)
        tx = slide.shapes.add_textbox(Inches(0.2), Inches(7.0),
                                      Inches(5), Inches(0.4))
        tf = tx.text_frame
        tf.text = f"Slide {i + 1} @ {_fmt_ts(s.get('timestamp', 0))}"
        # OCR text -> speaker notes (Outline-searchable)
        notes = (s.get("text", "") or "").strip()
        if notes:
            slide.notes_slide.placeholders[1].text = notes[:4000]
    prs.core_properties.title = title
    prs.core_properties.author = "SlideSnap"
    prs.save(out_path)
    return out_path
