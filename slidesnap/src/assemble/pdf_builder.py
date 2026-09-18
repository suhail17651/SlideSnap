"""Searchable PDF: page image + invisible OCR text layer + timestamp bookmarks."""
import os
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io


def _fmt_ts(sec):
    sec = float(sec)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_pdf(slides, out_path, title="SlideSnap Lecture Slides"):
    """slides: list of dicts with keys: rectified (BGR), text, words, timestamp.
    words: [{text, conf, box(x,y,w,h)}] in rectified-image pixel coords.
    """
    c = canvas.Canvas(out_path)
    c.setTitle(title)
    c.setAuthor("SlideSnap")
    for i, s in enumerate(slides):
        img = s["rectified"]
        h, w = img.shape[:2]
        # landscape page sized to image aspect (points; 1px ~ 0.5pt cap)
        page_w, page_h = 800, 800 * h / max(1, w)
        c.setPageSize((page_w, page_h))
        # draw image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=page_w, height=page_h)
        # invisible text layer positioned by word box
        words = s.get("words", []) or []
        if words:
            t = c.beginText()
            # invisible render mode (3) if supported
            try:
                t.setTextRenderMode(3)
            except Exception:
                pass
            for wd in words:
                x, y, bw, bh = wd["box"]
                # map image px -> page pts; PDF origin bottom-left
                px = x / float(w) * page_w
                py_top = y / float(h) * page_h
                py = page_h - py_top
                fs = max(4, bh / float(h) * page_h * 0.9)
                try:
                    t.setFont("Helvetica", fs)
                except Exception:
                    pass
                t.setTextOrigin(px, py)
                # escape parentheses
                txt = wd["text"].replace("\\", "").replace("(", "").replace(")", "")
                t.textOut(txt + " ")
            c.drawText(t)
        ts = s.get("timestamp", 0)
        label = f"Slide {i+1} @ {_fmt_ts(ts)}"
        c.bookmarkPage(f"slide{i}")
        c.addOutlineEntry(label, f"slide{i}", level=0, closed=0)
        c.showPage()
    c.save()
    return out_path
