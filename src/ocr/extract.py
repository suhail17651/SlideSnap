"""OCR backends: Tesseract (preferred) -> RapidOCR-ONNX (no system dep) -> off.

Priority:
  1. Tesseract via pytesseract (best word boxes on clean slides).
  2. RapidOCR-onnxruntime (pure pip, no system tesseract needed).
  3. Unavailable: pipeline still runs (PDF has images, empty text layer).

Return contract is identical for all backends:
  {text, words:[{text,conf,box(x,y,w,h)}], available, backend}
Box coords are in input-image pixels.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as C

# --- backend 1: tesseract ---
try:
    import pytesseract
    from pytesseract import Output as _TessOut
    try:
        pytesseract.get_tesseract_version()
        _HAS_TESS = True
    except Exception:
        _HAS_TESS = False
except Exception:
    _HAS_TESS = False

# --- backend 2: rapidocr ---
try:
    from rapidocr_onnxruntime import RapidOCR as _Rapid
    _rapid_engine = None

    def _engine():
        global _rapid_engine
        if _rapid_engine is None:
            _rapid_engine = _Rapid()
        return _rapid_engine
    _HAS_RAPID = True
except Exception:
    _HAS_RAPID = False


def has_ocr():
    return bool(_HAS_TESS or _HAS_RAPID)


def backend_name():
    if _HAS_TESS:
        return "tesseract"
    if _HAS_RAPID:
        return "rapidocr"
    return "none"


def _extract_tess(frame_bgr, psm):
    import pytesseract
    from pytesseract import Output
    cfg = f"--oem {C.TESSERACT_OEM} --psm {psm}"
    data = pytesseract.image_to_data(frame_bgr, config=cfg, output_type=Output.DICT)
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        t = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1
        if not t or conf < C.MIN_OCR_CONF:
            continue
        words.append({"text": t, "conf": conf / 100.0,
                      "box": (int(data["left"][i]), int(data["top"][i]),
                              int(data["width"][i]), int(data["height"][i]))})
    try:
        full = pytesseract.image_to_string(frame_bgr, config=cfg)
    except Exception:
        full = " ".join(w["text"] for w in words)
    return {"text": full, "words": words, "available": True, "backend": "tesseract"}


def _extract_rapid(frame_bgr):
    import cv2
    eng = _engine()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res, _ = eng(rgb)
    words = []
    texts = []
    if res:
        for box, txt, conf in res:
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            if conf < C.RAPID_MIN_CONF:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            words.append({"text": str(txt), "conf": conf, "box": (x, y, w, h)})
            texts.append(str(txt))
    return {"text": " ".join(texts), "words": words,
            "available": bool(words or texts), "backend": "rapidocr"}


def extract(frame_bgr, psm=None):
    """Return {text, words, available, backend}. Never raises."""
    psm = psm or C.TESSERACT_PSM
    if _HAS_TESS:
        try:
            return _extract_tess(frame_bgr, psm)
        except Exception:
            pass
    if _HAS_RAPID:
        try:
            return _extract_rapid(frame_bgr)
        except Exception:
            pass
    return {"text": "", "words": [], "available": False, "backend": "none"}
