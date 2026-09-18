# statement.md — SlideSnap

## Problem statement

Recorded lectures are hard to revise from: the slides are trapped inside
hours of video, un-searchable and un-printable. Students scrub timelines to
find one diagram; faculty cannot reuse recorded content; accessibility
services lack text versions of visual material.

## Scope

SlideSnap consumes a lecture recording (mp4/avi/mov/mkv/webm) and produces a
deduplicated, perspective-corrected, OCR'd PDF of every distinct slide, with
bookmarks linking each page back to its video timestamp. Fully CLI-runnable
(`python -m slidesnap VIDEO --out OUT/`); the Streamlit UI is an optional
demo. Out of scope: speaker diarization, audio transcription,
handwritten-equation parsing.

## Target users

Students revising before exams, faculty repurposing recorded content,
accessibility services needing text versions of visual material.

## High-level features

1. Temporal segmentation — slide-change timestamps from frame-differencing
   + histogram + SSIM fusion with hysteresis (no scene-detection libs).
2. Keyframe selection & rectification — sharpest unoccluded frame per slide,
   homography-warped to a top-down 1600×900 canvas.
3. Enhancement & OCR — illumination flattening, adaptive binarization,
   Hough deskew, Tesseract/RapidOCR word boxes.
4. Document assembly — searchable PDF (image + invisible text layer) with
   timestamp bookmarks; slide builds collapsed automatically.
