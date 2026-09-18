"""SlideSnap CLI entry: `python -m slidesnap ...` == `python slidesnap/pipeline.py ...`."""
from .pipeline import main, run

if __name__ == "__main__":
    raise SystemExit(main())
