"""Unit tests: end-to-end smoke test on the synthetic corpus (16 slides)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def test_pipeline_smoke():
    av = os.path.join(ROOT, "data", "samples", "a_clean.avi")
    if not os.path.isfile(av):
        pytest.skip("sample corpus not generated (run tools/make_samples.py)")
    import pipeline as P
    res = P.run(av, "/tmp/snap_test_smoke", debug=False, log=False)
    assert res.get("ok"), res
    assert res["slides"] == 16, res
    assert os.path.isfile("/tmp/snap_test_smoke/slides.pdf")
