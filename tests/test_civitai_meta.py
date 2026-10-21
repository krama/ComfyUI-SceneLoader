"""Tests for tools/fetch_civitai_meta weight parsing. Run: python tests/test_civitai_meta.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import fetch_civitai_meta as F  # noqa: E402

CASES = [
    ("SDXL Illustrious DoRA weighting begins at 0.0, with 0.05 - 0.4 as optimal range.", (0.05, 0.4)),
    ("Strength:0.8 - 1", (0.8, 1.0)),
    ("Recommend preferred weight 0.7~1", (0.7, 1.0)),
    ("Use 30 steps, cfg 5, weight 0.6", (0.6, 0.6)),
    ("Pony version: weight 0.4. Illustrious version: weight 0.8", (0.8, 0.8)),
    ("sweet spot 0.7", (0.7, 0.7)),
    ("Upscale 1.5x with denoise 0.4", None),
    ("No numbers here", None),
    ("weight 0.8, 30 steps", (0.8, 0.8)),
    ("strength 1 with cfg 7", (1.0, 1.0)),
]


def test_weights_from_text():
    for text, want in CASES:
        got = F.weights_from_text(text, "illustrious")
        assert got == want, (text, got, want)


def test_optional_tags():
    got = F.optional_tags("trigger: s1_dram Optional positive: dappled light Optional negative: 3d")
    assert got == {"extra_positive": "dappled light", "extra_negative": "3d"}, got


if __name__ == "__main__":
    for fn in (test_weights_from_text, test_optional_tags):
        fn()
        print(f"ok  {fn.__name__}")
