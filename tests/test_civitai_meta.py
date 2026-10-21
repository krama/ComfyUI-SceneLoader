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


def test_weights_from_workflow():
    import json
    import re
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())  # noqa: E731
    names = {norm("abc_v1")}
    rg = {"comfy": json.dumps({"n": [{"on": True, "lora": "illustrious\\style\\abc_v1.safetensors", "strength": 0.65}]})}
    ll = {"workflow": {"nodes": [{"inputs": {"lora_name": "abc_v1.safetensors", "strength_model": 0.7}}]}}
    two = {"comfy": json.dumps({"x": [{"lora": "other.safetensors", "strength": 0.3},
                                      {"lora": "sub/abc_v1.safetensors", "strength": 0.9}]})}
    assert F.weights_from_workflow(rg, names, norm) == [0.65]
    assert F.weights_from_workflow(ll, names, norm) == [0.7]
    assert F.weights_from_workflow(two, names, norm) == [0.9]


def test_usage_tips_priority():
    info = {"id": 1, "modelId": 2, "baseModel": "Illustrious", "description": "Use weight 0.6"}
    tips = {"strength": 1, "minStrength": 0.7, "maxStrength": 1.5}
    assert F.build_meta(info, "x", None, tips)["weight"] == {"min": 1.0, "max": 1.0, "source": "usage_tips"}
    assert F.build_meta(info, "x", None, {})["weight"]["source"] == "version_description"


if __name__ == "__main__":
    for fn in (test_weights_from_text, test_optional_tags, test_weights_from_workflow, test_usage_tips_priority):
        fn()
        print(f"ok  {fn.__name__}")
