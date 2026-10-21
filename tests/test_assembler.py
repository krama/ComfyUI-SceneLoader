"""Tests for sceneloader.assembler. Run: python tests/test_assembler.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader.assembler import Library, SceneError, assemble, report, to_lora_stack  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


def test_example_scene():
    r = assemble(Library(EXAMPLES), "forest_walk")
    pos = r["positive"]
    # Prompt order: quality -> character (core first, then children) -> background -> light -> camera -> style.
    order = ["masterpiece", "1girl", "walking", "long coat", "forest", "golden hour", "cowboy shot", "watercolor"]
    idx = [pos.index(t) for t in order]
    assert idx == sorted(idx), pos
    assert r["negative"] == "low quality, worst quality, bad anatomy, extra fingers, short hair", r["negative"]
    files = [x["file"] for x in r["loras"]]
    assert files[-1] == "style/example_watercolor.safetensors", files  # style always last
    ident = [x for x in r["loras"] if x["type"] == "character"][0]
    assert ident["weight"] == 0.9, ident  # identity weight not overridable by the scene
    assert any("identity LoRA" in n for n in r["notes"]), r["notes"]
    det = {x["file"]: x for x in r["loras_detailer"]}
    assert "slider/example_height_slider.safetensors" not in det  # dropped by type
    assert det["character/example_girl.safetensors"]["weight"] == 1.0  # override
    assert det["detail/example_detail.safetensors"]["weight"] == 0.5  # add
    stack = json.loads(to_lora_stack(r["loras"], "illustrious"))
    assert stack[0] == {"name": "illustrious/character/example_girl.safetensors", "weight": 0.9,
                        "enabled": True, "tags": ""}, stack[0]
    text = report(r, "illustrious", lambda n: None)
    assert "MISSING" in text and "[forest_walk/example_girl]" in text, text


def test_errors():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "blocks"))
        os.makedirs(os.path.join(d, "scenes"))
        with open(os.path.join(d, "blocks", "a.yaml"), "w") as fh:
            fh.write("children:\n  - use: b\n")
        with open(os.path.join(d, "blocks", "b.yaml"), "w") as fh:
            fh.write("children:\n  - use: a\n")
        with open(os.path.join(d, "scenes", "loop.yaml"), "w") as fh:
            fh.write("children:\n  - use: a\n")
        with open(os.path.join(d, "scenes", "missing.yaml"), "w") as fh:
            fh.write("children:\n  - use: nope\n")
        lib = Library(d)
        assert lib.scenes() == ["loop", "missing"]
        for name, needle in (("loop", "circular"), ("missing", "not found")):
            try:
                assemble(lib, name)
            except SceneError as e:
                assert needle in str(e), e
            else:
                raise AssertionError(f"{name}: expected SceneError")


if __name__ == "__main__":
    for fn in (test_example_scene, test_errors):
        fn()
        print(f"ok  {fn.__name__}")
