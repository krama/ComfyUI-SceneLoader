"""Tests for sceneloader.assembler. Run: python tests/test_assembler.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader.assembler import (Library, SceneError, assemble, camera_tags_from_json,  # noqa: E402
                                   report, to_lora_stack)

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


def cam(shot, tilt, pan):
    return json.dumps({"camera": {"shot_type": shot, "tilt_deg": tilt, "pan_deg": pan, "roll_deg": 0.0}})


def test_camera_tags():
    cases = [
        (cam("medium_shot", "tilt 0.0", "pan to right 180.0"), "", "upper body"),
        (cam("close_up", "tilt down 20.0", "pan to left 170.0"), "", "close-up, portrait, from above"),
        (cam("full_shot", "tilt down 80.0", "pan to right 179.0"), "", "full body, from above, bird's eye view"),
        (cam("medium_long_shot", "tilt up 25.0", "pan to right 100.0"), "", "cowboy shot, from below, from side"),
        (cam("wide_shot", "tilt 0.0", "pan to left 10.0"), "", "wide shot, very wide shot, from behind"),
        (cam("full_shot", "tilt 0.0", "pan 0.0"), "looking from behind, full shot", "full body, from behind"),
        (cam("full_shot", "tilt down 90.0", "pan 0.0"), "above object and tilt down at 90 degree",
         "full body, from above, bird's eye view"),
        (cam("full_shot", "tilt 0.0", "pan 0.0"), "Pan the camera 90 degrees to the right, full shot",
         "full body, from side"),
    ]
    for js, prompt, want in cases:
        got = camera_tags_from_json(js, prompt)
        assert got == want, (js, got, want)


def test_camera_override():
    lib = Library(EXAMPLES)
    r = assemble(lib, "forest_walk", "full body, from above")
    pos = r["positive"]
    assert "cowboy shot" not in pos and "full body, from above" in pos, pos
    assert pos.index("from above") < pos.index("watercolor"), pos  # camera stays before style
    assert any("replaced by 3d camera" in n for n in r["notes"]), r["notes"]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "scenes"))
        with open(os.path.join(d, "scenes", "s.yaml"), "w") as fh:
            fh.write("children:\n  - {name: bg, kind: background, tags: forest}\n"
                     "  - {name: st, kind: style, tags: watercolor}\n")
        r = assemble(Library(d), "s", "upper body")
        assert r["positive"] == "forest, upper body, watercolor", r["positive"]
        r = assemble(Library(d), "s", None)
        assert r["positive"] == "forest, watercolor", r["positive"]


if __name__ == "__main__":
    for fn in (test_example_scene, test_errors, test_camera_tags, test_camera_override):
        fn()
        print(f"ok  {fn.__name__}")
