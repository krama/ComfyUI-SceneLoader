"""Tests for sceneloader.composer. Run: python tests/test_composer.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader.assembler import Library, SceneError  # noqa: E402
from sceneloader.composer import compose, slot_options  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


# Create loras/illustrious/<rel> files with optional sidecars in a temp dir.
def make_loras(tmp, files):
    for rel, meta in files.items():
        path = os.path.join(tmp, "illustrious", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()
        if meta is not None:
            with open(os.path.splitext(path)[0] + ".sceneloader.json", "w") as fh:
                json.dump(meta, fh)
    return tmp


def test_compose_order_weights_triggers():
    with tempfile.TemporaryDirectory() as tmp:
        make_loras(tmp, {
            "style/ink.safetensors": {"weight": {"min": 0.6, "max": 0.9}, "trigger_sets": ["inkstyle", "ink alt"],
                                      "extra_positive": "monochrome", "extra_negative": "color"},
            "clothing/top/coat.safetensors": {"weight": {"min": 0.8, "max": 0.8}, "trigger_sets": ["red coat"]},
            "detail/sharp.safetensors": {"weight": {"min": 0.4, "max": 0.4}, "trigger_sets": ["sharpword"]},
            "pose/sit.safetensors": None,
        })
        lib = Library(EXAMPLES)
        rows = [
            {"slot": "style", "lora": "style/ink.safetensors", "trigger": 1, "extras": True},
            {"slot": "detail", "lora": "detail/sharp.safetensors"},
            {"slot": "pose", "lora": "pose/sit.safetensors"},
            {"slot": "clothing", "lora": "clothing/top/coat.safetensors", "weight": 0.5},
            {"slot": "character", "use": "characters/example_girl"},
            {"slot": "quality", "use": "quality/default"},
            {"slot": "concept", "none": True},
            {"slot": "detailer", "drop": ["style"]},
        ]
        r = compose(lib, tmp, "illustrious", rows)
        pos = r["positive"]
        order = ["masterpiece", "1girl", "red coat", "ink alt", "monochrome"]
        idx = [pos.index(t) for t in order]
        assert idx == sorted(idx), pos
        assert "inkstyle" not in pos and "sharpword" not in pos, pos  # chosen set only; detail is stack-only
        assert "color" in r["negative"], r["negative"]
        w = {x["file"]: x["weight"] for x in r["loras"]}
        assert w["style/ink.safetensors"] == 0.6  # lower bound of recommended range
        assert w["clothing/top/coat.safetensors"] == 0.5  # explicit row weight wins
        assert w["pose/sit.safetensors"] == 1.0  # no sidecar
        assert r["loras"][-1]["file"] == "style/ink.safetensors"  # style last
        assert all(x["type"] != "style" for x in r["loras_detailer"])


def test_not_chosen_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        lib = Library(EXAMPLES)
        for rows, msg in (([{"slot": "style"}], "not chosen"),
                          ([{"slot": "style", "lora": "style/nope.safetensors"}], "not found")):
            try:
                compose(lib, tmp, "illustrious", rows)
            except SceneError as e:
                assert msg in str(e), e
            else:
                raise AssertionError(f"expected SceneError for {rows}")


def test_slot_options():
    with tempfile.TemporaryDirectory() as tmp:
        make_loras(tmp, {"style/a.safetensors": None, "misc folder/b.safetensors": None, "root.safetensors": None})
        opts = slot_options(Library(EXAMPLES), tmp, "illustrious")
        assert list(opts)[:2] == ["quality", "character"], list(opts)
        assert opts["style"]["loras"] == ["style/a.safetensors"]
        assert opts["misc folder"]["loras"] == ["misc folder/b.safetensors"]
        assert opts["other"]["loras"] == ["root.safetensors"]
        assert "characters/example_girl" in opts["character"]["blocks"], opts["character"]


if __name__ == "__main__":
    for fn in (test_compose_order_weights_triggers, test_not_chosen_and_missing, test_slot_options):
        fn()
        print(f"ok  {fn.__name__}")
