"""Tests for sceneloader.composer. Run: python tests/test_composer.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader.assembler import Library, SceneError, assemble, to_lora_stack  # noqa: E402
from sceneloader.composer import (block_loras, camera_input_tags, compose, dedupe_tags, load_preset, save_preset,  # noqa: E402
                                  slot_options)

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


def test_block_lora_weights():
    lib = Library(EXAMPLES)
    found = {x["file"]: (x["weight"], x["type"]) for x in block_loras(lib, "characters/example_girl")}
    assert found == {"character/example_girl.safetensors": (0.9, "character"),
                     "slider/example_height_slider.safetensors": (1.0, "slider")}, found
    weights = {"slider/example_height_slider.safetensors": 0.3, "character/example_girl.safetensors": 0.5}
    r = compose(lib, "/nonexistent", "illustrious",
                [{"slot": "character", "use": "characters/example_girl", "weights": weights}])
    w = {x["file"]: x["weight"] for x in r["loras"]}
    assert w["slider/example_height_slider.safetensors"] == 0.3, w  # nested block LoRA overridden
    assert w["character/example_girl.safetensors"] == 0.9, w  # identity weight stays fixed
    assert any("identity LoRA" in n for n in r["notes"]), r["notes"]


def test_camera_input():
    lib = Library(EXAMPLES)
    with tempfile.TemporaryDirectory() as tmp:
        make_loras(tmp, {"camera/fisheye.safetensors": {"weight": {"min": 0.7}, "trigger_sets": ["fisheye"]}})
        rows = [{"slot": "quality", "use": "quality/default"}, {"slot": "background", "use": "backgrounds/forest_path"},
                {"slot": "camera", "lora": "camera/fisheye.safetensors"}, {"slot": "camera", "use": "cameras/cowboy_shot"}]
        # Not connected: camera slot rows are used (tags + LoRA).
        r = compose(lib, tmp, "illustrious", rows, None)
        assert "fisheye" in r["positive"] and "cowboy shot" in r["positive"], r["positive"]
        assert [x["file"] for x in r["loras"]] == ["camera/fisheye.safetensors"], r["loras"]
        # Connected: the input replaces the whole camera slot, in the camera prompt position.
        r = compose(lib, tmp, "illustrious", rows, "from above, full body")
        pos = r["positive"]
        assert "fisheye" not in pos and "cowboy shot" not in pos and r["loras"] == [], r
        assert pos.index("forest") < pos.index("from above"), pos
        assert sum("replaced by camera input" in n for n in r["notes"]) == 2, r["notes"]
        # Connected without any camera row: tags are still added.
        assert "from above" in compose(lib, tmp, "illustrious", rows[:2], "from above")["positive"]
    # Input value: text as-is, camera_json -> booru tags for illustrious, error for anima, empty -> None.
    cam = json.dumps({"camera": {"shot_type": "full_shot", "tilt_deg": "tilt down 30.0", "pan_deg": "pan 0"}})
    assert camera_input_tags("anima", "  a low angle shot ") == "a low angle shot"
    assert "full body" in camera_input_tags("illustrious", cam) and "from above" in camera_input_tags("illustrious", cam)
    assert camera_input_tags("illustrious", "  ") is None and camera_input_tags("illustrious", None) is None
    try:
        camera_input_tags("anima", cam)
    except SceneError as e:
        assert "prompt" in str(e), e
    else:
        raise AssertionError("anima must refuse camera_json")


def test_scene_preset_roundtrip():
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        lib_dir = os.path.join(tmp, "lib")
        shutil.copytree(EXAMPLES, lib_dir)
        loras = make_loras(os.path.join(tmp, "loras"), {"style/example_watercolor.safetensors": None})
        lib = Library(lib_dir)
        # A scene file loaded into the composer assembles exactly like Scene Loader.
        p = load_preset(lib, "forest_walk")
        assert p["warnings"] == [], p["warnings"]
        a, b = assemble(lib, "forest_walk"), compose(lib, loras, "illustrious", p["rows"])
        for k in ("positive", "negative"):
            assert a[k] == b[k], (k, a[k], b[k])
        for k in ("loras", "loras_detailer"):
            assert to_lora_stack(a[k], "illustrious") == to_lora_stack(b[k], "illustrious"), k
        # Save as a composer preset, load it back, refuse silent overwrite, reject bad names.
        rows = p["rows"] + [{"slot": "concept", "tags": "rain"}]
        save_preset(lib, "mine/walk", "illustrious", rows)
        back = load_preset(lib, "mine/walk")
        assert back["rows"] == rows and back["base"] == "illustrious", back
        assert "mine/walk" in lib.scenes() and lib.scene_bases()["mine/walk"] == "illustrious"
        for bad, exc in ((("mine/walk", False), FileExistsError), (("../x", True), SceneError)):
            try:
                save_preset(lib, bad[0], "illustrious", rows, overwrite=bad[1])
            except exc:
                pass
            else:
                raise AssertionError(f"expected {exc.__name__} for {bad}")
        save_preset(lib, "mine/walk", "illustrious", rows[:1], overwrite=True)
        assert load_preset(lib, "mine/walk")["rows"] == rows[:1]
        try:
            assemble(lib, "mine/walk")
        except SceneError as e:
            assert "Scene Composer" in str(e)
        else:
            raise AssertionError("Scene Loader must refuse composer presets")


def test_detailer_row():
    lib = Library(EXAMPLES)
    rows = [{"slot": "character", "use": "characters/example_girl"},
            {"slot": "detailer", "drop": ["slider"], "override": [{"type": "character", "weight": 1.0}],
             "add": [{"file": ""}, {"file": "detail/x.safetensors"}]}]
    r = compose(lib, "/nonexistent", "illustrious", rows)
    det = {x["file"]: x["weight"] for x in r["loras_detailer"]}
    assert det == {"character/example_girl.safetensors": 1.0, "detail/x.safetensors": 1.0}, det
    assert any("without a file" in n for n in r["notes"]), r["notes"]


def test_dedupe_tags():
    text, removed = dedupe_tags(
        "1girl, Smile, long_hair, (smile:1.2), long hair, (low quality, worst quality:1.4), worst quality, "
        "blue eyes, (blue eyes:1.1), <lora:x:0.5>, sketch, sketch, (low quality, worst quality:1.4)")
    assert text == "1girl, long_hair, (smile:1.2), (low quality, worst quality:1.4), (blue eyes:1.1), " \
                   "<lora:x:0.5>, sketch", text
    assert removed == ["Smile", "long hair", "worst quality", "blue eyes", "sketch",
                       "(low quality, worst quality:1.4)"], removed
    assert dedupe_tags("a, b, c") == ("a, b, c", [])
    assert dedupe_tags("") == ("", [])


def test_pose_pack_and_pose_image():
    import shutil
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import import_pose_pack as ipp
    with tempfile.TemporaryDirectory() as tmp:
        pack, lib_dir = os.path.join(tmp, "pack"), os.path.join(tmp, "lib")
        shutil.copytree(EXAMPLES, lib_dir)
        os.makedirs(os.path.join(pack, "POSES"))
        with open(os.path.join(pack, "POSES.yaml"), "w") as fh:
            fh.write("  POSES:\n    bent over:  bent over,\n    indian: sitting, indian style,\n    no-pic: x\n")
        for name in ("bent over", "indian"):
            open(os.path.join(pack, "POSES", name + ".jpeg"), "wb").write(b"img")
        assert ipp.main(["--pack", pack, "--library", lib_dir]) == 0
        lib = Library(lib_dir)
        opts = slot_options(lib, "/nonexistent", "illustrious")
        assert opts["pose"]["blocks"] == ["poses/bent_over", "poses/indian", "poses/no_pic"], opts["pose"]
        rows = [{"slot": "pose", "use": "poses/no_pic"}, {"slot": "pose", "use": "poses/indian"}]
        r = compose(lib, "/nonexistent", "illustrious", rows)
        assert "sitting, indian style" in r["positive"], r["positive"]
        assert r["pose_image"].endswith(os.path.join("blocks", "poses", "indian.jpeg")), r["pose_image"]
        assert compose(lib, "/nonexistent", "illustrious", rows[:1])["pose_image"] is None
        # A block pointing outside the library is ignored.
        with open(os.path.join(lib_dir, "blocks", "poses", "evil.yaml"), "w") as fh:
            fh.write("name: evil\nkind: pose\ntags: x\nimage: ../../../pack/POSES/indian.jpeg\n")
        r = compose(lib, "/nonexistent", "illustrious", [{"slot": "pose", "use": "poses/evil"}])
        assert r["pose_image"] is None and any("outside the library" in n for n in r["notes"]), r["notes"]


def test_pose_inbox_import():
    import shutil
    import zipfile
    import numpy as np
    from PIL import Image
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import import_pose_inbox as ipi
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        inbox, lib_dir = os.path.join(tmp, "inbox"), os.path.join(tmp, "lib")
        shutil.copytree(EXAMPLES, lib_dir)
        os.makedirs(inbox)
        photo = Image.fromarray(rng.integers(0, 255, (64, 48, 3), np.uint8))
        skeleton = np.zeros((64, 48, 3), np.uint8)
        skeleton[30:34, 5:40] = [255, 0, 0]
        depth = np.repeat(rng.integers(0, 255, (64, 48, 1), np.uint8), 3, 2)
        pack = os.path.join(tmp, "pk")
        os.makedirs(pack)
        photo.save(os.path.join(pack, "render.png"))
        Image.fromarray(skeleton).save(os.path.join(pack, "openpose.png"))
        Image.fromarray(depth).save(os.path.join(pack, "depth.png"))
        with zipfile.ZipFile(os.path.join(inbox, "Dancer Pose.zip"), "w") as z:
            for f in os.listdir(pack):
                z.write(os.path.join(pack, f), f"pack/{f}")
        # Multi-pose pack: wrapper folder, same file names in subfolders, a preview grid, a skeleton.
        multi = os.path.join(inbox, "Multi Pack", "Multi Pack v2")
        for sub in ("pose01", "pose02"):
            os.makedirs(os.path.join(multi, sub))
            photo.save(os.path.join(multi, sub, "reference.png"))
        Image.fromarray(skeleton).save(os.path.join(multi, "pose01", "openpose.png"))
        open(os.path.join(multi, "pose02", "reference.png.txt"), "w").write("arms up")
        photo.save(os.path.join(multi, "Preview_Grid.jpg"))
        photo.save(os.path.join(inbox, "kneel.jpg"))
        open(os.path.join(inbox, "kneel.txt"), "w").write("kneeling,\nfrom side")
        assert ipi.main(["--inbox", inbox, "--library", lib_dir]) == 0
        lib = Library(lib_dir)
        blocks = slot_options(lib, "/nonexistent", "illustrious")["pose"]["blocks"]
        assert "poses/civitai/dancer_pose" in blocks and "poses/civitai/kneel" in blocks, blocks
        assert "poses/civitai/multi_pack/pose01_reference" in blocks, blocks
        assert "poses/civitai/multi_pack/pose02_reference" in blocks, blocks
        assert not [b for b in blocks if "openpose" in b or "depth" in b or "preview" in b], blocks
        r = compose(lib, "/nonexistent", "illustrious", [{"slot": "pose", "use": "poses/civitai/multi_pack/pose02_reference"}])
        assert "arms up" in r["positive"] and r["pose_image"].endswith("pose02_reference.png"), r
        r = compose(lib, "/nonexistent", "illustrious", [{"slot": "pose", "use": "poses/civitai/kneel"}])
        assert "kneeling, from side" in r["positive"] and r["pose_image"].endswith("kneel.jpg"), r
        left = sorted(os.listdir(inbox))
        assert left == ["_imported"], left  # sources and tag sidecars moved
        # A second drop with the same name is kept apart, not overwritten.
        photo.save(os.path.join(inbox, "kneel.jpg"))
        assert ipi.main(["--inbox", inbox, "--library", lib_dir]) == 0
        assert sorted(os.listdir(os.path.join(inbox, "_imported"))) == ["Dancer Pose.zip", "Multi Pack", "kneel.2.jpg", "kneel.jpg",
                                                                              "kneel.txt"]


if __name__ == "__main__":
    for fn in (test_compose_order_weights_triggers, test_not_chosen_and_missing, test_slot_options,
               test_block_lora_weights, test_camera_input, test_scene_preset_roundtrip, test_detailer_row, test_dedupe_tags,
               test_pose_pack_and_pose_image, test_pose_inbox_import):
        fn()
        print(f"ok  {fn.__name__}")
