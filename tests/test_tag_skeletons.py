"""Skeleton geometry tags of tools/tag_skeletons.py. Run: python tests/test_tag_skeletons.py"""
import os
import sys
import tempfile

import numpy as np
import yaml
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tag_skeletons as ts  # noqa: E402
from sceneloader.pose_camera import draw_openpose  # noqa: E402

# Standing, facing the viewer (COCO-18; subject's right on the viewer's left), 512x768.
STAND = [(256, 100), (256, 160), (206, 165), (196, 250), (190, 330), (306, 165), (316, 250), (322, 330),
         (226, 360), (224, 500), (222, 640), (286, 360), (288, 500), (290, 640),
         (246, 90), (266, 90), (236, 95), (276, 95)]


# Copy of STAND with some points moved (dict index -> point or None).
def pose(**moves):
    p = [np.array(x, float) for x in STAND]
    for k, v in moves.items():
        p[int(k[1:])] = None if v is None else np.array(v, float)
    return p


# Tags read back from a drawn skeleton.
def tags(points, w=512, h=768):
    return ts.tags_from(ts.keypoints(ts.limb_ends(draw_openpose(points, w, h))))


def test_tags():
    assert tags(pose()) == ["standing", "full body"], tags(pose())
    sit = pose(j9=(224, 380), j12=(288, 380), j10=(222, 520), j13=(290, 520))
    assert tags(sit) == ["sitting", "full body"], tags(sit)
    kneel = pose(j9=(224, 520), j12=(288, 520), j10=(222, 530), j13=(290, 530))
    assert tags(kneel) == ["kneeling", "full body"], tags(kneel)
    rot = [None if q is None else np.array([768 - q[1], q[0]]) for q in pose()]  # turned 90 degrees
    t = tags(rot, 768, 512)
    assert t[0] == "lying", t
    up = pose(j3=(200, 120), j4=(200, 40), j6=(312, 120), j7=(312, 40))
    assert tags(up) == ["standing", "full body", "arms up"], tags(up)
    back = pose(j0=None, j14=None, j15=None)
    assert "from behind" in tags(back), tags(back)
    side = pose(j2=(250, 165), j5=(262, 165))
    assert "from side" in tags(side), tags(side)
    upper = pose(j8=None, j9=None, j10=None, j11=None, j12=None, j13=None)
    assert tags(upper) == ["upper body"], tags(upper)
    akimbo = pose(j3=(160, 260), j6=(352, 260), j4=(220, 350), j7=(292, 350))
    assert "hands on hips" in tags(akimbo), tags(akimbo)
    spread = pose(j10=(120, 640), j13=(392, 640), j9=(180, 500), j12=(332, 500))
    assert "spread legs" in tags(spread), tags(spread)


def test_write():
    with tempfile.TemporaryDirectory() as lib:
        d = os.path.join(lib, "blocks", "poses", "pack")
        os.makedirs(d)
        Image.fromarray(draw_openpose(pose(), 512, 768)).save(os.path.join(d, "s.png"))
        blocks = {"a": {"name": "a", "kind": "pose", "tags": "", "skeleton": "s.png"},
                  "b": {"name": "b", "kind": "pose", "tags": "", "image": "x.png", "skeleton": "s.png"},
                  "c": {"name": "c", "kind": "pose", "tags": "my tags", "skeleton": "s.png"}}
        for n, data in blocks.items():
            with open(os.path.join(d, n + ".yaml"), "w") as fh:
                yaml.safe_dump(data, fh)
        assert ts.main(["--library", lib]) == 0
        read = {n: yaml.safe_load(open(os.path.join(d, n + ".yaml"))) for n in blocks}
        assert read["a"]["tags"] == "standing, full body" and read["a"]["tags_source"] == "skeleton", read["a"]
        assert read["b"]["tags"] == "" and read["c"]["tags"] == "my tags", read
        assert ts.find_blocks(lib, [], False) == [] and len(ts.find_blocks(lib, [], True)) == 1


if __name__ == "__main__":
    for fn in (test_tags, test_write):
        fn()
        print(f"ok  {fn.__name__}")
