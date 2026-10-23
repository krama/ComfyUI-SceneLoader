"""Skeleton-only pose packs: import queues them, skeleton_to_figure makes blocks via a fake ComfyUI.

Run: python tests/test_skeleton_figure.py
"""
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import import_pose_inbox as ipi  # noqa: E402
import skeleton_to_figure as stf  # noqa: E402
from sceneloader.pose_camera import draw_openpose  # noqa: E402

POINTS = [(240, 100), (240, 160), (200, 160), (190, 240), (185, 310), (280, 160), (290, 240), (295, 310),
          (215, 330), (210, 450), (205, 560), (265, 330), (270, 450), (275, 560), (232, 92), (248, 92),
          (222, 98), (258, 98)]


def skeleton_png(path):
    Image.fromarray(draw_openpose(POINTS, 480, 640)).save(path)


def test_classify_and_import():
    with tempfile.TemporaryDirectory() as tmp:
        sk, line, photo = (os.path.join(tmp, n) for n in ("sk.png", "line.png", "photo.png"))
        skeleton_png(sk)
        a = np.zeros((640, 480, 3), np.uint8)
        a[100:500, 238:242] = 255
        Image.fromarray(a).save(line)
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 48, 3), np.uint8)).save(photo)
        assert ipi.classify(sk) == ("skeleton", None)
        assert ipi.classify(line)[0] == "skip" and ipi.classify(photo) == ("photo", None)

        inbox, lib = os.path.join(tmp, "inbox"), os.path.join(tmp, "lib")
        pack = os.path.join(inbox, "Skel Pack")
        os.makedirs(pack)
        for n in ("a", "b"):
            skeleton_png(os.path.join(pack, f"{n}.png"))
        open(os.path.join(pack, "b.png.txt"), "w").write("arms out")
        mixed = os.path.join(inbox, "Mixed")
        os.makedirs(mixed)
        skeleton_png(os.path.join(mixed, "openpose.png"))
        Image.open(photo).save(os.path.join(mixed, "ref.png"))
        assert ipi.main(["--inbox", inbox, "--library", lib]) == 0
        import yaml
        civ = os.path.join(lib, "blocks", "poses", "civitai")
        b = yaml.safe_load(open(os.path.join(civ, "skel_pack", "b.yaml")))
        assert b == {"name": "b", "kind": "pose", "tags": "arms out", "source": "civitai",
                     "skeleton": "b_skeleton.png"}, b
        assert os.path.isfile(os.path.join(civ, "skel_pack", "a_skeleton.png"))
        m = yaml.safe_load(open(os.path.join(civ, "mixed.yaml")))  # one folder: photo + skeleton
        assert m["image"] == "mixed.png" and m["skeleton"] == "mixed_skeleton.png", m
    print("ok  test_classify_and_import")


class FakeComfy(BaseHTTPRequestHandler):
    """Upload / prompt / history / view; the first prompt fails like NLF finding nobody."""
    prompts, uploads = [], []

    def log_message(self, *args):
        pass

    def _send(self, body, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/upload/image":
            self.uploads.append(data)
            self._send({"name": "x.png", "subfolder": stf.UPLOAD_SUBFOLDER, "type": "input"})
        else:
            self.prompts.append(json.loads(data)["prompt"])
            self._send({"prompt_id": str(len(self.prompts)), "number": 1, "node_errors": {}})

    def do_GET(self):
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[1]
            if pid == "1":
                entry = {"status": {"status_str": "error", "completed": False, "messages": [
                    ["execution_error", {"exception_message": "NLF detected no person in the image."}]]},
                    "outputs": {}}
            else:
                entry = {"status": {"status_str": "success", "completed": True, "messages": []},
                         "outputs": {"12": {"images": [{"filename": "f.png", "subfolder": "s", "type": "output"}]}}}
            self._send({pid: entry})
        elif self.path.startswith("/view?"):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (200, 100, 50)).save(buf, "PNG")
            self._send(buf.getvalue(), "image/png")


def test_pairing():
    k = ipi.pose_key
    assert k("o1") == k("r1") == ("", "1") and k("o1") != k("o11")
    assert k("set/lying,head_tilt,") == k("set/lying,head_tilt,--") == k("set/lying_head_tilt_openpose")
    assert k("x/pose_3_ref") == k("x/pose_3_openpose") != k("y/pose_3_ref")
    photos = [("r1", "P1"), ("r2", "P2"), ("solo/shot", "PS"), ("many/a", "A"),
                                                ("many/b", "B")]
    skels = [("o1", "S1"), ("o3", "S3"), ("solo/keypoints_rendered", "SS"), ("many/c", "C")]
    got = ipi.pair(photos, skels)
    assert got == [("many/a", "A", None), ("many/b", "B", None), ("many/c", None, "C"), ("o3", None, "S3"),
                   ("r1", "P1", "S1"), ("r2", "P2", None), ("solo/shot", "PS", "SS")], got
    # Role folders: poses/10 (skeleton) + previews/preview_10 (photo) are one pose.
    assert k("pack/poses/10") == k("pack/previews/preview_10") == ("pack", "10"), k("pack/previews/preview_10")
    print("ok  test_pairing")


def test_role_folders_and_exclude():
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        inbox, lib = os.path.join(tmp, "inbox"), os.path.join(tmp, "lib")
        root = os.path.join(inbox, "Thousands", "ControlnetPoses")
        photo = Image.fromarray(np.random.default_rng(2).integers(0, 255, (64, 48, 3), np.uint8))
        for sub in ("female/poses", "female/previews", "site"):
            os.makedirs(os.path.join(root, sub))
        for i in (1, 2):
            skeleton_png(os.path.join(root, "female", "poses", f"{i}.png"))
            photo.save(os.path.join(root, "female", "previews", f"preview_{i}.png"))
        photo.save(os.path.join(root, "female", "previews", "preview_cover.png"))  # no skeleton: pack preview
        photo.save(os.path.join(root, "site", "dog.png"))
        assert ipi.main(["--inbox", inbox, "--library", lib, "--exclude", "site/*"]) == 0
        d = os.path.join(lib, "blocks", "poses", "civitai", "thousands")
        blocks = sorted(f for f in os.listdir(d) if f.endswith(".yaml"))
        assert blocks == ["female_previews_preview_1.yaml", "female_previews_preview_2.yaml"], blocks
        b = yaml.safe_load(open(os.path.join(d, blocks[0])))
        assert b["image"].endswith(".png") and b["skeleton"].endswith("_skeleton.png"), b
    print("ok  test_role_folders_and_exclude")


def test_skeleton_to_figure():
    srv = HTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as lib:
            import yaml
            d = os.path.join(lib, "blocks", "poses", "civitai", "pk")
            os.makedirs(d)
            skeleton_png(os.path.join(d, "a_skeleton.png"))
            for n in ("a", "z"):
                yaml.safe_dump({"name": n, "kind": "pose", "tags": "arms out", "skeleton": "a_skeleton.png"},
                               open(os.path.join(d, f"{n}.yaml"), "w"), sort_keys=False)
            yaml.safe_dump({"name": "p", "kind": "pose", "image": "p.png", "skeleton": "a_skeleton.png"},
                           open(os.path.join(d, "p.yaml"), "w"))
            pending = stf.find_pending(lib)
            assert [r for _, r, _ in pending] == ["poses/civitai/pk/a", "poses/civitai/pk/z"], pending
            assert [r for _, r, _ in stf.find_pending(lib, ["*/z"])] == ["poses/civitai/pk/z"]
            rc = stf.main(["--library", lib, "--smplx-dir", "M", "--nlf-dir", "N", "--seed", "7", "--only", "*/a",
                           "--server", f"http://127.0.0.1:{srv.server_port}"])
            assert rc == 0
            data = yaml.safe_load(open(os.path.join(d, "a.yaml")))
            assert data == {"name": "a", "kind": "pose", "tags": "arms out", "skeleton": "a_skeleton.png",
                            "image": "a.png", "figure_seed": 8}, data
            assert os.path.isfile(os.path.join(d, "a.png")) and "image" not in yaml.safe_load(open(os.path.join(d, "z.yaml")))
            p = FakeComfy.prompts
            assert len(p) == 2 and p[0]["10"]["inputs"]["seed"] == 7 and p[1]["10"]["inputs"]["seed"] == 8
            assert p[0]["5"]["inputs"]["image"] == f"{stf.UPLOAD_SUBFOLDER}/x.png"
            assert (p[0]["8"]["inputs"]["width"], p[0]["8"]["inputs"]["height"]) == (896, 1152)  # 480x640 -> ~1 MP
            assert p[0]["15"]["inputs"]["image"] == ["11", 0] and p[0]["13"]["inputs"]["model_path"] == "M"
            assert b'name="image"; filename="a.png"' in FakeComfy.uploads[0]
            assert stf.main(["--library", lib, "--smplx-dir", "M", "--nlf-dir", "N", "--only", "nothing"]) == 0
    finally:
        srv.shutdown()
    print("ok  test_skeleton_to_figure")


if __name__ == "__main__":
    test_classify_and_import()
    test_pairing()
    test_role_folders_and_exclude()
    test_skeleton_to_figure()
