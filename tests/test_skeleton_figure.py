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


def test_classify_and_queue():
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
        civ = os.path.join(lib, "blocks", "poses", "civitai")
        assert sorted(os.listdir(os.path.join(civ, "skel_pack", "_pending"))) == ["a.png", "b.png", "b.txt"]
        assert not [f for f in os.listdir(os.path.join(civ, "skel_pack")) if f.endswith(".yaml")]
        assert os.path.isfile(os.path.join(civ, "mixed.yaml")), os.listdir(civ)  # photo wins
        assert not os.path.exists(os.path.join(civ, "_pending"))
    print("ok  test_classify_and_queue")


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


def test_skeleton_to_figure():
    srv = HTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as lib:
            pend = os.path.join(lib, "blocks", "poses", "civitai", "pk", "_pending")
            os.makedirs(pend)
            skeleton_png(os.path.join(pend, "a.png"))
            open(os.path.join(pend, "a.txt"), "w").write("arms out")
            rc = stf.main(["--library", lib, "--smplx-dir", "M", "--nlf-dir", "N", "--seed", "7",
                           "--server", f"http://127.0.0.1:{srv.server_port}"])
            assert rc == 0
            d = os.path.dirname(pend)
            assert sorted(os.listdir(d)) == ["_pending", "a.png", "a.yaml", "a_skeleton.png", "a_skeleton.txt"]
            assert os.listdir(pend) == []
            import yaml
            data = yaml.safe_load(open(os.path.join(d, "a.yaml")))
            assert data == {"name": "a", "kind": "pose", "tags": "arms out", "image": "a.png",
                            "skeleton": "a_skeleton.png", "source": "civitai", "figure_seed": 8}, data
            p = FakeComfy.prompts
            assert len(p) == 2 and p[0]["10"]["inputs"]["seed"] == 7 and p[1]["10"]["inputs"]["seed"] == 8
            assert p[0]["5"]["inputs"]["image"] == f"{stf.UPLOAD_SUBFOLDER}/x.png"
            assert (p[0]["8"]["inputs"]["width"], p[0]["8"]["inputs"]["height"]) == (896, 1152)  # 480x640 -> ~1 MP
            assert p[0]["15"]["inputs"]["image"] == ["11", 0] and p[0]["13"]["inputs"]["model_path"] == "M"
            assert b'name="image"; filename="a.png"' in FakeComfy.uploads[0]
            # Nothing queued: clean exit.
            assert stf.main(["--library", lib, "--smplx-dir", "M", "--nlf-dir", "N"]) == 0
    finally:
        srv.shutdown()
    print("ok  test_skeleton_to_figure")


if __name__ == "__main__":
    test_classify_and_queue()
    test_skeleton_to_figure()
