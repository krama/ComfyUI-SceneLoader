"""Check the virtual pose camera. Needs smplx + SMPL-X model files.

Run: python tests/test_pose_camera.py <models_dir>   (or set SMPLX_MODELS)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader import pose_camera as pc  # noqa: E402
from sceneloader import smplx_rig  # noqa: E402

T_POSE = {"body_pose": np.zeros(63)}
A_POSE = {"body_pose": np.zeros(63)}  # arms down, so framings can crop the legs
A_POSE["body_pose"][45 + 2], A_POSE["body_pose"][48 + 2] = -1.2, 1.2  # shoulders about Z


def test_shot_tags():
    m = lambda *idx: np.isin(np.arange(18), idx)  # noqa: E731
    head, hips, knees, feet = (0, 14, 15), (8, 11), (9, 12), (10, 13)
    assert pc.shot_tags(m(*head, 1, *hips, *knees, *feet)) == ["full body"]
    assert pc.shot_tags(m(*head, 1, *hips, *knees)) == ["cowboy shot", "feet out of frame"]
    assert pc.shot_tags(m(*head, 1, *hips)) == ["upper body"]
    assert pc.shot_tags(m(*head, 1)) == ["portrait"]
    assert pc.shot_tags(m(*hips, *knees, *feet)) == ["lower body", "head out of frame"]


def test_merge_camera_tags():
    assert pc.merge_camera_tags("1girl, (upper body:1.2), smile, From Above", ["full body"]) == \
        "1girl, smile, full body"
    assert pc.merge_camera_tags("", ["portrait", "from side"]) == "portrait, from side"


def check(models_dir):
    model = smplx_rig.load_model(models_dir, "neutral", 10)
    w, h = 1536, 640
    img, tags, pts = pc.render_pose(model, T_POSE, w, h)
    assert img.shape == (h, w, 3) and img.any() and tags == ["full body"], tags
    assert all(p is not None for p in pts), pts
    px = np.array(pts)
    assert pts[2][0] < pts[5][0], "facing the camera: person's right shoulder on the image left"
    span = px.max(0) - px.min(0)
    assert span[1] < 0.9 * h and span[1] > 0.6 * h, span  # body height fills the 640 px frame
    assert abs((px[:, 0].min() + px[:, 0].max()) / 2 - w / 2) < 0.1 * w  # centered

    _, tags, pts = pc.render_pose(model, T_POSE, w, h, yaw=180)
    assert "from behind" in tags and pts[0] is None and pts[14] is None and pts[15] is None, (tags, pts[:1])
    assert pts[2][0] > pts[5][0], "from behind: right shoulder on the image right"
    assert "from side" in pc.render_pose(model, T_POSE, w, h, yaw=90)[1]
    assert "from above" in pc.render_pose(model, T_POSE, w, h, elevation=45)[1]
    assert "from below" in pc.render_pose(model, T_POSE, w, h, elevation=-30)[1]

    _, tags, pts = pc.render_pose(model, A_POSE, 832, 1216, framing="portrait")
    assert tags == ["portrait"] and pts[0] is not None and pts[9] is None and pts[10] is None, pts
    _, _, pts = pc.render_pose(model, A_POSE, 832, 1216, framing="upper body")
    assert pts[1] is not None and pts[10] is None and pts[13] is None, pts
    # Shift moves the subject; zoom out shrinks it.
    a = np.array([p for p in pc.render_pose(model, T_POSE, w, h)[2]])
    b = np.array([p for p in pc.render_pose(model, T_POSE, w, h, shift_x=0.2)[2]])
    assert np.allclose(b[:, 0] - a[:, 0], 0.2 * w, atol=1)
    c = np.array([p for p in pc.render_pose(model, T_POSE, w, h, zoom=0.5)[2]])
    assert np.ptp(c[:, 1]) < 0.6 * np.ptp(a[:, 1])
    # Tags follow what is in frame, not the framing setting: zoomed in past the knees.
    _, tags, pts = pc.render_pose(model, A_POSE, w, h, zoom=1.58, shift_y=-0.29)
    assert pts[10] is None and pts[13] is None and "full body" not in tags, (tags, pts)
    _, tags, pts = pc.render_pose(model, A_POSE, w, h, zoom=1.3, shift_y=-0.15)
    assert pts[9] and pts[10] is None and tags[:2] == ["cowboy shot", "feet out of frame"], (tags, pts)
    print("ok  render_pose: fit, sides, back view, tags, framing, shift, zoom")
    # Depth map from the same camera: every visible joint lands on the body, background stays black.
    for kw in (dict(), dict(yaw=120, elevation=35, fov=70), dict(zoom=1.6, shift_y=-0.3)):
        cam = {}
        pts = pc.render_pose(model, A_POSE, w, h, cam=cam, **kw)[2]
        d = pc.depth_map(model, A_POSE, cam, w, h)
        assert d.shape == (h, w, 3) and d[0, 0].max() == 0 and d.max() == 255, kw
        on = [d[int(p[1]), int(p[0]), 0] > 0 for p in pts if p is not None]
        assert on and all(on), (kw, on)
    print("ok  depth_map: silhouette under the OpenPose joints")
    # Overscan: a waist-up shot gets a taller canvas holding the legs; the frame is a crop of it.
    kw = dict(zoom=1.6, shift_y=-0.3)
    r = pc.shot(model, A_POSE, w, h, body=pc.pose_body(model, A_POSE), overscan=1.3, **kw)
    c = r["canvas"]
    assert c["height"] > h * 1.3 and c["width"] % 64 == 0 and c["height"] % 64 == 0, c
    assert c["width"] * c["height"] <= 1.3 * w * h * 1.1, c
    x, y, cw, ch = c["crop"]
    assert y == 0 and abs(cw / ch - w / h) < 0.02 and x + cw <= c["width"] and ch < c["height"], c
    assert r["points"][10] is None and r["tags"][0] == "full body", r["tags"]  # frame: no feet; canvas: all
    import cv2
    crop = cv2.resize(r["map"][y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    for p in r["points"]:  # frame joints sit on the cropped canvas map
        if p is not None:
            yy, xx = int(p[1]), int(p[0])
            assert crop[max(0, yy - 6):yy + 7, max(0, xx - 6):xx + 7].any(), p
    assert r["map"][ch:].any(), "legs drawn below the frame"
    d = pc.depth_map(model, A_POSE, r["cam"], w, h, canvas=c)
    assert d.shape == (c["height"], c["width"], 3) and d[ch:].any()
    full = pc.shot(model, A_POSE, w, h, overscan=1.3)  # body fits: canvas is the frame
    assert full["canvas"]["crop"] == (0, 0, w, h) and full["map"].shape == (h, w, 3)
    print("ok  overscan: canvas holds the body, frame crop matches")
    check_js(model)


SHOTS = [dict(), dict(yaw=90), dict(yaw=180), dict(yaw=-35, elevation=45, fov=70),
         dict(framing="portrait", zoom=1.5, shift_x=0.1), dict(framing="cowboy shot", elevation=-25, shift_y=-0.2),
         dict(framing="upper body", yaw=150)]


# The browser view (web/pose_camera_math.js) must place the joints where render_pose does.
def check_js(model):
    import json
    import shutil
    import subprocess
    import tempfile
    node = shutil.which("node")
    if not node:
        print("skip pose_camera_math.js: node not found")
        return
    w, h = 1536, 640
    body = pc.pose_body(model, A_POSE)
    cams = [dict({"framing": "full body", "yaw": 0.0, "elevation": 0.0, "fov": 40.0, "zoom": 1.0,
                  "shift_x": 0.0, "shift_y": 0.0}, **s) for s in SHOTS]
    js = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "pose_camera_math.js")
    runner = ("import { shoot, unpack } from " + json.dumps("file://" + os.path.abspath(js)) + ";\n"
              "let s = ''; process.stdin.on('data', (d) => (s += d)); process.stdin.on('end', () => {\n"
              "  const { body, cams } = JSON.parse(s); const b = unpack(body);\n"
              "  console.log(JSON.stringify(cams.map((c) => shoot(b, c).points)));\n});\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.mjs")
        open(path, "w").write(runner)
        out = subprocess.run([node, path], input=json.dumps({"body": pc.view_payload(body, w, h), "cams": cams}),
                             capture_output=True, text=True, check=True).stdout
    for cam, js_pts in zip(cams, json.loads(out)):
        py_pts = pc.render_pose(model, A_POSE, w, h, body=body, **cam)[2]
        for i, (a, b) in enumerate(zip(py_pts, js_pts)):
            assert (a is None) == (b is None), (cam, i, a, b)
            if a is not None:
                assert abs(a[0] - b[0]) < 0.5 and abs(a[1] - b[1]) < 0.5, (cam, i, a, b)
    print(f"ok  pose_camera_math.js matches render_pose on {len(cams)} shots")


if __name__ == "__main__":
    test_shot_tags()
    test_merge_camera_tags()
    print("ok  test_shot_tags, test_merge_camera_tags")
    models = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMPLX_MODELS", "")
    if not models:
        print("skip render_pose: pass the models dir or set SMPLX_MODELS")
        sys.exit(0)
    check(models)
