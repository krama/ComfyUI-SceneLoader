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
    print("ok  render_pose: fit, sides, back view, tags, framing, shift, zoom")


if __name__ == "__main__":
    test_merge_camera_tags()
    print("ok  test_merge_camera_tags")
    models = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMPLX_MODELS", "")
    if not models:
        print("skip render_pose: pass the models dir or set SMPLX_MODELS")
        sys.exit(0)
    check(models)
