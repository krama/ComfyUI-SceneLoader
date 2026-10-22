"""Check SMPL-X -> OpenPose 3D keypoints. Needs smplx + SMPL-X model files.

Run: python tests/test_smplx_openpose.py <models_dir>   (or set SMPLX_MODELS)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader import smplx_openpose as so  # noqa: E402
from sceneloader import smplx_rig  # noqa: E402


def check(models_dir):
    model = smplx_rig.load_model(models_dir, "neutral", 10)
    w, h = 768, 1344
    front = so.smplx_to_openpose(model, {"body_pose": np.zeros(63)}, w, h)[0]
    kp = np.array(front["people"][0]["pose_keypoints_2d"]).reshape(18, 3)
    px = kp[:, :2] * [w, h]
    assert (kp[:, 2] == 1).all() and (px >= 0).all() and (px[:, 0] <= w).all() and (px[:, 1] <= h).all(), px
    # Facing the camera: the person's right side is on the image left (OpenPose/DWPose convention).
    assert px[2, 0] < px[5, 0] and px[8, 0] < px[11, 0], px[[2, 5, 8, 11]]
    assert px[0, 1] < px[1, 1] < px[8, 1] < px[9, 1] < px[10, 1], px[:, 1]  # nose above neck above hip...
    pts = front["_3d_pose_data"]["points"]
    rel = np.array([[p["x"], p["y"], p["z"]] for p in pts])
    assert np.allclose(rel[:, 0], px[:, 0] - w / 2) and np.allclose(rel[:, 1], -(px[:, 1] - h / 2))
    assert [p["id"] for p in pts] == list(range(18)) and len(front["_3d_pose_data"]["connections"]) == 17
    # Turned around (180 deg about Y): nose goes behind the neck in depth.
    back = so.smplx_to_openpose(model, {"global_orient": [0, np.pi, 0], "body_pose": np.zeros(63)}, w, h)[0]
    bz = {p["id"]: p["z"] for p in back["_3d_pose_data"]["points"]}
    fz = {p["id"]: p["z"] for p in pts}
    assert fz[0] > fz[1] and bz[0] < bz[1], (fz[0], fz[1], bz[0], bz[1])
    print("ok  smplx_to_openpose: front/back, sides, canvas fit, 3D/2D consistency")


if __name__ == "__main__":
    models = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMPLX_MODELS", "")
    if not models:
        print("skip smplx_to_openpose: pass the models dir or set SMPLX_MODELS")
        sys.exit(0)
    check(models)
