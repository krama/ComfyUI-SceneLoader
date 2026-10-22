"""SMPL-X pose -> OpenPose (COCO-18) keypoints with 3D data for the 3D OpenPose Editor.

The body is posed with the SMPL-X model (Y up, camera looking along -Z, as SMPLx-Estimator returns
it), projected orthographically and centered on a canvas of the sample image size. The result is a
ComfyUI POSE_KEYPOINT list whose first item also carries "_3d_pose_data" in the editor's format
(points in canvas pixels from the center, y up), so the editor opens with real depth.
"""
import numpy as np

# COCO-18 order used by OpenPose / DWPose; "neck" is the shoulder midpoint like OpenPose.
OPENPOSE_FROM_SMPLX = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
                       "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip",
                       "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
# Same limb pairs and colors as the editor (js/openpose.js connect_keypoints / connect_color).
CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
               [1, 11], [11, 12], [12, 13], [14, 0], [14, 16], [15, 0], [15, 17]]
COLORS = [[0, 0, 255], [255, 0, 0], [255, 170, 0], [255, 255, 0], [255, 85, 0], [170, 255, 0],
          [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
          [0, 85, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
FILL = 0.9  # share of the canvas the body occupies along its limiting side


# Pose the SMPL-X model (no translation) and return (vertices, joints dict by name).
def posed_body(model, params):
    import torch
    from smplx.joint_names import JOINT_NAMES

    def t(key, n):
        v = np.zeros(n, np.float32)
        src = np.asarray(params.get(key, v), np.float32).reshape(-1)[:n]
        v[:len(src)] = src
        return torch.tensor(v)[None]

    with torch.inference_mode(False), torch.no_grad():
        out = model(global_orient=t("global_orient", 3), body_pose=t("body_pose", 63),
                    betas=t("betas", int(model.num_betas)), expression=t("expression", 10),
                    left_hand_pose=t("left_hand_pose", 45), right_hand_pose=t("right_hand_pose", 45),
                    jaw_pose=t("jaw_pose", 3), leye_pose=t("leye_pose", 3), reye_pose=t("reye_pose", 3))
    verts = out.vertices[0].double().numpy()
    joints = out.joints[0].double().numpy()
    return verts, {name: joints[i] for i, name in enumerate(JOINT_NAMES[:len(joints)])}


# Return the 18 OpenPose joints (18, 3) from named SMPL-X joints.
def openpose_joints(named):
    pts = []
    for name in OPENPOSE_FROM_SMPLX:
        if name == "neck":
            pts.append((named["left_shoulder"] + named["right_shoulder"]) / 2.0)
        else:
            pts.append(named[name])
    return np.array(pts)


# Fit body to a WxH canvas: return (2D pixel coords (18, 2), editor 3D coords (18, 3)).
def fit_to_canvas(joints, verts, width, height):
    lo, hi = verts[:, :2].min(0), verts[:, :2].max(0)
    span = np.maximum(hi - lo, 1e-6)
    s = FILL * min(width / span[0], height / span[1])
    center = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, verts[:, 2].mean()])
    rel = (joints - center) * s                     # editor space: pixels from canvas center, y up
    px = np.stack([width / 2.0 + rel[:, 0], height / 2.0 - rel[:, 1]], 1)
    return px, rel


# Build the POSE_KEYPOINT list (normalized coords) with editor 3D data on the first item.
def pose_keypoint(px, rel, width, height):
    flat = []
    for x, y in px:
        flat += [float(x / width), float(y / height), 1.0]
    points = [{"id": i, "poseId": 0, "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "color": COLORS[i]}
              for i, p in enumerate(rel)]
    conns = [{"startId": a, "endId": b, "startPoseId": 0, "endPoseId": 0} for a, b in CONNECTIONS]
    return [{"people": [{"pose_keypoints_2d": flat}], "canvas_width": int(width), "canvas_height": int(height),
             "_3d_pose_data": {"points": points, "connections": conns, "version": "1.2"}}]


# SMPLX dict -> POSE_KEYPOINT with 3D data, for a canvas of the given size.
def smplx_to_openpose(model, params, width, height):
    verts, named = posed_body(model, params)
    px, rel = fit_to_canvas(openpose_joints(named), verts, width, height)
    return pose_keypoint(px, rel, width, height)
