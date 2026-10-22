"""Virtual camera around an SMPL-X pose: framing, angle, height -> OpenPose map + camera tags.

The body comes from SMPLx-Estimator (Y up, the sample photo's camera on +Z). yaw 0 / elevation 0
looks at the body like the sample photo did; yaw turns the camera around the body, elevation raises
it (positive = from above). The framing picks which body parts must fit; the camera distance is
solved so they fill the frame, then zoom and shift move the shot. The map is drawn like
controlnet_aux / DWPose (COCO-18, same colors), so OpenPose ControlNets read it as usual.
"""
import numpy as np

from .smplx_openpose import openpose_joints, posed_body

# Body parts that must be in the shot, as SMPL-X joints owning the vertices (by skin weight).
_LEGS_LOW = {"left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"}
_HIPS = {"left_hip", "right_hip"}
_HEAD = {"neck", "head", "jaw", "left_eye_smplhf", "right_eye_smplhf", "left_collar", "right_collar"}
FRAMING = {
    "full body": None,                                  # every vertex
    "cowboy shot": _LEGS_LOW,                           # down to mid thigh (knees and below excluded)
    "upper body": _LEGS_LOW | _HIPS | {"pelvis"},       # down to the waist
    "portrait": "head",                                 # head, neck and collarbones
}
FILL = 0.9  # share of the frame the framed parts take along the limiting side

# controlnet_aux draw_bodypose: limbs (0-based COCO-18 pairs) and colors (RGB), limb i uses COLORS[i].
LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
         [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
FACE = (0, 14, 15, 16, 17)  # nose, eyes, ears: hidden when turned away from the camera


# Rotation of the camera: yaw about Y, then elevation about the camera's X (degrees).
def camera_rotation(yaw, elevation):
    a, e = np.radians(yaw), np.radians(elevation)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), np.sin(e)], [0, -np.sin(e), np.cos(e)]])
    return ry @ rx  # columns: camera right, up, back (camera looks along -column 2)


# Vertices that must be in the shot for a framing (lbs weights pick each vertex's main joint).
def framing_vertices(model, verts, framing):
    from smplx.joint_names import JOINT_NAMES
    rule = FRAMING[framing]
    if rule is None:
        return verts
    owner = np.asarray(model.lbs_weights.detach().cpu().numpy()).argmax(1)
    names = np.array(JOINT_NAMES[:model.lbs_weights.shape[1]])[owner]
    keep = np.isin(names, list(_HEAD)) if rule == "head" else ~np.isin(names, list(rule))
    return verts[keep]


# Perspective projection: points (N, 3) -> (pixels (N, 2), depth (N,)) for a camera at eye.
def project(points, eye, rot, focal, width, height):
    cam = (points - eye) @ rot          # camera coords: x right, y up, z back
    depth = -cam[:, 2]
    safe = np.where(depth > 1e-6, depth, 1e-6)
    px = np.stack([width / 2 + focal * cam[:, 0] / safe, height / 2 - focal * cam[:, 1] / safe], 1)
    return px, depth


# Place the camera so the framed vertices fill the frame; returns (eye, rot, focal, pixel offset).
def solve_camera(framed, yaw, elevation, fov, zoom, shift_x, shift_y, width, height):
    rot = camera_rotation(yaw, elevation)
    focal = (height / 2) / np.tan(np.radians(fov) / 2)
    aim = (framed.min(0) + framed.max(0)) / 2
    dist = 3.0
    for _ in range(6):  # perspective: refit a few times, converges fast
        eye = aim + rot[:, 2] * dist
        px, _ = project(framed, eye, rot, focal, width, height)
        span = np.maximum(px.max(0) - px.min(0), 1e-6)
        dist *= max(span[0] / (FILL * width), span[1] / (FILL * height))
    dist /= max(zoom, 1e-3)
    eye = aim + rot[:, 2] * dist
    px, _ = project(framed, eye, rot, focal, width, height)
    center = (px.min(0) + px.max(0)) / 2
    offset = np.array([width / 2, height / 2]) - center + [shift_x * width, -shift_y * height]
    return eye, rot, focal, offset


# Booru camera tags for the shot: framing plus view direction relative to the body.
def camera_tags(named, rot, elevation, framing):
    tags = [framing]
    up = np.array([0.0, 1.0, 0.0])
    right = named["left_hip"] - named["right_hip"]       # person's left
    forward = np.cross(right, up)                        # chest direction (horizontal)
    to_cam = rot[:, 2].copy()
    to_cam[1] = 0
    if np.linalg.norm(forward) > 1e-6 and np.linalg.norm(to_cam) > 1e-6:
        c = np.dot(forward, to_cam) / (np.linalg.norm(forward) * np.linalg.norm(to_cam))
        if c < -0.5:
            tags.append("from behind")
        elif c < 0.5:
            tags.append("from side")
    if elevation >= 30:
        tags.append("from above")
    elif elevation <= -20:
        tags.append("from below")
    return tags


# Face points turned away from the camera are hidden, like a detector would miss them.
def visible_face(joints, eye):
    head = (joints[16] + joints[17]) / 2
    to_cam = eye - head
    vis = np.ones(18, bool)
    for i in FACE:
        v = joints[i] - head
        n = np.linalg.norm(v) * np.linalg.norm(to_cam)
        vis[i] = n < 1e-9 or np.dot(v, to_cam) / n > (-0.3 if i in (16, 17) else 0.0)
    return vis


# Draw COCO-18 keypoints (None = missing) like controlnet_aux draw_bodypose.
def draw_openpose(points, width, height):
    import cv2
    canvas = np.zeros((height, width, 3), np.uint8)
    stick = 4  # same as DWPose without xinsr scaling
    for i, (a, b) in enumerate(LIMBS):
        if points[a] is None or points[b] is None:
            continue
        (x1, y1), (x2, y2) = points[a], points[b]
        length = float(np.hypot(x1 - x2, y1 - y2))
        angle = float(np.degrees(np.arctan2(y1 - y2, x1 - x2)))
        poly = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(length / 2), stick),
                                int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in COLORS[i]])
    for i, p in enumerate(points):
        if p is not None:
            cv2.circle(canvas, (int(p[0]), int(p[1])), stick, COLORS[i], thickness=-1)
    return canvas


# SMPL-X pose + camera settings -> (OpenPose map uint8 (H, W, 3), tags list, points list).
def render_pose(model, params, width, height, framing="full body", yaw=0.0, elevation=0.0, fov=40.0,
                zoom=1.0, shift_x=0.0, shift_y=0.0):
    if framing not in FRAMING:
        raise ValueError(f"unknown framing {framing!r}, expected one of {list(FRAMING)}")
    verts, named = posed_body(model, params)
    joints = openpose_joints(named)
    framed = framing_vertices(model, verts, framing)
    eye, rot, focal, offset = solve_camera(framed, yaw, elevation, fov, zoom, shift_x, shift_y, width, height)
    px, depth = project(joints, eye, rot, focal, width, height)
    px = px + offset
    vis = visible_face(joints, eye) & (depth > 1e-3)
    vis &= (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    points = [tuple(px[i]) if vis[i] else None for i in range(18)]
    return draw_openpose(points, width, height), camera_tags(named, rot, elevation, framing), points


# Shot and view tags the camera owns; the same tags already in the prompt are replaced.
CAMERA_TAGS = {"full body", "cowboy shot", "upper body", "lower body", "portrait", "close-up", "wide shot",
               "from above", "from below", "from side", "from behind", "straight-on", "dutch angle",
               "feet out of frame", "head out of frame"}


# Remove camera tags (any weight syntax) from a comma-separated prompt and append the new ones.
def merge_camera_tags(prompt, tags):
    import re
    kept = []
    for tok in (t.strip() for t in prompt.split(",")):
        bare = re.sub(r":[\d.]+\)?$", "", tok).strip("()[]{} ").lower()
        if tok and bare not in CAMERA_TAGS:
            kept.append(tok)
    return ", ".join(kept + list(tags))
