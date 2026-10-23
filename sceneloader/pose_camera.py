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


# Every STEP-th vertex is used for framing; the browser view gets the same points (same fit).
STEP = 4


# Bool mask per framing over all vertices (lbs weights pick each vertex's main joint).
def framing_masks(model):
    from smplx.joint_names import JOINT_NAMES
    owner = np.asarray(model.lbs_weights.detach().cpu().numpy()).argmax(1)
    names = np.array(JOINT_NAMES[:model.lbs_weights.shape[1]])[owner]
    out = {}
    for framing, rule in FRAMING.items():
        if rule is None:
            out[framing] = np.ones(len(names), bool)
        elif rule == "head":
            out[framing] = np.isin(names, list(_HEAD))
        else:
            out[framing] = ~np.isin(names, list(rule))
    return out


# Posed body for the camera: (sampled vertices, {framing: mask over them}, 18 joints, named joints).
def pose_body(model, params):
    verts, named = posed_body(model, params)
    masks = {k: m[::STEP] for k, m in framing_masks(model).items()}
    return verts[::STEP], masks, openpose_joints(named), named


# Data for the browser view (web/pose_camera_math.js): same points and masks as render_pose.
def view_payload(body, width, height):
    verts, masks, joints, _ = body
    bits = np.zeros(len(verts), np.int64)
    for i, framing in enumerate(FRAMING):
        bits |= masks[framing].astype(np.int64) << i
    return {"width": int(width), "height": int(height), "framings": list(FRAMING), "fill": FILL,
            "verts": np.round(verts, 5).reshape(-1).tolist(), "bits": bits.tolist(),
            "joints": np.round(joints, 5).reshape(-1).tolist()}


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


# Shot tag from the joints inside the frame (not from the framing setting, which only fits the camera).
def shot_tags(inside):
    head = inside[[0, 14, 15, 16, 17]].any()
    feet, knees, hips = inside[[10, 13]].any(), inside[[9, 12]].any(), inside[[8, 11]].any()
    if feet:
        tags = ["full body"] if head else ["lower body"]
    elif knees:
        tags = ["cowboy shot"] + ([] if feet else ["feet out of frame"])
    elif hips:
        tags = ["upper body"]
    else:
        tags = ["portrait"]
    return tags + ([] if head else ["head out of frame"])


# Booru camera tags for the shot: what is in frame plus view direction relative to the body.
def camera_tags(named, rot, elevation, inside):
    tags = shot_tags(inside)
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


MAX_EXTEND = 2.0  # overscan canvas side at most this many times the frame side


# Overscan canvas around the frame that also holds the whole body (frame pixel coords, px (N, 2)).
# Returns {width, height, x0, y0, sx, sy, crop: (x, y, w, h)}: canvas pixel = (frame pixel - x0/y0) * sx/sy,
# with about pixels x frame pixels in total (never more resolution than the frame). None: body fits.
def overscan_canvas(px, width, height, pixels=1.3, margin=0.03):
    size = np.array([width, height], float)
    lo = np.minimum(px.min(0) - margin * size, 0)
    hi = np.maximum(px.max(0) + margin * size, size)
    if (lo > -1).all() and (hi < size + 1).all():
        return None
    ext = (hi - lo) - size                                   # extension per axis, split before / after
    cap = (MAX_EXTEND - 1) * size
    k = np.minimum(1.0, cap / np.maximum(ext, 1e-9))
    lo, hi = lo * k, size + (hi - size) * k
    span = hi - lo
    s = min(1.0, float(np.sqrt(pixels * width * height / (span[0] * span[1]))))
    cw, ch = (max(64, int(round(v * s / 64)) * 64) for v in span)
    sx = sy = min(cw / span[0], ch / span[1])  # one scale keeps the frame aspect; rounding slack goes right / down
    crop = (int(round(-lo[0] * sx)), int(round(-lo[1] * sy)), int(round(width * sx)), int(round(height * sy)))
    return {"width": cw, "height": ch, "x0": float(lo[0]), "y0": float(lo[1]), "sx": sx, "sy": sy, "crop": crop}


# Identity canvas: the frame itself.
def frame_canvas(width, height):
    return {"width": width, "height": height, "x0": 0.0, "y0": 0.0, "sx": 1.0, "sy": 1.0,
            "crop": (0, 0, width, height)}


# Frame pixel coords -> canvas pixel coords.
def to_canvas(px, canvas):
    return (px - [canvas["x0"], canvas["y0"]]) * [canvas["sx"], canvas["sy"]]


# One shot: OpenPose map on the canvas (frame, or overscan canvas holding the whole body), tags and
# points of the frame, solved camera and canvas. body: pose_body() result; overscan: pixels budget or None.
def shot(model, params, width, height, framing="full body", yaw=0.0, elevation=0.0, fov=40.0, zoom=1.0,
         shift_x=0.0, shift_y=0.0, body=None, overscan=None):
    if framing not in FRAMING:
        raise ValueError(f"unknown framing {framing!r}, expected one of {list(FRAMING)}")
    verts, masks, joints, named = body or pose_body(model, params)
    eye, rot, focal, offset = solve_camera(verts[masks[framing]], yaw, elevation, fov, zoom, shift_x, shift_y,
                                           width, height)
    px, depth = project(joints, eye, rot, focal, width, height)
    px = px + offset
    inside = (depth > 1e-3) & (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    face = visible_face(joints, eye)
    canvas = None
    if overscan:
        vpx, vdepth = project(verts, eye, rot, focal, width, height)
        canvas = overscan_canvas((vpx + offset)[vdepth > 1e-3], width, height, overscan)
    canvas = canvas or frame_canvas(width, height)
    cpx = to_canvas(px, canvas)
    on_canvas = (depth > 1e-3) & (cpx[:, 0] >= 0) & (cpx[:, 0] < canvas["width"]) & (cpx[:, 1] >= 0) \
        & (cpx[:, 1] < canvas["height"])
    canvas_points = [tuple(cpx[i]) if face[i] and on_canvas[i] else None for i in range(18)]
    return {"map": draw_openpose(canvas_points, canvas["width"], canvas["height"]),
            # The prompt describes what is generated: the canvas (whole body when overscanned).
            "tags": camera_tags(named, rot, elevation, on_canvas),
            "points": [tuple(px[i]) if face[i] and inside[i] else None for i in range(18)],
            "cam": {"eye": eye, "rot": rot, "focal": focal, "offset": offset}, "canvas": canvas}


# SMPL-X pose + camera settings -> (OpenPose map uint8 (H, W, 3), tags list, points list) for the frame.
# body: pose_body() result, to reuse it for view_payload; cam: dict that receives the solved camera.
def render_pose(model, params, width, height, framing="full body", yaw=0.0, elevation=0.0, fov=40.0,
                zoom=1.0, shift_x=0.0, shift_y=0.0, body=None, cam=None):
    r = shot(model, params, width, height, framing, yaw, elevation, fov, zoom, shift_x, shift_y, body)
    if cam is not None:
        cam.update(r["cam"])
    return r["map"], r["tags"], r["points"]


# Depth map of the posed body (full mesh, z-buffer) for the camera solved by render_pose / shot: uint8
# (H, W, 3), near = bright, background black, like MiDaS maps for depth ControlNets. canvas: shot()
# canvas to draw on (default: the frame).
def depth_map(model, params, cam, width, height, near=255, far=40, canvas=None):
    verts, _ = posed_body(model, params)
    faces = np.asarray(model.faces, np.int64)
    px, depth = project(verts, cam["eye"], cam["rot"], cam["focal"], width, height)
    canvas = canvas or frame_canvas(width, height)
    px = to_canvas(px + cam["offset"], canvas)
    width, height = canvas["width"], canvas["height"]
    zbuf = np.full((height, width), np.inf)
    tri, tz = px[faces], depth[faces]                       # (F, 3, 2), (F, 3)
    keep = (tz > 1e-3).all(1)
    lo, hi = np.floor(tri.min(1)).astype(int), np.ceil(tri.max(1)).astype(int)
    keep &= (hi[:, 0] >= 0) & (lo[:, 0] < width) & (hi[:, 1] >= 0) & (lo[:, 1] < height)
    for f in np.nonzero(keep)[0]:
        (x0, y0), (x1, y1) = np.maximum(lo[f], 0), np.minimum(hi[f], [width - 1, height - 1])
        if x1 < x0 or y1 < y0:
            continue
        (ax, ay), (bx, by), (cx, cy) = tri[f]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        w2 = 1.0 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not ins.any():
            continue
        z = w0 * tz[f, 0] + w1 * tz[f, 1] + w2 * tz[f, 2]
        region = zbuf[y0:y1 + 1, x0:x1 + 1]
        np.copyto(region, np.minimum(region, np.where(ins, z, np.inf)))
    hit = np.isfinite(zbuf)
    out = np.zeros((height, width), np.uint8)
    if hit.any():
        z0, z1 = zbuf[hit].min(), zbuf[hit].max()
        t = (zbuf[hit] - z0) / max(z1 - z0, 1e-6)
        out[hit] = np.round(near + (far - near) * t).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, 2)


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
