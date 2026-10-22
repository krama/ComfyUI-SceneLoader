"""Export an SMPL-X pose (SMPLx-Estimator "SMPLX" dict) as a skinned glTF binary (.glb).

The mesh is stored in the bind pose (shape + expression applied, all joint rotations zero) with
55 joints, top-4 skin weights from the model and the estimated pose as joint rotations, so Blender
or Cascadeur can keep posing it. Pose-corrective blendshapes are not exported (plain LBS).
"""
import json
import os
import struct

import numpy as np

SMPLX_JOINTS = 55
TOPK = 4


# Return the folder that holds smplx/SMPLX_<GENDER>.npz for a path that may be that folder,
# its smplx/ subfolder or the .npz file itself.
def model_parent(path):
    p = os.path.abspath(path)
    if p.endswith(".npz"):
        p = os.path.dirname(p)
    if os.path.basename(p).lower() == "smplx":
        p = os.path.dirname(p)
    return p


# Load SMPL-X with the same options as SMPLx-Estimator (full hand pose, flat hand mean).
def load_model(path, gender, num_betas):
    import smplx
    import torch
    with torch.inference_mode(False):
        return smplx.create(model_path=model_parent(path), model_type="smplx", gender=gender,
                            use_pca=False, flat_hand_mean=True, num_betas=num_betas, ext="npz").eval()


# Return the 55 joint rotations (axis-angle, shape (55, 3)) from an SMPLX dict.
def pose_axis_angles(params):
    get = lambda k, n: np.asarray(params.get(k, np.zeros(n)), np.float64).reshape(-1)[:n]  # noqa: E731
    parts = [get("global_orient", 3), get("body_pose", 63), get("jaw_pose", 3), get("leye_pose", 3),
             get("reye_pose", 3), get("left_hand_pose", 45), get("right_hand_pose", 45)]
    return np.concatenate(parts).reshape(SMPLX_JOINTS, 3)


# Convert axis-angle vectors (N, 3) to unit quaternions (N, 4) in glTF order x, y, z, w.
def axis_angle_to_quat(aa):
    angle = np.linalg.norm(aa, axis=1, keepdims=True)
    axis = np.divide(aa, angle, out=np.zeros_like(aa), where=angle > 1e-12)
    half = angle / 2.0
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=1)


# Convert unit quaternions (N, 4) x, y, z, w to rotation matrices (N, 3, 3).
def quat_to_matrix(q):
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


# Return world 4x4 matrices of joints from local rotations and translations, following parents.
def world_matrices(parents, rot, trans):
    out = np.zeros((len(parents), 4, 4))
    for j, p in enumerate(parents):
        local = np.eye(4)
        local[:3, :3] = rot[j]
        local[:3, 3] = trans[j]
        out[j] = local if p < 0 else out[p] @ local
    return out


# Linear blend skinning of bind-pose vertices with joint world matrices and inverse binds.
def skin_vertices(verts, joints_idx, weights, world, inv_bind):
    skin = world @ inv_bind                                  # (J, 4, 4)
    vh = np.concatenate([verts, np.ones((len(verts), 1))], 1)
    out = np.zeros((len(verts), 3))
    for k in range(joints_idx.shape[1]):
        m = skin[joints_idx[:, k]]                           # (V, 4, 4)
        out += weights[:, k:k + 1] * np.einsum("vij,vj->vi", m, vh)[:, :3]
    return out


# Build the rig: bind verts/joints, faces, parents, top-k weights, local transforms, names.
def build_rig(model, params):
    import torch
    from smplx.joint_names import JOINT_NAMES
    nb = int(model.num_betas)
    betas = np.zeros(nb)
    src = np.asarray(params.get("betas", []), np.float64).reshape(-1)[:nb]
    betas[:len(src)] = src
    expr = np.zeros(10)
    src = np.asarray(params.get("expression", []), np.float64).reshape(-1)[:10]
    expr[:len(src)] = src
    with torch.inference_mode(False), torch.no_grad():
        out = model(betas=torch.tensor(betas, dtype=torch.float32)[None],
                    expression=torch.tensor(expr, dtype=torch.float32)[None])  # all poses zero
    verts = out.vertices[0].double().numpy()
    joints = out.joints[0, :SMPLX_JOINTS].double().numpy()
    parents = model.parents.cpu().numpy().astype(int)
    parents[0] = -1
    lbs = model.lbs_weights.detach().cpu().double().numpy()  # (V, 55)
    idx = np.argsort(-lbs, axis=1)[:, :TOPK]
    w = np.take_along_axis(lbs, idx, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    quat = axis_angle_to_quat(pose_axis_angles(params))
    local_t = joints - np.where(parents[:, None] >= 0, joints[np.maximum(parents, 0)], 0)
    # Ground the posed body: pelvis above the origin, lowest vertex on Y=0 (glTF is Y-up).
    world = world_matrices(parents, quat_to_matrix(quat), local_t)
    inv_bind = np.tile(np.eye(4), (SMPLX_JOINTS, 1, 1))
    inv_bind[:, :3, 3] = -joints
    posed = skin_vertices(verts, idx, w, world, inv_bind)
    local_t[0] = [0.0, joints[0, 1] - posed[:, 1].min(), 0.0]
    return {"verts": verts, "faces": np.asarray(model.faces, np.uint32), "joints": joints,
            "parents": parents, "idx": idx.astype(np.uint16), "weights": w.astype(np.float32),
            "quat": quat, "local_t": local_t, "inv_bind": inv_bind,
            "names": list(JOINT_NAMES[:SMPLX_JOINTS])}


# Serialize a rig dict to GLB bytes (one skinned mesh, 55 joint nodes).
def rig_to_glb(rig):
    blobs, views, accessors = [], [], []

    def add(arr, comp, typ, target=None, minmax=False):
        data = np.ascontiguousarray(arr).tobytes()
        offset = sum(len(b) for b in blobs)
        blobs.append(data + b"\0" * (-len(data) % 4))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    pos = add(rig["verts"].astype(np.float32), 5126, "VEC3", 34962, minmax=True)
    jnt = add(rig["idx"], 5123, "VEC4", 34962)
    wgt = add(rig["weights"], 5126, "VEC4", 34962)
    ind = add(rig["faces"].reshape(-1, 1), 5125, "SCALAR", 34963)
    ibm = add(rig["inv_bind"].transpose(0, 2, 1).reshape(-1, 16).astype(np.float32), 5126, "MAT4")

    children = [[] for _ in rig["parents"]]
    for j, p in enumerate(rig["parents"]):
        if p >= 0:
            children[p].append(j)
    nodes = []
    for j, name in enumerate(rig["names"]):
        node = {"name": name, "translation": rig["local_t"][j].tolist(), "rotation": rig["quat"][j].tolist()}
        if children[j]:
            node["children"] = children[j]
        nodes.append(node)
    nodes.append({"name": "smplx_body", "mesh": 0, "skin": 0})
    doc = {
        "asset": {"version": "2.0", "generator": "ComfyUI-SceneLoader smplx_rig"},
        "scene": 0,
        "scenes": [{"nodes": [0, len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": [{"name": "smplx_body", "primitives": [{
            "attributes": {"POSITION": pos, "JOINTS_0": jnt, "WEIGHTS_0": wgt}, "indices": ind}]}],
        "skins": [{"name": "smplx_skeleton", "inverseBindMatrices": ibm, "skeleton": 0,
                   "joints": list(range(len(rig["names"])))}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    bin_ = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(bin_), 0x004E4942) + bin_)


# Export an SMPLX dict to a skinned GLB file; returns the rig (for checks and logging).
def export_rigged_glb(params, path, model=None):
    if model is None:
        model = load_model(params["model_path"], params.get("gender", "neutral"),
                           max(10, len(np.asarray(params.get("betas", [])).reshape(-1))))
    rig = build_rig(model, params)
    with open(path, "wb") as fh:
        fh.write(rig_to_glb(rig))
    return rig
