"""Check smplx_rig GLB export against the smplx library. Needs smplx + SMPL-X model files.

Run: python tests/test_smplx_rig.py <models_dir>   (or set SMPLX_MODELS; models_dir holds smplx/*.npz)
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader import smplx_rig  # noqa: E402


# Read a GLB back into its JSON document and binary chunk.
def read_glb(path):
    data = open(path, "rb").read()
    magic, version, total = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and total == len(data), (magic, version, total)
    jlen, jtype = struct.unpack("<II", data[12:20])
    assert jtype == 0x4E4F534A
    doc = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack("<II", data[20 + jlen:28 + jlen])
    assert btype == 0x004E4942
    return doc, data[28 + jlen:28 + jlen + blen]


# Return accessor data as a numpy array.
def accessor(doc, bin_, i):
    acc = doc["accessors"][i]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    raw = bin_[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    return np.frombuffer(raw, dtype).reshape(acc["count"], width)


# Pose the GLB mesh from its own nodes/skin, like an importer would.
def skin_from_glb(doc, bin_):
    prim = doc["meshes"][0]["primitives"][0]
    verts = accessor(doc, bin_, prim["attributes"]["POSITION"]).astype(np.float64)
    idx = accessor(doc, bin_, prim["attributes"]["JOINTS_0"]).astype(int)
    w = accessor(doc, bin_, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    skin = doc["skins"][0]
    ibm = accessor(doc, bin_, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    parents = [-1] * len(skin["joints"])
    for j, node in enumerate(doc["nodes"][:len(parents)]):
        for c in node.get("children", []):
            parents[c] = j
    q = np.array([n["rotation"] for n in doc["nodes"][:len(parents)]])
    t = np.array([n["translation"] for n in doc["nodes"][:len(parents)]])
    world = smplx_rig.world_matrices(parents, smplx_rig.quat_to_matrix(q), t)
    return smplx_rig.skin_vertices(verts, idx, w, world, ibm), parents


def test_glb_matches_smplx(models_dir):
    import torch
    rng = np.random.default_rng(7)
    params = {
        "model_path": models_dir, "gender": "neutral",
        "global_orient": rng.uniform(-1.0, 1.0, 3), "body_pose": rng.uniform(-0.5, 0.5, 63),
        "left_hand_pose": rng.uniform(-0.4, 0.4, 45), "right_hand_pose": rng.uniform(-0.4, 0.4, 45),
        "jaw_pose": rng.uniform(0, 0.3, 3), "leye_pose": np.zeros(3), "reye_pose": np.zeros(3),
        "betas": rng.uniform(-1.5, 1.5, 10), "expression": rng.uniform(-1, 1, 10), "transl": [0.3, 0.1, 2.0],
    }
    model = smplx_rig.load_model(models_dir, "neutral", 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rig.glb")
        smplx_rig.export_rigged_glb(params, path, model)
        doc, bin_ = read_glb(path)
        got, parents = skin_from_glb(doc, bin_)
        assert parents == list(model.parents.numpy()[:55].clip(-1)), "joint hierarchy differs"
        assert [n["name"] for n in doc["nodes"][:3]] == ["pelvis", "left_hip", "right_hip"]
    # Reference: smplx LBS without pose correctives, same grounding (pelvis xz at 0, min y at 0).
    t = lambda k: torch.tensor(np.asarray(params[k], np.float32))[None]  # noqa: E731
    with torch.inference_mode(False), torch.no_grad():
        model.posedirs.zero_()
        out = model(global_orient=t("global_orient"), body_pose=t("body_pose"), betas=t("betas"),
                    expression=t("expression"), left_hand_pose=t("left_hand_pose"),
                    right_hand_pose=t("right_hand_pose"), jaw_pose=t("jaw_pose"),
                    leye_pose=t("leye_pose"), reye_pose=t("reye_pose"))
    ref = out.vertices[0].double().numpy()
    pelvis = out.joints[0, 0].double().numpy()
    ref -= [pelvis[0], ref[:, 1].min(), pelvis[2]]
    # Top-4 weights vs the full 55: allow a few mm on far-from-joint vertices.
    err = np.linalg.norm(got - ref, axis=1)
    print(f"vertex error: mean {err.mean() * 1000:.2f} mm, max {err.max() * 1000:.2f} mm")
    assert err.mean() < 0.002 and err.max() < 0.02, (err.mean(), err.max())
    assert abs(got[:, 1].min()) < 1e-5, got[:, 1].min()


if __name__ == "__main__":
    models = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMPLX_MODELS", "")
    if not models:
        print("skip test_glb_matches_smplx: pass the models dir or set SMPLX_MODELS")
        sys.exit(0)
    test_glb_matches_smplx(models)
    print("ok  test_glb_matches_smplx")
