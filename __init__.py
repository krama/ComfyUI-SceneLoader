"""Scene Loader - builds a scene from structured blocks (character, background, light) with LoRAs."""
import json
import os

import folder_paths
import torch
from nodes import SaveImage

from .sceneloader.prompt_log import build_log
from .sceneloader.composer import (NAME_RE, block_loras, camera_input_tags, compose, load_preset, lora_info,
                                   save_preset, sidecar, slot_options)
from .sceneloader.assembler import (Library, SceneError, assemble, camera_tags_from_json, report,
                                    scene_base, to_lora_stack, KNOWN_BASES)

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


# Resolve the block library: SCENELOADER_LIBRARY env, then ./library, then ./examples.
def library_root():
    env = os.environ.get("SCENELOADER_LIBRARY")
    if env:
        return env
    lib = os.path.join(PKG_DIR, "library")
    return lib if os.path.isdir(lib) else os.path.join(PKG_DIR, "examples")


class SceneLoader:
    @classmethod
    def INPUT_TYPES(cls):
        scenes = Library(library_root()).scenes() or ["(no scenes)"]
        return {"required": {
            "scene": (scenes, {"tooltip": "Scene file from library/scenes"}),
            "base": (list(KNOWN_BASES), {"tooltip": "LoRA base folder under loras/"}),
        }, "optional": {
            "camera_json": ("STRING", {"forceInput": True,
                                       "tooltip": "Advanced Camera Control camera_json; replaces camera block"}),
            "camera_prompt": ("STRING", {"forceInput": True,
                                         "tooltip": "Advanced Camera Control prompt; used as-is for anima"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "lora_stack", "lora_stack_detailer", "info")
    FUNCTION = "load"
    CATEGORY = "SceneLoader"

    # Re-run when any library file changes on disk.
    @classmethod
    def IS_CHANGED(cls, scene, base, camera_json=None, camera_prompt=None):
        return Library(library_root()).fingerprint()

    # Return camera tags from 3D inputs: natural prompt for anima, booru tags for illustrious.
    @staticmethod
    def camera_override(base, camera_json, camera_prompt):
        if not camera_json and not camera_prompt:
            return None
        if base == "anima" and camera_prompt:
            return camera_prompt.strip()
        if camera_json:
            return camera_tags_from_json(camera_json, camera_prompt or "")
        return camera_prompt.strip()

    # Assemble the selected scene into prompts and SimplePowerLora stacks.
    def load(self, scene, base, camera_json=None, camera_prompt=None):
        lib = Library(library_root())
        try:
            camera = self.camera_override(base, camera_json, camera_prompt)
            if camera is not None:
                print(f"[SceneLoader] {scene}: 3d camera -> {camera}")
            sb = scene_base(lib, scene)
            if sb and sb != base:
                raise SceneError(f"scene '{scene}' is written for base '{sb}', but base is '{base}'")
            result = assemble(lib, scene, camera)
        except SceneError as e:
            print(f"[SceneLoader] {scene}: {e}")
            raise
        resolver = lambda name: folder_paths.get_full_path("loras", name)  # noqa: E731
        info = report(result, base, resolver)
        missing = info.count("  MISSING")
        print(f"[SceneLoader] {scene} ({base}): {len(result['loras'])} loras, "
              f"{len(result['loras_detailer'])} detailer loras, {missing} missing")
        return (result["positive"], result["negative"], to_lora_stack(result["loras"], base),
                to_lora_stack(result["loras_detailer"], base), info)


class SceneComposer:
    """Builds a scene from slot rows (library blocks and LoRAs) edited in the node UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": (list(KNOWN_BASES), {"tooltip": "LoRA base folder under loras/"}),
            "composition": ("STRING", {"multiline": True, "default": "[]",
                                       "tooltip": "Slot rows as JSON; edited by the composer UI"}),
        }, "optional": {
            "camera": ("STRING", {"forceInput": True,
                                  "tooltip": "Replaces the camera slot when connected: text tags/prompt, or "
                                             "Advanced Camera Control camera_json (booru tags for illustrious)"}),
        }}

    RETURN_TYPES = SceneLoader.RETURN_TYPES + ("IMAGE", "BOOLEAN", "IMAGE", "BOOLEAN")
    RETURN_NAMES = SceneLoader.RETURN_NAMES + ("pose_image", "has_pose_image", "pose_skeleton", "use_controlnet")
    OUTPUT_TOOLTIPS = (None,) * len(SceneLoader.RETURN_TYPES) + (
        "Sample image of the pose block (-> NLF 3D pose); without one, nodes using it are skipped",
        "True when the pose has a sample image: 3D pose + Pose Camera",
        "OpenPose skeleton of the pose block, a ready 2D ControlNet map; without one, users are skipped",
        "True when the pose has a sample image or a skeleton: apply the pose ControlNet")
    FUNCTION = "compose"
    CATEGORY = "SceneLoader"

    # Re-run when any library file changes on disk.
    @classmethod
    def IS_CHANGED(cls, base, composition, camera=None, **legacy):
        return Library(library_root()).fingerprint()

    # Assemble the rows into prompts and SimplePowerLora stacks.
    # Graphs saved before the single camera input still send camera_json / camera_prompt (legacy).
    def compose(self, base, composition, camera=None, camera_json=None, camera_prompt=None, **legacy):
        lib = Library(library_root())
        try:
            rows = json.loads(composition or "[]")
            if not isinstance(rows, list):
                raise SceneError("composition must be a JSON list of rows")
            if not camera and (camera_json or camera_prompt):
                # Same choice the old inputs had: prompt text for anima, camera_json for illustrious.
                camera = camera_prompt if base == "anima" and camera_prompt else (camera_json or camera_prompt)
                print("[SceneComposer] legacy camera_json/camera_prompt input used; "
                      "reconnect it to the 'camera' input (right-click node -> Fix node)")
            cam = camera_input_tags(base, camera)
            if cam is not None:
                print(f"[SceneComposer] camera input -> {cam}")
            result = compose(lib, lora_root(base), base, rows, cam)
        except (SceneError, ValueError) as e:
            print(f"[SceneComposer] {base}: {e}")
            raise SceneError(str(e))
        resolver = lambda name: folder_paths.get_full_path("loras", name)  # noqa: E731
        info = report(result, base, resolver)
        print(f"[SceneComposer] {base}: {len(rows)} rows, {len(result['loras'])} loras, "
              f"{len(result['loras_detailer'])} detailer loras, {info.count('  MISSING')} missing")
        return (result["positive"], result["negative"], to_lora_stack(result["loras"], base),
                to_lora_stack(result["loras_detailer"], base), info, *pose_outputs(result.get("pose_image")),
                pose_outputs(result.get("pose_skeleton"), "skeleton")[0],
                bool(result.get("pose_image") or result.get("pose_skeleton")))


# (pose_image, has_pose_image): without a sample the image output blocks its users silently, so the
# pose branch (NLF, Pose Camera, previews) does not run; If/Else switches take the plain path.
def pose_outputs(path, what="sample image"):
    if path:
        return load_pose_image(path), True
    try:
        from comfy_execution.graph_utils import ExecutionBlocker
    except ImportError:  # outside ComfyUI (tests)
        return load_pose_image(None), False
    print(f"[SceneComposer] pose row has no {what}: nodes using it are skipped")
    return ExecutionBlocker(None), False


# Load a pose sample image as a ComfyUI IMAGE (1, H, W, 3); no image gives a 64x64 black frame.
def load_pose_image(path):
    if not path:
        return torch.zeros(1, 64, 64, 3)
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


# Return the configured loras folder that holds the <base>/ subfolder (first one as fallback).
def lora_root(base):
    paths = folder_paths.get_folder_paths("loras")
    hit = next((p for p in paths if os.path.isdir(os.path.join(p, base))), None)
    if hit is None:
        print(f"[SceneComposer] no loras folder contains '{base}/': {paths}")
    return hit or paths[0]


class SMPLXRiggedGLB:
    """Saves an SMPL-X pose (SMPLx-Estimator output) as a skinned GLB for Blender / Cascadeur."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "smplx": ("SMPLX", {"tooltip": "SMPL-X pose from SMPLx-Estimator (e.g. Body: NLF)"}),
            "filename_prefix": ("STRING", {"default": "sceneloader/pose/rigged"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "SceneLoader"

    # Write <output>/<prefix>_NNNNN_.glb with 55 joints, skin weights and the pose as joint rotations.
    def export(self, smplx, filename_prefix="sceneloader/pose/rigged"):
        from .sceneloader.smplx_rig import export_rigged_glb
        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        path = os.path.join(full_dir, f"{name}_{counter:05}_.glb")
        try:
            rig = export_rigged_glb(smplx, path)
        except (OSError, KeyError, ValueError) as e:
            print(f"[SceneLoader] rigged GLB export failed: {e}")
            raise
        print(f"[SceneLoader] rigged GLB: {path} ({len(rig['verts'])} verts, {len(rig['names'])} joints)")
        return {"ui": {"text": [path]}, "result": (path,)}


class SMPLXToOpenPose3D:
    """SMPL-X pose -> OpenPose keypoints with 3D data for the 3D OpenPose Editor (pose_point input)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "smplx": ("SMPLX", {"tooltip": "SMPL-X pose from SMPLx-Estimator (e.g. Body: NLF)"}),
            "image": ("IMAGE", {"tooltip": "Sample image; sets the canvas size (also feed it to pose_image)"}),
        }}

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_point",)
    FUNCTION = "convert"
    CATEGORY = "SceneLoader"

    # Pose SMPL-X, take the 18 OpenPose joints and center them on a canvas of the image size.
    def convert(self, smplx, image):
        from .sceneloader.smplx_openpose import smplx_to_openpose
        h, w = int(image.shape[1]), int(image.shape[2])
        model = _smplx_model(smplx["model_path"], smplx.get("gender", "neutral"))
        pose = smplx_to_openpose(model, smplx, w, h)
        print(f"[SceneLoader] SMPL-X -> OpenPose 3D: 18 points on {w}x{h}")
        return (pose,)


class PoseCamera:
    """Virtual camera around an SMPL-X pose: OpenPose map for ControlNet + camera tags in the prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        from .sceneloader.pose_camera import FRAMING
        deg = lambda d, lo, hi, tip: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": 1.0, "tooltip": tip})  # noqa: E731
        return {
            "required": {
                "smplx": ("SMPLX", {"tooltip": "SMPL-X pose from SMPLx-Estimator (e.g. Body: NLF)"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "tooltip": "Generation width"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "tooltip": "Generation height"}),
                "framing": (list(FRAMING), {"default": "full body", "tooltip": "Body parts that must fit"}),
                "yaw": deg(0.0, -180.0, 180.0, "Turn the camera around the body; 0 = like the sample photo"),
                "elevation": deg(0.0, -80.0, 89.0, "Camera height angle: + from above, - from below"),
                "fov": deg(40.0, 10.0, 120.0, "Lens field of view; small = flat tele, large = wide angle"),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 4.0, "step": 0.05,
                                   "tooltip": "> 1 moves closer (crops), < 1 adds room around"}),
                "shift_x": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                                      "tooltip": "Move the subject in the frame, share of width (+ right)"}),
                "shift_y": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                                      "tooltip": "Move the subject in the frame, share of height (+ up)"}),
                "stop_on_new_pose": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Run with a new pose stops here (nothing after this node "
                                                            "runs): set the camera in the view, Run again to render"}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True,
                                      "tooltip": "Prompt to get the camera tags; its own camera tags are replaced"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_map", "positive", "camera_tags", "depth_map")
    OUTPUT_TOOLTIPS = ("OpenPose map for ControlNet OpenPose", "Prompt with the camera tags", "Camera tags",
                       "Depth of the body from the same camera (near = bright) for a depth ControlNet")
    FUNCTION = "shoot"
    CATEGORY = "SceneLoader"

    # Run every time: whether to stop depends on the pose seen by the previous run, not on the inputs.
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # Render the OpenPose map for the camera and put its tags into the prompt.
    # A pose this node has not shown yet stops the run (outputs blocked) so the camera can be set first.
    def shoot(self, smplx, width, height, framing, yaw, elevation, fov, zoom, shift_x, shift_y,
              stop_on_new_pose=True, prompt="", unique_id=None):
        from .sceneloader.pose_camera import depth_map, merge_camera_tags, pose_body, render_pose, view_payload
        model = _smplx_model(smplx["model_path"], smplx.get("gender", "neutral"))
        body = pose_body(model, smplx)
        key = pose_key(smplx)
        new_pose = _POSE_SEEN.get(unique_id) != key
        _POSE_SEEN[unique_id] = key
        cam = {}
        img, tags, points = render_pose(model, smplx, width, height, framing, yaw, elevation, fov, zoom,
                                        shift_x, shift_y, body=body, cam=cam)
        text = ", ".join(tags)
        seen = sum(p is not None for p in points)
        print(f"[SceneLoader] Pose Camera {width}x{height}: {text}; {seen}/18 points in frame "
              f"(yaw {yaw:g}, elevation {elevation:g}, fov {fov:g}, zoom {zoom:g})")
        # The body goes to the node's 3D view (web/pose_camera.js) for mouse camera control.
        ui = {"pose_body": [view_payload(body, width, height)], "camera_tags": [text]}
        if stop_on_new_pose and new_pose:
            from comfy_execution.graph_utils import ExecutionBlocker
            print("[SceneLoader] Pose Camera: new pose, run stopped here; set the camera and Run again to render")
            ui["camera_status"] = ["New pose: set the camera, then Run again to render."]
            return {"ui": ui, "result": (ExecutionBlocker(None),) * 4}
        depth = depth_map(model, smplx, cam, width, height)
        to_image = lambda a: torch.from_numpy(a.astype("float32") / 255.0)[None]  # noqa: E731
        result = (to_image(img), merge_camera_tags(prompt or "", tags), text, to_image(depth))
        return {"ui": ui, "result": result}


# Last pose shown by each Pose Camera node (unique_id -> pose_key); a new pose stops the run once.
_POSE_SEEN = {}


# Hash of the pose parameters (not the camera) of an SMPLX dict.
def pose_key(smplx):
    import hashlib
    import numpy as np
    h = hashlib.sha1()
    for k in ("global_orient", "body_pose", "betas", "left_hand_pose", "right_hand_pose", "jaw_pose", "expression"):
        if k in smplx:
            h.update(k.encode() + np.asarray(smplx[k], np.float32).tobytes())
    return h.hexdigest()


_SMPLX_CACHE = {}


# Load SMPL-X once per (path, gender); the model is CPU-only and small.
def _smplx_model(path, gender):
    from .sceneloader.smplx_rig import load_model
    key = (path, gender)
    if key not in _SMPLX_CACHE:
        _SMPLX_CACHE[key] = load_model(path, gender, 10)
    return _SMPLX_CACHE[key]


class SceneSave(SaveImage):
    """SaveImage that also returns a generation log (final prompt and settings) for Scene Log."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "sceneloader/image"}),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True, "tooltip": "Final positive prompt"}),
                "negative": ("STRING", {"forceInput": True, "tooltip": "Final negative prompt"}),
                "info": ("STRING", {"forceInput": True, "tooltip": "Scene Loader info"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    FUNCTION = "save_with_log"
    CATEGORY = "SceneLoader"

    # Save PNGs via SaveImage; the log is only returned to the UI (no files written).
    def save_with_log(self, images, filename_prefix="sceneloader/image", positive="", negative="",
                      info="", prompt=None, extra_pnginfo=None):
        out = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
        height, width = images[0].shape[0], images[0].shape[1]
        logs = []
        for k, item in enumerate(out["ui"]["images"]):
            img = images[min(k, images.shape[0] - 1)]
            has_nan = bool(torch.isnan(img).any())
            all_black = bool(torch.nan_to_num(img).max() == 0)
            text = build_log(item["filename"], width, height, prompt, positive, negative, info,
                             has_nan=has_nan, all_black=all_black)
            logs.append(text)
            flag = " - WARNING: NaN/black image" if (has_nan or all_black) else ""
            print(f"[SceneLoader] saved {item['subfolder']}/{item['filename']}{flag}")
        joined = "\n".join(logs)
        return {"ui": {**out["ui"], "text": [joined]}, "result": (images, joined)}


class SceneLog:
    """Read-only view of a Scene Save log; also filled by the frontend when only Scene Save runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"log": ("STRING", {"forceInput": True})},
            "optional": {"text": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Display only; filled after each run"})},
        }

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "SceneLoader"

    # Pass the log to the UI text field.
    def show(self, log, text=""):
        return {"ui": {"text": [log]}}


# Expose {scene: base} for the frontend scene filter.
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/sceneloader/scenes")
    async def sceneloader_scenes(request):
        return web.json_response({"scenes": Library(library_root()).scene_bases()})

    # Expose composer slots for a base, with recommended weight and trigger sets per LoRA.
    @PromptServer.instance.routes.get("/sceneloader/composer/options")
    async def sceneloader_composer_options(request):
        base = request.query.get("base", "")
        if base not in KNOWN_BASES:
            return web.json_response({"error": f"unknown base {base!r}"}, status=400)
        root = lora_root(base)
        slots = slot_options(Library(library_root()), root, base)
        meta = {}
        for opt in slots.values():
            for rel in opt["loras"]:
                m = sidecar(root, base, rel)
                meta[rel] = {"w": (m.get("weight") or {}).get("min", 1.0), "t": m.get("trigger_sets") or [],
                             "x": bool(m.get("extra_positive") or m.get("extra_negative")), "i": lora_info(m)}
        lib = Library(library_root())
        blocks, block_text = {}, {}
        for opt in slots.values():
            for ref in opt["blocks"]:
                try:
                    with open(lib.path("blocks", ref), encoding="utf-8") as fh:
                        block_text[ref] = fh.read()
                    blocks[ref] = block_loras(lib, ref)
                except (SceneError, OSError, ValueError) as e:  # broken block: show it without LoRAs
                    print(f"[SceneComposer] block {ref}: {e}")
                    blocks[ref] = []
                    block_text.setdefault(ref, f"error: {e}")
        return web.json_response({"slots": slots, "loras": meta, "blocks": blocks, "block_text": block_text})

    # Load a scene file (old scene tree or composer preset) as composer rows.
    @PromptServer.instance.routes.get("/sceneloader/composer/scene")
    async def sceneloader_composer_scene_get(request):
        name = request.query.get("name", "")
        if not NAME_RE.match(name):
            return web.json_response({"error": f"bad scene name {name!r}"}, status=400)
        try:
            return web.json_response(load_preset(Library(library_root()), name))
        except (SceneError, OSError, ValueError) as e:
            print(f"[SceneComposer] load scene {name}: {e}")
            return web.json_response({"error": str(e)}, status=400)

    # Save composer rows as library/scenes/<name>.yaml; 409 when it exists and overwrite is false.
    @PromptServer.instance.routes.post("/sceneloader/composer/scene")
    async def sceneloader_composer_scene_post(request):
        try:
            body = await request.json()
            path = save_preset(Library(library_root()), str(body.get("name") or ""), body.get("base"),
                               body.get("rows"), bool(body.get("overwrite")))
        except FileExistsError as e:
            return web.json_response({"error": f"exists: {e}"}, status=409)
        except (SceneError, OSError, ValueError) as e:
            print(f"[SceneComposer] save scene: {e}")
            return web.json_response({"error": str(e)}, status=400)
        print(f"[SceneComposer] saved scene {path}")
        return web.json_response({"path": os.path.relpath(path, library_root())})

    # Full LoRA info (untruncated description) for the composer popup.
    @PromptServer.instance.routes.get("/sceneloader/composer/lora_info")
    async def sceneloader_composer_lora_info(request):
        base, rel = request.query.get("base", ""), request.query.get("file", "").replace("\\", "/")
        if base not in KNOWN_BASES or not rel or os.path.isabs(rel) or ".." in rel.split("/"):
            return web.json_response({"error": "bad base or file"}, status=400)
        return web.json_response({"text": lora_info(sidecar(lora_root(base), base, rel), None)})
except Exception as e:  # server not available (tests, CLI)
    print(f"[SceneLoader] scene filter route not registered: {e}")

NODE_CLASS_MAPPINGS = {"SceneLoader": SceneLoader, "SceneComposer": SceneComposer,
                       "SceneSave": SceneSave, "SceneLog": SceneLog, "SMPLXRiggedGLB": SMPLXRiggedGLB,
                       "SMPLXToOpenPose3D": SMPLXToOpenPose3D, "PoseCamera": PoseCamera}
WEB_DIRECTORY = "./web"
NODE_DISPLAY_NAME_MAPPINGS = {"SceneLoader": "Scene Loader", "SceneComposer": "Scene Composer",
                              "SceneSave": "Scene Save", "SceneLog": "Scene Log",
                              "SMPLXRiggedGLB": "SMPL-X -> Rigged GLB",
                              "SMPLXToOpenPose3D": "SMPL-X -> OpenPose 3D", "PoseCamera": "Pose Camera"}

print(f"[SceneLoader] loaded - {len(NODE_CLASS_MAPPINGS)} nodes, library: {library_root()}")
