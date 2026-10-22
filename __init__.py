"""Scene Loader - builds a scene from structured blocks (character, background, light) with LoRAs."""
import json
import os

import folder_paths
import torch
from nodes import SaveImage

from .sceneloader.prompt_log import build_log
from .sceneloader.composer import block_loras, compose, lora_info, sidecar, slot_options
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
        }, "optional": SceneLoader.INPUT_TYPES()["optional"]}

    RETURN_TYPES = SceneLoader.RETURN_TYPES
    RETURN_NAMES = SceneLoader.RETURN_NAMES
    FUNCTION = "compose"
    CATEGORY = "SceneLoader"

    # Re-run when any library file changes on disk.
    @classmethod
    def IS_CHANGED(cls, base, composition, camera_json=None, camera_prompt=None):
        return Library(library_root()).fingerprint()

    # Assemble the rows into prompts and SimplePowerLora stacks.
    def compose(self, base, composition, camera_json=None, camera_prompt=None):
        lib = Library(library_root())
        try:
            rows = json.loads(composition or "[]")
            if not isinstance(rows, list):
                raise SceneError("composition must be a JSON list of rows")
            camera = SceneLoader.camera_override(base, camera_json, camera_prompt)
            result = compose(lib, lora_root(base), base, rows, camera)
        except (SceneError, ValueError) as e:
            print(f"[SceneComposer] {base}: {e}")
            raise SceneError(str(e))
        resolver = lambda name: folder_paths.get_full_path("loras", name)  # noqa: E731
        info = report(result, base, resolver)
        print(f"[SceneComposer] {base}: {len(rows)} rows, {len(result['loras'])} loras, "
              f"{len(result['loras_detailer'])} detailer loras, {info.count('  MISSING')} missing")
        return (result["positive"], result["negative"], to_lora_stack(result["loras"], base),
                to_lora_stack(result["loras_detailer"], base), info)


# Return the configured loras folder that holds the <base>/ subfolder (first one as fallback).
def lora_root(base):
    paths = folder_paths.get_folder_paths("loras")
    hit = next((p for p in paths if os.path.isdir(os.path.join(p, base))), None)
    if hit is None:
        print(f"[SceneComposer] no loras folder contains '{base}/': {paths}")
    return hit or paths[0]


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
except Exception as e:  # server not available (tests, CLI)
    print(f"[SceneLoader] scene filter route not registered: {e}")

NODE_CLASS_MAPPINGS = {"SceneLoader": SceneLoader, "SceneComposer": SceneComposer,
                       "SceneSave": SceneSave, "SceneLog": SceneLog}
WEB_DIRECTORY = "./web"
NODE_DISPLAY_NAME_MAPPINGS = {"SceneLoader": "Scene Loader", "SceneComposer": "Scene Composer",
                              "SceneSave": "Scene Save", "SceneLog": "Scene Log"}

print(f"[SceneLoader] loaded - {len(NODE_CLASS_MAPPINGS)} nodes, library: {library_root()}")
