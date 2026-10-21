"""Scene Loader - builds a scene from structured blocks (character, background, light) with LoRAs."""
import os

import folder_paths
import torch
from nodes import SaveImage

from .sceneloader.prompt_log import build_log
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
except Exception as e:  # server not available (tests, CLI)
    print(f"[SceneLoader] scene filter route not registered: {e}")

NODE_CLASS_MAPPINGS = {"SceneLoader": SceneLoader, "SceneSave": SceneSave, "SceneLog": SceneLog}
WEB_DIRECTORY = "./web"
NODE_DISPLAY_NAME_MAPPINGS = {"SceneLoader": "Scene Loader", "SceneSave": "Scene Save", "SceneLog": "Scene Log"}

print(f"[SceneLoader] loaded - {len(NODE_CLASS_MAPPINGS)} nodes, library: {library_root()}")
