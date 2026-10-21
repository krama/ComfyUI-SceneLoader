"""Scene Loader - builds a scene from structured blocks (character, background, light) with LoRAs."""
import os

import folder_paths

from .sceneloader.assembler import (Library, SceneError, assemble, camera_tags_from_json, report,
                                    to_lora_stack)

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
            "base": (["illustrious", "anima"], {"tooltip": "LoRA base folder under loras/"}),
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


NODE_CLASS_MAPPINGS = {"SceneLoader": SceneLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"SceneLoader": "Scene Loader"}

print(f"[SceneLoader] loaded - {len(NODE_CLASS_MAPPINGS)} nodes, library: {library_root()}")
