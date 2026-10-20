# ComfyUI-SceneLoader

Scene preparation nodes for ComfyUI: a scene is a tree of entities
(character, background, light, ...), each carrying its own prompt parts and LoRAs.

## Layout

- `__init__.py` — node registration
- `blocks/` — reusable scene parts (poses, actions, weather, backgrounds, characters)
- `scenes/` — composed scenes
- `templates/` — text templates mixed via selectors
- `tools/` — maintenance scripts for the model library (not loaded by ComfyUI)

## Install

Symlink the repo into ComfyUI `custom_nodes/` and restart ComfyUI.
