# ComfyUI-SceneLoader

Scene preparation nodes for ComfyUI: a scene is a tree of blocks
(character, background, light, camera, style), each carrying its own prompt tags and LoRAs.

## Node: Scene Loader

Inputs: `scene` (file from `scenes/`), `base` (LoRA base folder: `illustrious` or `anima`).

| Output | Connect to |
| --- | --- |
| `positive` | Prompt Merge Edit `prompt` |
| `negative` | CLIP Text Encode (negative) `text` |
| `lora_stack` | Simple Power Lora `lora_stack` |
| `lora_stack_detailer` | Simple Power Lora in the detailer graph |
| `info` | any text display: prompt order and LoRA sources |

## Library

Looked up in order: `SCENELOADER_LIBRARY` env, `./library` (git-ignored, may be a symlink),
`./examples`. Layout: `blocks/` (reusable blocks), `scenes/` (scenes referencing blocks).

Block format:

```yaml
name: anna
kind: character        # scene|quality|character|part|pose|action|expression|props|background|light|camera|style|detailer
tags: "1girl, red hair"
negative: "short hair"
loras:
  - {file: "character/anna.safetensors", weight: 0.9}   # type = first folder unless set
children: []
```

A block may reference a library block with `use: characters/anna` and add tags or LoRAs locally.
Identity (`character`) LoRA weights cannot be changed by the scene. Style LoRAs go last.
A `kind: detailer` child of the scene sets the detailer delta: `add`, `drop` (types), `override`.

## Tests

`python tests/test_assembler.py`

## Tools

`tools/` holds model library maintenance scripts (dedupe, versions, quarantine, relocate,
LoRA classification). They are not loaded by ComfyUI.
