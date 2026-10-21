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

## 3D camera (optional)

Connect ComfyUI-AdvancedCameraPrompts outputs to Scene Loader `camera_json` and `camera_prompt`
(Load 3D -> Advanced Camera Control Prompt Generator). When connected, they replace the scene
camera block: `anima` uses the natural-language prompt, `illustrious` gets booru tags
(shot type -> close-up/upper body/cowboy shot/full body/wide shot, tilt -> from above/below,
pan -> from side/from behind).

`tools/make_mannequin.py --out <ComfyUI input>/3d/mannequin.obj` writes a box figure facing +Z
for framing (height and proportions are arguments).

## Tests

`python tests/test_assembler.py`

## Tools

`tools/` holds model library maintenance scripts (dedupe, versions, quarantine, relocate,
LoRA classification). They are not loaded by ComfyUI.
