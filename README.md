# ComfyUI-SceneLoader

Scene preparation nodes for ComfyUI: a scene is a tree of blocks
(character, clothing, pose, background, light, camera, style), each carrying its own prompt tags
and LoRAs. A pose can come from a sample image: it is lifted to a 3D body, and a virtual camera
around that body gives the ControlNet maps and the camera tags of the shot.

## Nodes

| Node | What it does |
| --- | --- |
| Scene Composer | Builds the scene from slot rows (library blocks, LoRAs, tags) edited in the node |
| Scene Loader | Same output from a scene file (`scenes/`); older, kept for existing graphs |
| Pose Camera | Virtual camera around an SMPL-X pose: OpenPose and depth maps, camera tags, latent |
| Pose Camera Crop | Cuts the frame out of an overscan render (see Pose Camera) |
| SMPL-X -> OpenPose 3D | SMPL-X pose as OpenPose keypoints with 3D data for 3D OpenPose editors |
| SMPL-X -> Rigged GLB | SMPL-X pose as a skinned glTF (55 joints) for Blender / Cascadeur |
| Scene Save / Scene Log | SaveImage plus a generation log (final prompt and settings) |

## Scene Composer

Inputs: `base` (LoRA base folder: `illustrious` or `anima`), `composition` (rows, edited in the
node UI), `pose_camera` (on in graphs with Pose Camera), optional `camera` (tags, or Advanced
Camera Control `camera_json`; replaces the camera rows).

Rows: one per slot (quality, character, clothing, pose, nsfw, background, lighting, camera,
detail, ...). A row uses a library block, a LoRA from `loras/<base>/<slot>/` with weight and
trigger words, or plain tags. Presets save and load whole compositions. A detailer section sets
the LoRA stack for the detailer pass. Duplicate tags are removed; the ⓘ button shows the LoRA's
Civitai description and usage tips (`tools/fetch_civitai_meta.py`).

| Output | Connect to |
| --- | --- |
| `positive`, `negative` | prompt text (Prompt Merge Edit / CLIP Text Encode) |
| `lora_stack`, `lora_stack_detailer` | Simple Power Lora |
| `info` | text display: prompt order, LoRA sources, notes |
| `pose_image` | sample image of the pose block -> NLF (3D pose) |
| `has_pose_image` | If/Else switches around the 3D pose branch |
| `pose_skeleton` | OpenPose skeleton of the pose block, a ready 2D ControlNet map |
| `use_controlnet` | If/Else switches around the pose ControlNet |

A pose without a sample image or skeleton blocks `pose_image` / `pose_skeleton` silently, so the
nodes using them are skipped instead of failing. With `pose_camera` on and a pose that has a
sample image, camera rows (tags and LoRAs) and the `camera` input are left out: Pose Camera sets
the camera.

## Pose Camera

Needs [ComfyUI-SMPLx-Estimator](https://github.com/ameliacode/ComfyUI-SMPLx-Estimator)
(NLF model, SMPL-X model files) and `smplx`.

Chain: Scene Composer `pose_image` -> Image Scale To Total Pixels (~1 MP) -> Load SMPL-X /
Load NLF -> NLF SMPL-X Estimator -> Pose Camera.

- **Camera:** `framing` (full body / cowboy shot / upper body / portrait: what must fit), `yaw`
  (around the body, 0 = like the sample), `elevation` (+ from above), `fov`, `zoom`, `shift_x/y`.
  The 3D view in the node shows the body and the frame: drag = yaw / elevation, wheel = zoom,
  shift+drag = shift; ⤢ opens a large view. The sliders follow the mouse.
- **Two runs, no pause:** with `stop_on_new_pose` on, the first run with a new pose stops at
  Pose Camera (nothing after it runs); set the camera, run again to render.
- **Tags:** shot tags follow the joints inside the frame (full body, cowboy shot + feet out of
  frame, upper body, portrait, head out of frame) plus view tags (from side / behind / above /
  below); the camera tags already in the prompt are replaced.
- **Overscan:** when the body leaves the frame, the maps and the empty latent cover a larger
  canvas with the whole body (at most 2x per side, `overscan` x frame pixels), so cut-off parts
  follow the pose; Pose Camera Crop after VAE Decode cuts the frame out and scales it back.

| Output | Connect to |
| --- | --- |
| `pose_map` | ControlNet OpenPose image |
| `positive` | prompt text (Composer positive with the camera tags) |
| `camera_tags` | text display |
| `depth_map` | depth ControlNet image (body depth from the same camera) |
| `latent` | KSampler `latent_image` |
| `crop` | Pose Camera Crop |

## Library

Looked up in order: `SCENELOADER_LIBRARY` env, `./library` (git-ignored, may be a symlink),
`./examples`. Layout: `blocks/` (reusable blocks, subfolders allowed), `scenes/` (scenes and
composer presets).

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

Pose blocks may also carry `image:` (sample photo / render -> 3D pose) and `skeleton:`
(OpenPose map), files next to the block.

A block may reference a library block with `use: characters/anna` and add tags or LoRAs locally.
Identity (`character`) LoRA weights cannot be changed by the scene. Style LoRAs go last.
A `kind: detailer` child of the scene sets the detailer delta: `add`, `drop` (types), `override`.

## Tools

Run with the ComfyUI venv python; none of them is loaded by ComfyUI.

| Tool | Purpose |
| --- | --- |
| `import_pose_inbox.py` | Pose images / zips / folders (Civitai downloads) -> pose blocks; photos and skeletons paired by name; `--dry-run`, `--skip` |
| `import_pose_pack.py` | Pose card pack (`POSES.yaml` + images) -> pose blocks |
| `tag_poses.py` | Pose tags for pose blocks from their images with a WD14 tagger (ONNX); keeps only pose / gesture / viewpoint tags |
| `skeleton_to_figure.py` | Skeleton-only pose blocks get a figure image via a running ComfyUI (ControlNet + NLF check); `--only`, `--limit` |
| `fetch_civitai_meta.py` | Civitai metadata (weights, triggers, usage tips) next to each LoRA |
| `classify_loras.py`, `relocate.py` | Propose / apply a LoRA folder layout (plan CSV) |
| `dedupe_scan.py`, `versions_scan.py`, `quarantine.py` | Find duplicates and old versions, move them to a quarantine folder |
| `make_mannequin.py` | Box mannequin (.obj) for Load 3D framing |

`tools/patches/` holds small local patches for third-party nodes (SMPLx-Estimator resizable
viewer, 3D OpenPose editor 3D passthrough); apply with `git apply` in the node's folder.

## Tests

`python tests/test_<name>.py` for each file in `tests/`. The SMPL-X tests need the model files:
pass their folder as the first argument or set `SMPLX_MODELS`; `test_pose_camera.py` also
checks the browser view math under `node` when it is installed.
