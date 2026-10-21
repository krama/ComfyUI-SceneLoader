"""Build a human-readable generation log from a ComfyUI API prompt. Pure Python, no ComfyUI imports."""
import datetime

# Loader classes and the input that names the loaded file.
MODEL_LOADERS = {"SimpleCheckpointLoaderPlus": "ckpt", "CheckpointLoaderSimple": "ckpt_name",
                 "UNETLoader": "unet_name"}
SAMPLERS = ("KSampler", "KSamplerAdvanced", "FaceDetailer")


# Return nodes of the given classes as (node_id, node) sorted by numeric id.
def _nodes(prompt, classes):
    out = [(k, v) for k, v in (prompt or {}).items() if v.get("class_type") in classes]
    return sorted(out, key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)


# Resolve an input value; a link [node_id, slot] is followed to a seed/value widget of its source.
def _value(prompt, value):
    if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
        src = prompt[str(value[0])].get("inputs", {})
        for key in ("seed", "value", "noise_seed"):
            if key in src and not isinstance(src[key], list):
                return src[key]
        return "linked"
    return value


# Return the node display title from API _meta, else its class name.
def _title(node):
    return (node.get("_meta") or {}).get("title") or node.get("class_type", "?")


# Return a list of "label: value" lines describing scene, models and samplers in the prompt.
def describe_prompt(prompt):
    prompt = prompt or {}
    lines = []
    for _, node in _nodes(prompt, ("SceneLoader",)):
        inp = node.get("inputs", {})
        camera = " · 3d camera" if isinstance(inp.get("camera_json"), list) else ""
        lines.append(f"scene: {inp.get('scene')} ({inp.get('base')}){camera}")
    for _, node in _nodes(prompt, tuple(MODEL_LOADERS)):
        lines.append(f"model: {node['inputs'].get(MODEL_LOADERS[node['class_type']])}")
    for _, node in _nodes(prompt, ("CLIPLoader",)):
        lines.append(f"text encoder: {node['inputs'].get('clip_name')}")
    for _, node in _nodes(prompt, ("VAELoader",)):
        lines.append(f"vae: {node['inputs'].get('vae_name')}")
    for _, node in _nodes(prompt, ("CLIPSetLastLayer",)):
        lines.append(f"clip skip: {_value(prompt, node['inputs'].get('stop_at_clip_layer'))}")
    for _, node in _nodes(prompt, SAMPLERS):
        i = node.get("inputs", {})
        lines.append(f"sampler [{_title(node)}]: seed {_value(prompt, i.get('seed', i.get('noise_seed')))}, "
                     f"{i.get('sampler_name')}/{i.get('scheduler')}, {i.get('steps')} steps, "
                     f"cfg {i.get('cfg')}, denoise {i.get('denoise', 1.0)}")
    return lines


# Return warning lines: broken image (NaN/all black) and clip skip -1 (last CLIP layer).
def warnings(prompt, has_nan=False, all_black=False):
    out = []
    if has_nan or all_black:
        what = "contains NaN" if has_nan else "is completely black"
        out.append(f"WARNING: image {what} - the model produced NaN; check clip skip, then unload models")
    for _, node in _nodes(prompt or {}, ("CLIPSetLastLayer",)):
        if _value(prompt, node.get("inputs", {}).get("stop_at_clip_layer")) == -1:
            out.append("WARNING: clip skip -1 uses the last CLIP layer; SDXL/Illustrious mixes can return NaN "
                       "(black images) - use -2")
    return out


# Build the full log text for one saved image.
def build_log(filename, width, height, prompt, positive="", negative="", info="", now=None,
              has_nan=False, all_black=False):
    now = now or datetime.datetime.now()
    parts = warnings(prompt, has_nan, all_black)
    parts += [f"image: {filename}", f"date: {now:%Y-%m-%d %H:%M:%S}", f"size: {width}x{height}"]
    parts += describe_prompt(prompt)
    parts += ["", "positive:", positive or "(not connected)", "", "negative:", negative or "(not connected)"]
    if info:
        parts += ["", "scene assembly:", info]
    return "\n".join(parts) + "\n"
