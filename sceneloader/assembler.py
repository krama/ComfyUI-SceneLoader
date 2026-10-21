"""Scene assembly: resolve a scene tree of blocks into prompt text and LoRA stacks.

Pure Python (PyYAML only) so it can be tested without ComfyUI.
"""
import json
import os

import yaml

# Prompt position per block kind; earlier tokens get more attention (SD prompt priority).
KIND_RANK = {
    "scene": 0, "quality": 0,
    "character": 1, "part": 1, "pose": 1, "action": 1, "expression": 1,
    "props": 2,
    "background": 3, "weather": 3,
    "light": 4,
    "camera": 5,
    "style": 6,
}
# Order of character children by name or kind: body -> clothing -> pose -> action -> expression.
CHILD_RANK = {"body": 0, "clothing": 1, "pose": 2, "action": 3, "expression": 4}
MAX_DEPTH = 32


class SceneError(ValueError):
    """Raised for invalid scene or block definitions."""


# Load one YAML file and require a mapping at the top level.
def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SceneError(f"{path}: top level must be a mapping")
    return data


# Return the type of a LoRA entry: explicit type, else first folder of its path.
def lora_type(entry):
    if entry.get("type"):
        return str(entry["type"])
    parts = str(entry["file"]).replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "other"


# Normalise a LoRA list from YAML into dicts with file, weight and type.
def norm_loras(items, where):
    out = []
    for i, item in enumerate(items or []):
        if not isinstance(item, dict) or not item.get("file"):
            raise SceneError(f"{where}: loras[{i}] needs a 'file'")
        entry = {"file": str(item["file"]).replace("\\", "/"), "weight": float(item.get("weight", 1.0))}
        entry["type"] = lora_type({**item, **entry})
        out.append(entry)
    return out


# Join non-empty comma-separated tag strings into one clean string.
def join_tags(parts):
    tokens = []
    for p in parts:
        for t in str(p or "").split(","):
            t = t.strip()
            if t:
                tokens.append(t)
    return ", ".join(tokens)


# Booru tags per shot_type from ComfyUI-AdvancedCameraPrompts camera_json.
SHOT_TAGS = {
    "extreme_close_up": "close-up, portrait", "close_up": "close-up, portrait",
    "medium_close_up": "upper body", "medium_shot": "upper body",
    "medium_long_shot": "cowboy shot", "full_shot": "full body",
    "wide_shot": "wide shot, very wide shot", "extreme_wide_shot": "wide shot, very wide shot",
}


# Return the signed number at the end of strings like "tilt down 30.0" or "pan to left 12.5".
def _tail_number(text):
    try:
        return float(str(text).split()[-1])
    except (ValueError, IndexError):
        return 0.0


# Convert AdvancedCameraPrompts camera_json (+ its prompt) into booru camera tags.
def camera_tags_from_json(camera_json, camera_prompt=""):
    try:
        cam = json.loads(camera_json)["camera"]
    except (ValueError, KeyError, TypeError) as e:
        raise SceneError(f"camera_json: cannot parse ({e})")
    tags = [SHOT_TAGS.get(cam.get("shot_type") or "", "")]
    tilt = str(cam.get("tilt_deg") or "")
    tilt_val = _tail_number(tilt)
    if tilt.startswith("tilt down") and tilt_val >= 75:
        tags.append("from above, bird's eye view")
    elif tilt.startswith("tilt down") and tilt_val >= 15:
        tags.append("from above")
    elif tilt.startswith("tilt up") and tilt_val >= 15:
        tags.append("from below")
    # Their yaw is measured from the back view: front camera gives ~180, so angle = 180 - |pan|.
    pan = abs(_tail_number(cam.get("pan_deg")))
    view = 180.0 - pan
    if pan == 0.0 and "Pan the camera" in (camera_prompt or "") and "behind" not in camera_prompt:
        view = 90.0  # yaw collapses to 0 for an exact side view; the prompt text disambiguates
    if tilt_val >= 75:
        return join_tags(tags)  # near-vertical camera: horizontal angle is undefined in their math
    if view > 150:
        tags.append("from behind")
    elif view >= 30:
        tags.append("from side")
    return join_tags(tags)


class Library:
    """Block library rooted at a directory with blocks/ and scenes/ subfolders."""

    def __init__(self, root):
        self.root = os.path.abspath(root)

    # List scene names (paths under scenes/ without extension).
    def scenes(self):
        base = os.path.join(self.root, "scenes")
        out = []
        for d, _, files in os.walk(base):
            for f in files:
                if f.endswith((".yaml", ".yml")):
                    rel = os.path.relpath(os.path.join(d, f), base)
                    out.append(os.path.splitext(rel)[0].replace(os.sep, "/"))
        return sorted(out)

    # Return the newest modification time of any library file, for cache invalidation.
    def fingerprint(self):
        latest = 0.0
        for d, _, files in os.walk(self.root):
            for f in files:
                latest = max(latest, os.path.getmtime(os.path.join(d, f)))
        return latest

    # Return the file path for a scene or block reference.
    def path(self, folder, ref):
        for ext in (".yaml", ".yml"):
            p = os.path.join(self.root, folder, *ref.split("/")) + ext
            if os.path.isfile(p):
                return p
        raise SceneError(f"{folder}/{ref}: file not found under {self.root}")


# Merge a local block onto a library block referenced by `use`.
def merge_block(base, local, notes, where):
    out = dict(base)
    for key in ("name", "kind"):
        if local.get(key):
            out[key] = local[key]
    out["tags"] = join_tags([base.get("tags"), local.get("tags")])
    out["negative"] = join_tags([base.get("negative"), local.get("negative")])
    out["params"] = {**(base.get("params") or {}), **(local.get("params") or {})}
    loras = [dict(x) for x in base.get("loras") or []]
    for item in local.get("loras") or []:
        hit = [x for x in loras if x["file"] == item["file"]]
        if not hit:
            loras.append(dict(item))
        elif hit[0]["type"] == "character":
            notes.append(f"{where}: identity LoRA {item['file']} keeps weight {hit[0]['weight']}")
        else:
            hit[0]["weight"] = item["weight"]
    out["loras"] = loras
    out["children"] = list(base.get("children") or []) + list(local.get("children") or [])
    return out


# Resolve `use` references recursively and normalise every block in the tree.
def resolve(lib, raw, notes, where, stack=()):
    if len(stack) > MAX_DEPTH:
        raise SceneError(f"{where}: nesting deeper than {MAX_DEPTH}")
    if not isinstance(raw, dict):
        raise SceneError(f"{where}: block must be a mapping")
    local = dict(raw)
    local["loras"] = norm_loras(local.get("loras"), where)
    ref = local.pop("use", None)
    if ref:
        if ref in stack:
            raise SceneError(f"{where}: circular use of {ref} ({' -> '.join(stack + (ref,))})")
        base = resolve(lib, load_yaml(lib.path("blocks", ref)), notes, f"blocks/{ref}", stack + (ref,))
        block = merge_block(base, local, notes, where)
        block.setdefault("name", ref.split("/")[-1])
    else:
        block = local
        block["tags"] = join_tags([block.get("tags")])
        block["negative"] = join_tags([block.get("negative")])
        block["params"] = block.get("params") or {}
    block["kind"] = str(block.get("kind") or "part")
    block["name"] = str(block.get("name") or block["kind"])
    kids = []
    for i, child in enumerate(block.get("children") or []):
        kids.append(resolve(lib, child, notes, f"{where}/children[{i}]", stack))
    block["children"] = kids
    return block


# Return children sorted for prompt order; character children follow CHILD_RANK.
def ordered_children(block):
    kids = [c for c in block["children"] if c["kind"] != "detailer"]
    if block["kind"] == "character":
        return sorted(kids, key=lambda c: CHILD_RANK.get(c["name"], CHILD_RANK.get(c["kind"], 99)))
    return sorted(kids, key=lambda c: KIND_RANK.get(c["kind"], 99))


# Walk the tree depth-first in prompt order, yielding (block, source path).
def walk(block, path=""):
    here = f"{path}/{block['name']}" if path else block["name"]
    yield block, here
    for child in ordered_children(block):
        yield from walk(child, here)


# Apply a detailer delta (add, drop, override) to a LoRA stack and return the new stack.
def apply_detailer(stack, delta, notes):
    drop = {str(t) for t in delta.get("drop") or []}
    out = [dict(x) for x in stack if x["type"] not in drop]
    if drop:
        notes.append(f"detailer: drop types {sorted(drop)} -> removed {len(stack) - len(out)}")
    for i, ov in enumerate(delta.get("override") or []):
        match = [x for x in out if (ov.get("file") and x["file"] == ov["file"])
                 or (ov.get("type") and x["type"] == ov["type"])]
        if not match:
            notes.append(f"detailer: override[{i}] matched nothing")
        for x in match:
            x["weight"] = float(ov.get("weight", x["weight"]))
    for item in norm_loras(delta.get("add"), "detailer.add"):
        out.append({**item, "source": "detailer"})
    return out


# Assemble a scene; camera_override (tags) replaces camera block tags. Returns prompts and stacks.
def assemble(lib, scene_name, camera_override=None):
    notes = []
    raw = load_yaml(lib.path("scenes", scene_name))
    raw.setdefault("kind", "scene")
    raw.setdefault("name", scene_name.split("/")[-1])
    root = resolve(lib, raw, notes, f"scenes/{scene_name}")

    pos, neg, loras = [], [], []
    placed = camera_override is None
    for block, src in walk(root):
        if block["kind"] == "camera" and camera_override is not None:
            if not placed and camera_override:
                pos.append((camera_override, f"{src} <- 3d camera"))
            placed = True
            notes.append(f"{src}: camera tags replaced by 3d camera")
        elif block["kind"] == "style" and not placed:
            if camera_override:
                pos.append((camera_override, "3d camera"))
            placed = True
        if block["tags"] and not (block["kind"] == "camera" and camera_override is not None):
            pos.append((block["tags"], src))
        if block["negative"]:
            neg.append(block["negative"])
        for item in block["loras"]:
            if any(x["file"] == item["file"] for x in loras):
                notes.append(f"{src}: duplicate LoRA {item['file']} ignored")
                continue
            loras.append({**item, "source": src})
    if not placed and camera_override:
        pos.append((camera_override, "3d camera"))
    # Style LoRAs change global model weights, so they always go last.
    loras = [x for x in loras if x["type"] != "style"] + [x for x in loras if x["type"] == "style"]

    deltas = [c for c in root["children"] if c["kind"] == "detailer"]
    det = loras
    for d in deltas:
        det = apply_detailer(det, d, notes)
    return {
        "positive": join_tags(t for t, _ in pos),
        "positive_parts": pos,
        "negative": join_tags(neg),
        "loras": loras,
        "loras_detailer": det,
        "notes": notes,
    }


# Serialise a LoRA list to the SimplePowerLora lora_stack JSON with the base folder prefix.
def to_lora_stack(loras, base):
    items = []
    for x in loras:
        name = x["file"] if not base or x["file"].startswith(base + "/") else f"{base}/{x['file']}"
        items.append({"name": name, "weight": x["weight"], "enabled": True, "tags": ""})
    return json.dumps(items, ensure_ascii=False)


# Build a human-readable assembly report; resolver(name) returns a path or None.
def report(result, base, resolver=None):
    lines = ["# prompt order"]
    lines += [f"  [{src}] {tags}" for tags, src in result["positive_parts"]]
    for title, key in (("lora stack", "loras"), ("lora stack (detailer)", "loras_detailer")):
        lines.append(f"# {title}")
        for x in result[key]:
            name = x["file"] if x["file"].startswith(base + "/") else f"{base}/{x['file']}"
            missing = resolver is not None and resolver(name) is None
            lines.append(f"  [{x['source']}] {name} @{x['weight']} ({x['type']})" + ("  MISSING" if missing else ""))
    if result["notes"]:
        lines.append("# notes")
        lines += [f"  {n}" for n in result["notes"]]
    return "\n".join(lines)
