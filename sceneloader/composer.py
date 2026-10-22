"""Scene Composer: turn slot rows (library blocks and LoRAs) into a scene and assemble it.

Rows are dicts: {"slot", "use", "weights"?: {lora file: weight}} for a block,
{"slot", "lora", "weight"?, "trigger"?, "extras"?} for a LoRA, {"slot", "tags", "negative"?} for
plain tags, {"slot", "none": true} to skip, {"slot": "detailer", "drop"/"add"/"override"} for the
detailer delta, or a row with nothing chosen (an error at run time).
"""
import json
import os
import re

import yaml

from .assembler import SceneError, assemble_raw, join_tags, load_yaml, resolve, scene_base

# Slot -> (prompt kind, block folder under library/blocks or None). Order is the UI order.
SLOTS = {
    "quality": ("quality", "quality"),
    "character": ("character", "characters"),
    "body": ("body", None),
    "clothing": ("clothing", "clothing"),
    "expression": ("expression", None),
    "pose": ("pose", None),
    "action": ("action", None),
    "concept": ("concept", None),
    "nsfw": ("nsfw", None),
    "background": ("background", "backgrounds"),
    "lighting": ("light", "lights"),
    "camera": ("camera", "cameras"),
    "style": ("style", None),
    "detail": ("detail", None),
    "slider": ("slider", None),
}
# LoRA types that change the model only; their trigger words are not added to the prompt.
STACK_ONLY = {"detail", "slider"}
LORA_EXT = (".safetensors", ".ckpt", ".pt")
SIDECAR = ".sceneloader.json"
# Camera slot value that takes tags from the camera_json/camera_prompt inputs (3D camera).
CAMERA_INPUT = "3d camera"


# Return LoRA files (relative to loras/<base>, "/" separated) grouped by top folder.
def lora_files(loras_dir, base):
    root = os.path.join(loras_dir, base)
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(LORA_EXT):
                rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                slot = rel.split("/")[0] if "/" in rel else "other"
                out.setdefault(slot, []).append(rel)
    return {k: sorted(v, key=str.lower) for k, v in out.items()}


# Return {slot: {"kind", "blocks": [...], "loras": [...]}} in prompt order; unknown folders follow.
def slot_options(lib, loras_dir, base):
    loras = lora_files(loras_dir, base)
    out = {}
    for slot in list(SLOTS) + sorted(set(loras) - set(SLOTS)):
        kind, folder = SLOTS.get(slot, ("concept", None))
        blocks = []
        if folder:
            d = os.path.join(lib.root, "blocks", folder)
            if os.path.isdir(d):
                blocks = sorted(f"{folder}/{os.path.splitext(f)[0]}" for f in os.listdir(d)
                                if f.endswith((".yaml", ".yml")))
        if slot == "quality":
            blocks = [b for b in blocks if b.split("/")[1] in (base, "default")]
        if slot == "camera":
            blocks = [CAMERA_INPUT] + blocks
        out[slot] = {"kind": kind, "blocks": blocks, "loras": loras.get(slot, [])}
    return out


# Return LoRAs a library block brings (including nested blocks): [{"file", "weight", "type"}].
def block_loras(lib, ref):
    block = resolve(lib, {"use": ref}, [], f"blocks/{ref}")
    out, stack = [], [block]
    while stack:
        b = stack.pop(0)
        out += [{"file": x["file"], "weight": x["weight"], "type": x["type"]} for x in b["loras"]]
        stack += b["children"]
    return out


# Return a human-readable LoRA sidecar summary; description cut to `limit` chars (None = full).
def lora_info(meta, limit=300):
    if not meta:
        return "no metadata (.sceneloader.json): weight 1.0, no trigger words"
    w = meta.get("weight") or {}
    civ = meta.get("civitai") or {}
    lines = [f"{meta.get('name') or '?'} | {meta.get('version') or '?'} | {meta.get('base') or '?'}",
             f"weight: {w.get('min', 1.0)}-{w.get('max', 1.0)} (source: {w.get('source', 'default')})"]
    tips = {k: v for k, v in (meta.get("usage_tips") or {}).items() if v is not None}
    if tips:
        lines.append("civitai usage tips: " + ", ".join(f"{k}={v}" for k, v in tips.items()))
    for i, t in enumerate(meta.get("trigger_sets") or []):
        lines.append(f"trigger set {i + 1}: {t}")
    for key in ("extra_positive", "extra_negative"):
        if meta.get(key):
            lines.append(f"{key}: {meta[key]}")
    if civ.get("modelId"):
        lines.append(f"https://civitai.com/models/{civ['modelId']}?modelVersionId={civ.get('versionId')}")
    desc = str(meta.get("description") or "").strip()
    if desc and limit:
        desc = " ".join(desc.split())
        desc = desc[:limit] + ("..." if len(desc) > limit else "")
    if desc:
        lines.append(desc)
    return "\n".join(lines)


# Read <lora>.sceneloader.json; missing or broken sidecar gives {}.
def sidecar(loras_dir, base, rel):
    path = os.path.join(loras_dir, base, os.path.splitext(rel)[0] + SIDECAR)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# Return (weight, trigger tags, extra positive, extra negative) for a LoRA row.
def lora_defaults(meta, row):
    weight = row.get("weight")
    if weight is None:
        weight = (meta.get("weight") or {}).get("min", 1.0)  # lower bound of the recommended range
    sets = meta.get("trigger_sets") or []
    trig = row.get("trigger", 0)
    if trig == "all":
        tags = join_tags(sets)
    elif trig == "none" or not sets:
        tags = ""
    else:
        tags = sets[min(int(trig), len(sets) - 1)]
    pos, neg = ("", "")
    if row.get("extras"):
        pos, neg = meta.get("extra_positive") or "", meta.get("extra_negative") or ""
    return float(weight), tags, pos, neg


# Convert composer rows into a scene mapping (the scene file format).
def rows_to_scene(rows, loras_dir, base, notes, camera=None):
    children = []
    # A "3d camera" row replaces every other camera-slot row (their tags and LoRAs).
    cam3d = any(r.get("slot") == "camera" and r.get("use") == CAMERA_INPUT and not r.get("none") for r in rows)
    for i, row in enumerate(rows):
        slot = str(row.get("slot") or "")
        kind = SLOTS.get(slot, ("concept", None))[0]
        where = f"row {i + 1} ({slot})"
        if row.get("none"):
            continue
        if cam3d and slot == "camera" and row.get("use") != CAMERA_INPUT:
            what = row.get("use") or row.get("lora") or row.get("tags") or "empty row"
            notes.append(f"{where}: {what} replaced by {CAMERA_INPUT}")
            continue
        if slot == "detailer":
            delta = {k: row[k] for k in ("drop", "add", "override") if row.get(k)}
            if delta.get("add"):
                empty = [a for a in delta["add"] if not (isinstance(a, dict) and a.get("file"))]
                if empty:
                    notes.append(f"{where}: {len(empty)} detailer LoRA row(s) without a file ignored")
                delta["add"] = [a for a in delta["add"] if a not in empty]
            children.append({"kind": "detailer", "name": "detailer", **delta})
        elif row.get("use") == CAMERA_INPUT:
            if camera is None:
                raise SceneError(f"{where}: '{CAMERA_INPUT}' needs camera_json or camera_prompt connected")
            children.append({"kind": kind, "name": CAMERA_INPUT, "tags": camera})
            notes.append(f"{where}: tags from 3d camera input")
        elif row.get("use"):
            # "weights" overrides the strength of LoRAs inside the block (identity LoRAs stay fixed).
            loras = [{"file": f, "weight": float(w)} for f, w in (row.get("weights") or {}).items()]
            children.append({"use": row["use"], "kind": kind, "tags": row.get("tags", ""), "loras": loras})
        elif row.get("lora"):
            rel = str(row["lora"]).replace("\\", "/")
            if not os.path.isfile(os.path.join(loras_dir, base, rel)):
                raise SceneError(f"{where}: LoRA not found: {base}/{rel}")
            weight, tags, pos, neg = lora_defaults(sidecar(loras_dir, base, rel), row)
            if slot in STACK_ONLY:
                tags, pos = "", ""
            children.append({"kind": kind, "name": os.path.splitext(os.path.basename(rel))[0],
                             "tags": join_tags([tags, pos, row.get("tags")]), "negative": neg,
                             "loras": [{"file": rel, "weight": weight}]})
        elif row.get("tags"):
            children.append({"kind": kind, "name": slot, "tags": row["tags"], "negative": row.get("negative", "")})
        else:
            raise SceneError(f"{where}: not chosen (pick a block or LoRA, type tags, or set none)")
    return {"kind": "scene", "name": "composer", "children": children}


FOLDER_SLOT = {folder: slot for slot, (_, folder) in SLOTS.items() if folder}
KIND_SLOT = {kind: slot for slot, (kind, _) in SLOTS.items()}
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+(/[A-Za-z0-9_\-]+)*$")


# Convert a scene-file tree into composer rows (nested children are flattened); returns (rows, warnings).
def scene_to_rows(raw):
    rows, warnings = [], []

    def walk(children, where):
        for i, c in enumerate(children or []):
            here = f"{where}[{i}]"
            if not isinstance(c, dict):
                warnings.append(f"{here}: not a mapping, skipped")
                continue
            if c.get("kind") == "detailer":
                rows.append({"slot": "detailer", **{k: c[k] for k in ("drop", "add", "override") if c.get(k)}})
                continue
            loras = [x for x in c.get("loras") or [] if isinstance(x, dict) and x.get("file")]
            if c.get("use"):
                ref = str(c["use"])
                row = {"slot": KIND_SLOT.get(c.get("kind"), FOLDER_SLOT.get(ref.split("/")[0], "concept")), "use": ref}
                if c.get("tags"):
                    row["tags"] = c["tags"]
                if loras:
                    row["weights"] = {x["file"]: float(x.get("weight", 1.0)) for x in loras}
                rows.append(row)
            else:
                slot = KIND_SLOT.get(c.get("kind"), "concept")
                if c.get("tags") or c.get("negative"):
                    rows.append({"slot": slot, "tags": c.get("tags") or "", "negative": c.get("negative") or ""})
                for x in loras:  # inline LoRAs keep the scene weight and add no trigger words
                    rows.append({"slot": slot, "lora": x["file"], "weight": float(x.get("weight", 1.0)),
                                 "trigger": "none"})
                if not (c.get("tags") or c.get("negative") or loras or c.get("children")):
                    warnings.append(f"{here}: empty block skipped")
            walk(c.get("children"), f"{here}.children")

    walk(raw.get("children"), "children")
    return rows, warnings


# Load a scene file as composer rows: {"base", "rows", "warnings"}.
def load_preset(lib, name):
    raw = load_yaml(lib.path("scenes", name))
    if isinstance(raw.get("composer"), list):
        return {"base": raw.get("base"), "rows": raw["composer"], "warnings": []}
    rows, warnings = scene_to_rows(raw)
    return {"base": scene_base(lib, name), "rows": rows, "warnings": warnings}


# Save composer rows as library/scenes/<name>.yaml; returns the path. Refuses to overwrite unless asked.
def save_preset(lib, name, base, rows, overwrite=False):
    if not NAME_RE.match(name or ""):
        raise SceneError(f"bad scene name {name!r}: use letters, digits, _ - and / for subfolders")
    if not isinstance(rows, list):
        raise SceneError("rows must be a list")
    path = os.path.join(lib.root, "scenes", *name.split("/")) + ".yaml"
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"name": name.split("/")[-1], "base": base, "composer": rows}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Scene Composer preset: open it in the Scene Composer node.\n")
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, width=120)
    return path


# Assemble composer rows; camera tags are used only by a "3d camera" row. Same result as assemble.
def compose(lib, loras_dir, base, rows, camera=None):
    notes = []
    scene = rows_to_scene(rows, loras_dir, base, notes, camera)
    if camera is not None and not any(r.get("use") == CAMERA_INPUT for r in rows):
        notes.append(f"camera input connected but unused: set a camera row to '{CAMERA_INPUT}'")
    return assemble_raw(lib, scene, "composer", None, notes)
