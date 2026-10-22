"""Scene Composer: turn slot rows (library blocks and LoRAs) into a scene and assemble it.

Rows are dicts: {"slot", "use", "weights"?: {lora file: weight}} for a block,
{"slot", "lora", "weight"?, "trigger"?, "extras"?} for a LoRA, {"slot", "none": true} to skip,
or a row with neither "use" nor "lora" (not chosen).
"""
import json
import os

from .assembler import SceneError, assemble_raw, join_tags, resolve

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
def rows_to_scene(rows, loras_dir, base, notes):
    children = []
    for i, row in enumerate(rows):
        slot = str(row.get("slot") or "")
        kind = SLOTS.get(slot, ("concept", None))[0]
        where = f"row {i + 1} ({slot})"
        if row.get("none"):
            continue
        if slot == "detailer":
            children.append({"kind": "detailer", "name": "detailer",
                             **{k: row[k] for k in ("drop", "add", "override") if row.get(k)}})
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
        else:
            raise SceneError(f"{where}: not chosen (pick a block or LoRA, or set none)")
    return {"kind": "scene", "name": "composer", "children": children}


# Assemble composer rows; same result shape as assembler.assemble.
def compose(lib, loras_dir, base, rows, camera_override=None):
    notes = []
    scene = rows_to_scene(rows, loras_dir, base, notes)
    return assemble_raw(lib, scene, "composer", camera_override, notes)
