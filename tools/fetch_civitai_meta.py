#!/usr/bin/env python3
"""Fetch Civitai model metadata for LoRAs and write <lora>.sceneloader.json next to each file.

Reads modelId/versionId from <lora>.civitai.info, queries the public Civitai API once per model,
and records name, tags, trigger sets, author's optional positive/negative tags and a recommended
weight range. Weight source priority: version description > model description > author's example
images (exact LoRA match) > default 1.0. Existing sidecars are skipped unless --force.
"""
import argparse
import csv
import datetime
import glob
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://civitai.com/api/v1/models/{}"
SIDECAR = ".sceneloader.json"
BASE_WORDS = {"illustrious": ("illustrious", "noob", "ilxl", "ill"), "anima": ("anima",),
              "pony": ("pony",), "sdxl": ("sdxl",), "flux": ("flux",), "sd15": ("sd1.5", "sd 1.5", "sd15")}
WEIGHT_RE = re.compile(
    r"(?:weighting|weight|strength|str|lora\s*(?:scale|weight)|权重|强度)\s*(?:is|of|at|around|between|from|[:=~])?\s*"
    r"(\d(?:\.\d+)?)(?:\s*(?:-|–|~|to|and|/)\s*(\d(?:\.\d+)?))?(?!\s*(?:steps|%|px))", re.I)
LORA_TAG_RE = re.compile(r"<lora:[^:>]+:(\d(?:\.\d+)?)>", re.I)
# "0.05 - 0.4 as optimal range", "recommended range 0.6-0.8", "sweet spot 0.7".
RANGE_RE = re.compile(
    r"(\d(?:\.\d+)?)\s*(?:-|–|~|to)\s*(\d(?:\.\d+)?)\s*(?:is\s*|as\s*(?:the\s*)?|are\s*)?(?:optimal|recommended|ideal|best|sweet\s*spot)"
    r"|(?:optimal|recommended|ideal|best|sweet\s*spot)\s*(?:range|weight|value|strength)?\s*(?:is|:|of|at)?\s*"
    r"(\d(?:\.\d+)?)(?:\s*(?:-|–|~|to)\s*(\d(?:\.\d+)?))?", re.I)
NOT_WEIGHT_RE = re.compile(r"\b(cfg|steps?|denoise|denoising|clip\s*skip|hires|upscale|seed)\b", re.I)


def log(msg):
    print(f"[civitai-meta] {msg}", flush=True)


# Strip HTML to plain text.
def plain(text):
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text or "", flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"[ \t]+", " ", text).strip()


# Return the base key ("illustrious", "anima", ...) for a Civitai baseModel string.
def base_key(base_model):
    b = (base_model or "").lower()
    for key, words in BASE_WORDS.items():
        if any(w in b for w in words):
            return key
    return b


# Return (min, max) weights stated in text for our base, or None; sentences about other bases are skipped.
def weights_from_text(text, our_base):
    values = []
    for sentence in re.split(r"(?<=[.!?\n])\s+", text or ""):
        low = sentence.lower()
        other = [k for k, words in BASE_WORDS.items() if k != our_base and any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)]
        ours = any(re.search(rf"\b{re.escape(w)}\b", low) for w in BASE_WORDS.get(our_base, ()))
        if other and not ours:
            continue
        for m in WEIGHT_RE.finditer(sentence):
            window = sentence[m.start():m.end() + 8]
            if NOT_WEIGHT_RE.search(window):
                continue
            values += [float(x) for x in m.groups() if x]
        for m in RANGE_RE.finditer(sentence):
            if not NOT_WEIGHT_RE.search(sentence[m.start():m.end() + 8]):
                values += [float(x) for x in m.groups() if x]
        values += [float(x) for x in LORA_TAG_RE.findall(sentence)]
    values = [v for v in values if 0.05 <= v <= 2.0]
    return (min(values), max(values)) if values else None


# Return (min, max) weights this exact LoRA used in the author's example images, or None.
def weights_from_examples(info, stem):
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())  # noqa: E731
    vid = info.get("id")
    hashes = {((f.get("hashes") or {}).get("AutoV2") or "").lower() for f in info.get("files", [])} - {""}
    names = {norm(stem)} | {norm((f.get("name") or "").rsplit(".", 1)[0]) for f in info.get("files", [])}
    ws = []
    for im in info.get("images") or []:
        meta = im.get("meta") or {}
        for r in meta.get("civitaiResources") or []:
            if r.get("modelVersionId") == vid and r.get("weight") is not None:
                ws.append(float(r["weight"]))
        for r in meta.get("resources") or []:
            if r.get("weight") is not None and ((r.get("hash") or "").lower() in hashes or norm(r.get("name")) in names):
                ws.append(float(r["weight"]))
        for n, w in re.findall(r"<lora:([^:>]+):(-?\d+(?:\.\d+)?)", meta.get("prompt") or ""):
            if norm(n) in names:
                ws.append(float(w))
    ws = [w for w in ws if 0.05 <= w <= 2.0]
    return (min(ws), max(ws)) if ws else None


# Return the author's optional positive and negative tags from description text.
def optional_tags(text):
    out = {}
    for key, label in (("extra_positive", "positive"), ("extra_negative", "negative")):
        m = re.search(rf"(?:optional|recommended|suggested)\s+{label}\s*(?:prompt|tags?)?\s*[:：]\s*([^\n]{{1,200}}?)"
                      r"(?=\s*(?:optional|recommended|suggested)\s|\n|$)", text or "", re.I)
        out[key] = m.group(1).strip(" .,;") if m else ""
    return out


# GET a Civitai model with retries on 429/5xx.
def fetch_model(model_id, retries=4):
    req = urllib.request.Request(API.format(model_id), headers={"User-Agent": "ComfyUI-SceneLoader/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    return None


# Build sidecar data for one LoRA from its civitai.info and the fetched model (may be None).
def build_meta(info, stem, model):
    vid = info.get("id")
    version = next((v for v in (model or {}).get("modelVersions", []) if v.get("id") == vid), {})
    our_base = base_key(info.get("baseModel") or version.get("baseModel"))
    vdesc, mdesc = plain(version.get("description") or info.get("description")), plain((model or {}).get("description"))
    weight, source = None, "default"
    for text, src in ((vdesc, "version_description"), (mdesc, "model_description")):
        weight = weights_from_text(text, our_base)
        if weight:
            source = src
            break
    if not weight:
        weight = weights_from_examples(info, stem)
        source = "examples" if weight else "default"
    weight = weight or (1.0, 1.0)
    words = version.get("trainedWords") or info.get("trainedWords") or []
    return {
        "civitai": {"modelId": info.get("modelId"), "versionId": vid,
                    "fetched": datetime.date.today().isoformat(), "found": model is not None},
        "name": (model or {}).get("name") or (info.get("model") or {}).get("name"),
        "version": version.get("name") or info.get("name"),
        "base": info.get("baseModel") or version.get("baseModel"),
        "tags": (model or {}).get("tags") or [],
        "trigger_sets": [w for w in words if isinstance(w, str) and w.strip()],
        **optional_tags(vdesc + "\n" + mdesc),
        "weight": {"min": weight[0], "max": weight[1], "source": source},
        "description": (vdesc + "\n\n" + mdesc).strip()[:4000],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--subdir", default="loras", help="folder under models root to scan")
    ap.add_argument("--limit", type=int, default=0, help="process at most N LoRAs (0 = all)")
    ap.add_argument("--sample", type=int, default=0, help="pick N LoRAs spread across folders")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--force", action="store_true", help="rewrite existing sidecars")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    infos = sorted(glob.glob(os.path.join(args.models_root, args.subdir, "**", "*.civitai.info"), recursive=True))
    if args.sample:
        step = max(1, len(infos) // args.sample)
        infos = infos[::step][:args.sample]
    if args.limit:
        infos = infos[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    models, rows, fetched = {}, [], 0
    for n, path in enumerate(infos, 1):
        stem = path[:-len(".civitai.info")]
        side = stem + SIDECAR
        if os.path.exists(side) and not args.force:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, ValueError) as e:
            log(f"skip unreadable {path}: {e}")
            continue
        mid = info.get("modelId")
        if mid and mid not in models:
            try:
                models[mid] = fetch_model(mid)
            except Exception as e:  # network or API error: keep local data only
                log(f"fetch failed for model {mid}: {e}")
                models[mid] = None
            fetched += 1
            time.sleep(args.sleep)
        meta = build_meta(info, os.path.basename(stem), models.get(mid))
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        w = meta["weight"]
        rows.append([os.path.relpath(stem, args.models_root), w["min"], w["max"], w["source"],
                     len(meta["trigger_sets"]), meta["extra_positive"][:60], meta["extra_negative"][:60]])
        log(f"{n}/{len(infos)} {os.path.basename(stem)}: weight {w['min']}-{w['max']} ({w['source']})")
    report = os.path.join(args.out, "civitai-meta-report.csv")
    with open(report, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["lora", "weight_min", "weight_max", "source", "trigger_sets", "extra_positive", "extra_negative"])
        wr.writerows(rows)
    log(f"written {len(rows)} sidecars, {fetched} API calls -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
