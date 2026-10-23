#!/usr/bin/env python3
"""Fill pose block tags from their sample images with a WD14 tagger (ONNX), keeping only pose tags.

Pose blocks (blocks/poses/**.yaml) with an image and empty tags get the tagger's general tags that
describe the pose (sitting, arms up, hand on hip, from side, ...); clothing, hair, colors and the
rest are dropped. --force re-tags blocks that already have tags. The model folder holds model.onnx
and selected_tags.csv (e.g. SmilingWolf/wd-eva02-large-tagger-v3). Needs onnxruntime, numpy,
Pillow, PyYAML (the ComfyUI venv has them).
"""
import argparse
import csv
import fnmatch
import os
import re
import sys

import numpy as np
import yaml
from PIL import Image

# Pose, gesture and viewpoint vocabulary; a tag is kept when it matches and is not an item of clothing.
POSE_RE = re.compile(
    r"\b(standing|sitting|seiza|wariza|yokozuwari|kneeling|lying|squatting|crouching|leaning|bent over|"
    r"all fours|on (back|stomach|side)|fetal position|indian style|crossed (legs|arms|ankles)|"
    r"spread (legs|arms)|legs? (up|apart|together)|knees? (up|together|apart)|feet (up|together|apart)|"
    r"knees to chest|arms? (up|behind|support|at sides)|outstretched|reaching|stretching|arched back|"
    r"hands? (on|in|to|up|between|behind|together)|hand on|hands on|own hands|clasped|interlocked fingers|"
    r"head (tilt|rest)|looking (back|at viewer|away|up|down|to the side|ahead|at another)|"
    r"from (above|below|side|behind)|contrapposto|v|peace sign|double v|salute|pointing|waving|"
    r"hugging|holding|carrying|walking|running|jumping|dancing|fighting stance|straddling|split|"
    r"on one knee|one knee|on one leg|standing on one leg|leg lift|hip|pose|crawling|bending|"
    r"hand gesture|finger to mouth|covering|shushing|thumbs up|heart hands|arm up|arms up|"
    r"head down|head back|upside[- ]down|twisting|turning)\b")
NOT_POSE_RE = re.compile(r"warmer|glove|sleeve|tattoo|bracelet|nail|wear\b|socks|thighhigh|boot|shoe|"
                         r"holding hands|hair|fan\b|mask|ribbon|bow\b|strap|garter")


def log(msg):
    print(f"[tag-poses] {msg}", flush=True)


# True for tags that describe the pose, gesture or viewpoint (not clothing or other items).
def is_pose_tag(text):
    return bool(POSE_RE.search(text)) and not NOT_POSE_RE.search(text)


# Tagger: (session, input name, size, general tag names list, their indices).
def load_tagger(folder, provider):
    import onnxruntime as ort
    providers = [provider] if provider else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(os.path.join(folder, "model.onnx"), providers=providers)
    inp = sess.get_inputs()[0]
    with open(os.path.join(folder, "selected_tags.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    general = [(i, r["name"]) for i, r in enumerate(rows) if r["category"] == "0"]
    return sess, inp.name, int(inp.shape[1]), general


# WD14 v3 input: RGB on white, padded square, resized, BGR float32 (1, S, S, 3).
def prepare(path, size):
    with Image.open(path) as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    side = max(im.size)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    a = np.asarray(sq.resize((size, size), Image.BICUBIC), np.float32)[:, :, ::-1]
    return np.ascontiguousarray(a[None])


# Pose tags of one image, most probable first.
def pose_tags(tagger, path, threshold):
    sess, name, size, general = tagger
    probs = sess.run(None, {name: prepare(path, size)})[0][0]
    out = []
    for i, tag in sorted(general, key=lambda g: -probs[g[0]]):
        if probs[i] < threshold:
            break
        text = tag.replace("_", " ") if len(tag) > 3 else tag
        if is_pose_tag(text):
            out.append(text)
    return out


# Pose blocks to tag: [(yaml path, ref, data, image path)].
def find_blocks(library, only, force):
    root = os.path.join(library, "blocks")
    out = []
    for d, dirs, files in os.walk(os.path.join(root, "poses")):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(d, f)
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            ref = os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, "/")
            if not isinstance(data, dict) or not data.get("image"):
                continue
            # Coarse tags from tag_skeletons.py are replaced once the block has a sample image.
            if data.get("tags") and data.get("tags_source") != "skeleton" and not force:
                continue
            if only and not any(fnmatch.fnmatch(ref, g) for g in only):
                continue
            out.append((path, ref, data, os.path.join(d, str(data["image"]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", required=True, help="Scene Loader library root")
    ap.add_argument("--model", required=True, help="tagger folder with model.onnx and selected_tags.csv")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--only", action="append", default=[], metavar="GLOB", help="block refs to tag")
    ap.add_argument("--force", action="store_true", help="re-tag blocks that already have tags")
    ap.add_argument("--dry-run", action="store_true", help="print the tags, write nothing")
    ap.add_argument("--provider", default="", help="onnxruntime provider (default CPUExecutionProvider)")
    a = ap.parse_args(argv)
    blocks = find_blocks(a.library, a.only, a.force)
    if not blocks:
        log("nothing to tag")
        return 0
    tagger = load_tagger(a.model, a.provider)
    done = empty = 0
    for k, (path, ref, data, image) in enumerate(blocks, 1):
        try:
            tags = pose_tags(tagger, image, a.threshold)
        except (OSError, ValueError) as e:
            log(f"{ref}: {e}")
            continue
        log(f"[{k}/{len(blocks)}] {ref}: {', '.join(tags) or '(no pose tags)'}")
        if not tags:
            empty += 1
            continue
        done += 1
        if not a.dry_run:
            data["tags"] = ", ".join(tags)
            data.pop("tags_source", None)
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    log(("DRY RUN: " if a.dry_run else "") + f"{done} blocks tagged, {empty} without pose tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
