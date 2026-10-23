#!/usr/bin/env python3
"""Import pose sample images dropped into an inbox folder (Civitai downloads) as library pose blocks.

The inbox takes images, .zip packs and folders. A single image becomes poses/civitai/<slug>; a pack
with several photos becomes poses/civitai/<pack>/<path in pack>, one block per photo; a pack with
one photo becomes poses/civitai/<pack>. Each block is <slug>.yaml ({name, kind: pose, tags, image})
with the image next to it, so Scene Composer offers it in the pose slot and returns it as
pose_image. A source without photos but with OpenPose skeletons queues them in <dir>/_pending/;
tools/skeleton_to_figure.py turns each into a figure image (ControlNet) and writes the block.
Other control maps (lineart, depth / canny) and pack previews (preview, cover, grid, thumb,
banner, showcase in the file name) are skipped and listed.
Tags come from a <image>.txt next to the image, if any. Imported sources are moved to
<inbox>/_imported/. Needs Pillow and numpy (run with the ComfyUI venv python).
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_pose_pack import IMAGE_EXT, slug  # noqa: E402

DONE_DIR = "_imported"
PENDING_DIR = "_pending"  # skeletons waiting for tools/skeleton_to_figure.py
OUT_SUBDIR = ("poses", "civitai")
PREVIEW_RE = re.compile(r"preview|cover|grid|thumb|banner|showcase", re.I)


def log(msg):
    print(f"[import-pose-inbox] {msg}", flush=True)


# Classify an image: ("photo", None), ("skeleton", None) or ("skip", reason).
def classify(path):
    if PREVIEW_RE.search(os.path.splitext(os.path.basename(path))[0]):
        return "skip", "pack preview / cover by file name"
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((256, 256), Image.NEAREST), np.int16)
    dark = a.max(2) < 20
    if dark.mean() > 0.85:
        lit = a[~dark]
        # OpenPose limbs are saturated colors; lineart / scribble is white or gray on black.
        if len(lit) and (lit.max(1) - lit.min(1)).mean() > 80:
            return "skeleton", None
        return "skip", "mostly black (lineart / scribble map)"
    if np.abs(np.diff(a, axis=2)).mean() < 2:
        return "skip", "grayscale (depth / canny map)"
    return "photo", None


# Queue an OpenPose skeleton for skeleton_to_figure.py: <out_dir>/_pending/<slug>.png (+ .txt tags).
def queue_skeleton(out_dir, name, img, force):
    s = slug(name)
    pending = os.path.join(out_dir, PENDING_DIR)
    if os.path.exists(os.path.join(out_dir, s + ".yaml")) and not force:
        return None
    os.makedirs(pending, exist_ok=True)
    with Image.open(img) as im:
        im.convert("RGB").save(os.path.join(pending, s + ".png"))
    tags = tags_for(img)
    if tags:
        with open(os.path.join(pending, s + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(tags)
    return s


# Return the pack's content root: wrapper folders holding only one subfolder are skipped.
def pack_root(folder):
    while True:
        items = [x for x in os.listdir(folder) if not x.startswith(("_", "."))]
        if len(items) == 1 and os.path.isdir(os.path.join(folder, items[0])):
            folder = os.path.join(folder, items[0])
        else:
            return folder


# Return [(path inside the pack without extension, image path)] for every image under a folder.
def images_in(folder):
    out = []
    for d, dirs, files in os.walk(folder):
        dirs[:] = sorted(x for x in dirs if not x.startswith(("_", ".")))
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXT):
                path = os.path.join(d, f)
                out.append((os.path.splitext(os.path.relpath(path, folder))[0].replace(os.sep, "/"), path))
    return out


# Read tags from <image>.txt or <stem>.txt next to the image.
def tags_for(path):
    for p in (path + ".txt", os.path.splitext(path)[0] + ".txt"):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return ", ".join(t.strip() for t in fh.read().replace("\n", ",").split(",") if t.strip())
    return ""


# Write one block; returns the slug or None when it exists and force is off.
def write_block(out_dir, name, img, force):
    s = slug(name)
    os.makedirs(out_dir, exist_ok=True)
    block = os.path.join(out_dir, s + ".yaml")
    if os.path.exists(block) and not force:
        return None
    dst = s + os.path.splitext(img)[1].lower()
    shutil.copy2(img, os.path.join(out_dir, dst))
    data = {"name": s, "kind": "pose", "tags": tags_for(img), "image": dst, "source": "civitai"}
    with open(block, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return s


# Unpack a zip into dest, refusing members that would land outside it.
def unzip(src, dest):
    with zipfile.ZipFile(src) as z:
        for m in z.infolist():
            target = os.path.realpath(os.path.join(dest, m.filename))
            if target.startswith(os.path.realpath(dest) + os.sep) and not m.is_dir():
                z.extract(m, dest)


# Move a processed source (and its tag sidecar) to the done folder without overwriting.
def move_done(inbox, src, stem):
    done_dir = os.path.join(inbox, DONE_DIR)
    os.makedirs(done_dir, exist_ok=True)
    sidecars = [p for p in (src + ".txt", os.path.join(inbox, stem + ".txt")) if os.path.isfile(p)]
    for path in [src] + sidecars[:1]:
        name = os.path.basename(path)
        target, n = os.path.join(done_dir, name), 1
        while os.path.exists(target):
            n += 1
            root, ext = os.path.splitext(name)
            target = os.path.join(done_dir, f"{root}.{n}{ext}")
        shutil.move(path, target)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="folder with dropped images / zips / folders")
    ap.add_argument("--library", required=True, help="Scene Loader library root")
    ap.add_argument("--force", action="store_true", help="overwrite existing pose blocks")
    args = ap.parse_args(argv)
    base_dir = os.path.join(args.library, "blocks", *OUT_SUBDIR)
    ref_base = "/".join(OUT_SUBDIR)
    written = kept = skipped = pending = 0
    with tempfile.TemporaryDirectory() as tmp:
        for entry in sorted(os.listdir(args.inbox)):
            src = os.path.join(args.inbox, entry)
            stem = os.path.splitext(entry)[0]
            if entry.startswith(("_", ".")) or entry.lower().endswith(".txt"):
                continue  # done folder, hidden files, tag sidecars (moved with their image)
            if os.path.isdir(src) or entry.lower().endswith(".zip"):
                root = src
                if not os.path.isdir(src):
                    root = os.path.join(tmp, slug(stem))
                    unzip(src, root)
                found, pack = images_in(pack_root(root)), slug(stem)
            elif entry.lower().endswith(IMAGE_EXT):
                found, pack = [(stem, src)], None
            else:
                log(f"{entry}: not an image, zip or folder, left in the inbox")
                continue
            photos, skeletons = [], []
            for rel, img in found:
                kind, why = classify(img)
                if kind == "skip":
                    skipped += 1
                    log(f"{entry}: skip {rel}: {why}")
                else:
                    (photos if kind == "photo" else skeletons).append((rel, img))
            # Skeletons are used only when the source has no photos (else they duplicate the photos).
            items, action = (photos, "photo") if photos else (skeletons, "skeleton")
            if skeletons and photos:
                skipped += len(skeletons)
                log(f"{entry}: {len(skeletons)} OpenPose skeleton(s) skipped, photos are used")
            if not items:
                log(f"{entry}: no pose photo or OpenPose skeleton found")
            for rel, img in items:
                if pack is None or len(items) == 1:
                    out_dir, name, ref = base_dir, pack or rel, ref_base
                else:
                    out_dir, name, ref = os.path.join(base_dir, pack), rel, f"{ref_base}/{pack}"
                if action == "photo":
                    s = write_block(out_dir, name, img, args.force)
                else:
                    s = queue_skeleton(out_dir, name, img, args.force)
                if s and action == "photo":
                    written += 1
                    log(f"{entry}: {ref}/{s}")
                elif s:
                    pending += 1
                    log(f"{entry}: {ref}/{s} queued: OpenPose skeleton, run skeleton_to_figure.py")
                else:
                    kept += 1
                    log(f"{entry}: {ref}/{slug(name)} exists, kept (use --force)")
            move_done(args.inbox, src, stem)
    log(f"{written} blocks written, {pending} skeletons queued, {kept} kept, {skipped} images skipped "
        f"-> {base_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
