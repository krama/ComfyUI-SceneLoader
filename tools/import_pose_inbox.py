#!/usr/bin/env python3
"""Import pose sample images dropped into an inbox folder (Civitai downloads) as library pose blocks.

The inbox takes images, .zip packs and folders. A single image becomes poses/civitai/<slug>; a pack
with several photos becomes poses/civitai/<pack>/<path in pack>, one block per photo; a pack with
one photo becomes poses/civitai/<pack>. Each block is <slug>.yaml ({name, kind: pose, tags, image})
with the image next to it, so Scene Composer offers it in the pose slot and returns it as
pose_image. Control maps (OpenPose skeletons on black, depth / canny in grayscale) and pack
previews (preview, cover, grid, thumb, banner, showcase in the file name) are skipped and listed.
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
OUT_SUBDIR = ("poses", "civitai")
PREVIEW_RE = re.compile(r"preview|cover|grid|thumb|banner|showcase", re.I)


def log(msg):
    print(f"[import-pose-inbox] {msg}", flush=True)


# Return why an image is not a pose photo (control map or pack preview), or None for a photo.
def skip_reason(path):
    if PREVIEW_RE.search(os.path.splitext(os.path.basename(path))[0]):
        return "pack preview / cover by file name"
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((128, 128)), np.int16)
    if (a.max(2) < 20).mean() > 0.85:
        return "mostly black (OpenPose / lineart map)"
    if np.abs(np.diff(a, axis=2)).mean() < 2:
        return "grayscale (depth / canny map)"
    return None


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
    written = kept = skipped = 0
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
            photos = []
            for rel, img in found:
                why = skip_reason(img)
                if why:
                    skipped += 1
                    log(f"{entry}: skip {rel}: {why}")
                else:
                    photos.append((rel, img))
            if not photos:
                log(f"{entry}: no pose photo found (NLF needs a person in the image)")
            for rel, img in photos:
                if pack is None or len(photos) == 1:
                    out_dir, name, ref = base_dir, pack or rel, ref_base
                else:
                    out_dir, name, ref = os.path.join(base_dir, pack), rel, f"{ref_base}/{pack}"
                s = write_block(out_dir, name, img, args.force)
                if s:
                    written += 1
                    log(f"{entry}: {ref}/{s}")
                else:
                    kept += 1
                    log(f"{entry}: {ref}/{slug(name)} exists, kept (use --force)")
            move_done(args.inbox, src, stem)
    log(f"{written} blocks written, {kept} kept, {skipped} images skipped -> {base_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
