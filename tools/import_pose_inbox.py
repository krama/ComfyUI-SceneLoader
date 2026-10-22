#!/usr/bin/env python3
"""Import pose sample images dropped into an inbox folder (Civitai downloads) as library pose blocks.

The inbox takes images, .zip packs and folders. Each photo/render becomes
<library>/blocks/poses/civitai/<slug>.yaml ({name, kind: pose, tags, image}) with the image next to
it, so Scene Composer offers it as poses/civitai/<slug> and returns it as pose_image. Control maps
in packs (OpenPose skeletons on black, depth / canny in grayscale) are skipped: the pose is rebuilt
in 3D from the photo. Tags come from a <image>.txt next to the image, if any. Imported sources are
moved to <inbox>/_imported/. Needs Pillow and numpy (run with the ComfyUI venv python).
"""
import argparse
import os
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


def log(msg):
    print(f"[import-pose-inbox] {msg}", flush=True)


# Return why an image is a control map rather than a photo, or None for a photo.
def control_map(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((128, 128)), np.int16)
    if (a.max(2) < 20).mean() > 0.85:
        return "mostly black (OpenPose / lineart map)"
    if np.abs(np.diff(a, axis=2)).mean() < 2:
        return "grayscale (depth / canny map)"
    return None


# Yield (source label, image path) for every image under a folder.
def images_in(folder, label):
    for d, dirs, files in os.walk(folder):
        dirs[:] = sorted(x for x in dirs if not x.startswith(("_", ".")))
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXT):
                yield label, os.path.join(d, f)


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
    block = os.path.join(out_dir, s + ".yaml")
    if os.path.exists(block) and not force:
        return None
    dst = s + os.path.splitext(img)[1].lower()
    shutil.copy2(img, os.path.join(out_dir, dst))
    data = {"name": s, "kind": "pose", "tags": tags_for(img), "image": dst, "source": "civitai"}
    with open(block, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="folder with dropped images / zips / folders")
    ap.add_argument("--library", required=True, help="Scene Loader library root")
    ap.add_argument("--force", action="store_true", help="overwrite existing pose blocks")
    args = ap.parse_args(argv)
    out_dir = os.path.join(args.library, "blocks", "poses", "civitai")
    done_dir = os.path.join(args.inbox, DONE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    written = kept = skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for entry in sorted(os.listdir(args.inbox)):
            src = os.path.join(args.inbox, entry)
            if entry.startswith(("_", ".")):
                continue
            stem = os.path.splitext(entry)[0]
            if os.path.isdir(src):
                found = list(images_in(src, stem))
            elif entry.lower().endswith(".zip"):
                dest = os.path.join(tmp, slug(stem))
                with zipfile.ZipFile(src) as z:
                    for m in z.infolist():  # no path traversal out of dest
                        target = os.path.realpath(os.path.join(dest, m.filename))
                        if target.startswith(os.path.realpath(dest) + os.sep) and not m.is_dir():
                            z.extract(m, dest)
                found = list(images_in(dest, stem))
            elif entry.lower().endswith(IMAGE_EXT):
                found = [("", src)]
            elif entry.lower().endswith(".txt"):
                continue  # tag sidecar, moved together with its image
            else:
                log(f"{entry}: not an image, zip or folder, left in the inbox")
                continue
            photos = []
            for label, img in found:
                why = control_map(img)
                if why:
                    skipped += 1
                    log(f"{entry}: skip {os.path.basename(img)}: {why}")
                else:
                    photos.append((label, img))
            for label, img in photos:
                base = os.path.splitext(os.path.basename(img))[0]
                name = base if not label else label if len(photos) == 1 else f"{label}_{base}"
                s = write_block(out_dir, name, img, args.force)
                if s:
                    written += 1
                    log(f"{entry}: poses/civitai/{s}")
                else:
                    kept += 1
                    log(f"{entry}: poses/civitai/{slug(name)} exists, kept (use --force)")
            os.makedirs(done_dir, exist_ok=True)
            sidecars = [p for p in (src + ".txt", os.path.join(args.inbox, stem + ".txt")) if os.path.isfile(p)]
            for path in [src] + sidecars[:1]:
                name = os.path.basename(path)
                target, n = os.path.join(done_dir, name), 1
                while os.path.exists(target):  # never overwrite an earlier import
                    n += 1
                    root, ext = os.path.splitext(name)
                    target = os.path.join(done_dir, f"{root}.{n}{ext}")
                shutil.move(path, target)
    log(f"{written} blocks written, {kept} kept, {skipped} control maps skipped -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
