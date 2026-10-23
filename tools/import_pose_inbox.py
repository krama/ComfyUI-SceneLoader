#!/usr/bin/env python3
"""Import pose images dropped into an inbox folder (Civitai downloads) as library pose blocks.

The inbox takes images, .zip packs and folders. Each image is a photo / render of a person, an
OpenPose skeleton (colored limbs on black) or something to skip (lineart, depth / canny, pack
previews named preview / cover / grid / thumb / banner / showcase). Photos and skeletons of the
same pose are paired by name (role words like openpose / depth / ref and o1 / r1 prefixes are
ignored) or, failing that, by sharing a folder alone. Each pose becomes <slug>.yaml
({name, kind: pose, tags, image?, skeleton?}) with its files next to it:
  - image: Scene Composer returns it as pose_image -> 3D pose (NLF) and camera (Pose Camera);
  - skeleton only: returned as pose_skeleton, used as a ready 2D ControlNet map (no camera);
    tools/skeleton_to_figure.py can add a figure image later for the 3D path.
A single image goes to poses/civitai/<slug>, a pack with one pose to poses/civitai/<pack>, a pack
with several to poses/civitai/<pack>/<path in pack>. Tags come from <image>.txt next to it.
Imported sources are moved to <inbox>/_imported/. Needs Pillow and numpy (ComfyUI venv python).
"""
import argparse
import fnmatch
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_pose_pack import IMAGE_EXT, slug  # noqa: E402

DONE_DIR = "_imported"
OUT_SUBDIR = ("poses", "civitai")
PREVIEW_RE = re.compile(r"preview|cover|grid|thumb|banner|showcase", re.I)
# Words that name a file's role, not the pose; dropped when pairing photos with skeletons.
ROLE_RE = re.compile(r"open_?pose|dw_?pose|keypoints?|skeleton|skel|depth|canny|line_?art|normal(map)?|"
                     r"rendered|render|reference|ref|original|orig|photo|image|img|sample|source|src|pose")


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


# Pairing key of a path inside a pack: (folder, stem without role words / o1-r1 style prefixes).
def pose_key(rel):
    folder, stem = os.path.split(rel.lower())
    key = re.sub(r"[^a-z0-9]+", "", ROLE_RE.sub("", stem))
    key = re.sub(r"^[a-z](?=\d+$)", "", key)  # o1 / r1 -> 1
    return folder, key or re.sub(r"[^a-z0-9]+", "", stem)


# Pair photos with skeletons: [(name, photo or None, skeleton or None)], one entry per pose.
def pair(photos, skeletons):
    groups = defaultdict(lambda: ([], []))
    for i, items in enumerate((photos, skeletons)):
        for rel, path in items:
            groups[pose_key(rel)][i].append((rel, path))
    out, lone, total = [], defaultdict(lambda: ([], [])), defaultdict(lambda: [0, 0])
    for (folder, _), (ph, sk) in sorted(groups.items()):
        total[folder][0] += len(ph)
        total[folder][1] += len(sk)
        if ph and sk:
            out += [(rel, path, sk[0][1]) for rel, path in ph]
        else:
            lone[folder][0 if ph else 1].extend(ph or sk)
    for folder, (ph, sk) in sorted(lone.items()):
        if total[folder] == [1, 1]:  # the folder holds one pose, file names unrelated
            out.append((ph[0][0], ph[0][1], sk[0][1]))
            continue
        out += [(rel, path, None) for rel, path in ph] + [(rel, None, path) for rel, path in sk]
    return sorted(out, key=lambda e: e[0])


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


# Write one pose block (photo and / or skeleton); returns the slug or None when it exists.
def write_block(out_dir, name, photo, skeleton, force):
    s = slug(name)
    block = os.path.join(out_dir, s + ".yaml")
    if os.path.exists(block) and not force:
        return None
    os.makedirs(out_dir, exist_ok=True)
    data = {"name": s, "kind": "pose", "tags": tags_for(photo) if photo else "", "source": "civitai"}
    if photo:
        data["image"] = s + os.path.splitext(photo)[1].lower()
        shutil.copy2(photo, os.path.join(out_dir, data["image"]))
    if skeleton:
        data["skeleton"] = s + "_skeleton.png"
        with Image.open(skeleton) as im:
            im.convert("RGB").save(os.path.join(out_dir, data["skeleton"]))
        data["tags"] = data["tags"] or tags_for(skeleton)
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
    ap.add_argument("--skip", action="append", default=[], metavar="GLOB",
                    help="inbox entries to leave untouched (repeatable), e.g. 'thousandsOfPoses*'")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be imported")
    args = ap.parse_args(argv)
    base_dir = os.path.join(args.library, "blocks", *OUT_SUBDIR)
    ref_base = "/".join(OUT_SUBDIR)
    n = defaultdict(int)
    with tempfile.TemporaryDirectory() as tmp:
        for entry in sorted(os.listdir(args.inbox)):
            src = os.path.join(args.inbox, entry)
            stem = os.path.splitext(entry)[0]
            if entry.startswith(("_", ".")) or entry.lower().endswith(".txt"):
                continue  # done folder, hidden files, tag sidecars (moved with their image)
            if any(fnmatch.fnmatch(entry, g) for g in args.skip):
                log(f"{entry}: skipped (--skip)")
                continue
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
                    n["skipped"] += 1
                    log(f"{entry}: skip {rel}: {why}")
                else:
                    (photos if kind == "photo" else skeletons).append((rel, img))
            poses = pair(photos, skeletons)
            if not poses:
                log(f"{entry}: no pose photo or OpenPose skeleton found")
            for rel, photo, skeleton in poses:
                if pack is None or len(poses) == 1:
                    out_dir, name, ref = base_dir, pack or rel, ref_base
                else:
                    out_dir, name, ref = os.path.join(base_dir, pack), rel, f"{ref_base}/{pack}"
                kind = "photo+skeleton" if photo and skeleton else "photo" if photo else "skeleton"
                s = slug(name) if args.dry_run else write_block(out_dir, name, photo, skeleton, args.force)
                if s:
                    n[kind] += 1
                    log(f"{entry}: {ref}/{s} ({kind})")
                else:
                    n["kept"] += 1
                    log(f"{entry}: {ref}/{slug(name)} exists, kept (use --force)")
            if not args.dry_run:
                move_done(args.inbox, src, stem)
    log(("DRY RUN, nothing written: " if args.dry_run else "")
        + f"{n['photo'] + n['photo+skeleton']} poses with photo ({n['photo+skeleton']} with skeleton), "
        f"{n['skeleton']} skeleton-only, {n['kept']} kept, {n['skipped']} images skipped -> {base_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
