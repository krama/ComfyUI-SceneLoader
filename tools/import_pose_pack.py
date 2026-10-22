#!/usr/bin/env python3
"""Import a pose card pack (POSES.yaml + POSES/<name>.<img>) as library blocks with sample images.

Writes <library>/blocks/poses/<slug>.yaml ({name, kind: pose, tags, image}) and copies the image
next to it, so Scene Composer offers the pose in its pose slot and returns the image as pose_image.
Existing blocks are kept unless --force.
"""
import argparse
import os
import re
import shutil
import sys

import yaml

IMAGE_EXT = (".jpeg", ".jpg", ".png", ".webp")


def log(msg):
    print(f"[import-pose-pack] {msg}", flush=True)


# Turn a pose name like "bent over" or "all-fours" into a file-safe slug.
def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


# Return {pose name: tags} from the pack's POSES.yaml (top key "POSES" or a flat mapping).
def read_pack(yaml_path):
    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    poses = data.get("POSES", data) if isinstance(data, dict) else {}
    return {str(k): ", ".join(t.strip() for t in str(v or "").split(",") if t.strip()) for k, v in poses.items()}


# Return the image file for a pose name in the images dir, or None.
def find_image(images_dir, name):
    for ext in IMAGE_EXT:
        p = os.path.join(images_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True, help="folder with POSES.yaml and POSES/ images")
    ap.add_argument("--library", required=True, help="Scene Loader library root")
    ap.add_argument("--force", action="store_true", help="overwrite existing pose blocks")
    args = ap.parse_args(argv)
    poses = read_pack(os.path.join(args.pack, "POSES.yaml"))
    out_dir = os.path.join(args.library, "blocks", "poses")
    os.makedirs(out_dir, exist_ok=True)
    written = skipped = no_image = 0
    for name, tags in sorted(poses.items()):
        s = slug(name)
        block = os.path.join(out_dir, s + ".yaml")
        if os.path.exists(block) and not args.force:
            skipped += 1
            continue
        img = find_image(os.path.join(args.pack, "POSES"), name)
        data = {"name": s, "kind": "pose", "tags": tags}
        if img:
            dst = s + os.path.splitext(img)[1].lower()
            shutil.copy2(img, os.path.join(out_dir, dst))
            data["image"] = dst
        else:
            no_image += 1
            log(f"{name}: no image found, block without pose_image")
        with open(block, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        written += 1
    log(f"{written} blocks written, {skipped} kept, {no_image} without image -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
