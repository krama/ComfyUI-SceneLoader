#!/usr/bin/env python3
"""Fill pose block tags from OpenPose skeleton images (no sample image), by body geometry.

Skeleton-only pose blocks (blocks/poses/**.yaml with "skeleton" and no "image") get coarse tags:
standing / sitting / kneeling / lying / all fours / upside-down, arms up, hands on hips, from side,
from behind, spread legs, full body / cowboy shot / upper body. Keypoints are read back from the
limb colors of the controlnet_aux / DWPose drawing (COCO-18). Written tags carry
"tags_source: skeleton", so tag_poses.py replaces them once the block gets a sample image.
--force re-tags blocks that have skeleton tags; tags from other sources are never touched.
"""
import argparse
import fnmatch
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader.pose_camera import COLORS, LIMBS  # noqa: E402

TOL = 14  # max per-channel difference from a limb color (antialiased edges are left out)
MIN_PIX = 12  # fewer limb pixels than this = limb not drawn
NECK_LIMBS = (0, 1, 6, 9, 12)  # limbs starting at the neck


def log(msg):
    print(f"[tag-skeletons] {msg}", flush=True)


# Ends of each limb ellipse (two points along its main axis), None when the limb is not drawn.
def limb_ends(rgb):
    a = rgb.astype(np.int16)
    out = []
    for i in range(len(LIMBS)):
        color = np.array([int(c * 0.6) for c in COLORS[i]], np.int16)
        ys, xs = np.nonzero(np.abs(a - color).max(-1) <= TOL)
        if len(xs) < MIN_PIX:
            out.append(None)
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        cen = pts.mean(0)
        axis = np.linalg.svd(pts - cen, full_matrices=False)[2][0]
        proj = (pts - cen) @ axis
        out.append((cen + axis * proj.min(), cen + axis * proj.max()))
    return out


# COCO-18 keypoints (None = missing) from limb ends: the neck is the end the neck limbs share,
# then each limb's far end from its known start joint.
def keypoints(ends):
    pts = [None] * 18
    cands = [ends[i] for i in NECK_LIMBS if ends[i]]
    if len(cands) < 2:
        return pts
    best = None
    for k, limb in enumerate(cands):
        for e in limb:
            near = [min(limb2, key=lambda p: np.linalg.norm(p - e)) for j, limb2 in enumerate(cands) if j != k]
            score = sum(np.linalg.norm(p - e) for p in near)
            if best is None or score < best[0]:
                best = (score, np.mean([e] + near, 0))
    pts[1] = best[1]
    parent = {b: a for a, b in LIMBS}
    for i, (a, b) in enumerate(LIMBS):
        if ends[i] is None or pts[b] is not None:
            continue
        if pts[a] is None:
            # Start limb hidden (e.g. a shoulder drawn over by its joint): the end nearer the
            # start joint's parent is the start joint.
            if pts[parent.get(a, a)] is None:
                continue
            ref = pts[parent[a]]
            pts[a] = min(ends[i], key=lambda p: np.linalg.norm(p - ref))
        pts[b] = max(ends[i], key=lambda p: np.linalg.norm(p - pts[a]))
    return pts


# Coarse pose tags from COCO-18 keypoints in image coordinates (y down).
def tags_from(p):
    neck = p[1]
    if neck is None:
        return []
    tags = []
    hips = [p[i] for i in (8, 11) if p[i] is not None]
    knees = [p[i] for i in (9, 12) if p[i] is not None]
    ankles = [p[i] for i in (10, 13) if p[i] is not None]
    wrists = [p[i] for i in (4, 7) if p[i] is not None]
    face = [p[i] for i in (0, 14, 15) if p[i] is not None]
    if not hips:
        shoulders = [p[i] for i in (2, 5) if p[i] is not None]
        if len(shoulders) < 2:
            return []
        size = float(np.linalg.norm(shoulders[0] - shoulders[1])) * 1.5  # torso length estimate
        tags.append("upper body")
    else:
        mid = np.mean(hips, 0)
        v = mid - neck
        size = float(np.linalg.norm(v)) or 1.0
        tilt = float(np.degrees(np.arctan2(abs(v[0]), v[1])))  # 0 = hips straight below the neck
        if tilt > 120:
            tags.append("upside-down")
        elif tilt > 55:
            low = max(neck[1], mid[1]) + 0.3 * size  # hands and knees down on the floor
            if len(wrists) and len(knees) and all(w[1] > low for w in wrists) and all(k[1] > low for k in knees):
                tags.append("all fours")
            else:
                tags.append("lying")
        elif knees:
            kd = float(np.mean([k[1] for k in knees]) - mid[1]) / size
            ad = float(np.mean([a[1] for a in ankles]) - np.mean([k[1] for k in knees])) / size if ankles else None
            if kd < 0.35:
                tags.append("sitting")
            elif ad is not None and ad > 0.45 and kd > 0.55:
                tags.append("standing")
            elif ad is not None and abs(ad) < 0.3:
                tags.append("kneeling")
            if len(ankles) == 2 and tilt <= 55:
                hip_w = float(np.linalg.norm(hips[0] - hips[1])) if len(hips) == 2 else 0.3 * size
                if abs(ankles[0][0] - ankles[1][0]) > max(2.5 * hip_w, 0.8 * size):
                    tags.append("spread legs")
        tags.append("full body" if ankles else "cowboy shot" if knees else "upper body")
    head_y = min([f[1] for f in face] + [neck[1] - 0.25 * size])
    up = sum(1 for w in wrists if w[1] < head_y)
    if up == 2:
        tags.append("arms up")
    elif up == 1:
        tags.append("arm up")
    # Hand on hip: wrist at a hip with the elbow pushed out past the shoulder (hanging arms do not count).
    on_hip = 0
    for s, e, w in ((2, 3, 4), (5, 6, 7)):
        if hips and all(p[i] is not None for i in (s, e, w)):
            cx = neck[0]
            out = abs(p[e][0] - cx) > abs(p[s][0] - cx) + 0.2 * size
            on_hip += out and min(np.linalg.norm(p[w] - h) for h in hips) < 0.25 * size
    if on_hip == 2:
        tags.append("hands on hips")
    elif on_hip == 1:
        tags.append("hand on hip")
    if not face and (p[16] is not None or p[17] is not None or p[2] is not None):
        tags.append("from behind")
    elif (p[2] is None) != (p[5] is None) or (p[2] is not None and p[5] is not None
                                               and abs(p[2][0] - p[5][0]) < 0.3 * size):
        tags.append("from side")  # one shoulder hidden, or both close together
    return tags


# Pose tags of one skeleton image.
def skeleton_tags(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return tags_from(keypoints(limb_ends(rgb)))


# Skeleton-only pose blocks to tag: [(yaml path, ref, data, skeleton path)].
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
            if not isinstance(data, dict) or data.get("image") or not data.get("skeleton"):
                continue
            if data.get("tags") and (data.get("tags_source") != "skeleton" or not force):
                continue
            ref = os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, "/")
            if only and not any(fnmatch.fnmatch(ref, g) for g in only):
                continue
            out.append((path, ref, data, os.path.join(d, str(data["skeleton"]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", required=True, help="Scene Loader library root")
    ap.add_argument("--only", action="append", default=[], metavar="GLOB", help="block refs to tag")
    ap.add_argument("--force", action="store_true", help="re-tag blocks that already have skeleton tags")
    ap.add_argument("--dry-run", action="store_true", help="print the tags, write nothing")
    a = ap.parse_args(argv)
    blocks = find_blocks(a.library, a.only, a.force)
    done = empty = 0
    for k, (path, ref, data, skeleton) in enumerate(blocks, 1):
        try:
            tags = skeleton_tags(skeleton)
        except (OSError, ValueError) as e:
            log(f"{ref}: {e}")
            continue
        log(f"[{k}/{len(blocks)}] {ref}: {', '.join(tags) or '(no tags)'}")
        if not tags:
            empty += 1
            continue
        done += 1
        if not a.dry_run:
            data["tags"] = ", ".join(tags)
            data["tags_source"] = "skeleton"
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    log(("DRY RUN: " if a.dry_run else "") + f"{done} blocks tagged, {empty} without tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
