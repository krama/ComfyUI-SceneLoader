#!/usr/bin/env python3
"""Move model files listed in a plan CSV (and their sidecars) into a quarantine folder.

Dry-run by default; pass --apply to move. Every move is appended to a manifest CSV
(old_path,new_path) that --undo replays backwards. Moves are same-filesystem renames only.
"""
import argparse
import csv
import os
import sys

# Sidecar suffixes that travel with a model; matched as "<stem><suffix>".
SIDECAR_SUFFIXES = (".civitai.info", ".sceneloader.json", ".preview.png", ".preview.jpg", ".preview.jpeg",
                    ".preview.webp", ".rgthree-info.json", ".metadata.json", ".json",
                    ".png", ".jpg", ".jpeg", ".webp", ".txt", ".yaml")


def log(msg):
    print(f"[quarantine] {msg}", flush=True)


# Return the model file plus existing sidecars that share its stem.
def bundle(path):
    stem = os.path.splitext(path)[0]
    out = [path]
    for suf in SIDECAR_SUFFIXES:
        cand = stem + suf
        if os.path.isfile(cand) and cand not in out:
            out.append(cand)
    return out


# Read relative model paths with action == quarantine from a plan CSV.
def read_plan(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["path"] for r in rows if r.get("action", "").strip() == "quarantine"]


# Rename src to dst after checking both sides; return False if skipped.
def move(src, dst, apply):
    if not os.path.exists(src):
        log(f"skip, missing: {src}")
        return False
    if os.path.exists(dst):
        log(f"skip, destination exists: {dst}")
        return False
    if apply:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--quarantine-root", required=True)
    ap.add_argument("--plan", help="CSV with columns action,path (path relative to models root)")
    ap.add_argument("--manifest", required=True, help="CSV log of moves, appended on --apply")
    ap.add_argument("--undo", action="store_true", help="move files from the manifest back")
    ap.add_argument("--apply", action="store_true", help="perform moves (default: dry-run)")
    args = ap.parse_args()
    models = os.path.abspath(args.models_root)
    quar = os.path.abspath(args.quarantine_root)
    os.makedirs(quar, exist_ok=True) if args.apply else None
    probe = quar if os.path.exists(quar) else os.path.dirname(quar)
    if os.stat(models).st_dev != os.stat(probe).st_dev:
        log("models root and quarantine root are on different filesystems; refusing to copy")
        return 2

    if args.undo:
        with open(args.manifest, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        moved = sum(move(r["new_path"], r["old_path"], args.apply) for r in reversed(rows))
        log(f"{'restored' if args.apply else 'would restore'} {moved}/{len(rows)} files")
        if args.apply and moved == len(rows):
            os.replace(args.manifest, args.manifest + ".undone")
        return 0

    if not args.plan:
        ap.error("--plan is required unless --undo")
    pairs = []
    for rel in read_plan(args.plan):
        src = os.path.join(models, rel)
        for f in bundle(src):
            pairs.append((f, os.path.join(quar, os.path.relpath(f, models))))
    new = not os.path.exists(args.manifest)
    moved, size = 0, 0
    with open(args.manifest, "a", newline="", encoding="utf-8") if args.apply else open(os.devnull, "w") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        if args.apply and new:
            w.writerow(["old_path", "new_path"])
        for src, dst in pairs:
            sz = os.path.getsize(src) if os.path.exists(src) else 0
            if move(src, dst, args.apply):
                moved += 1
                size += sz
                if args.apply:
                    w.writerow([src, dst])
                    fh.flush()
                log(f"{'moved' if args.apply else 'would move'}: {os.path.relpath(src, models)}")
    log(f"{'moved' if args.apply else 'would move'} {moved}/{len(pairs)} files, {size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
