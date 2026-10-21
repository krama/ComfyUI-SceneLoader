#!/usr/bin/env python3
"""Move model files (and their sidecars) inside the models root according to a plan CSV.

Plan columns: path,dest (both relative to models root; dest is a directory). Rows whose
action column exists and is not "move" are skipped. Dry-run by default; --apply moves and
appends to a manifest CSV (old_path,new_path) that quarantine.py --undo can replay.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quarantine import bundle, log as _qlog, move  # noqa: E402


def log(msg):
    print(f"[relocate] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.models_root)
    with open(args.plan, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("action", "move").strip() == "move"]

    pairs = []
    for r in rows:
        src = os.path.join(root, r["path"])
        dest_dir = os.path.join(root, r["dest"].strip("/"))
        if os.path.dirname(src) == dest_dir:
            continue
        for f in bundle(src):
            pairs.append((f, os.path.join(dest_dir, os.path.basename(f))))

    new = not os.path.exists(args.manifest)
    moved = 0
    with open(args.manifest, "a", newline="", encoding="utf-8") if args.apply else open(os.devnull, "w") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        if args.apply and new:
            w.writerow(["old_path", "new_path"])
        for src, dst in pairs:
            if move(src, dst, args.apply):
                moved += 1
                if args.apply:
                    w.writerow([src, dst])
                    fh.flush()
                log(f"{'moved' if args.apply else 'would move'}: {os.path.relpath(src, root)} -> "
                    f"{os.path.relpath(os.path.dirname(dst), root)}/")
    log(f"{'moved' if args.apply else 'would move'} {moved}/{len(pairs)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
