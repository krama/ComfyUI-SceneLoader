#!/usr/bin/env python3
"""Propose quarantine of older model versions and of whole model families. Read-only.

Versions are grouped by Civitai modelId + baseModel from .civitai.info. The newest version
(publishedAt, else createdAt) is kept. Older ones are marked "quarantine" only when the
version names differ only in version tokens; otherwise the group is a collection of variants
and is marked "review". Output CSV is compatible with quarantine.py (action,path columns).
"""
import argparse
import csv
import json
import os
import re
import sys

MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
# Version token inside a version name: "v1", "V2.0", "1.5", "v3.1b".
VERSION_TOKEN_RE = re.compile(r"\bv?\d+(\.\d+)*[a-z]?\b", re.I)


# Return a version name with version tokens and punctuation removed, for comparison.
def name_core(name):
    core = VERSION_TOKEN_RE.sub(" ", str(name).lower())
    return re.sub(r"[^a-z]+", " ", core).strip()


def log(msg):
    print(f"[versions] {msg}", flush=True)


# Return parsed .civitai.info for a model path when it identifies a version, else None.
def read_info(path):
    side = os.path.splitext(path)[0] + ".civitai.info"
    try:
        with open(side, encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return None
    return info if info.get("id") and info.get("modelId") else None


# Return the release timestamp string used to order versions (ISO 8601 sorts lexically).
def released(info):
    return info.get("publishedAt") or info.get("createdAt") or ""


# Return True when relpath lies inside one of the family directories.
def in_family(rel, families):
    return any(rel == f or rel.startswith(f.rstrip("/") + "/") for f in families)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--family", action="append", default=[],
                    help="directory relative to models root to quarantine whole (repeatable)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    root = os.path.abspath(args.models_root)
    os.makedirs(args.out, exist_ok=True)
    families = [f.strip("/") for f in args.family]

    rows, groups, no_info = [], {}, 0
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in sorted(files):
            if not f.lower().endswith(MODEL_EXT):
                continue
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            size = os.path.getsize(p)
            info = read_info(p)
            if in_family(rel, families):
                rows.append(dict(group="family", model_id=(info or {}).get("modelId", ""),
                                 base_model=(info or {}).get("baseModel", ""), action="quarantine",
                                 released="", version_name="", size_gb=f"{size / 1e9:.3f}",
                                 path=rel, reason="family " + next(x for x in families if in_family(rel, [x]))))
                continue
            if not info:
                no_info += 1
                continue
            key = (info["modelId"], info.get("baseModel", ""))
            groups.setdefault(key, []).append((p, rel, size, info))

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    for gi, (key, members) in enumerate(sorted(multi.items(), key=lambda x: str(x[0])), 1):
        members.sort(key=lambda m: released(m[3]), reverse=True)
        # Same name once version tokens are removed means a version history, not variants.
        numeric = len({name_core(m[3].get("name", "")) for m in members}) == 1
        same_version = len({m[3]["id"] for m in members}) == 1
        for j, (p, rel, size, info) in enumerate(members):
            if j == 0:
                action, reason = "keep", "newest"
            elif same_version:
                action, reason = "review", "same version id twice (renamed copy?)"
            elif numeric:
                action, reason = "quarantine", f"older than {members[0][3].get('name')}"
            else:
                action, reason = "review", "variant names, likely a collection"
            rows.append(dict(group=gi, model_id=key[0], base_model=key[1], action=action,
                             released=released(info)[:10], version_name=info.get("name", ""),
                             size_gb=f"{size / 1e9:.3f}", path=rel, reason=reason))

    out_csv = os.path.join(args.out, "versions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "model_id", "base_model", "action", "released",
                                           "version_name", "size_gb", "path", "reason"])
        w.writeheader()
        w.writerows(rows)

    def total(pred):
        sel = [r for r in rows if pred(r)]
        return len(sel), sum(float(r["size_gb"]) for r in sel)

    log(f"{len(multi)} multi-version groups, {no_info} files without sidecar identity skipped")
    for name, pred in [("family quarantine", lambda r: r["group"] == "family"),
                       ("old version quarantine", lambda r: r["group"] != "family" and r["action"] == "quarantine"),
                       ("review", lambda r: r["action"] == "review")]:
        n, gb = total(pred)
        log(f"{name}: {n} files, {gb:.1f} GB")
    log(f"-> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
