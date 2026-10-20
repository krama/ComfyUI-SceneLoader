#!/usr/bin/env python3
"""Find byte-identical model files and propose which copy to keep. Read-only: never moves files.

Hash sources, cheapest first: size grouping -> size-verified SHA256 from .civitai.info
-> head/tail fingerprint -> full SHA256 (only where fingerprints collide). Hashes are cached.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
FP_CHUNK = 16 * 1024 * 1024
# Folder that "owns" a file type; a copy inside it wins over copies elsewhere.
PREFERRED_DIRS = ("text_encoders", "vae", "diffusion_models", "loras", "checkpoints",
                  "controlnet", "upscale_models", "clip", "unet")


def log(msg):
    print(f"[dedupe] {msg}", flush=True)


# Walk the models root and return [(abs_path, size, mtime_ns)] for model files.
def list_models(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith((".", "_to_delete"))]
        for f in files:
            if f.lower().endswith(MODEL_EXT):
                p = os.path.join(d, f)
                st = os.stat(p)
                out.append((p, st.st_size, st.st_mtime_ns))
    return out


# Return the sidecar SHA256 whose recorded size matches the local file, else None.
def sidecar_sha256(path, size):
    side = os.path.splitext(path)[0] + ".civitai.info"
    try:
        with open(side, encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return None
    for entry in info.get("files") or []:
        if not isinstance(entry, dict):
            continue
        sha = ((entry.get("hashes") or {}).get("SHA256") or "").strip().lower()
        kb = entry.get("sizeKB")
        if sha and isinstance(kb, (int, float)) and abs(kb * 1024 - size) <= 2048:
            return sha
    return None


# Return a fingerprint of size plus SHA256 over the first and last FP_CHUNK bytes.
def fingerprint(path, size):
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(FP_CHUNK))
        if size > 2 * FP_CHUNK:
            fh.seek(-FP_CHUNK, os.SEEK_END)
            h.update(fh.read(FP_CHUNK))
    return h.hexdigest()


# Return the full-file SHA256 of path.
def full_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# Load cached full hashes keyed by (path, size, mtime_ns).
def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cache[(r["path"], int(r["size"]), int(r["mtime_ns"]))] = r["sha256"]
    return cache


# Write the hash cache back to disk.
def save_cache(path, cache):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "size", "mtime_ns", "sha256"])
        for (p, s, m), sha in sorted(cache.items()):
            w.writerow([p, s, m, sha])


# Rank a copy for keeping: lower tuple wins (preferred folder, has sidecar, shorter path).
def keep_rank(path, root):
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    top = rel.split("/", 1)[0]
    dir_rank = PREFERRED_DIRS.index(top) if top in PREFERRED_DIRS else len(PREFERRED_DIRS)
    has_side = os.path.exists(os.path.splitext(path)[0] + ".civitai.info")
    return (dir_rank, 0 if has_side else 1, len(rel), rel)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", required=True, help="ComfyUI models directory")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    root = os.path.abspath(args.models_root)
    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "hash-cache.csv")
    cache = load_cache(cache_path)
    t0 = time.time()

    files = list_models(root)
    log(f"{len(files)} model files, {sum(s for _, s, _ in files) / 1e9:.1f} GB under {root}")
    by_size = {}
    for f in files:
        by_size.setdefault(f[1], []).append(f)
    groups = [g for g in by_size.values() if len(g) > 1]
    log(f"{len(groups)} size groups with >1 file ({sum(len(g) for g in groups)} files); rest unique")

    sha = {}
    stats = {"sidecar": 0, "cache": 0, "fingerprint": 0, "full": 0}
    for g in groups:
        for p, s, m in g:
            v = sidecar_sha256(p, s)
            if v:
                sha[p] = v
                stats["sidecar"] += 1
            elif (p, s, m) in cache:
                sha[p] = cache[(p, s, m)]
                stats["cache"] += 1
        unknown = [f for f in g if f[0] not in sha]
        if not unknown:
            continue
        # Fingerprint every member so known and unknown copies compare on one key.
        fps = {}
        for p, s, m in g:
            fps.setdefault(fingerprint(p, s), []).append((p, s, m))
            stats["fingerprint"] += 1
        for members in fps.values():
            if len(members) < 2:
                continue
            for p, s, m in members:
                if p not in sha:
                    log(f"full hash: {os.path.relpath(p, root)} ({s / 1e9:.2f} GB)")
                    sha[p] = full_sha256(p)
                    cache[(p, s, m)] = sha[p]
                    stats["full"] += 1
    save_cache(cache_path, cache)

    by_sha = {}
    for p, v in sha.items():
        by_sha.setdefault(v, []).append(p)
    dups = sorted((v, ps) for v, ps in by_sha.items() if len(ps) > 1)
    out_csv = os.path.join(args.out, "duplicates.csv")
    reclaim = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "sha256", "size_gb", "action", "path"])
        for i, (v, ps) in enumerate(dups, 1):
            ps = sorted(ps, key=lambda p: keep_rank(p, root))
            size = os.path.getsize(ps[0])
            for j, p in enumerate(ps):
                w.writerow([i, v, f"{size / 1e9:.3f}", "keep" if j == 0 else "quarantine",
                            os.path.relpath(p, root)])
                if j:
                    reclaim += size
    log(f"hash sources: {stats}")
    log(f"{len(dups)} duplicate groups, {reclaim / 1e9:.1f} GB reclaimable -> {out_csv}")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
