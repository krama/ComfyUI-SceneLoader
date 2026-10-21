#!/usr/bin/env python3
"""Propose a target folder for every LoRA from keyword rules. Read-only, writes a plan CSV.

Text matched per file: Civitai model name, version name, file name and trained words, with
leetspeak normalised. Stage 1 picks a slot (character, clothing, background, ...); stage 2
picks a subfolder for slots that define one. Output is compatible with relocate.py.
"""
import argparse
import csv
import json
import os
import re
import sys

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth")


def log(msg):
    print(f"[classify] {msg}", flush=True)


# Normalise leetspeak and punctuation into a space-padded lowercase string.
def normalise(text):
    text = text.lower()
    text = re.sub(r"(?<=[a-z])[0-9@]+|[0-9@]+(?=[a-z])", lambda m: m.group(0).translate(LEET), text)
    return " " + re.sub(r"[^a-z]+", " ", text) + " "


# Return (name_text, full_text): name = model name + file name; full adds version and trained words.
def describe(path):
    name = [os.path.splitext(os.path.basename(path))[0]]
    extra = []
    try:
        with open(os.path.splitext(path)[0] + ".civitai.info", encoding="utf-8") as fh:
            info = json.load(fh)
        name.append((info.get("model") or {}).get("name") or "")
        extra.append(info.get("name") or "")
        extra += [w for w in info.get("trainedWords") or [] if isinstance(w, str)]
    except (OSError, ValueError):
        pass
    return normalise(" ".join(name)), normalise(" ".join(name + extra))


# Match the name text first; fall back to the full text. Returns (key, hits, weak).
def match(texts, table):
    key, hits = best_match(texts[0], table)
    if key:
        return key, hits, False
    key, hits = best_match(texts[1], table)
    return key, hits, bool(key)


# Return (best_key, hits) for the rule table whose keywords match text most often.
def best_match(text, table):
    best, best_hits = None, []
    for key, words in table.items():
        hits = [w for w in words if f" {w} " in text or (len(w) >= 5 and w in text)]
        if len(hits) > len(best_hits):
            best, best_hits = key, hits
    return best, best_hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--rules", required=True, help="JSON rules file (see lora_rules.example.json)")
    ap.add_argument("--base", action="append", default=[], help="LoRA base folder, e.g. loras/illustrious")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    root = os.path.abspath(args.models_root)
    with open(args.rules, encoding="utf-8") as fh:
        rules = json.load(fh)
    slots, subs = rules["slots"], rules.get("subfolders", {})
    trusted = set(rules.get("trusted_slots", []))
    merge = rules.get("merge", {})
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for base in args.base:
        broot = os.path.join(root, base)
        for d, _, files in os.walk(broot):
            for f in sorted(files):
                if not f.lower().endswith(MODEL_EXT):
                    continue
                p = os.path.join(d, f)
                rel_dir = os.path.relpath(d, broot).replace(os.sep, "/")
                cur_slot = rel_dir.split("/")[0] if rel_dir != "." else ""
                texts = describe(p)
                slot, hits, weak = match(texts, slots)
                action, note = "move", ""
                cur_slot_m = merge.get(cur_slot, cur_slot)
                if cur_slot_m in trusted or not slot:
                    if slot and slot != cur_slot_m:
                        note = f"rules suggest {slot}"
                        action = "review"
                    slot = cur_slot_m
                elif weak and cur_slot_m and slot != cur_slot_m:
                    # Only trained words matched: suggest, keep the file where it is.
                    note, action, slot = f"weak match: {slot}", "review", cur_slot_m
                sub, sub_hits = (None, [])
                if slot in subs:
                    sub, sub_hits, _ = match(texts, subs[slot])
                    sub = sub or "_unsorted"
                dest = "/".join(x for x in (base, slot, sub) if x)
                if dest == os.path.relpath(d, root).replace(os.sep, "/") and action == "move":
                    action = "keep"
                if not slot:
                    action, dest = "keep", os.path.relpath(d, root)
                rows.append(dict(action=action, path=os.path.relpath(p, root), dest=dest,
                                 slot_hits=" ".join(hits), sub_hits=" ".join(sub_hits), note=note))

    out_csv = os.path.join(args.out, "lora-plan.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["action", "path", "dest", "slot_hits", "sub_hits", "note"])
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    log(f"{len(rows)} LoRAs: {counts} -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
