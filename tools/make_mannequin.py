#!/usr/bin/env python3
"""Generate a box mannequin (.obj) for ComfyUI Load 3D camera framing.

The figure stands on y=0 and faces +Z (nose and toes point +Z), matching the front view of
ComfyUI-AdvancedCameraPrompts. Default height 0.425 units = 1.7 m at its 4 m per grid unit.
"""
import argparse
import os
import sys


# Return 8 vertices and 6 quad faces (1-based offset) for an axis-aligned box.
def box(cx, cy, cz, sx, sy, sz, offset):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx + dx * hx, cy + dy * hy, cz + dz * hz)
         for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    f = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    return v, [tuple(i + offset for i in face) for face in f]


# Return named boxes (center x,y,z and size x,y,z) for a figure of height h, scaled by build.
def parts(h, shoulders, hips):
    u = h / 8.0  # classic 8-heads proportion
    leg_h, torso_h = 4 * u, 2.6 * u
    return {
        "leg_l": (-0.45 * u * hips, leg_h / 2, 0, 0.55 * u, leg_h, 0.6 * u),
        "leg_r": (0.45 * u * hips, leg_h / 2, 0, 0.55 * u, leg_h, 0.6 * u),
        "foot_l": (-0.45 * u * hips, 0.1 * u, 0.35 * u, 0.5 * u, 0.2 * u, 1.1 * u),
        "foot_r": (0.45 * u * hips, 0.1 * u, 0.35 * u, 0.5 * u, 0.2 * u, 1.1 * u),
        "pelvis": (0, leg_h + 0.4 * u, 0, 1.6 * u * hips, 0.8 * u, 0.9 * u),
        "torso": (0, leg_h + 0.8 * u + torso_h / 2, 0, 1.7 * u * shoulders, torso_h, 0.9 * u),
        "arm_l": (-1.05 * u * shoulders, leg_h + 2.0 * u, 0, 0.4 * u, 2.9 * u, 0.45 * u),
        "arm_r": (1.05 * u * shoulders, leg_h + 2.0 * u, 0, 0.4 * u, 2.9 * u, 0.45 * u),
        "neck": (0, 7.1 * u, 0, 0.4 * u, 0.3 * u, 0.4 * u),
        "head": (0, 7.6 * u, 0, 0.75 * u, 0.85 * u, 0.85 * u),
        "nose": (0, 7.6 * u, 0.5 * u, 0.15 * u, 0.15 * u, 0.2 * u),  # front marker (+Z)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output .obj path (ComfyUI input/3d/...)")
    ap.add_argument("--height", type=float, default=0.425, help="figure height in grid units")
    ap.add_argument("--shoulders", type=float, default=1.0, help="shoulder width factor")
    ap.add_argument("--hips", type=float, default=1.0, help="hip width factor")
    args = ap.parse_args()
    lines = ["# box mannequin, faces +Z, feet on y=0"]
    count = 0
    for name, (cx, cy, cz, sx, sy, sz) in parts(args.height, args.shoulders, args.hips).items():
        v, f = box(cx, cy, cz, sx, sy, sz, count)
        lines.append(f"o {name}")
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in v]
        lines += ["f " + " ".join(map(str, face)) for face in f]
        count += len(v)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[mannequin] wrote {args.out}: {count} vertices, height {args.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
