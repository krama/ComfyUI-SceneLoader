#!/usr/bin/env python3
"""Give skeleton-only pose blocks a figure image (for the 3D pose path), via a running ComfyUI.

Pose blocks with a skeleton but no image (import_pose_inbox.py) work as 2D ControlNet maps. For
each one picked (--only / --limit) this uploads the skeleton, queues a small graph (checkpoint ->
ControlNet OpenPose -> KSampler -> VAE decode -> SaveImage, plus NLF on the result as a check) and
waits. When NLF finds a person, the figure is saved as <slug>.png next to the block and its yaml
gets image: (and figure_seed:); when it does not, the next seed is tried.
Stdlib + PyYAML (the ComfyUI venv has it). Uses the GPU of the ComfyUI it talks to.
"""
import argparse
import fnmatch
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

import yaml

UPLOAD_SUBFOLDER = "sceneloader_pending"
POSITIVE = ("masterpiece, best quality, 1girl, solo, full body, plain grey bodysuit, barefoot, "
            "simple background, white background")
NEGATIVE = ("lowres, bad anatomy, bad hands, extra limbs, missing limbs, multiple people, 2girls, "
            "cropped, out of frame, text, watermark, worst quality, low quality")


def log(msg):
    print(f"[skeleton-to-figure] {msg}", flush=True)


# Return [(block yaml, ref, skeleton png)] for pose blocks with a skeleton and no image.
def find_pending(library, only=()):
    out = []
    root = os.path.join(library, "blocks")
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
            ref = os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, "/")
            if only and not any(fnmatch.fnmatch(ref, g) for g in only):
                continue
            out.append((path, ref, os.path.join(d, str(data["skeleton"]))))
    return out


# Latent size near 1 MP with the skeleton's aspect, multiples of 64.
def latent_size(width, height, megapixels=1.0):
    s = (megapixels * 1_000_000 / (width * height)) ** 0.5
    return max(64, round(width * s / 64) * 64), max(64, round(height * s / 64) * 64)


# API prompt: skeleton -> ControlNet OpenPose -> figure image -> SaveImage, NLF as the check.
def build_prompt(a, image_name, width, height, seed, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.checkpoint}},
        "2": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": a.clip_skip}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": a.positive, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["2", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": a.controlnet}},
        "7": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "control_net": ["6", 0], "image": ["5", 0],
            "vae": ["9", 0], "strength": a.strength, "start_percent": 0.0, "end_percent": 1.0}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
        "10": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0],
            "seed": seed, "steps": a.steps, "cfg": a.cfg, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["9", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
        "13": {"class_type": "LoadSMPLX", "inputs": {
            "model_source": "local", "gender": "neutral", "model_path": a.smplx_dir, "hf_token": ""}},
        "14": {"class_type": "LoadNLF", "inputs": {
            "smplx_model": ["13", 0], "model_source": "local", "model_path": a.nlf_dir, "hf_token": "",
            "device": "auto"}},
        "15": {"class_type": "NLFSMPLXEstimator", "inputs": {"model": ["14", 0], "image": ["11", 0]}},
    }


class Comfy:
    """Minimal ComfyUI HTTP client: upload, queue, wait, download."""

    def __init__(self, server):
        self.server = server.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _req(self, path, data=None, headers=None):
        req = urllib.request.Request(self.server + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    # Upload an image to <input>/<subfolder>/<name>; returns the LoadImage value.
    def upload(self, path, name, subfolder):
        boundary = uuid.uuid4().hex
        with open(path, "rb") as fh:
            data = fh.read()
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in (("subfolder", subfolder), ("overwrite", "true"), ("type", "input"))]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                     f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        res = json.loads(self._req("/upload/image", body,
                                   {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def queue(self, prompt):
        body = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode()
        res = json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))
        if res.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected the graph: {res['node_errors']}")
        return res["prompt_id"]

    # Wait for a prompt; returns its history entry.
    def wait(self, prompt_id, timeout, poll=2.0):
        end = time.time() + timeout
        while time.time() < end:
            entry = json.loads(self._req(f"/history/{prompt_id}")).get(prompt_id)
            if entry and entry.get("status", {}).get("completed") is not None:
                return entry
            time.sleep(poll)
        raise TimeoutError(f"prompt {prompt_id} not done in {timeout}s")

    def download(self, image):
        q = urllib.parse.urlencode({k: image.get(k, "") for k in ("filename", "subfolder", "type")})
        return self._req(f"/view?{q}")


# Error message of a failed history entry, or None when it succeeded.
def failure(entry):
    status = entry.get("status", {})
    if status.get("status_str") == "success":
        return None
    errs = [m[1].get("exception_message", "") for m in status.get("messages", []) if m[0] == "execution_error"]
    return "; ".join(e.strip() for e in errs) or status.get("status_str") or "unknown error"


# Make the figure for one skeleton-only block; returns True when its yaml got an image.
def convert(comfy, a, block, name, skeleton):
    with open(skeleton, "rb") as fh:  # PNG IHDR: width, height
        head = fh.read(24)
    width, height = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    w, h = latent_size(width, height)
    image_name = comfy.upload(skeleton, f"{name}.png", UPLOAD_SUBFOLDER)
    for t in range(a.tries):
        seed = a.seed + t
        pid = comfy.queue(build_prompt(a, image_name, w, h, seed, f"sceneloader/pose_figures/{name}"))
        entry = comfy.wait(pid, a.timeout)
        err = failure(entry)
        if err:
            log(f"{name}: seed {seed} failed: {err[:160]}")
            continue
        images = entry.get("outputs", {}).get("12", {}).get("images", [])
        if not images:
            log(f"{name}: seed {seed}: no image in the SaveImage output")
            continue
        with open(block, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        data["image"] = os.path.splitext(os.path.basename(block))[0] + ".png"
        data["figure_seed"] = seed
        with open(os.path.join(os.path.dirname(block), data["image"]), "wb") as fh:
            fh.write(comfy.download(images[0]))
        with open(block, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", required=True, help="Scene Loader library root")
    ap.add_argument("--smplx-dir", required=True, help="SMPL-X model folder (LoadSMPLX model_path)")
    ap.add_argument("--nlf-dir", required=True, help="NLF model folder (LoadNLF model_path)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--checkpoint", default="illustrious/lilithsLullaby_v20.safetensors")
    ap.add_argument("--vae", default="SDXL_anime_natural_vae.safetensors")
    ap.add_argument("--controlnet", default="noobaiXLControlnet_openposeModel.safetensors")
    ap.add_argument("--positive", default=POSITIVE)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--clip-skip", type=int, default=-2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=1.0, help="ControlNet strength")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tries", type=int, default=3, help="seeds to try when NLF finds no person")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per generation")
    ap.add_argument("--only", action="append", default=[], metavar="GLOB",
                    help="block refs to convert (repeatable), e.g. 'poses/civitai/sportlikeposes*'")
    ap.add_argument("--limit", type=int, default=0, help="convert at most N skeletons (0 = all)")
    a = ap.parse_args(argv)
    pending = find_pending(a.library, a.only)
    if a.limit:
        pending = pending[:a.limit]
    if not pending:
        log("nothing queued")
        return 0
    comfy = Comfy(a.server)
    done = failed = 0
    for block, ref, skeleton in pending:
        log(f"{ref}: generating figure ({done + failed + 1}/{len(pending)})")
        if convert(comfy, a, block, os.path.basename(ref), skeleton):
            done += 1
            log(f"{ref}: ok")
        else:
            failed += 1
            log(f"{ref}: no figure after {a.tries} seeds, stays skeleton-only (try --seed / --strength)")
    log(f"{done} converted, {failed} still skeleton-only")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
