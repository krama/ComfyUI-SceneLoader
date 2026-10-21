"""Tests for sceneloader.prompt_log. Run: python tests/test_prompt_log.py"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sceneloader.prompt_log import build_log  # noqa: E402

PROMPT = {
    "1": {"class_type": "SimpleCheckpointLoaderPlus", "inputs": {"ckpt": "illustrious/model.safetensors"}},
    "2": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": ["9", 0]}},
    "9": {"class_type": "PrimitiveInt", "inputs": {"value": -2}},
    "4": {"class_type": "SceneLoader", "inputs": {"scene": "forest_walk", "base": "illustrious",
                                                  "camera_json": ["20", 1]}},
    "7": {"class_type": "Seed (rgthree)", "inputs": {"seed": 1234}},
    "10": {"class_type": "KSampler", "_meta": {"title": "Base"},
           "inputs": {"seed": ["7", 0], "steps": 25, "cfg": 5.0, "sampler_name": "euler",
                      "scheduler": "karras", "denoise": 1.0}},
}


def test_build_log():
    text = build_log("base_00001_.png", 1344, 768, PROMPT, "pos tags", "neg tags", "info text",
                     now=datetime.datetime(2026, 10, 21, 18, 0, 0))
    for needle in ("image: base_00001_.png", "date: 2026-10-21 18:00:00", "size: 1344x768",
                   "scene: forest_walk (illustrious) · 3d camera", "model: illustrious/model.safetensors",
                   "clip skip: -2", "sampler [Base]: seed 1234, euler/karras, 25 steps, cfg 5.0, denoise 1.0",
                   "positive:\npos tags", "negative:\nneg tags", "scene assembly:\ninfo text"):
        assert needle in text, (needle, text)
    empty = build_log("x.png", 1, 1, None)
    assert "positive:\n(not connected)" in empty, empty


if __name__ == "__main__":
    test_build_log()
    print("ok  test_build_log")
