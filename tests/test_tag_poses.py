"""Pose tag filter of tools/tag_poses.py (no tagger model needed). Run: python tests/test_tag_poses.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import tag_poses as tp  # noqa: E402

KEEP = ["sitting", "standing", "crossed legs", "hand on own hip", "arms up", "looking back", "from side",
        "on stomach", "all fours", "v", "contrapposto", "spread legs", "head tilt", "knees up", "kneeling"]
DROP = ["1girl", "long hair", "blue eyes", "thighhighs", "arm warmers", "hair ribbon", "holding hands",
        "black gloves", "breasts", "smile", "sleeveless", "high heels", "bracelet"]

if __name__ == "__main__":
    bad = [t for t in KEEP if not tp.is_pose_tag(t)] + [t for t in DROP if tp.is_pose_tag(t)]
    assert not bad, bad
    print("ok  is_pose_tag")
