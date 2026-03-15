"""Convert GameFactory (GF-Minecraft) metadata to pose JSON for WorldCompass and AR training.

Reads metadata/*.json (actions: pos, pitch, yaw per frame), builds c2w from Minecraft
Y-up + yaw/pitch convention, and writes per-clip pose files in both formats:
- WorldCompass: {"extrinsic": c2w 4x4, "K": 3x3}
- AR dataset:    {"w2c": 4x4, "intrinsic": 3x3} (same K, raw; AR code normalizes on load)

Usage:
    python prepare_gamefactory_pose.py \
        --data_dir /path/to/game_factory/data_2003/data_2003 \
        --output_pose_dir /path/to/output/poses \
        --max_frames 2000
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from tqdm import tqdm

# Fixed K matching prepare_custom_action (960x540 view)
DEFAULT_K = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]


def parse_args():
    p = argparse.ArgumentParser(description="GF-Minecraft metadata -> pose JSON (WorldCompass + AR).")
    p.add_argument("--data_dir", type=str, required=True, help="Dir with metadata/ (and optionally video/).")
    p.add_argument("--output_pose_dir", type=str, required=True, help="Output dir for poses/<clip_name>.json")
    p.add_argument("--max_frames", type=int, default=2000, help="Max frames per clip to export (1..max_frames).")
    p.add_argument("--skip_existing", action="store_true", help="Skip clip if pose file already exists.")
    return p.parse_args()


def yaw_pitch_to_c2w(pos_xyz, yaw_deg: float, pitch_deg: float):
    """Build 4x4 c2w (camera-to-world) from Minecraft pos and angles.

    Minecraft: Y up, yaw=0 => -Z (south), pitch=0 => horizontal.
    Camera convention: right, up, -forward as columns; translation = pos.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    # Forward in world: -Z when yaw=0, pitch=0
    cx, cy = math.cos(p), math.cos(y)
    sx, sy = math.sin(p), math.sin(y)
    forward = (-sy * cx, sx, -cy * cx)
    up_w = (0.0, 1.0, 0.0)
    # Right = up x forward (right-handed)
    right = (
        up_w[1] * forward[2] - up_w[2] * forward[1],
        up_w[2] * forward[0] - up_w[0] * forward[2],
        up_w[0] * forward[1] - up_w[1] * forward[0],
    )
    # View direction for camera is -forward
    R = [
        [right[0], up_w[0], -forward[0]],
        [right[1], up_w[1], -forward[1]],
        [right[2], up_w[2], -forward[2]],
    ]
    x, y, z = pos_xyz[0], pos_xyz[1], pos_xyz[2]
    return [
        [R[0][0], R[0][1], R[0][2], x],
        [R[1][0], R[1][1], R[1][2], y],
        [R[2][0], R[2][1], R[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def inv4x4(m):
    """Invert 4x4 matrix (list of lists)."""
    return np.linalg.inv(np.array(m)).tolist()


def process_clip(metadata_path: str, out_path: str, max_frames: int, K: list, skip_existing: bool) -> bool:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    actions = data.get("actions") or {}
    if not actions:
        return False
    keys_sorted = sorted([k for k in actions if k.isdigit()], key=int)
    if not keys_sorted:
        return False
    out = {}
    for i, k in enumerate(keys_sorted):
        if i >= max_frames:
            break
        a = actions[k]
        pos = a.get("pos", [0.0, 0.0, 0.0])
        yaw = float(a.get("yaw", 0.0))
        pitch = float(a.get("pitch", 0.0))
        c2w = yaw_pitch_to_c2w(pos, yaw, pitch)
        w2c = inv4x4(c2w)
        # WorldCompass: extrinsic + K; AR: w2c + intrinsic (same raw K)
        out[k] = {
            "extrinsic": c2w,
            "K": K,
            "w2c": w2c,
            "intrinsic": K,
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return True


def main():
    args = parse_args()
    meta_dir = os.path.join(args.data_dir, "metadata")
    if not os.path.isdir(meta_dir):
        print(f"Error: metadata dir not found: {meta_dir}", file=sys.stderr)
        sys.exit(1)
    pose_dir = os.path.join(args.output_pose_dir, "poses")
    os.makedirs(pose_dir, exist_ok=True)

    names = []
    for f in os.listdir(meta_dir):
        if f.endswith(".json"):
            base = f[:-5]
            names.append((os.path.join(meta_dir, f), base))
    names.sort(key=lambda x: x[1])

    done = 0
    for meta_path, base in tqdm(names, desc="pose"):
        # Match video naming: seed_N_part_N -> pose file seed_N_part_N.json
        out_path = os.path.join(pose_dir, base + ".json")
        if args.skip_existing and os.path.isfile(out_path):
            done += 1
            continue
        if process_clip(meta_path, out_path, args.max_frames, DEFAULT_K, args.skip_existing):
            done += 1
    print(f"Wrote {done} pose files under {pose_dir}")


if __name__ == "__main__":
    main()
