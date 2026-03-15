"""Add pose_path to each entry in latents.json for middle training / RL.

Latents from prepare_image_text_latent_simple have latent_path, image_path, caption.
Pose files from prepare_gamefactory_pose are poses/seed_N_part_N.json per video.
Image names from 2.1 are seed_N_part_f<frame>.jpg (same N as video seed_N_part_N).
We derive pose_path from image_path and write latents_with_pose.json.

Usage:
    python merge_latents_pose.py \
        --latents_json /scratch/.../latents_output_3.15/latents.json \
        --poses_dir /scratch/.../gf_prepare_output/poses \
        --output_json /scratch/.../latents_output_3.15/latents_with_pose.json
"""

import argparse
import json
import os
import re


def image_basename_to_pose_basename(basename: str) -> str | None:
    """From image basename like seed_1_part_f62.jpg return pose basename seed_1_part_1.json."""
    base = basename.rsplit(".", 1)[0] if "." in basename else basename
    # seed_N_part_f<frame> (from 2.1) -> pose seed_N_part_N.json (from 2.2, one pose per video)
    m = re.match(r"^seed_(\d+)_part(_f\d+)?$", base, re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"seed_{n}_part_{n}.json"
    return None


def main():
    p = argparse.ArgumentParser(description="Add pose_path to latents.json for GameFactory.")
    p.add_argument("--latents_json", type=str, required=True, help="Path to latents.json from prepare_image_text_latent_simple.")
    p.add_argument("--poses_dir", type=str, required=True, help="Directory containing poses/seed_N_part_N.json files.")
    p.add_argument("--output_json", type=str, required=True, help="Output path for latents_with_pose.json.")
    args = p.parse_args()

    with open(args.latents_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        items = [items]

    out = []
    missing = 0
    for item in items:
        image_path = item.get("image_path") or item.get("image_path_abs")
        if not image_path:
            missing += 1
            continue
        basename = os.path.basename(image_path)
        pose_basename = image_basename_to_pose_basename(basename)
        if not pose_basename:
            missing += 1
            continue
        pose_path = os.path.join(args.poses_dir, pose_basename)
        if not os.path.isfile(pose_path):
            missing += 1
            continue
        new_item = dict(item)
        new_item["pose_path"] = os.path.abspath(pose_path)
        out.append(new_item)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} items to {args.output_json} (skipped {missing} without pose).")


if __name__ == "__main__":
    main()
