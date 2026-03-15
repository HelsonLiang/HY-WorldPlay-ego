"""Build image + caption list from GameFactory (GF-Minecraft) for prepare_image_text_latent_simple.

Reads annotation.csv and extracts one frame per clip from the corresponding video,
then writes a JSON in the format expected by prepare_image_text_latent_simple.py:
  [{"image_path": "/path/to/image.jpg", "caption": "..."}, ...]

Usage:
    python prepare_gamefactory_image_caption.py \
        --data_dir /path/to/game_factory/data_2003 \
        --output_image_dir /path/to/output/frames \
        --output_json /path/to/output/image_caption_list.json \
        --frame_mode start

    # With optional args
    python prepare_gamefactory_image_caption.py \
        --data_dir /path/to/data_2003 \
        --output_image_dir ./frames \
        --output_json ./image_caption_list.json \
        --frame_mode middle \
        --skip_existing \
        --image_ext .jpg
"""

import argparse
import csv
import json
import os
import re
import sys

# Optional: use OpenCV for frame extraction
try:
    import cv2
except ImportError:
    cv2 = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build image_path + caption JSON from GameFactory annotation and videos."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root of a GF-Minecraft split (e.g. data_2003 or data_269), "
        "containing video/, annotation.csv.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Directory to save extracted frames (created if missing).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON path: list of {image_path, caption} for prepare_image_text_latent_simple.",
    )
    parser.add_argument(
        "--frame_mode",
        type=str,
        default="start",
        choices=["start", "middle", "end"],
        help="Which frame to extract per clip: start, middle, or end.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip writing image if file already exists (still add to JSON).",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Extension for saved images, e.g. .jpg or .png.",
    )
    parser.add_argument(
        "--image_quality",
        type=int,
        default=95,
        help="JPEG quality when image_ext is .jpg (1-100). Ignored for PNG.",
    )
    return parser.parse_args()


def _norm_cols(row):
    """Normalize CSV column names (strip spaces)."""
    return {k.strip(): v for k, v in row.items()}


def _resolve_video_path(video_dir: str, name: str) -> str | None:
    """Return path to video file; name may or may not have .mp4.

    GF-Minecraft CSV may list 'seed_N_part.mp4' while files are 'seed_N_part_N.mp4'.
    """
    base = name.strip()
    if not base:
        return None
    base = base.replace(".mp4", "")
    candidates = [
        base + ".mp4",
        base,
        base + ".mp4",
    ]
    # GF-Minecraft: seed_1_part -> seed_1_part_1.mp4
    m = re.match(r"^seed_(\d+)_part$", base, re.IGNORECASE)
    if m:
        n = m.group(1)
        candidates.insert(0, f"seed_{n}_part_{n}.mp4")
    for candidate in candidates:
        path = os.path.join(video_dir, candidate)
        if os.path.isfile(path):
            return path
    return None


def _frame_index(row, frame_mode: str, get_col) -> int:
    """Compute 0-based frame index from annotation row."""
    try:
        start = int(get_col(row, "Start frame index") or 0)
    except (ValueError, TypeError):
        start = 0
    try:
        end = int(get_col(row, "End frame index") or start)
    except (ValueError, TypeError):
        end = start
    if frame_mode == "start":
        return max(0, start)
    if frame_mode == "end":
        return max(0, end)
    return max(0, (start + end) // 2)


def extract_frame(video_path: str, frame_idx: int, out_path: str, jpeg_quality: int) -> bool:
    """Extract one frame from video and save to out_path. Returns True on success."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for frame extraction. Install with: pip install opencv-python")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    # OpenCV uses BGR; save as-is (JPEG/PNG accept common formats)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imwrite(out_path, frame)


def main():
    args = parse_args()
    video_dir = os.path.join(args.data_dir, "video")
    ann_path = os.path.join(args.data_dir, "annotation.csv")

    if not os.path.isdir(video_dir):
        print(f"Error: video directory not found: {video_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(ann_path):
        print(f"Error: annotation file not found: {ann_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_image_dir, exist_ok=True)

    # CSV columns: Original video name, Start frame index, End frame index, Prompt
    with open(ann_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [_norm_cols(r) for r in reader]

    if not rows:
        print("Warning: annotation.csv is empty.", file=sys.stderr)
        out_list = []
    else:
        # Infer column names (allow slight variants)
        def get_col(r, *keys):
            for k in keys:
                for dk in r:
                    if dk.strip().lower().replace(" ", "") == k.lower().replace(" ", ""):
                        return r[dk]
            return ""

        out_list = []
        for i, row in enumerate(rows):
            name = get_col(row, "Original video name")
            prompt = get_col(row, "Prompt")
            if not name:
                continue
            video_path = _resolve_video_path(video_dir, name)
            if not video_path:
                print(f"Skip (video not found): {name}", file=sys.stderr)
                continue
            frame_idx = _frame_index(row, args.frame_mode, get_col)
            # Safe filename: avoid path separators and long names
            base_name = os.path.splitext(os.path.basename(name))[0]
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in base_name)
            image_name = f"{safe_name}_f{frame_idx}{args.image_ext}"
            image_path = os.path.join(args.output_image_dir, image_name)
            image_path_abs = os.path.abspath(image_path)

            if not args.skip_existing or not os.path.isfile(image_path):
                if not extract_frame(
                    video_path, frame_idx, image_path, args.image_quality
                ):
                    print(f"Skip (frame extract failed): {video_path} frame {frame_idx}", file=sys.stderr)
                    continue
            out_list.append({"image_path": image_path_abs, "caption": prompt or ""})

        print(f"Processed {len(out_list)} clips, wrote images to {args.output_image_dir}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out_list)} items to {args.output_json}")


if __name__ == "__main__":
    main()
