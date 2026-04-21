#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def count_cameras(path: Path) -> int:
    with open(path, "r") as f:
        return len(json.load(f))


def read_prompt(path: Path) -> str:
    if not path.exists():
        return ""

    with open(path, "r") as f:
        data = json.load(f)

    value = data.get("prompt_dynamic_object", "")
    if isinstance(value, str):
        return value
    return ""


def build_rows(
    root: Path,
    camera_file: str,
    context_video_name: str,
    target_video_name: str,
    prompt_file: str,
    train_split_ratio: float,
) -> list[dict[str, str | int]]:
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    valid_scene_dirs = []
    for scene_dir in scene_dirs:
        if not (scene_dir / camera_file).exists():
            continue
        if not (scene_dir / context_video_name).exists():
            continue
        if not (scene_dir / target_video_name).exists():
            continue
        valid_scene_dirs.append(scene_dir)

    total = len(valid_scene_dirs)
    if total == 0:
        return []

    split_idx = int(round(total * train_split_ratio))
    split_idx = min(max(split_idx, 1), max(total - 1, 1)) if total > 1 else total

    rows = []
    for idx, scene_dir in enumerate(valid_scene_dirs):
        split = "train"
        if total > 1 and idx >= split_idx:
            split = "val"

        rows.append(
            {
                "scene": scene_dir.name,
                "split": split,
                "num_cameras": count_cameras(scene_dir / camera_file),
                "has_context_video": 1,
                "has_target_video": 1,
                "prompt_dynamic_object": read_prompt(scene_dir / prompt_file),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a meta.csv for filtering DAVIS scenes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the DAVIS root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument("--camera-file", default="cameras.json")
    parser.add_argument("--context-video-name", default="inpaint_result_effecterase.mp4")
    parser.add_argument("--target-video-name", default="video_input.mp4")
    parser.add_argument("--prompt-file", default="dynamic_prompts.json")
    parser.add_argument("--train-split-ratio", type=float, default=0.9)
    args = parser.parse_args()

    rows = build_rows(
        root=args.root,
        camera_file=args.camera_file,
        context_video_name=args.context_video_name,
        target_video_name=args.target_video_name,
        prompt_file=args.prompt_file,
        train_split_ratio=args.train_split_ratio,
    )
    if not rows:
        raise ValueError(f"No valid DAVIS scenes found under {args.root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "split",
        "num_cameras",
        "has_context_video",
        "has_target_video",
        "prompt_dynamic_object",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} DAVIS scenes to {args.output}")


if __name__ == "__main__":
    main()
