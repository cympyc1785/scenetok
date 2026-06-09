"""Build `dynamicverse_{standard,unseen}.json` evaluation indices.

Scans `WorldTraj/dynamicverse/*` for scenes that have all of
`inpaint_result.mp4` + `cameras.json` + a non-empty `prompt_scene` in
`prompts.json`. Splits into:

  - **standard**: random subset (size `--n_standard`, seed `--seed`) of scenes
    from the "seen" pool (all subdatasets except `unseen_subdatasets`).
    Train uses the full seen pool minus this standard subset.
  - **unseen**: random subset (`--n_unseen`) of scenes from `unseen_subdatasets`
    (default `DAVIS`). Never used for training.

Each scene gets fixed context / target indices (`[0,4,9,13,...,48]` × 12 for
context, `range(37)` for target — matches Wan-VAE 4N+1 layout assuming 50f
clips). Output JSON shape mirrors the other `assets/evaluation_index/*` files:

  {scene_name: {"context": [...], "target": [...]}}
"""
import argparse
import json
import random
from pathlib import Path

DEFAULT_ROOT = Path("WorldTraj/dynamicverse")
DEFAULT_OUT = Path("assets/evaluation_index")
DEFAULT_EXCLUDED = ["dynpose-100k", "logs"]
DEFAULT_UNSEEN = ["DAVIS"]

# 12 context indices spread over [0, 48] (50-frame DAVIS-style clips).
DEFAULT_CONTEXT = [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 48]
# 37 target frames = 1 + 9*4 (Wan VAE 4N+1 with num_target_views=10, td=4).
DEFAULT_TARGET = list(range(37))


def has_valid_prompt(scene_dir: Path, prompt_file: str, prompt_key: str) -> bool:
    p = scene_dir / prompt_file
    if not p.exists():
        return False
    try:
        d = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(d, dict):
        return False
    for entry in d.values():
        if isinstance(entry, dict):
            v = entry.get(prompt_key)
            if isinstance(v, str) and v.strip():
                return True
            if isinstance(v, dict):
                for sk in ("concise", "detail"):
                    if isinstance(v.get(sk), str) and v.get(sk).strip():
                        return True
    return False


def collect(
    root: Path,
    excluded: set[str],
    unseen: set[str],
    video_name: str,
    camera_file: str,
    prompt_file: str,
    prompt_key: str,
) -> tuple[list[str], list[str]]:
    seen_scenes: list[str] = []
    unseen_scenes: list[str] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name in excluded:
            continue
        bucket = unseen_scenes if sub.name in unseen else seen_scenes
        for scene_dir in sorted(sub.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not (scene_dir / video_name).exists():
                continue
            if not (scene_dir / camera_file).exists():
                continue
            if not has_valid_prompt(scene_dir, prompt_file, prompt_key):
                continue
            bucket.append(scene_dir.name)
    return seen_scenes, unseen_scenes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--excluded", nargs="*", default=DEFAULT_EXCLUDED)
    p.add_argument("--unseen", nargs="*", default=DEFAULT_UNSEEN)
    p.add_argument("--video_name", default="inpaint_result.mp4")
    p.add_argument("--camera_file", default="cameras.json")
    p.add_argument("--prompt_file", default="prompts.json")
    p.add_argument("--prompt_key", default="prompt_scene")
    p.add_argument("--n_standard", type=int, default=32)
    p.add_argument("--n_unseen", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seen, unseen = collect(
        args.root,
        set(args.excluded),
        set(args.unseen),
        args.video_name,
        args.camera_file,
        args.prompt_file,
        args.prompt_key,
    )
    print(f"seen pool : {len(seen)} scenes")
    print(f"unseen pool: {len(unseen)} scenes")

    rng = random.Random(args.seed)
    standard = sorted(rng.sample(seen, min(args.n_standard, len(seen))))
    unseen_pick = sorted(rng.sample(unseen, min(args.n_unseen, len(unseen))))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, scenes in [
        ("dynamicverse_standard.json", standard),
        ("dynamicverse_unseen.json", unseen_pick),
    ]:
        obj = {
            s: {"context": list(DEFAULT_CONTEXT), "target": list(DEFAULT_TARGET)}
            for s in scenes
        }
        path = args.out_dir / name
        with path.open("w") as f:
            json.dump(obj, f, indent=2)
        print(f"wrote {path}: {len(obj)} scenes")


if __name__ == "__main__":
    main()
