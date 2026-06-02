"""Build three video path-list files for AC3D-style MSV comparison:

  1) camera-motion only           — random DL3DV 1K scenes (static scene
                                     filmed with moving camera). Path entries
                                     are scene directories; `read_frames`
                                     reads `images/frame_*.png` from each.
  2) scene-motion only            — dynamicverse videos with all
                                     `camera_tags_per_seg.json` segments
                                     labeled `static + static`.
  3) camera+scene motion          — dynamicverse videos with at least one
                                     non-static segment label.

Sampling is reproducible via `--seed`. The lists are written to
`<out_dir>/{dl3dv_camera, dynamicverse_scene, dynamicverse_camera_scene}.txt`
with one absolute path per line.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
DL3DV_ROOT = REPO_ROOT / "DATA/DL3DV/DL3DV-960/train/1K"
DYN_ROOT = REPO_ROOT / "WorldTraj/dynamicverse"
DYN_SUBSETS = [
    "DAVIS", "dynamic_replica", "MOSE", "MVS-Synth",
    "SAV", "spring", "uvo", "VOST", "youtube_vis",
]


def _read_blacklist(root: Path) -> set[str]:
    p = root / "blacklist.csv"
    if not p.exists():
        return set()
    import csv
    with p.open() as f:
        rdr = csv.DictReader(f)
        return {row.get("scene", "") for row in rdr if row.get("scene")}


def collect_dl3dv(n: int, seed: int) -> list[str]:
    blacklist = _read_blacklist(REPO_ROOT / "DATA/DL3DV/DL3DV-960/train")
    candidates = []
    for scene in sorted(DL3DV_ROOT.iterdir()):
        if not scene.is_dir():
            continue
        if scene.name in blacklist:
            continue
        # Need at least an `images/` subdir with png frames
        if not (scene / "images").is_dir():
            continue
        candidates.append(scene)
    print(f"[dl3dv] candidates after blacklist+image-dir filter: {len(candidates)}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked = candidates[:n]
    return [str(p.absolute()) for p in picked]


def collect_dynamicverse(static_only: bool, n: int, seed: int) -> list[str]:
    """If `static_only`, scenes whose every segment label is 'static + static'.
    Otherwise scenes with at least one non-static segment."""
    pool = []
    for subset in DYN_SUBSETS:
        for tag_path in sorted((DYN_ROOT / subset).glob("*/viz/camera_tags_per_seg.json")):
            scene_dir = tag_path.parents[1]
            vid = scene_dir / "video_input.mp4"
            if not vid.exists():
                continue
            try:
                with tag_path.open() as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            is_static = all(
                v.get("label", {}).get("trans+rot") == "static + static"
                for v in d.values()
            )
            if static_only == is_static:
                pool.append(vid)
    label = "static (scene-motion only)" if static_only else "non-static (camera+scene)"
    print(f"[dynamicverse {label}] pool: {len(pool)}")
    rng = random.Random(seed)
    rng.shuffle(pool)
    picked = pool[:n]
    return [str(p.absolute()) for p in picked]


def write_list(paths: list[str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"[saved] {out_path}  ({len(paths)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="per-bucket count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="DATA/T2V/motion_lists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    dl3dv = collect_dl3dv(args.n, args.seed)
    dyn_scene = collect_dynamicverse(static_only=True, n=args.n, seed=args.seed + 1)
    dyn_camsc = collect_dynamicverse(static_only=False, n=args.n, seed=args.seed + 2)

    if len(dl3dv) < args.n:
        print(f"[warn] DL3DV: only {len(dl3dv)} of {args.n} available")
    if len(dyn_scene) < args.n:
        print(f"[warn] dynamicverse scene-motion: only {len(dyn_scene)} of {args.n}")
    if len(dyn_camsc) < args.n:
        print(f"[warn] dynamicverse camera+scene: only {len(dyn_camsc)} of {args.n}")

    write_list(dl3dv, out_dir / "dl3dv_camera.txt")
    write_list(dyn_scene, out_dir / "dynamicverse_scene.txt")
    write_list(dyn_camsc, out_dir / "dynamicverse_camera_scene.txt")


if __name__ == "__main__":
    main()
