"""Classify videos into motion types using optical flow heuristics.

Approach
--------
For each video, compute consecutive-frame optical flow `(T-1, H, W, 2)`.
Decompose into:
  - `global`   = per-frame mean flow vector (over all pixels)  → camera-induced
  - `residual` = flow − global, per pixel                       → scene-induced

Then:
  - `global_mag`   = mean ‖global‖  over time           (camera motion magnitude)
  - `residual_mag` = mean ‖residual‖ over time & space (scene motion magnitude)

Classification rules (defaults; thresholds tunable via CLI):
  - both `global_mag < static_eps` and `residual_mag < static_eps`  → "static"
  - `global_mag  ≥ camera_ratio × residual_mag` AND `global_mag  > active_eps`  → "camera"
  - `residual_mag ≥ scene_ratio  × global_mag`  AND `residual_mag > active_eps` → "scene"
  - otherwise                                                                  → "both"

Output
------
1) `<out_dir>/motion_classification.csv` with columns
   `video,label,global_mag,residual_mag,ratio_g_over_r`
2) Optional: symlink / copy videos into `<out_dir>/<label>/<basename>.mp4`
   so downstream MSV scripts can glob them by group.

Usage
-----
    python scripts/classify_motion_type.py \\
        --videos "DATA/T2V/videos/wan_ti2v_5b/*.mp4" \\
        --out_dir DATA/T2V/videos/wan_ti2v_5b_by_motion \\
        --link symlink

Then run MSV analysis per group, e.g.:
    python scripts/calculate_msv_temporal.py \\
        --videos "DATA/T2V/videos/wan_ti2v_5b_by_motion/camera/*.mp4" \\
        --save_npy .../msv_temporal_camera.npy
"""
import argparse
import csv
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_msv import read_frames, flow_farneback, flow_raft  # noqa: E402


def classify_video(
    video_path: str,
    flow_backend: str,
    device: str,
    resize: int | None,
    max_frames: int | None,
    static_eps: float,
    active_eps: float,
    camera_ratio: float,
    scene_ratio: float,
) -> dict:
    frames = read_frames(video_path, max_frames=max_frames, resize=resize)
    if flow_backend == "farneback":
        flows = flow_farneback(frames)
    elif flow_backend == "raft":
        flows = flow_raft(frames, device=device)
    else:
        raise ValueError(f"Unknown flow backend: {flow_backend}")
    # flows: (T-1, H, W, 2)
    global_flow = flows.mean(axis=(1, 2))                # (T-1, 2)
    residual = flows - global_flow[:, None, None, :]     # (T-1, H, W, 2)
    global_mag = float(np.linalg.norm(global_flow, axis=-1).mean())
    residual_mag = float(np.linalg.norm(residual, axis=-1).mean())

    # Label
    if global_mag < static_eps and residual_mag < static_eps:
        label = "static"
    elif global_mag >= camera_ratio * residual_mag and global_mag > active_eps:
        label = "camera"
    elif residual_mag >= scene_ratio * global_mag and residual_mag > active_eps:
        label = "scene"
    else:
        label = "both"

    return {
        "video": os.path.basename(video_path),
        "label": label,
        "global_mag": global_mag,
        "residual_mag": residual_mag,
        "ratio_g_over_r": global_mag / (residual_mag + 1e-9),
    }


def main():
    ap = argparse.ArgumentParser(description="Motion-type classifier via flow heuristics")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow", choices=["farneback", "raft"], default="farneback")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--static_eps", type=float, default=0.3,
                    help="below both global & residual → 'static'")
    ap.add_argument("--active_eps", type=float, default=0.5,
                    help="motion magnitude must exceed this to count as active")
    ap.add_argument("--camera_ratio", type=float, default=2.0,
                    help="global_mag must be ≥ this × residual_mag for 'camera'")
    ap.add_argument("--scene_ratio", type=float, default=1.5,
                    help="residual_mag must be ≥ this × global_mag for 'scene'")
    ap.add_argument("--link", choices=["none", "symlink", "copy"], default="symlink",
                    help="how to group videos into <out_dir>/<label>/ subdirs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("camera", "scene", "both", "static"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    for pat in args.videos:
        paths.extend(sorted(glob.glob(pat)) or [pat])

    rows = []
    counts = {"camera": 0, "scene": 0, "both": 0, "static": 0}
    for i, p in enumerate(paths, 1):
        try:
            r = classify_video(
                p,
                flow_backend=args.flow,
                device=args.device,
                resize=(None if args.resize < 0 else args.resize),
                max_frames=args.max_frames,
                static_eps=args.static_eps,
                active_eps=args.active_eps,
                camera_ratio=args.camera_ratio,
                scene_ratio=args.scene_ratio,
            )
        except Exception as exc:
            print(f"[err] {os.path.basename(p)}: {exc}")
            continue
        rows.append(r)
        counts[r["label"]] += 1
        print(f"[{i}/{len(paths)}] {r['video']:>10s}  "
              f"g={r['global_mag']:6.2f}  res={r['residual_mag']:6.2f}  "
              f"g/r={r['ratio_g_over_r']:5.2f}  → {r['label']}")

        if args.link != "none":
            src = Path(p).resolve()
            dst = out_dir / r["label"] / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.link == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)

    csv_path = out_dir / "motion_classification.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "global_mag", "residual_mag", "ratio_g_over_r"])
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"[saved] {csv_path}")
    print(f"[counts] " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
