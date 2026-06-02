"""Overlay two (or more) MSV curves with shaded min-max bands — for
comparing motion types (e.g. camera-motion vs scene-motion videos),
mirroring the AC3D paper's purple-vs-orange plot.

Each `--input NAME=PATH=COLOR` tuple loads a per-video MSV matrix
(`(N, n_bins)` from `calculate_msv*.py`), computes mean / min / max,
and plots them on shared axes.

Usage:
    python scripts/plot_msv_compare.py \\
        --input "camera=DATA/.../msv_camera.npy=purple" \\
        --input "scene=DATA/.../msv_scene.npy=orange" \\
        --plot DATA/.../msv_compare.png --log
"""
import argparse

import numpy as np


def parse_input(spec: str) -> tuple[str, str, str | None, float | None, int | None]:
    """`NAME=PATH[=COLOR[=FPS[=WINDOW]]]`. FPS/WINDOW override globals per-group."""
    parts = spec.split("=")
    if len(parts) == 2:
        return parts[0], parts[1], None, None, None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], None, None
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], float(parts[3]), None
    if len(parts) == 5:
        return parts[0], parts[1], parts[2], float(parts[3]), int(parts[4])
    raise ValueError(f"--input must be NAME=PATH[=COLOR[=FPS[=WINDOW]]], got: {spec}")


def main():
    ap = argparse.ArgumentParser(description="Overlay MSV curves with min-max bands")
    ap.add_argument("--input", action="append", required=True,
                    help="NAME=PATH[=COLOR] — repeat for each group")
    ap.add_argument("--plot", required=True)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--title", default="Motion Spectral Volume")
    ap.add_argument("--xlabel", default=None,
                    help="override default x-axis label")
    ap.add_argument("--fps", type=float, default=None,
                    help="video FPS. With --window_size, x-axis = Hz")
    ap.add_argument("--window_size", type=int, default=None)
    ap.add_argument("--amplitude", action="store_true",
                    help="sqrt power → amplitude on y-axis")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))

    for spec in args.input:
        name, path, color, group_fps, group_window = parse_input(spec)
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected (N, n_bins), got {arr.shape}")
        if args.amplitude:
            arr = np.sqrt(np.maximum(arr, 0.0))
        N, n_bins = arr.shape
        mean_v = arr.mean(axis=0)
        min_v = arr.min(axis=0)
        max_v = arr.max(axis=0)
        eff_fps = group_fps if group_fps is not None else args.fps
        eff_window = group_window if group_window is not None else args.window_size
        if eff_fps is not None and eff_window is not None:
            x = np.arange(n_bins) * (eff_fps / eff_window)
        else:
            x = np.linspace(0, 1, n_bins)
        plt.fill_between(x, min_v, max_v, color=color, alpha=0.20)
        plt.plot(x, mean_v, color=color, lw=2.0, label=f"{name}  (n={N})")

    if args.log:
        plt.yscale("log")
    if args.xlabel:
        plt.xlabel(args.xlabel)
    elif args.fps is not None and args.window_size is not None:
        plt.xlabel("temporal frequency  (Hz)")
    else:
        plt.xlabel("temporal frequency  (0=low, 1=Nyquist)")
    plt.ylabel("amplitude" if args.amplitude else "motion energy")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"[saved] {args.plot}")


if __name__ == "__main__":
    main()
