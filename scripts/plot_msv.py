"""Plot Motion Spectral Volume curves from a saved per-video MSV matrix.

Input: `.npy` of shape `(N, n_bins)` produced by `scripts/calculate_msv.py`
(each row is the radial MSV for one video).

Output: line plot of the mean curve plus a shaded min–max band over all videos.

Usage:
    python scripts/plot_msv.py \\
        --npy DATA/T2V/videos/wan_ti2v_5b_msv.npy \\
        --plot DATA/T2V/videos/wan_ti2v_5b_msv.png \\
        --log
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Plot MSV mean ± min/max band")
    ap.add_argument("--npy", required=True, help="(N, n_bins) saved MSV matrix")
    ap.add_argument("--plot", required=True, help=".png output path")
    ap.add_argument("--log", action="store_true", help="log-scale y axis")
    ap.add_argument("--title", default="Motion Spectral Volume")
    ap.add_argument("--color", default="tab:blue")
    ap.add_argument("--fps", type=float, default=None,
                    help="video FPS. If set together with --window_size, x-axis is in Hz "
                         "(bin k → k * fps / window_size). Otherwise normalized 0-1.")
    ap.add_argument("--window_size", type=int, default=None,
                    help="window size that was used in calculate_msv_temporal.py "
                         "(needed together with --fps for Hz x-axis)")
    ap.add_argument("--amplitude", action="store_true",
                    help="treat npy as power (|FFT|^2) and plot sqrt as amplitude (|FFT|)")
    ap.add_argument("--ylabel", default=None, help="override y-axis label")
    ap.add_argument("--xlabel", default=None, help="override x-axis label")
    args = ap.parse_args()

    arr = np.load(args.npy)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (N, n_bins), got shape {arr.shape}")
    N, n_bins = arr.shape

    if args.amplitude:
        arr = np.sqrt(np.maximum(arr, 0.0))

    mean_msv = arr.mean(axis=0)
    min_msv = arr.min(axis=0)
    max_msv = arr.max(axis=0)

    if args.fps is not None and args.window_size is not None:
        x = np.arange(n_bins) * (args.fps / args.window_size)
        default_xlabel = "temporal frequency  (Hz)"
    else:
        x = np.linspace(0, 1, n_bins)
        default_xlabel = "normalized frequency  (0=low, 1=high)"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.fill_between(x, min_msv, max_msv, color=args.color, alpha=0.25, label="min–max range")
    plt.plot(x, mean_msv, color=args.color, lw=2.0, label="mean MSV")
    if args.log:
        plt.yscale("log")
    plt.xlabel(args.xlabel or default_xlabel)
    plt.ylabel(args.ylabel or ("amplitude" if args.amplitude else "motion energy"))
    plt.title(f"{args.title}  (n={N} videos)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"[saved] {args.plot}")


if __name__ == "__main__":
    main()
