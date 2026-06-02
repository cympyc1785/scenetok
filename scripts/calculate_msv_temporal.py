"""Temporal Motion Spectral Volume (MSV) — AC3D-style.

AC3D 논문(Bahmani et al., 2024)의 다음 분석을 재현한다:

    Average magnitude of motion spectral volumes along spatial, temporal
    offset, and video batch dimensions for scenes with different motion
    types. We compute the flow of each video in a sliding window manner
    with temporal offsets and average the frequencies across all offsets.
    Frequency refers to the temporal frequency.

레시피
-------
1) 비디오 → 연속 프레임 optical flow → 각 픽셀의 flow magnitude 시계열
   `(T-1, H, W)`.
2) 시간축을 따라 sliding window (`window_size`, `hop`)를 적용. 각 window에
   대해 Hann window 곱하고 시간축으로 1D FFT.
3) 각 window의 power = |FFT|² 를 spatial (y, x) 평균 → `(n_freq,)`.
4) 모든 sliding window들과 (선택적으로 여러 비디오)에 대해 평균 →
   최종 1D temporal-frequency curve.

논문의 "temporal offset" 차원은 **sliding-window의 시작 위치**로 해석.
DC bin은 보존하지 않음 (`np.fft.rfft`에 윈도윙을 먼저 적용 → DC = 평균
모션). detrend 별도로 안 함.

사용 예 (계산만, plot은 `scripts/plot_msv.py`)
----------------------------------------------
    python scripts/calculate_msv_temporal.py \\
        --videos "DATA/T2V/videos/wan_ti2v_5b/*.mp4" \\
        --flow farneback --window_size 32 --hop 16 \\
        --save_npy DATA/T2V/videos/wan_ti2v_5b_msv_temporal.npy

    python scripts/plot_msv.py \\
        --npy DATA/T2V/videos/wan_ti2v_5b_msv_temporal.npy \\
        --plot DATA/T2V/videos/wan_ti2v_5b_msv_temporal.png --log
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Reuse flow / video utilities from the spatial MSV script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_msv import read_frames, flow_farneback, flow_raft  # noqa: E402


def temporal_msv_from_flows(
    flows: np.ndarray,
    window_size: int,
    hop: int,
    use_hann: bool = True,
) -> Optional[np.ndarray]:
    """flows (T-1, H, W, 2) → temporal MSV (n_freq,). None if video too short."""
    Tm1, H, W, _ = flows.shape
    mag = np.linalg.norm(flows, axis=-1)  # (T-1, H, W)

    if Tm1 < window_size:
        return None

    n_freq = window_size // 2 + 1
    accum = np.zeros(n_freq, dtype=np.float64)
    n_windows = 0

    hann = np.hanning(window_size).reshape(-1, 1, 1) if use_hann else None

    for start in range(0, Tm1 - window_size + 1, hop):
        w = mag[start : start + window_size]  # (window_size, H, W)
        if hann is not None:
            w = w * hann
        F = np.fft.rfft(w, axis=0)  # (n_freq, H, W)
        power = np.abs(F) ** 2  # (n_freq, H, W)
        accum += power.mean(axis=(1, 2))  # avg over spatial
        n_windows += 1

    return accum / max(n_windows, 1)  # (n_freq,)


def compute_temporal_msv(
    video_path: str,
    window_size: int,
    hop: int,
    flow_backend: str = "farneback",
    device: str = "cuda",
    max_frames: Optional[int] = None,
    resize: Optional[int] = 256,
    use_hann: bool = True,
) -> Optional[np.ndarray]:
    frames = read_frames(video_path, max_frames=max_frames, resize=resize)
    if flow_backend == "farneback":
        flows = flow_farneback(frames)
    elif flow_backend == "raft":
        flows = flow_raft(frames, device=device)
    else:
        raise ValueError(f"Unknown flow backend: {flow_backend}")
    return temporal_msv_from_flows(flows, window_size=window_size, hop=hop, use_hann=use_hann)


def main():
    ap = argparse.ArgumentParser(description="Temporal MSV (AC3D-style)")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--flow", choices=["farneback", "raft"], default="farneback")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window_size", type=int, default=32,
                    help="sliding window length along time (frames)")
    ap.add_argument("--hop", type=int, default=16,
                    help="window stride along time (frames)")
    ap.add_argument("--no_hann", action="store_true",
                    help="disable Hann window (default: on)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--resize", type=int, default=256,
                    help="짧은 변 기준 리사이즈 픽셀 (음수면 비활성)")
    ap.add_argument("--save_npy", required=True,
                    help="per-video temporal MSV 저장 경로 (N, n_freq) .npy")
    args = ap.parse_args()

    paths: List[str] = []
    for p in args.videos:
        paths.extend(sorted(glob.glob(p)) or [p])
    resize = None if (args.resize is not None and args.resize < 0) else args.resize

    all_msv = []
    for p in paths:
        msv = compute_temporal_msv(
            p,
            window_size=args.window_size,
            hop=args.hop,
            flow_backend=args.flow,
            device=args.device,
            max_frames=args.max_frames,
            resize=resize,
            use_hann=not args.no_hann,
        )
        if msv is None:
            print(f"[skip] {os.path.basename(p)}  (video shorter than window)")
            continue
        all_msv.append(msv)
        print(f"[ok] {os.path.basename(p)}  "
              f"low/high 에너지비 = {msv[: max(1, args.window_size // 8)].sum() / (msv.sum() + 1e-12):.3f}")

    if not all_msv:
        print("[error] no valid videos processed")
        return

    stacked = np.stack(all_msv, axis=0)
    np.save(args.save_npy, stacked)
    print(f"[saved] {args.save_npy}  shape={stacked.shape}  (N_videos, n_freq)")


if __name__ == "__main__":
    main()
