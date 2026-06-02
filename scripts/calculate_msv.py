"""
Motion Spectral Volume (MSV) analysis from video.

AC3D 논문(Bahmani et al., 2024)에서 사용한 분석 방식의 재현 구현.
공식 레포(snap-research/ac3d)에는 이 분석 코드가 없어서, 논문의 정의를 따라
직접 작성한 것이다.

핵심 레시피
-----------
1) 비디오에서 연속 프레임 간 optical flow (u, v) 추정
2) flow field를 2D 공간 FFT -> power spectrum
3) DC(중심) 기준 radial frequency bin 으로 power 를 집계(radial averaging)
4) 모든 프레임/비디오에 대해 평균 -> "주파수 대역별 모션 에너지" 곡선 = MSV
   (저주파 쪽 에너지가 크면 = 부드럽고 전역적인 모션, 예: 카메라 모션)

flow 백엔드 두 가지를 제공한다.
  - "farneback": OpenCV 내장, GPU 불필요, 의존성 가벼움 (기본값)
  - "raft":      torchvision RAFT, 논문에 더 가까움, GPU 권장

사용 예 — 계산만 (.npy 저장). plot은 별도의 `scripts/plot_msv.py` 사용.
-------
    python scripts/calculate_msv.py --videos "a.mp4" "b.mp4" --flow farneback \\
        --save_npy out.npy
    python scripts/plot_msv.py --npy out.npy --plot out.png --log
"""

import argparse
import glob
import os
from typing import List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# 1) 비디오 -> 프레임
# ----------------------------------------------------------------------------
def read_frames(path: str, max_frames: Optional[int] = None,
                resize: Optional[int] = None) -> np.ndarray:
    """비디오 또는 이미지 시퀀스 디렉토리를 (T, H, W, 3) uint8 RGB로 읽는다.

    `path`가 디렉토리면 `images/frame_*.png` (또는 .jpg) 패턴으로 정렬해
    프레임을 순차 로드한다 (DL3DV 등 scene-dir 입력용). 그 외엔 cv2.VideoCapture.
    """
    import os
    import cv2
    frames: list[np.ndarray] = []

    def _maybe_resize(rgb):
        if resize is None:
            return rgb
        h, w = rgb.shape[:2]
        scale = resize / min(h, w)
        return cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    if os.path.isdir(path):
        # Scene directory mode: look for images under `images/` (DL3DV layout)
        # or directly under `path`.
        import glob as _glob
        img_dir = os.path.join(path, "images") if os.path.isdir(os.path.join(path, "images")) else path
        candidates = sorted(_glob.glob(os.path.join(img_dir, "frame_*.png")))
        if not candidates:
            candidates = sorted(_glob.glob(os.path.join(img_dir, "frame_*.jpg")))
        if not candidates:
            candidates = sorted(_glob.glob(os.path.join(img_dir, "*.png")) +
                                _glob.glob(os.path.join(img_dir, "*.jpg")))
        if not candidates:
            raise IOError(f"이미지가 없습니다: {img_dir}")
        for img_path in candidates:
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(_maybe_resize(rgb))
            if max_frames is not None and len(frames) >= max_frames:
                break
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"비디오를 열 수 없습니다: {path}")
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(_maybe_resize(rgb))
            if max_frames is not None and len(frames) >= max_frames:
                break
        cap.release()

    if len(frames) < 2:
        raise ValueError(f"프레임이 2개 미만입니다: {path}")
    return np.stack(frames, axis=0)


# ----------------------------------------------------------------------------
# 2) Optical flow 백엔드
# ----------------------------------------------------------------------------
def flow_farneback(frames: np.ndarray) -> np.ndarray:
    """OpenCV Farneback. 반환 (T-1, H, W, 2)."""
    import cv2
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flows.append(flow)
    return np.stack(flows, axis=0)


def flow_raft(frames: np.ndarray, device: str = "cuda") -> np.ndarray:
    """torchvision RAFT (large). 반환 (T-1, H, W, 2)."""
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    import torch.nn.functional as F

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()

    # RAFT 는 8의 배수 해상도를 선호하므로 패딩
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    t = t * 2.0 - 1.0  # [-1, 1]
    _, _, h, w = t.shape
    ph, pw = (-h) % 8, (-w) % 8
    t = F.pad(t, (0, pw, 0, ph))
    t = t.to(device)

    flows = []
    with torch.no_grad():
        for i in range(t.shape[0] - 1):
            pred = model(t[i:i + 1], t[i + 1:i + 2])[-1]  # 마지막 반복 결과
            f = pred[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
            flows.append(f)
    return np.stack(flows, axis=0)


# ----------------------------------------------------------------------------
# 3) flow -> radial power spectrum (한 프레임)
# ----------------------------------------------------------------------------
def _radial_power(flow_uv: np.ndarray, n_bins: int) -> np.ndarray:
    """
    하나의 flow field (H, W, 2)에 대한 radial power spectrum.
    u, v 각각 2D FFT -> |.|^2 합산 -> 중심으로부터의 거리로 binning.
    """
    H, W, _ = flow_uv.shape
    # 경계 누설(spectral leakage) 완화를 위한 2D Hann 윈도우
    wy = np.hanning(H)[:, None]
    wx = np.hanning(W)[None, :]
    win = wy * wx

    power = np.zeros((H, W), dtype=np.float64)
    for c in range(2):
        f = np.fft.fftshift(np.fft.fft2(flow_uv[..., c] * win))
        power += np.abs(f) ** 2

    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.indices((H, W))
    # 정규화된 반경 [0, 1] : 0 = DC(저주파), 1 = 최대 주파수
    r = np.sqrt(((yy - cy) / (H / 2.0)) ** 2 + ((xx - cx) / (W / 2.0)) ** 2)
    r = np.clip(r, 0, 1)

    bin_idx = np.minimum((r * n_bins).astype(int), n_bins - 1)
    out = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(out, bin_idx.ravel(), power.ravel())
    np.add.at(counts, bin_idx.ravel(), 1.0)
    counts[counts == 0] = 1.0
    return out / counts  # bin 당 평균 power


# ----------------------------------------------------------------------------
# 4) 비디오 한 개의 MSV
# ----------------------------------------------------------------------------
def msv_from_flows(flows: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """flows (T-1, H, W, 2) -> MSV (n_bins,). 모든 프레임 평균."""
    spec = np.zeros(n_bins, dtype=np.float64)
    for t in range(flows.shape[0]):
        spec += _radial_power(flows[t], n_bins)
    return spec / flows.shape[0]


def compute_msv(video_path: str, flow_backend: str = "farneback",
                device: str = "cuda", n_bins: int = 64,
                max_frames: Optional[int] = None,
                resize: Optional[int] = 256) -> np.ndarray:
    frames = read_frames(video_path, max_frames=max_frames, resize=resize)
    if flow_backend == "farneback":
        flows = flow_farneback(frames)
    elif flow_backend == "raft":
        flows = flow_raft(frames, device=device)
    else:
        raise ValueError(f"알 수 없는 flow backend: {flow_backend}")
    return msv_from_flows(flows, n_bins=n_bins)


# ----------------------------------------------------------------------------
# 5) CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Motion Spectral Volume (MSV) 분석")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="비디오 파일 경로들 (glob 패턴도 가능)")
    ap.add_argument("--flow", choices=["farneback", "raft"], default="farneback")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--resize", type=int, default=256,
                    help="짧은 변 기준 리사이즈 픽셀 (None 으로 두려면 -1)")
    ap.add_argument("--save_npy", required=True,
                    help="per-video MSV 저장 경로 (.npy, shape=(N, n_bins))")
    args = ap.parse_args()

    paths: List[str] = []
    for p in args.videos:
        paths.extend(sorted(glob.glob(p)) or [p])
    resize = None if args.resize is not None and args.resize < 0 else args.resize

    all_msv = []
    for p in paths:
        msv = compute_msv(p, flow_backend=args.flow, device=args.device,
                          n_bins=args.n_bins, max_frames=args.max_frames,
                          resize=resize)
        all_msv.append(msv)
        print(f"[ok] {os.path.basename(p)}  "
              f"low/high 에너지비 = {msv[:args.n_bins//4].sum() / (msv.sum()+1e-12):.3f}")

    stacked = np.stack(all_msv, axis=0)  # (N, n_bins)
    np.save(args.save_npy, stacked)
    print(f"[saved] {args.save_npy}  shape={stacked.shape}")


if __name__ == "__main__":
    main()