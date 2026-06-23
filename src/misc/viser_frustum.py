"""Viser helpers: start a server and visualize camera-pose view frustums
(optionally textured with the context image on the view plane).

Conventions
-----------
- Camera poses are **c2w** (camera-to-world) 4x4 matrices in the **OpenCV**
  convention (+X right, +Y down, +Z forward) — same as this repo's
  `preprocess_batch` / DL3DV `convert_poses`. viser's `add_camera_frustum`
  uses the same OpenCV frustum convention, so c2w rotation→wxyz and
  translation→position map directly.
- Intrinsics are the repo's **normalized** 3x3 (fx, fy ∈ [0,1] = focal / image
  side, cx≈cy≈0.5). Vertical fov = 2·atan(0.5 / fy); aspect = fy / fx = W / H.
  Pass `intrinsics_normalized=False` if you hand pixel-unit K (then `image_hw`
  is required to recover fov).

Typical use:
    from src.misc.viser_frustum import start_server, add_view_frustums
    server = start_server()
    add_view_frustums(server, c2w, intrinsics=intr, images=imgs)
    # keep the process alive (server runs in a background thread)
"""
from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ───────────────────────── helpers ─────────────────────────
def _to_np(x):
    if x is None:
        return None
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation → quaternion (w,x,y,z). Robust Shepperd's method."""
    m = R.astype(np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def path_points_from_origins(origins: np.ndarray) -> np.ndarray:
    """(N,3) camera origins → (N-1,2,3) consecutive segments for add_line_segments."""
    o = np.asarray(origins, dtype=np.float32)
    return np.stack([o[:-1], o[1:]], axis=1)


def _prep_image(img) -> Optional[np.ndarray]:
    """→ (H,W,3) uint8, or None. Accepts (3,H,W)/(H,W,3), float [0,1]/[-1,1] or uint8."""
    if img is None:
        return None
    a = _to_np(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))          # CHW → HWC
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        if a.min() < -0.01:                     # [-1,1] → [0,1]
            a = a * 0.5 + 0.5
        a = np.clip(a * 255.0 if a.max() <= 1.0 + 1e-3 else a, 0, 255).astype(np.uint8)
    return a[..., :3]


def _fov_aspect(K_norm: np.ndarray, fallback_fov: float, fallback_aspect: float,
                normalized: bool, image_hw: Optional[tuple]) -> tuple[float, float]:
    if K_norm is None:
        return fallback_fov, fallback_aspect
    fx, fy = float(K_norm[0, 0]), float(K_norm[1, 1])
    if normalized:
        vfov = 2.0 * math.atan(0.5 / max(fy, 1e-6))
        aspect = fy / max(fx, 1e-6)             # = W/H
    else:
        if image_hw is None:
            return fallback_fov, fallback_aspect
        H, W = image_hw
        vfov = 2.0 * math.atan((H / 2.0) / max(fy, 1e-6))
        aspect = (W / max(fx, 1e-6)) / (H / max(fy, 1e-6))
    return vfov, aspect


# ───────────────────────── public API ─────────────────────────
def start_server(host: str = "0.0.0.0", port: int = 8080, label: str | None = "scenetok-viser"):
    """Start (and return) a viser server. Server runs in a background thread."""
    import viser
    server = viser.ViserServer(host=host, port=port, label=label)
    return server


def add_view_frustums(
    server,
    c2w,                                        # (N,4,4) or (4,4)
    intrinsics=None,                            # (N,3,3) or (3,3), normalized K
    images=None,                                # (N,H,W,3)/(N,3,H,W) or list, optional
    *,
    fov: float = math.radians(60.0),            # fallback vfov [rad] if no intrinsics
    aspect: float = 16 / 9,                     # fallback aspect if no intrinsics
    intrinsics_normalized: bool = True,
    scale: float = 0.3,
    color_start: tuple = (40, 120, 255),        # 첫 카메라 색 (기본 파랑)
    color_end: tuple = (255, 80, 40),           # 마지막 카메라 색 (기본 주황)
    prefix: str = "context",
    add_world_axes: bool = True,
    add_gui_color: bool = True,                 # viser 패널에 시작/끝 색 picker 추가
    draw_path: bool = True,                     # 카메라 origin 첫→마지막 잇는 경로선
    path_color: tuple = (0, 0, 0),              # 경로선 색 (검정 고정)
    path_line_width: float = 3.0,
    return_all: bool = False,                   # True면 제거 가능한 모든 handle 반환(reload용)
):
    """Add one camera frustum per pose; color lerps first→last (order cue),
    texture view plane with `images[i]` if given. If `add_gui_color`, adds two
    RGB pickers (start/end) that recolor all frustums live. Returns handles."""
    def _lerp_color(c0, c1, t):
        return tuple(int(round(c0[k] + (c1[k] - c0[k]) * t)) for k in range(3))
    c2w = _to_np(c2w)
    if c2w.ndim == 2:
        c2w = c2w[None]
    N = c2w.shape[0]
    K = _to_np(intrinsics)
    if K is not None and K.ndim == 2:
        K = K[None]
    imgs = images
    if imgs is not None and not isinstance(imgs, (list, tuple)):
        imgs = _to_np(imgs)
        imgs = [imgs[i] for i in range(imgs.shape[0])] if imgs.ndim == 4 else [imgs]

    extra = []                                   # 제거 가능한 비-frustum handle (axes/path/gui)
    if add_world_axes:
        extra.append(server.scene.add_frame("/world", show_axes=True, axes_length=0.5, axes_radius=0.01))

    def _color_at(i):                            # t = i/(N-1), first→last
        t = 0.0 if N <= 1 else i / (N - 1)
        return _lerp_color(color_start, color_end, t)

    handles = []
    for i in range(N):
        R, t = c2w[i, :3, :3], c2w[i, :3, 3]
        img = _prep_image(imgs[i]) if imgs is not None and i < len(imgs) else None
        hw = img.shape[:2] if img is not None else None
        vfov, asp = _fov_aspect(K[i] if K is not None else None, fov, aspect,
                                intrinsics_normalized, hw)
        h = server.scene.add_camera_frustum(
            f"/{prefix}/cam_{i:03d}",
            fov=float(vfov), aspect=float(asp), scale=scale, color=_color_at(i),
            image=img, wxyz=_rotmat_to_wxyz(R), position=tuple(float(v) for v in t),
        )
        handles.append(h)

    # 카메라 origin들을 첫→마지막 순서로 잇는 경로선
    path_handle = None
    if draw_path and N >= 2:
        origins = c2w[:, :3, 3].astype(np.float32)                  # (N,3)
        cols = np.tile(np.array(path_color, np.uint8), (N - 1, 2, 1))
        path_handle = server.scene.add_line_segments(
            f"/{prefix}/path", points=path_points_from_origins(origins), colors=cols,
            line_width=path_line_width)
        path_handle._is_frustum_path = True                         # launcher가 식별용
        extra.append(path_handle)

    if add_gui_color:
        gui_start = server.gui.add_rgb(f"{prefix} start color", color_start)
        gui_end = server.gui.add_rgb(f"{prefix} end color", color_end)
        def _recolor(_=None):
            c0, c1 = gui_start.value, gui_end.value
            for i, h in enumerate(handles):
                t = 0.0 if N <= 1 else i / (N - 1)
                h.color = _lerp_color(c0, c1, t)
        gui_start.on_update(_recolor)
        gui_end.on_update(_recolor)
        extra += [gui_start, gui_end]
    return (handles + extra) if return_all else handles
