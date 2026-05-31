"""
Sparse multi-view NVS demo for a DL3DV scene using GEN3C's forward_warp.

Pipeline (mirrors `cosmos_predict1/diffusion/inference/cache_3d.py:Cache3D_BufferSelector`):
  1. Load DA3 depth + extrinsics + intrinsics from `mini_npz/results.npz` (self-consistent).
  2. Pick target = frame 20; pick top-K source frames closest in camera position.
  3. Forward-warp each source's RGB image (+ depth) into the target view.
  4. Combine warps via mask-weighted average (top-K overlap fusion).
  5. VA-VAE encode → decode the fused image for visualization.
  6. Dump GT target, per-source warps, fused warp, VA-VAE round-trip.

Crop to (272, 496) — closest multiple of 16 to the depth-native (280, 504),
so we don't resize depth or intrinsics (only crop) and VA-VAE accepts the shape.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub NVIDIA Warp — `forward_warp_utils_pytorch` imports `warp` at module
# level for `foreground_masking=True` (ray-triangle intersection), but we
# don't enable foreground_masking here. Inject a no-op module so the import
# succeeds without installing `warp-lang`.
import types as _types
if "warp" not in sys.modules:
    _stub = _types.ModuleType("warp")
    _stub.init = lambda: None
    sys.modules["warp"] = _stub

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.model.GEN3C.cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    forward_warp,
)
from src.model.autoencoder.vavae.vavae import AutoencoderVA


# ---------- config ----------
SCENE = "001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
DATA_ROOT = REPO_ROOT / "DATA" / "DL3DV" / "DL3DV-960" / "train" / "1K" / SCENE
NPZ = DATA_ROOT / "da3" / "exports" / "mini_npz" / "results.npz"
TRANSFORMS = DATA_ROOT / "transforms.json"
IMG_DIR = DATA_ROOT / "images"
TARGET_FRAME = 20
TOPK = 5
CROP_H, CROP_W = 272, 496  # multiples of 16, close to depth native (280, 504)
# CAM_SOURCE ∈ {"npz", "transforms"}: which extrinsics+intrinsics source to use.
# - "npz":        DA3 mini_npz cameras (self-consistent with depth, OpenCV w2c).
# - "transforms": nerfstudio transforms.json c2w (OpenGL convention -> OpenCV
#                 flip applied) + global intrinsics scaled to depth resolution.
CAM_SOURCE = "transforms"
OUT_DIR = REPO_ROOT / "tmp" / f"gen3c_sparse_nvs_{CAM_SOURCE}_cam"
CKPT = "checkpoints/vavae-imagenet256-f16d32-dinov2.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)


# ---------- load depth/cameras ----------
print(f"Loading {NPZ}")
npz = np.load(NPZ)
depths = npz["depth"]              # (N, 280, 504)
extr34 = npz["extrinsics"]         # (N, 3, 4) — assume w2c (DA3/COLMAP convention)
intrs = npz["intrinsics"]          # (N, 3, 3) at 504x280 resolution

N, Hd, Wd = depths.shape
print(f"  {N} frames at depth res {Hd}x{Wd}")
print(f"  depth range: {depths.min():.2f} ~ {depths.max():.2f}")

# Center-crop maths
crop_top = (Hd - CROP_H) // 2
crop_left = (Wd - CROP_W) // 2
print(f"Crop offsets: top={crop_top}, left={crop_left}; final {CROP_H}x{CROP_W}")

# Crop depth + adjust intrinsics
depths_c = depths[:, crop_top : crop_top + CROP_H, crop_left : crop_left + CROP_W]

if CAM_SOURCE == "npz":
    # DA3-native: extrinsics already w2c at depth resolution; intrinsics
    # already in pixel coords of (280, 504) — crop only shifts cx, cy.
    intrs_c = intrs.copy()
    intrs_c[:, 0, 2] -= crop_left
    intrs_c[:, 1, 2] -= crop_top
    w2c44 = np.zeros((N, 4, 4), dtype=np.float32)
    w2c44[:, :3] = extr34
    w2c44[:, 3, 3] = 1.0
elif CAM_SOURCE == "transforms":
    # transforms.json: per-frame c2w in OpenGL convention (nerfstudio default).
    # Global intrinsics fl_x, fl_y, cx, cy at full image res (W_full, H_full).
    with open(TRANSFORMS) as _f:
        _tj = json.load(_f)
    W_full = float(_tj["w"])
    H_full = float(_tj["h"])
    fl_x = float(_tj["fl_x"])
    fl_y = float(_tj["fl_y"])
    cx = float(_tj["cx"])
    cy = float(_tj["cy"])
    # Scale full-res intrinsics to depth-native (504, 280), then apply crop shift.
    sx = Wd / W_full
    sy = Hd / H_full
    K_depth = np.array(
        [[fl_x * sx, 0.0, cx * sx],
         [0.0, fl_y * sy, cy * sy],
         [0.0, 0.0, 1.0]], dtype=np.float32,
    )
    K_depth[0, 2] -= crop_left
    K_depth[1, 2] -= crop_top
    intrs_c = np.broadcast_to(K_depth, (N, 3, 3)).copy()

    # Build (N, 4, 4) w2c from c2w (transforms.json transform_matrix).
    # OpenGL (Y up, -Z fwd) -> OpenCV (Y down, +Z fwd) via flip(Y, Z) on cam axes.
    M_gl2cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    c2w_all = np.stack(
        [np.asarray(fr["transform_matrix"], dtype=np.float32) for fr in _tj["frames"]],
        axis=0,
    )  # (N, 4, 4)
    c2w_cv = c2w_all @ M_gl2cv
    w2c44 = np.linalg.inv(c2w_cv).astype(np.float32)
else:
    raise ValueError(f"Unknown CAM_SOURCE={CAM_SOURCE}")

# ---------- to tensors ----------
depths_t = torch.from_numpy(depths_c).unsqueeze(1).float().to(DEVICE)   # (N, 1, H, W)
intrs_t = torch.from_numpy(intrs_c).float().to(DEVICE)                  # (N, 3, 3)
w2cs_t = torch.from_numpy(w2c44).float().to(DEVICE)                     # (N, 4, 4)


# ---------- load image list ----------
with open(TRANSFORMS) as f:
    tj = json.load(f)
img_paths = [IMG_DIR / Path(fr["file_path"]).name for fr in tj["frames"]]
assert len(img_paths) == N, f"frame count mismatch: tj={len(img_paths)} npz={N}"


def load_img(path, depth_res_h, depth_res_w, crop_t, crop_l, crop_h, crop_w):
    """Load image, resize to depth-native res, then center-crop to (crop_h, crop_w).
    Returns (3, crop_h, crop_w) tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((depth_res_w, depth_res_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t[:, crop_t : crop_t + crop_h, crop_l : crop_l + crop_w]


# ---------- target + top-K source selection ----------
target_idx = TARGET_FRAME
print(f"\nTarget: frame {target_idx}")
target_img = load_img(img_paths[target_idx], Hd, Wd, crop_top, crop_left, CROP_H, CROP_W).to(DEVICE)
target_w2c = w2cs_t[target_idx : target_idx + 1]
target_intr = intrs_t[target_idx : target_idx + 1]

# Top-K closest source frames by camera position (inv(w2c) -> c2w; t-column = pos in world)
c2w_all = torch.linalg.inv(w2cs_t)
cam_pos_all = c2w_all[:, :3, 3]                                          # (N, 3)
target_pos = cam_pos_all[target_idx]
dists = (cam_pos_all - target_pos).norm(dim=-1)                          # (N,)
dists[target_idx] = float("inf")
source_indices = dists.topk(TOPK, largest=False).indices.tolist()
print(f"Top-{TOPK} closest source frames: {source_indices}")
print(f"  distances: {[round(float(dists[i]), 4) for i in source_indices]}")


# ---------- forward warp each source -> target ----------
warped_imgs, warped_masks = [], []
for src_idx in source_indices:
    src_img = load_img(img_paths[src_idx], Hd, Wd, crop_top, crop_left, CROP_H, CROP_W).to(DEVICE).unsqueeze(0)
    src_depth = depths_t[src_idx : src_idx + 1]
    src_w2c = w2cs_t[src_idx : src_idx + 1]
    src_intr = intrs_t[src_idx : src_idx + 1]

    warped, mask, _warped_depth, _flow = forward_warp(
        frame1=src_img,
        mask1=None,
        depth1=src_depth,
        transformation1=src_w2c,
        transformation2=target_w2c,
        intrinsic1=src_intr,
        intrinsic2=target_intr,
        is_image=True,
        is_depth=True,
    )
    warped_imgs.append(warped)
    warped_masks.append(mask)

warped_imgs = torch.cat(warped_imgs, dim=0)    # (K, 3, H, W) in [-1, 1]
warped_masks = torch.cat(warped_masks, dim=0)  # (K, 1, H, W) in {0, 1}
print(f"\nPer-source mask coverage: {[round(m.mean().item(), 3) for m in warped_masks]}")


# ---------- combine: mask-weighted average (top-K overlap fusion) ----------
weight = warped_masks
combined_img = (warped_imgs * weight).sum(0) / (weight.sum(0) + 1e-7)   # (3, H, W) in [-1, 1]
combined_mask = (weight.sum(0) > 0).float()                            # (1, H, W)
print(f"Combined mask coverage: {combined_mask.mean().item():.3f}")
combined_img = torch.where(
    combined_mask > 0, combined_img, torch.full_like(combined_img, -1.0)
)


# ---------- VA-VAE encode→decode round-trip ----------
print(f"\nLoading VA-VAE from {CKPT}")
vae = AutoencoderVA(cfg=None).from_pretrained(CKPT).to(DEVICE).eval()

x = combined_img.unsqueeze(0).to(DEVICE)                  # (1, 3, H, W) in [-1, 1]
z = vae.encode(x).latent_dist.mode().float()              # (1, 32, H/16, W/16)
y = vae.decode(z.to(x.dtype)).sample                      # (1, 3, H, W) in ~[-1, 1]
recon = y[0].float().cpu().clamp(-1, 1)
print(f"VAE latent shape: {tuple(z.shape)}")


# ---------- save outputs ----------
def save_png(t, path, mask=None):
    """t: (3, H, W) in [-1, 1] -> PNG. Optionally apply mask (0=black)."""
    img01 = ((t.detach().cpu().float() + 1) / 2).clamp(0, 1)
    if mask is not None:
        m = mask.detach().cpu().float().clamp(0, 1)
        if m.ndim == 3:
            m = m[0:1]
        img01 = img01 * m
    T.functional.to_pil_image(img01).save(path)


def save_mask(m, path):
    """m: (1, H, W) -> grayscale PNG."""
    img = m.detach().cpu().float().squeeze(0).clamp(0, 1)
    T.functional.to_pil_image(img).save(path)


save_png(target_img, OUT_DIR / "00_target_gt.png")
for i, src_idx in enumerate(source_indices):
    save_png(warped_imgs[i], OUT_DIR / f"01_warped_src{i:02d}_frame{src_idx:03d}.png",
             mask=warped_masks[i])
    save_mask(warped_masks[i], OUT_DIR / f"01_mask_src{i:02d}_frame{src_idx:03d}.png")
save_png(combined_img, OUT_DIR / "02_combined.png")
save_mask(combined_mask, OUT_DIR / "02_combined_mask.png")
save_png(recon, OUT_DIR / "03_combined_vae_recon.png")


# ---------- side-by-side grid ----------
def to01(t):
    return ((t.detach().cpu().float() + 1) / 2).clamp(0, 1)


def pad_to(t, h, w):
    """Center-pad a (3,h0,w0) image to (3,h,w) with black."""
    h0, w0 = t.shape[-2:]
    pad_t = (h - h0) // 2
    pad_l = (w - w0) // 2
    out = torch.zeros(3, h, w)
    out[:, pad_t : pad_t + h0, pad_l : pad_l + w0] = t
    return out


# row 1: target GT | combined | VAE recon
# row 2: K warped sources
H, W = CROP_H, CROP_W
row1 = torch.cat([to01(target_img), to01(combined_img), to01(recon)], dim=-1)        # (3, H, 3W)
row2_items = [to01(warped_imgs[i] * warped_masks[i]) for i in range(TOPK)]
# pad/truncate row2 to 3 panels width
while len(row2_items) < 3:
    row2_items.append(torch.zeros(3, H, W))
row2_items = row2_items[:3]
row2 = torch.cat(row2_items, dim=-1)
grid = torch.cat([row1, row2], dim=-2)                                                # (3, 2H, 3W)
T.functional.to_pil_image(grid).save(OUT_DIR / "grid_overview.png")


# ---------- metrics ----------
target_cpu = target_img.detach().cpu()
l1_warp = (combined_img.detach().cpu() - target_cpu).abs().mean().item()
l1_recon = (recon - target_cpu).abs().mean().item()
l1_warp_in_mask = (
    ((combined_img.detach().cpu() - target_cpu).abs() * combined_mask.detach().cpu()).sum()
    / (combined_mask.detach().cpu().sum() * 3 + 1e-7)
).item()
print(f"\nL1 (combined warp vs target GT, full image): {l1_warp:.4f}")
print(f"L1 (combined warp vs target GT, mask-only):    {l1_warp_in_mask:.4f}")
print(f"L1 (VAE recon of combined vs target GT):       {l1_recon:.4f}")
print(f"\nFiles written to: {OUT_DIR}")
