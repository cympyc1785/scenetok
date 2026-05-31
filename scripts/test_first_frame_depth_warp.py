"""Direct test of `multi_view_warp_to_target` on a single DL3DV scene.

Picks one scene + one target view (frame 20) + several context views, calls
`multi_view_warp_to_target` exactly as `_build_first_frame_from_depth` would
in `T2VWrapper`, and dumps the rendered first-frame RGB + visibility mask
+ side-by-side grid.

Run:
  CUDA_VISIBLE_DEVICES=2 python scripts/test_first_frame_depth_warp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from PIL import Image

from src.model.warp import multi_view_warp_to_target
from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg


# ---------- config ----------
SCENE = "001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
DATA_ROOT = REPO_ROOT / "DATA" / "DL3DV" / "DL3DV-960" / "train" / "1K" / SCENE
NPZ_PATH = DATA_ROOT / "da3" / "exports" / "mini_npz" / "results.npz"
TRANSFORMS = DATA_ROOT / "transforms.json"
IMG_DIR = DATA_ROOT / "images"
TARGET_FRAME = 20
CONTEXT_INDICES = [0, 5, 10, 15, 25, 30, 35, 40]   # sparse multi-view context
TOPK = 2                                              # top-K closest source views by camera distance
TARGET_SHAPE = (480, 832)                            # H, W — matches dataset target shape
OUT_DIR = REPO_ROOT / "tmp" / f"first_frame_depth_warp_test_topk{TOPK}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)


# ---------- load npz (depths + cameras) ----------
print(f"Loading {NPZ_PATH}")
npz = np.load(NPZ_PATH)
depth_full = torch.from_numpy(npz["depth"]).float()             # (N, Hd, Wd)
extr34 = torch.from_numpy(npz["extrinsics"]).float()             # (N, 3, 4) w2c
intr_npz = torch.from_numpy(npz["intrinsics"]).float()           # (N, 3, 3) at (Hd, Wd)
N, Hd, Wd = depth_full.shape
print(f"  {N} frames at depth-native res {Hd}x{Wd}")


# ---------- load image list ----------
with open(TRANSFORMS) as f:
    tj = json.load(f)
img_paths = [IMG_DIR / Path(fr["file_path"]).name for fr in tj["frames"]]
assert len(img_paths) == N, f"frame count mismatch: tj={len(img_paths)} npz={N}"


# ---------- resize npz to TARGET_SHAPE and adjust intrinsics ----------
H, W = TARGET_SHAPE
sx, sy = W / Wd, H / Hd
print(f"Resizing depth → {H}x{W} (sx={sx:.3f}, sy={sy:.3f})")


def select_npz(idx_long: torch.Tensor):
    """For a list of frame indices, return depth/(scaled intrinsics)/(4x4 w2c)
    all at TARGET_SHAPE = (H, W)."""
    d = depth_full[idx_long].unsqueeze(1)
    d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
    K = intr_npz[idx_long].clone()
    K[..., 0, 0] *= sx
    K[..., 1, 1] *= sy
    K[..., 0, 2] *= sx
    K[..., 1, 2] *= sy
    E = torch.eye(4).unsqueeze(0).repeat(idx_long.shape[0], 1, 1)
    E[:, :3] = extr34[idx_long]
    return d, K, E


def load_img_at_target(path: Path) -> torch.Tensor:
    """Full-res RGB → resize to (H, W) → tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------- assemble (B=1, V, ...) tensors ----------
ctx_idx_t = torch.tensor(CONTEXT_INDICES, dtype=torch.long)
ctx_imgs = torch.stack([load_img_at_target(img_paths[i]) for i in CONTEXT_INDICES], dim=0).unsqueeze(0).to(DEVICE)  # (1, V, 3, H, W)
ctx_depth, ctx_K, ctx_w2c = (t.unsqueeze(0).to(DEVICE) for t in select_npz(ctx_idx_t))

tgt_idx_t = torch.tensor([TARGET_FRAME], dtype=torch.long)
tgt_depth_unused, tgt_K, tgt_w2c = select_npz(tgt_idx_t)
tgt_K = tgt_K[0:1].to(DEVICE)
tgt_w2c = tgt_w2c[0:1].to(DEVICE)
tgt_img_gt = load_img_at_target(img_paths[TARGET_FRAME]).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)


# ---------- warp ----------
print(
    f"Warping: ctx_imgs {tuple(ctx_imgs.shape)}, ctx_depth {tuple(ctx_depth.shape)}, "
    f"ctx_w2c {tuple(ctx_w2c.shape)}, ctx_K {tuple(ctx_K.shape)}, "
    f"tgt_w2c {tuple(tgt_w2c.shape)}, tgt_K {tuple(tgt_K.shape)}"
)
warped, mask, per_warps, per_masks = multi_view_warp_to_target(
    context_images=ctx_imgs,
    context_depths=ctx_depth,
    context_w2cs=ctx_w2c,
    context_intrinsics=ctx_K,
    target_w2c=tgt_w2c,
    target_intrinsics=tgt_K,
    topk=TOPK,
    return_per_view=True,
)
print(f"warped: {tuple(warped.shape)}, mask: {tuple(mask.shape)}")
print(f"per-view warps: {tuple(per_warps.shape)}, per-view masks: {tuple(per_masks.shape)}")
print(f"Union mask coverage: {mask.mean().item():.3f}")
print(f"Per-source coverage: {[round(per_masks[0, i].mean().item(), 3) for i in range(per_masks.shape[1])]}")


# ---------- save outputs ----------
def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().cpu().float() + 1) / 2).clamp(0, 1)


def save_rgb(t: torch.Tensor, path: Path, m: torch.Tensor | None = None):
    """t: (3, H, W) in [-1, 1]. Optional mask (1, H, W) to black-out invisible pixels."""
    img = to01(t)
    if m is not None:
        img = img * m.detach().cpu().float().clamp(0, 1)
    TT.functional.to_pil_image(img).save(path)


def save_mask(m: torch.Tensor, path: Path):
    """m: (1, H, W) ∈ [0, 1]."""
    TT.functional.to_pil_image(m.detach().cpu().float().squeeze(0).clamp(0, 1)).save(path)


save_rgb(tgt_img_gt[0], OUT_DIR / "00_target_gt.png")
save_rgb(warped[0], OUT_DIR / "01_rendered_first_frame.png")
save_rgb(warped[0], OUT_DIR / "01_rendered_first_frame_masked.png", m=mask[0])
save_mask(mask[0], OUT_DIR / "02_visibility_mask.png")


# ---------- Wan VAE round-trip (mirrors `T2VWrapper.get_first_frame_latents`) ----------
print("\nLoading Wan VAE (latent_channels=48)…")
wan_vae = AutoencoderWan(WanKwargsCfg(latent_channels=48)).from_pretrained(
    "checkpoints/Wan2.2_VAE.pth"
).to(DEVICE).eval()
wan_vae.requires_grad_(False)

# Wrapper convention: warped is now in [0, 1] (holes → 0). VAE.encode expects
# [-1, 1] so do `* 2 - 1` here (matching `first_stage_encode`'s normalization).
warped_01 = ((warped + 1.0) / 2.0).clamp(0.0, 1.0)                   # (1, 3, H, W) in [0, 1]
warped_m11 = warped_01 * 2.0 - 1.0                                   # (1, 3, H, W) in [-1, 1]

# Pad 1 first frame with 3 zeros along V dim, encode the 4-frame block,
# take the first latent ([:, :1]).
first_frame_v = warped_m11.unsqueeze(1).to(DEVICE)                   # (1, 1, 3, H, W)
padding = torch.zeros(1, 3, 3, H, W, device=DEVICE, dtype=first_frame_v.dtype)
ff_video = torch.cat([first_frame_v, padding], dim=1)                # (1, 4, 3, H, W)
ff_latent = wan_vae.encode(ff_video)[:, :1]                          # (1, 1, 48, h, w)
print(f"  warped first-frame latent: {tuple(ff_latent.shape)}")
ff_decoded = wan_vae.decode(ff_latent)[:, :1]                        # (1, 1, 3, H, W) in [-1, 1]
ff_decoded_img = ff_decoded[0, 0].clamp(-1, 1)

# GT round-trip
gt_m11 = tgt_img_gt * 2.0 - 1.0 if tgt_img_gt.min() >= 0 else tgt_img_gt  # tgt_img_gt was loaded in [-1, 1] already
gt_video = torch.cat([gt_m11.unsqueeze(1), padding], dim=1)
gt_latent = wan_vae.encode(gt_video)[:, :1]
gt_decoded = wan_vae.decode(gt_latent)[:, :1]
gt_decoded_img = gt_decoded[0, 0].clamp(-1, 1)

save_rgb(ff_decoded_img, OUT_DIR / "04_warped_vae_recon.png")
save_rgb(gt_decoded_img, OUT_DIR / "05_gt_vae_recon.png")

# Latent mask (avg_pool 16) as the wrapper builds — show what soft blend sees.
mask_latent = F.avg_pool2d(mask.float(), kernel_size=16, stride=16)   # (1, 1, h, w)
print(f"  mask_latent shape: {tuple(mask_latent.shape)}, "
      f"min={mask_latent.min().item():.3f}, max={mask_latent.max().item():.3f}")
mask_latent_full = F.interpolate(mask_latent, size=(H, W), mode="nearest")[0]
save_mask(mask_latent_full, OUT_DIR / "03_visibility_mask_avgpool16.png")

# Soft-blended latent at decode time: warped latent * m + (1-m) * gt latent
# (stands in for model's denoised prediction).
soft_blended_latent = mask_latent.unsqueeze(1) * ff_latent + (1 - mask_latent.unsqueeze(1)) * gt_latent
soft_decoded = wan_vae.decode(soft_blended_latent)[:, :1]
save_rgb(soft_decoded[0, 0].clamp(-1, 1), OUT_DIR / "06_soft_blend_warp+gt_vae_recon.png")

# Resolve which CONTEXT_INDICES the module's top-K selection actually picked
# (mirrors the logic inside `multi_view_warp_to_target`).
_c2w_ctx = torch.linalg.inv(ctx_w2c.float())
_cam_pos_ctx = _c2w_ctx[..., :3, 3]
_c2w_tgt = torch.linalg.inv(tgt_w2c.float())
_cam_pos_tgt = _c2w_tgt[..., :3, 3].unsqueeze(1)
_dists = (_cam_pos_ctx - _cam_pos_tgt).norm(dim=-1)[0]                # (V,)
_top_local_idx = _dists.topk(per_warps.shape[1], largest=False).indices.tolist()
_top_original_frames = [CONTEXT_INDICES[i] for i in _top_local_idx]
print(f"Top-{per_warps.shape[1]} chosen context frames: {_top_original_frames}")
print(f"  distances: {[round(float(_dists[i]), 4) for i in _top_local_idx]}")

# Per-source warps and masks (indexed by the top-K selection order)
for i, src_idx in enumerate(_top_original_frames):
    save_rgb(per_warps[0, i], OUT_DIR / f"src{i:02d}_frame{src_idx:03d}_warp.png", m=per_masks[0, i])
    save_mask(per_masks[0, i], OUT_DIR / f"src{i:02d}_frame{src_idx:03d}_mask.png")


# Also dump the avg-pooled latent mask (what `get_first_frame_latents` would use)
mask_latent = F.avg_pool2d(mask.float(), kernel_size=16, stride=16)   # (1, 1, h, w)
# Upsample back for visualization
mask_latent_vis = F.interpolate(mask_latent, size=(H, W), mode="nearest")
save_mask(mask_latent_vis[0], OUT_DIR / "03_visibility_mask_avgpool16.png")
print(f"Latent mask range (h, w)=({mask_latent.shape[-2]}, {mask_latent.shape[-1]}): "
      f"min={mask_latent.min().item():.3f}, max={mask_latent.max().item():.3f}")


# Side-by-side grid: row 1 = (GT | rendered | visibility), row 2 = first 3 sources
row1 = torch.cat([
    to01(tgt_img_gt[0]),
    to01(warped[0]),
    mask[0].expand(3, -1, -1).cpu(),
], dim=-1)
row2_items = []
for i in range(min(3, per_warps.shape[1])):
    img = to01(per_warps[0, i]) * per_masks[0, i].cpu().expand(3, -1, -1)
    row2_items.append(img)
while len(row2_items) < 3:
    row2_items.append(torch.zeros(3, H, W))
row2 = torch.cat(row2_items, dim=-1)
grid = torch.cat([row1, row2], dim=-2)
TT.functional.to_pil_image(grid).save(OUT_DIR / "grid_overview.png")

# L1 fidelity (where visible)
gt = tgt_img_gt[0].cpu()
warp_cpu = warped[0].cpu()
mask_cpu = mask[0].cpu()
l1_visible = (
    ((warp_cpu - gt).abs() * mask_cpu).sum() / (mask_cpu.sum() * 3 + 1e-7)
).item()
print(f"\nL1 (rendered vs target GT, visible-pixels only): {l1_visible:.4f}")
print(f"Files written to: {OUT_DIR}")
