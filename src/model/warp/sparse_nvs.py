"""Sparse multi-view forward warp + aggregation.

Reusable wrapper around GEN3C's
`cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch.forward_warp`
that takes N context views (image + depth + w2c + intrinsics) and a target
camera, warps each context into the target view, and aggregates via
mask-weighted average — mirrors the top-K overlap fusion used in GEN3C's
`Cache3D_BufferSelector.render_cache`.

All inputs are torch tensors. Forward warp expects:
  - frame1     : (B, 3, H, W) image in [-1, 1] (we pass through unchanged)
  - depth1     : (B, 1, H, W) depth in metric units consistent with extrinsics
  - transformation1/2 : (B, 4, 4) **w2c** matrices (despite the docstring saying c2w,
                       the math in `compute_transformed_points` requires w2c)
  - intrinsic1/2 : (B, 3, 3) at the image/depth resolution

This module hides the per-source loop and the batch-of-N aggregation.
"""
from __future__ import annotations

import sys
import types
from typing import Optional

import torch

# `forward_warp_utils_pytorch` imports `warp` at module level for the optional
# `foreground_masking` branch (ray-triangle intersection). We don't enable
# that branch — inject a no-op stub so the import succeeds without the
# NVIDIA Warp package being installed.
if "warp" not in sys.modules:
    _stub = types.ModuleType("warp")
    _stub.init = lambda: None  # type: ignore[attr-defined]
    sys.modules["warp"] = _stub

# GEN3C(cosmos-predict1) is a vendored, gitignored dep only needed for depth-warp
# (condition_latents_input_type=first_frame_depth*). Guard so the module imports
# even when GEN3C is absent on this machine; only fails if the warp is invoked.
try:
    from src.model.GEN3C.cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
        forward_warp,
    )
except ModuleNotFoundError:
    forward_warp = None


@torch.no_grad()
def multi_view_warp_to_target(
    context_images: torch.Tensor,
    context_depths: torch.Tensor,
    context_w2cs: torch.Tensor,
    context_intrinsics: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    *,
    topk: Optional[int] = None,
    return_per_view: bool = False,
):
    """Forward-warp N context views into a single target view and aggregate.

    Args:
        context_images:     (B, V, 3, H, W) in [-1, 1]
        context_depths:     (B, V, 1, H, W) — metric depth
        context_w2cs:       (B, V, 4, 4)
        context_intrinsics: (B, V, 3, 3) at (H, W)
        target_w2c:         (B, 4, 4)
        target_intrinsics:  (B, 3, 3) at (H, W)
        topk: if set, pick K context views closest in camera position per
            batch item before warping; otherwise warp all V views.
        return_per_view: if True, additionally return per-view warps/masks
            (B, V_kept, 3, H, W), (B, V_kept, 1, H, W).

    Returns:
        warped: (B, 3, H, W) in [-1, 1] — mask-weighted average of per-view warps.
            Pixels with no valid contribution from any source are filled with -1.
        mask:   (B, 1, H, W) — union mask (1 where any source contributed).
        (optional) per_view_warps, per_view_masks if `return_per_view=True`.
    """
    if forward_warp is None:
        raise ModuleNotFoundError(
            "src.model.GEN3C (cosmos-predict1) is required for depth-warp "
            "(condition_latents_input_type=first_frame_depth*); vendor it under src/model/GEN3C."
        )
    B, V, _, H, W = context_images.shape
    assert context_depths.shape == (B, V, 1, H, W), context_depths.shape
    assert context_w2cs.shape == (B, V, 4, 4), context_w2cs.shape
    assert context_intrinsics.shape == (B, V, 3, 3), context_intrinsics.shape
    assert target_w2c.shape == (B, 4, 4), target_w2c.shape
    assert target_intrinsics.shape == (B, 3, 3), target_intrinsics.shape
    dtype = context_images.dtype
    device = context_images.device

    # Optional top-K source selection per batch item (by camera position).
    if topk is not None and topk < V:
        # Camera position in world: -R^T t  (for w2c [R|t]).
        c2w_ctx = torch.linalg.inv(context_w2cs.float())               # (B, V, 4, 4)
        cam_pos_ctx = c2w_ctx[..., :3, 3]                              # (B, V, 3)
        c2w_tgt = torch.linalg.inv(target_w2c.float())                 # (B, 4, 4)
        cam_pos_tgt = c2w_tgt[..., :3, 3].unsqueeze(1)                 # (B, 1, 3)
        dists = (cam_pos_ctx - cam_pos_tgt).norm(dim=-1)               # (B, V)
        topk_idx = dists.topk(topk, dim=1, largest=False).indices       # (B, K)
        # Gather. (B, K, ...) — done per batch for clarity.
        out_imgs, out_depths, out_w2cs, out_intrs = [], [], [], []
        for b in range(B):
            idx_b = topk_idx[b]
            out_imgs.append(context_images[b, idx_b])
            out_depths.append(context_depths[b, idx_b])
            out_w2cs.append(context_w2cs[b, idx_b])
            out_intrs.append(context_intrinsics[b, idx_b])
        context_images = torch.stack(out_imgs, dim=0)
        context_depths = torch.stack(out_depths, dim=0)
        context_w2cs = torch.stack(out_w2cs, dim=0)
        context_intrinsics = torch.stack(out_intrs, dim=0)
        V = topk

    # Forward-warp each view independently.
    per_view_warps = []
    per_view_masks = []
    # Repeat target camera across V to reuse the same forward_warp call shape (B*V, ...).
    target_w2c_b1 = target_w2c.unsqueeze(1).expand(B, V, 4, 4)
    target_intr_b1 = target_intrinsics.unsqueeze(1).expand(B, V, 3, 3)

    # Flatten (B, V, ...) -> (B*V, ...) for one forward_warp call.
    flat_imgs = context_images.reshape(B * V, 3, H, W)
    flat_depths = context_depths.reshape(B * V, 1, H, W)
    flat_src_w2cs = context_w2cs.reshape(B * V, 4, 4)
    flat_src_intrs = context_intrinsics.reshape(B * V, 3, 3)
    flat_tgt_w2cs = target_w2c_b1.reshape(B * V, 4, 4)
    flat_tgt_intrs = target_intr_b1.reshape(B * V, 3, 3)

    warped, mask, *_ = forward_warp(
        frame1=flat_imgs,
        mask1=None,
        depth1=flat_depths,
        transformation1=flat_src_w2cs,
        transformation2=flat_tgt_w2cs,
        intrinsic1=flat_src_intrs,
        intrinsic2=flat_tgt_intrs,
        is_image=True,
        is_depth=True,
    )
    warped = warped.reshape(B, V, 3, H, W).to(dtype)
    mask = mask.reshape(B, V, 1, H, W).to(dtype)

    # Mask-weighted average across V sources.
    sum_w = mask.sum(dim=1)                                  # (B, 1, H, W)
    combined = (warped * mask).sum(dim=1) / (sum_w + 1e-7)   # (B, 3, H, W)
    union_mask = (sum_w > 0).to(dtype)                       # (B, 1, H, W)
    # Fill holes with -1 (matches `is_image=True` background in forward_warp).
    combined = torch.where(union_mask > 0, combined, torch.full_like(combined, -1.0))

    if return_per_view:
        return combined, union_mask, warped, mask
    return combined, union_mask
