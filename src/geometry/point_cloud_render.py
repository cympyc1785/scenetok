import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from .projection import (
    homogenize_points,
    project_camera_space,
    sample_image_grid,
    transform_cam2world,
    transform_world2cam,
    unproject,
)


def unproject_depth_to_world(
    depth: Float[Tensor, "*batch H W"],
    extrinsics: Float[Tensor, "*batch 4 4"],
    intrinsics: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch H W 3"]:
    """Unproject a depth map into world-space points using the source camera."""
    *batch_dims, h, w = depth.shape
    device, dtype = depth.device, depth.dtype

    xy, _ = sample_image_grid((h, w), device=device, dtype=dtype)
    for _ in batch_dims:
        xy = xy.unsqueeze(0)
    xy = xy.expand(*batch_dims, h, w, 2)

    intrinsics_b = intrinsics[..., None, None, :, :].expand(*batch_dims, h, w, 3, 3)
    cam_points = unproject(xy, depth, intrinsics_b)

    extrinsics_b = extrinsics[..., None, None, :, :].expand(*batch_dims, h, w, 4, 4)
    world_points = transform_cam2world(homogenize_points(cam_points), extrinsics_b)[..., :3]
    return world_points


def render_point_cloud_zbuffer(
    points_world: Float[Tensor, "batch N 3"],
    colors: Float[Tensor, "batch N C"],
    extrinsics_tgt: Float[Tensor, "batch 4 4"],
    intrinsics_tgt: Float[Tensor, "batch 3 3"],
    out_hw: tuple[int, int],
    valid: Bool[Tensor, "batch N"] | None = None,
) -> tuple[
    Float[Tensor, "batch C H W"],
    Float[Tensor, "batch H W"],
    Bool[Tensor, "batch H W"],
]:
    """Splat world-space points onto the target image plane with a per-pixel z-buffer.

    Returns (rendered_color, rendered_depth, mask). Empty pixels have depth=inf.
    """
    b, n, c = colors.shape
    h_out, w_out = out_hw
    device, dtype = points_world.device, points_world.dtype
    inf = torch.tensor(float("inf"), device=device, dtype=dtype)

    # Move points into the target camera frame, then project (sharing the inverse).
    points_h = homogenize_points(points_world)
    extr_b = extrinsics_tgt[:, None, :, :].expand(b, n, 4, 4)
    intr_b = intrinsics_tgt[:, None, :, :].expand(b, n, 3, 3)
    cam_tgt = transform_world2cam(points_h, extr_b)[..., :3]
    z_tgt = cam_tgt[..., 2]
    in_front = z_tgt >= 0
    xy_tgt = project_camera_space(cam_tgt, intr_b)

    u = (xy_tgt[..., 0] * w_out).long()
    v = (xy_tgt[..., 1] * h_out).long()
    in_bounds = (u >= 0) & (u < w_out) & (v >= 0) & (v < h_out)
    valid_all = in_front & in_bounds & (z_tgt > 0)
    if valid is not None:
        valid_all = valid_all & valid

    flat_idx = (v.clamp(0, h_out - 1) * w_out + u.clamp(0, w_out - 1))

    # Per-pixel min depth via scatter_reduce (invalid points pushed to +inf).
    z_for_scatter = torch.where(valid_all, z_tgt, inf.expand_as(z_tgt))
    out_depth = torch.full((b, h_out * w_out), float("inf"), device=device, dtype=dtype)
    out_depth.scatter_reduce_(1, flat_idx, z_for_scatter, reduce="amin", include_self=True)

    # Identify the winner per source point (its depth equals min depth at its target pixel).
    min_at_src = out_depth.gather(1, flat_idx)
    is_winner = valid_all & (z_tgt == min_at_src)

    # Scatter colors only for winners.
    out_color = torch.zeros(b, h_out * w_out, c, device=device, dtype=dtype)
    out_mask = torch.zeros(b, h_out * w_out, dtype=torch.bool, device=device)

    winner_pos = is_winner.nonzero(as_tuple=False)
    if winner_pos.numel() > 0:
        b_idx = winner_pos[:, 0]
        n_idx = winner_pos[:, 1]
        tgt_pix = flat_idx[b_idx, n_idx]
        out_color[b_idx, tgt_pix] = colors[b_idx, n_idx]
        out_mask[b_idx, tgt_pix] = True

    out_color = rearrange(out_color, "b (h w) c -> b c h w", h=h_out, w=w_out)
    out_depth = rearrange(out_depth, "b (h w) -> b h w", h=h_out, w=w_out)
    out_mask = rearrange(out_mask, "b (h w) -> b h w", h=h_out, w=w_out)
    return out_color, out_depth, out_mask


def reproject_image(
    image: Float[Tensor, "batch C H W"],
    depth: Float[Tensor, "batch H W"],
    extrinsics_src: Float[Tensor, "batch 4 4"],
    intrinsics_src: Float[Tensor, "batch 3 3"],
    extrinsics_tgt: Float[Tensor, "batch 4 4"],
    intrinsics_tgt: Float[Tensor, "batch 3 3"],
    out_hw: tuple[int, int] | None = None,
) -> tuple[
    Float[Tensor, "batch C Hout Wout"],
    Float[Tensor, "batch Hout Wout"],
    Bool[Tensor, "batch Hout Wout"],
]:
    """Lift a source view to a point cloud via depth, then render under the target camera.

    Conventions: extrinsics are cam2world (4x4); intrinsics are normalized (3x3, coords in [0, 1]).
    """
    b, c, h, w = image.shape
    if out_hw is None:
        out_hw = (h, w)

    world_points = unproject_depth_to_world(depth, extrinsics_src, intrinsics_src)
    points_flat = rearrange(world_points, "b h w xyz -> b (h w) xyz")
    colors_flat = rearrange(image, "b c h w -> b (h w) c")

    valid = depth.reshape(b, h * w) > 0

    return render_point_cloud_zbuffer(
        points_flat,
        colors_flat,
        extrinsics_tgt,
        intrinsics_tgt,
        out_hw,
        valid=valid,
    )
