"""Test depth-based reprojection on a DL3DV scene.

Unproject frame 0 depth via cam 0 (transforms.json), project to cam 3, save outputs.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.geometry.point_cloud_render import reproject_image


SCENE = Path(
    "DATA/DL3DV/DL3DV-960/train/1K/"
    "001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
)
OUT = Path("DATA/test_reproject_out")
SRC_IDX, TGT_IDX = 0, 8
UPSAMPLE = 2  # Super-sample source by this factor to reduce splatting grid holes.


def opengl_to_opencv(c2w: torch.Tensor) -> torch.Tensor:
    """Convert nerfstudio (OpenGL) c2w to OpenCV by flipping camera Y, Z axes."""
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=c2w.dtype))
    return c2w @ flip


def load_image(path: Path, hw: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(t: torch.Tensor, path: Path) -> None:
    arr = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def save_depth_vis(d: torch.Tensor, path: Path) -> None:
    arr = d.cpu().numpy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vis = np.zeros_like(arr, dtype=np.uint8)
    else:
        lo, hi = float(finite.min()), float(finite.max())
        vis = np.where(
            np.isfinite(arr),
            ((arr - lo) / max(hi - lo, 1e-8) * 255).clip(0, 255),
            0,
        ).astype(np.uint8)
    Image.fromarray(vis).save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(SCENE / "da3/exports/mini_npz/results.npz")
    depth_np = data["depth"]
    K_da3 = data["intrinsics"]
    _, H, W = depth_np.shape

    with open(SCENE / "transforms.json") as f:
        meta = json.load(f)
    frames = meta["frames"]

    # Normalized intrinsics from da3 (matches depth resolution exactly).
    fx, fy = float(K_da3[SRC_IDX, 0, 0]), float(K_da3[SRC_IDX, 1, 1])
    cx, cy = float(K_da3[SRC_IDX, 0, 2]), float(K_da3[SRC_IDX, 1, 2])
    K_norm = torch.tensor(
        [[fx / W, 0.0, cx / W], [0.0, fy / H, cy / H], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    c2w_src = opengl_to_opencv(torch.tensor(frames[SRC_IDX]["transform_matrix"], dtype=torch.float32))
    c2w_tgt = opengl_to_opencv(torch.tensor(frames[TGT_IDX]["transform_matrix"], dtype=torch.float32))

    img_src = load_image(SCENE / frames[SRC_IDX]["file_path"], (H, W))
    img_tgt = load_image(SCENE / frames[TGT_IDX]["file_path"], (H, W))
    depth_src = torch.from_numpy(depth_np[SRC_IDX])

    # Super-sample source: nearest for depth (preserves edges), bilinear for RGB.
    if UPSAMPLE > 1:
        depth_in = F.interpolate(
            depth_src[None, None], scale_factor=UPSAMPLE, mode="nearest"
        )[0, 0]
        img_in = F.interpolate(
            img_src[None], scale_factor=UPSAMPLE, mode="bilinear", align_corners=False
        )[0]
    else:
        depth_in, img_in = depth_src, img_src

    rendered, depth_tgt, mask = reproject_image(
        image=img_in[None].to(device),
        depth=depth_in[None].to(device),
        extrinsics_src=c2w_src[None].to(device),
        intrinsics_src=K_norm[None].to(device),
        extrinsics_tgt=c2w_tgt[None].to(device),
        intrinsics_tgt=K_norm[None].to(device),
        out_hw=(H, W),
    )

    save_image(img_src, OUT / "src_image.png")
    save_image(img_tgt, OUT / "tgt_gt.png")
    save_image(rendered[0], OUT / "rendered_to_tgt.png")
    save_depth_vis(depth_src, OUT / "src_depth.png")
    Image.fromarray((mask[0].cpu().numpy().astype(np.uint8) * 255)).save(OUT / "rendered_mask.png")

    print(f"depth range: [{depth_src.min():.3f}, {depth_src.max():.3f}]")
    print(f"mask coverage: {mask.float().mean().item():.3f}")
    print(f"outputs saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
