"""LagerNVS inference worker for viser_server.py.

Runs in the `lagernvs` conda env (py3.10/3.11 + torch 2.8) as a SUBPROCESS —
viser_server runs in the `scenetok` env and cannot import LagerNVS in-process, so
it writes a payload + spawns this script with the lagernvs python.

Renders the viser-EDITED target cameras (not LagerNVS's auto VGGT path) with the
general_512 model: builds Plücker rays + cam_tokens from the bundle's context
images + (context/target) c2w + target intrinsics, following the exact
camera-based normalization in `vis.create_target_camera_path` (multi-view):
  scene_scale = 1.35 · max‖ctx c2w translation‖ ; translations /= scene_scale ;
  cam_tokens[...,9] = max‖normalized ctx translation‖, cam_tokens[...,10] = 0 ;
  cond rays = 0 (model ignores input poses), target rays from our cameras.
Feeding our own cameras skips `create_target_camera_path` → no VGGT 4GB download.

Payload (.pt) keys (written by viser_server):
  context_image_paths : list[str]   PNG paths of the Vc context views
  context_c2w         : (Vc,4,4)     OpenCV c2w, relative to context[0]
  target_c2w          : (Vt,4,4)     OpenCV c2w, relative to context[0] (edited)
  target_intrinsics_norm : (Vt,3,3)  NORMALIZED intrinsics (fx/W, cx/W, …)
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="LagerNVS render of viser target cameras")
    ap.add_argument("--repo", required=True, help="submodules/lagernvs path (added to sys.path)")
    ap.add_argument("--payload", required=True, help=".pt payload from viser_server")
    ap.add_argument("--output", required=True, help="output mp4 path")
    ap.add_argument("--ckpt", required=True, help="general_512 model.pt path")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--mode", default="resize", choices=["resize", "square_crop"])
    ap.add_argument("--attention_type", default="bidirectional_cross_attention",
                    choices=["bidirectional_cross_attention", "full_attention"])
    args = ap.parse_args()

    # LagerNVS modules import relative to the repo root → put it on sys.path so
    # this worker can live in OUR repo (scripts/) yet import `models`/`vis`/`vggt`.
    sys.path.insert(0, str(Path(args.repo).resolve()))

    import torch
    from models.encoder_decoder import EncDec_VitB8
    from vggt.utils.load_fn import load_and_preprocess_images
    from vis import render_chunked, compute_plucker_coordinates
    from eval.export import save_video

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16
             if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    pl = torch.load(args.payload, map_location="cpu")
    image_names = list(pl["context_image_paths"])
    num_cond = len(image_names)

    # Preprocess context images with LagerNVS's own loader (resize, longer side
    # = target_size, patch_size=8) so H,W match the model's expectations.
    images = load_and_preprocess_images(
        image_names, mode=args.mode, target_size=args.target_size, patch_size=8
    ).to(device).unsqueeze(0)  # (1, Vc, 3, H, W)
    H, W = images.shape[-2], images.shape[-1]

    ctx_c2w = pl["context_c2w"].float()             # (Vc,4,4) OpenCV, rel ctx0
    tgt_c2w = pl["target_c2w"].float()              # (Vt,4,4)
    tgt_K = pl["target_intrinsics_norm"].float()    # (Vt,3,3) normalized

    # Re-relativize to context[0] (idempotent if already rel ctx0) for safety.
    first_inv = torch.linalg.inv(ctx_c2w[0:1])
    ctx_c2w = first_inv @ ctx_c2w
    tgt_c2w = first_inv @ tgt_c2w

    # Camera-based scene normalization (vis.create_target_camera_path multi-view).
    scene_scale = 1.35 * torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1))
    scene_scale = torch.clamp(scene_scale, min=1e-6)
    ctx_c2w[:, :3, 3] /= scene_scale
    tgt_c2w[:, :3, 3] /= scene_scale
    camera_scale = torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)).item()

    Vt = tgt_c2w.shape[0]
    total = num_cond + Vt
    cam_tokens = torch.zeros(1, total, 11)
    cam_tokens[:, :, 9] = camera_scale
    cam_tokens[:, :, 10] = 0.0

    # Normalized intrinsics → pixel units at the render (H, W).
    fx = tgt_K[:, 0, 0] * W
    fy = tgt_K[:, 1, 1] * H
    cx = tgt_K[:, 0, 2] * W
    cy = tgt_K[:, 1, 2] * H
    tgt_fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).unsqueeze(0)  # (1, Vt, 4)

    target_rays = compute_plucker_coordinates(
        tgt_c2w.unsqueeze(0), tgt_fxfycxcy, (H, W))           # (1, Vt, 6, H, W)
    cond_rays = torch.zeros(1, num_cond, 6, H, W)
    rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
    cam_tokens = cam_tokens.to(device)

    print(f"[lagernvs-infer] Vc={num_cond} Vt={Vt} render={H}x{W} "
          f"scene_scale={float(scene_scale):.4f} camera_scale={camera_scale:.4f}")

    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type=args.attention_type)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(device).eval()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            video_out = render_chunked(
                model, (images, rays, cam_tokens), num_cond_views=num_cond)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_out[0], str(out))
    print(f"[lagernvs-infer] saved {tuple(video_out.shape)} -> {out}")


if __name__ == "__main__":
    main()
