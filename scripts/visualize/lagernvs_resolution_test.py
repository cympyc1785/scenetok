"""Check that LagerNVS general_512 runs at arbitrary (non-512, non-square)
resolutions, e.g. 256x448 and 480x832.

Loads a DL3DV context bundle, FORCES the context images to an exact (H,W) via
interpolation, builds Plücker rays / cam_tokens (mirroring lagernvs_infer.py's
normalization), runs render_chunked, and reports success + output shape. Saves a
sample mp4 per resolution so coherence can be eyeballed.

Run in the `lagernvs` conda env:
  CUDA_VISIBLE_DEVICES=3 <lagernvs_py> scripts/visualize/lagernvs_resolution_test.py \
      --repo submodules/lagernvs --ckpt submodules/lagernvs/checkpoints/lagernvs_general_512/model.pt \
      --bundle results/context_views_dl3dv_c16_37/a4c20f668ce179db.pt \
      --hw 256,448 480,832 --n_targets 12
"""
import argparse
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--hw", nargs="+", default=["256,448", "480,832"])
    ap.add_argument("--n_targets", type=int, default=12)
    ap.add_argument("--out_dir", default="results/lagernvs_resolution_test")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from models.encoder_decoder import EncDec_VitB8
    from vis import render_chunked, compute_plucker_coordinates

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    print(f"[res-test] loading model on {device} ({dtype})", flush=True)
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(device).eval()

    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    imgs = b["images"].float()                       # (Vc,3,h0,w0)
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    Vc, _, h0, w0 = imgs.shape
    ctx_c2w = b["c2w"].float()                        # (Vc,4,4)
    ctx_K = b["intrinsics"].float()                   # (Vc,3,3)
    tgt_c2w = b["target_c2w"].float()[: args.n_targets]
    tgt_K = b["target_intrinsics"].float()[: args.n_targets]

    # Normalize intrinsics to [0,1] (divide pixel K by the ORIGINAL image size).
    def norm_K(K):
        if K[..., 0, 0].max() <= 2.0:                # already normalized
            return K
        K = K.clone()
        K[..., 0, :] /= w0
        K[..., 1, :] /= h0
        return K
    ctx_Kn, tgt_Kn = norm_K(ctx_K), norm_K(tgt_K)

    # relativize + scale exactly like lagernvs_infer.py
    first_inv = torch.linalg.inv(ctx_c2w[0:1])
    ctx_c2w = first_inv @ ctx_c2w
    tgt_c2w = first_inv @ tgt_c2w
    scene_scale = torch.clamp(1.35 * torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)), min=1e-6)
    ctx_c2w[:, :3, 3] /= scene_scale
    tgt_c2w[:, :3, 3] /= scene_scale
    camera_scale = torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)).item()
    Vt = tgt_c2w.shape[0]

    out_root = Path(args.out_dir)
    for hw in args.hw:
        H, W = (int(x) for x in hw.split(","))
        assert H % 8 == 0 and W % 8 == 0, f"H,W must be /8 (patch_size): {H},{W}"
        try:
            images = F.interpolate(imgs, size=(H, W), mode="bilinear", align_corners=False)
            images = images.to(device).unsqueeze(0)          # (1,Vc,3,H,W)

            def fxfycxcy(Kn):
                return torch.stack([Kn[:, 0, 0] * W, Kn[:, 1, 1] * H,
                                    Kn[:, 0, 2] * W, Kn[:, 1, 2] * H], dim=-1).unsqueeze(0)
            target_rays = compute_plucker_coordinates(tgt_c2w.unsqueeze(0), fxfycxcy(tgt_Kn), (H, W))
            cond_rays = torch.zeros(1, Vc, 6, H, W)
            rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
            cam_tokens = torch.zeros(1, Vc + Vt, 11, device=device)
            cam_tokens[:, :, 9] = camera_scale

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                video_out = render_chunked(model, (images, rays, cam_tokens), num_cond_views=Vc)
            video = video_out[0].detach().cpu().float().clamp(0, 1)   # (Vt,3,H,W)
            d = out_root / f"{H}x{W}"
            d.mkdir(parents=True, exist_ok=True)
            frames = (video.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
            pil = [Image.fromarray(f) for f in frames]
            pil[0].save(str(d / "render.gif"), save_all=True, append_images=pil[1:],
                        duration=int(1000 / max(args.fps, 1)), loop=0)
            print(f"[res-test] OK  {H}x{W} -> out {tuple(video.shape)}  saved {d}/render.gif", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[res-test] FAIL {H}x{W}: {e}", flush=True)

    print("[res-test] done.", flush=True)


if __name__ == "__main__":
    main()
