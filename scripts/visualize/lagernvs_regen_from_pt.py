"""Render saved viser camera trajectories (poses.pt) with a POSED LagerNVS
checkpoint (re10k_2v / dl3dv_2-6_v), at one or more resolutions.

Mirrors lagernvs_infer.py's posed render (context Plücker rays from context
cameras), but batch-driven from a dir of poses.pt (each: target_c2w_edited rel→
ctx0 + source_bundle). Context views/cameras come from the bundle (subsampled to
`--num_cond`); targets from poses.pt. Both frames are rel→ctx0 (ctx0 = identity).

Run in the `lagernvs` conda env, e.g.:
  CUDA_VISIBLE_DEVICES=3 <lagernvs_py> scripts/visualize/lagernvs_regen_from_pt.py \
    --repo submodules/lagernvs \
    --ckpt submodules/lagernvs/checkpoints/lagernvs_re10k_2v_256/model.pt \
    --attention_type full_attention --num_cond 2 \
    --src_dir results/viser_generate/va-wan_dl3dv_256x448 \
    --out_dir results/viser_generate/lagernvs_re10k_2v_256 --hw 256,256 480,832
"""
import argparse
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attention_type", required=True,
                    choices=["full_attention", "bidirectional_cross_attention"])
    ap.add_argument("--num_cond", type=int, required=True, help="num conditioning views (re10k=2, dl3dv=6)")
    ap.add_argument("--src_dir", required=True, help="dir of per-pose subdirs each with poses.pt")
    ap.add_argument("--out_dir", required=True, help="output root; <out_dir>_<HxW>/<pose>/render.gif")
    ap.add_argument("--hw", nargs="+", default=["256,256", "480,832"])
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--unposed", action="store_true",
                    help="general_512 mode: zero context rays + zero-pose cam_tokens "
                         "(model ignores context cameras). Targets stay posed via rays.")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from models.encoder_decoder import EncDec_VitB8
    from vis import render_chunked, compute_plucker_coordinates
    from data.camera_utils import (
        get_full_res_crop_dims_constant_ar, adjust_intrinsics_for_crop_and_resize)
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[lnvs-regen] loading {Path(args.ckpt).parent.name} ({args.attention_type}, num_cond={args.num_cond})", flush=True)
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type=args.attention_type)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(device).eval()

    src = Path(args.src_dir)
    subdirs = sorted([d for d in src.iterdir() if d.is_dir() and d.name != "backup" and (d / "poses.pt").exists()])
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        subdirs = [d for d in subdirs if any(k in d.name for k in keys)]
    print(f"[lnvs-regen] {len(subdirs)} pose dirs", flush=True)

    bundle_cache = {}
    def get_bundle(p):
        if p not in bundle_cache:
            bundle_cache[p] = torch.load(p, map_location="cpu", weights_only=False)
        return bundle_cache[p]

    def adj_fxfycxcy(Kn, h0, w0, crop_hw, H, W):
        """normalized K (V,3,3) → original pixels → adjust for center-crop+resize
        to (H,W) so the FOV/aspect stays geometrically correct (NOT a stretch)."""
        fx = Kn[:, 0, 0] * w0; fy = Kn[:, 1, 1] * h0
        cx = Kn[:, 0, 2] * w0; cy = Kn[:, 1, 2] * h0
        fx, fy, cx, cy = adjust_intrinsics_for_crop_and_resize(
            (fx, fy, cx, cy), (h0, w0), crop_hw, (H, W))
        return torch.stack([fx, fy, cx, cy], dim=-1).unsqueeze(0)

    ok = fail = 0
    for d in subdirs:
        obj = torch.load(d / "poses.pt", map_location="cpu", weights_only=False)
        tgt_c2w = torch.as_tensor(np.asarray(obj["target_c2w_edited"]), dtype=torch.float32)  # (Vt,4,4) rel ctx0
        bpath = obj.get("source_bundle")
        if bpath is None or not Path(bpath).exists():
            print(f"[lnvs-regen] SKIP {d.name}: bundle missing"); fail += 1; continue
        b = get_bundle(bpath)
        imgs0 = b["images"].float()
        if imgs0.max() > 1.5:
            imgs0 = imgs0 / 255.0
        Vc0, _, h0, w0 = imgs0.shape
        cidx = np.linspace(0, Vc0 - 1, args.num_cond).round().astype(int)   # includes 0 → ctx0 identity
        ctx_imgs = imgs0[cidx]
        ctx_c2w = b["c2w"].float()[cidx]                  # rel ctx0
        ctx_Kn = b["intrinsics"].float()[cidx]            # normalized
        tgt_Kn = b["target_intrinsics"].float()           # (Vt,3,3) normalized
        Vt = tgt_c2w.shape[0]
        tgt_Kn = tgt_Kn[:Vt] if tgt_Kn.shape[0] >= Vt else tgt_Kn[:1].repeat(Vt, 1, 1)

        # normalize cameras like lagernvs_infer (ctx0=I → first_inv no-op, kept for parity)
        first_inv = torch.linalg.inv(ctx_c2w[0:1])
        ctx_c2w = first_inv @ ctx_c2w
        tgt_c2w = first_inv @ tgt_c2w
        scale = torch.clamp(1.35 * torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)), min=1e-6)
        ctx_c2w[:, :3, 3] /= scale
        tgt_c2w[:, :3, 3] /= scale
        camera_scale = torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)).item()

        for hw in args.hw:
            H, W = (int(x) for x in hw.split(","))
            out_d = Path(f"{args.out_dir}_{H}x{W}") / d.name
            try:
                # center-crop the context images to the TARGET aspect, then resize
                # to (H,W) — preserves geometry (vs naive stretch). Intrinsics get
                # the matching crop+resize adjustment so rays stay correct.
                crop_h, crop_w = get_full_res_crop_dims_constant_ar((h0, w0), (H, W))
                top, left = (h0 - crop_h) // 2, (w0 - crop_w) // 2
                cropped = ctx_imgs[:, :, top:top + crop_h, left:left + crop_w]
                # Match official load_and_preprocess_images: PIL BICUBIC resize of the
                # center crop (NOT bilinear F.interpolate) so context input is identical
                # to the LagerNVS inference pipeline at this resolution.
                resized = []
                for im in cropped:
                    pil = Image.fromarray((im.permute(1, 2, 0).numpy() * 255).astype("uint8"))
                    pil = pil.resize((W, H), Image.Resampling.BICUBIC)
                    resized.append(torch.from_numpy(np.asarray(pil).astype("float32") / 255.0).permute(2, 0, 1))
                images = torch.stack(resized).to(device).unsqueeze(0)
                crop_hw = (crop_h, crop_w)
                tgt_ff = adj_fxfycxcy(tgt_Kn, h0, w0, crop_hw, H, W)   # (1,Vt,4) pixel
                target_rays = compute_plucker_coordinates(tgt_c2w.unsqueeze(0), tgt_ff, (H, W))
                if args.unposed:
                    # general_512: context cameras ignored → zero cond rays + zero-pose
                    # cam_tokens (mirror vis.create_target_camera_path). Targets posed via rays.
                    cond_rays = torch.zeros(1, args.num_cond, 6, H, W)
                    rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
                    cam_tokens = torch.zeros(1, args.num_cond + Vt, 11, device=device)
                    cam_tokens[:, :, 9] = camera_scale
                else:
                    # POSED (mirror data.normalization.build_cam_cond): per-view 9-dim pose
                    # encoding (extri_intri_to_pose_encoding of c2w + pixel K) + [camera_scale, 0].
                    ctx_ff = adj_fxfycxcy(ctx_Kn, h0, w0, crop_hw, H, W)
                    cond_rays = compute_plucker_coordinates(ctx_c2w.unsqueeze(0), ctx_ff, (H, W))
                    rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
                    all_c2w = torch.cat([ctx_c2w, tgt_c2w], dim=0)
                    all_ff = torch.cat([ctx_ff, tgt_ff], dim=1)
                    pose_encoding = extri_intri_to_pose_encoding(
                        all_c2w[:, :3, :4].unsqueeze(0), all_ff, image_size_hw=(H, W)).squeeze(0)
                    scale_tok = torch.tensor([[camera_scale, 0.0]], dtype=pose_encoding.dtype).expand(pose_encoding.shape[0], 2)
                    cam_tokens = torch.cat([pose_encoding, scale_tok], dim=-1).unsqueeze(0).to(device)
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                    video = render_chunked(model, (images, rays, cam_tokens), num_cond_views=args.num_cond)
                video = video[0].detach().cpu().float().clamp(0, 1)
                out_d.mkdir(parents=True, exist_ok=True)
                frames = (video.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
                pil = [Image.fromarray(f) for f in frames]
                pil[0].save(str(out_d / "render.gif"), save_all=True, append_images=pil[1:],
                            duration=int(1000 / max(args.fps, 1)), loop=0)
                print(f"[lnvs-regen] OK {d.name} {H}x{W} -> {tuple(video.shape)}", flush=True)
                ok += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[lnvs-regen] FAIL {d.name} {H}x{W}: {e}", flush=True); fail += 1
    print(f"[lnvs-regen] done. ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
