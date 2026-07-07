"""Stage 2 (lagernvs env): render viser trajectories with the LagerNVS *diffusion*
decoder (checkpoint_latest.pt) via DDIM sampling.

The repo has no sampler (train_diffusion.py only does the forward objective), so we
implement DDIM (eta=0) over the training DDPM cosine schedule with x0-prediction.
Inputs (images / Plücker rays / cam_tokens) are built exactly like
scripts/visualize/lagernvs_infer.py from the Stage-1 payloads, but at the LagerNVS
training resolution (256x256 square) with a training-size cond-view count.

Run with the lagernvs (reco) python, cwd = the lagernvs repo:
  CUDA_VISIBLE_DEVICES=N reco_python scripts/visualize/lagernvs_diffusion_sample.py \
      --payload_dir <stage1 out> --out_dir <...> --ckpt output/.../checkpoint_latest.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import einops
from PIL import Image


def cosine_alphas_cumprod(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    acp = f / f[0]
    betas = (1 - acp[1:] / acp[:-1]).clamp(max=0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()  # (T,)


def build_inputs(payload, num_cond, size, mode, load_and_preprocess_images, compute_plucker, device):
    """Mirror lagernvs_infer.render(): images / rays / cam_tokens at (size,size)."""
    names = list(payload["context_image_paths"])
    # Subsample context to a training-size cond count (evenly across available views).
    if num_cond < len(names):
        idx = np.linspace(0, len(names) - 1, num_cond).round().astype(int)
        names = [names[i] for i in idx]
        ctx_sel = torch.as_tensor(idx)
    else:
        ctx_sel = torch.arange(len(names))
    n_cond = len(names)
    images = load_and_preprocess_images(names, mode=mode, target_size=size, patch_size=8).to(device).unsqueeze(0)
    H, W = images.shape[-2], images.shape[-1]

    ctx_c2w = payload["context_c2w"].float()[ctx_sel]     # (n_cond,4,4) world
    tgt_c2w = payload["target_c2w"].float()               # (T,4,4) world
    tgt_K = payload["target_intrinsics_norm"].float()     # (T,3,3) normalized

    first_inv = torch.linalg.inv(ctx_c2w[0:1])
    ctx_c2w = first_inv @ ctx_c2w
    tgt_c2w = first_inv @ tgt_c2w
    scene_scale = torch.clamp(1.35 * torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)), min=1e-6)
    ctx_c2w[:, :3, 3] /= scene_scale
    tgt_c2w[:, :3, 3] /= scene_scale
    camera_scale = torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)).item()

    T = tgt_c2w.shape[0]
    cam_tokens = torch.zeros(1, n_cond + T, 11)
    cam_tokens[:, :, 9] = camera_scale
    fx, fy = tgt_K[:, 0, 0] * W, tgt_K[:, 1, 1] * H
    cx, cy = tgt_K[:, 0, 2] * W, tgt_K[:, 1, 2] * H
    tgt_fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).unsqueeze(0)
    target_rays = compute_plucker(tgt_c2w.unsqueeze(0), tgt_fxfycxcy, (H, W))
    cond_rays = torch.zeros(1, n_cond, 6, H, W)           # unposed cond (general recipe)
    rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
    return images, rays, cam_tokens.to(device), n_cond, (H, W)


@torch.no_grad()
def ddim_sample(model, images, rays, cam_tokens, n_cond, hw, acp, steps, device, dtype, seed=0):
    H, W = hw
    input_images = images[:, :n_cond]
    cam_token = cam_tokens[:, :n_cond]
    target_rays = rays[:, n_cond:]
    v_t = target_rays.shape[1]
    # Encoder runs ONCE (rec_tokens are t-independent).
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        rec = model.reconstructor(input_images, cam_token)          # (1,v_in,p,c)
    rec = einops.rearrange(rec, "b v p c -> b (v p) c")
    rec = einops.repeat(rec, "b np d -> (b v) np d", v=v_t)

    T = acp.shape[0]
    ts = torch.linspace(T - 1, 0, steps).round().long().tolist()
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, v_t, 3, H, W, generator=g, device=device)    # x_T in [-1,1] space
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else 0
        a_t = acp[t].to(device); a_p = acp[t_prev].to(device)
        t_batch = torch.full((1, v_t), t, device=device, dtype=torch.long)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            pred_x0 = model.renderer(rec, target_rays, noisy_img=x, timestep=t_batch)  # [0,1]
        pred_x0c = (2.0 * pred_x0.float() - 1.0)
        eps = (x - a_t.sqrt() * pred_x0c) / (1.0 - a_t).clamp(min=1e-8).sqrt()
        x = a_p.sqrt() * pred_x0c + (1.0 - a_p).clamp(min=0).sqrt() * eps
    return ((x + 1.0) * 0.5).clamp(0, 1)[0]                          # (v_t,3,H,W) [0,1]


def save_video(frames, path, fps):
    import imageio.v3 as iio
    arr = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    iio.imwrite(path, arr, fps=fps, codec="libx264")
    gif = Path(path).with_suffix(".gif")
    imgs = [Image.fromarray(a) for a in arr]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, disposal=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/data1/cympyc1785/lagernvs")
    ap.add_argument("--payload_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="output/smoke_diffusion/checkpoints/checkpoint_latest.pt")
    ap.add_argument("--num_cond", type=int, default=6, help="training cond-view count (2-6)")
    ap.add_argument("--size", type=int, default=256, help="training resolution (square)")
    ap.add_argument("--mode", default="square_crop", choices=["resize", "square_crop"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from models.encoder_decoder import EncDec_VitB8
    from vggt.utils.load_fn import load_and_preprocess_images
    from vis import compute_plucker_coordinates

    device = "cuda"
    dtype = torch.bfloat16
    model = EncDec_VitB8(freeze_vggt=True, pretrained_vggt=False,
                         attention_to_features_type="bidirectional_cross_attention",
                         diffusion=True).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[diff-sample] ckpt iter={ck.get('iter_idx')} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    acp = cosine_alphas_cumprod(args.timesteps)

    pdir = Path(args.payload_dir)
    subdirs = sorted([d for d in pdir.iterdir() if (d / "payload.pt").exists()])
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        subdirs = [d for d in subdirs if any(k in d.name for k in keys)]
    print(f"[diff-sample] {len(subdirs)} payloads | num_cond={args.num_cond} size={args.size} steps={args.steps}")

    ok = 0
    for d in subdirs:
        payload = torch.load(d / "payload.pt", map_location="cpu", weights_only=False)
        images, rays, cam_tokens, n_cond, hw = build_inputs(
            payload, args.num_cond, args.size, args.mode,
            load_and_preprocess_images, compute_plucker_coordinates, device)
        t0 = time.time()
        frames = ddim_sample(model, images, rays, cam_tokens, n_cond, hw, acp,
                             args.steps, device, dtype, seed=args.seed)
        out_d = Path(args.out_dir) / d.name
        out_d.mkdir(parents=True, exist_ok=True)
        save_video(frames, str(out_d / "generated.mp4"), args.fps)
        (out_d / "meta.txt").write_text(
            f"ckpt={args.ckpt} iter={ck.get('iter_idx')} num_cond={n_cond} size={args.size} "
            f"mode={args.mode} steps={args.steps} T={args.timesteps} scene={payload.get('scene')}\n")
        print(f"[diff-sample] OK {d.name} -> {tuple(frames.shape)} ({time.time()-t0:.1f}s)")
        ok += 1
    print(f"[diff-sample] done. ok={ok} -> {args.out_dir}")


if __name__ == "__main__":
    main()
