"""Iterative Wan2.2 VAE encode-decode roundtrip on the FIRST validation-standard
GT video, to inspect how geometry survives repeated VAE passes.

- Rebuilds the exact standard val dataloader (shuffle=False, seed=0) of an
  experiment config, takes batch 0 / sample 0 -> batch["target"]["latent"]
  (GT RGB target video, [0,1]).
- Repeats encode->decode N times, feeding each decode back into the next encode.
- Height-concats [GT, iter1, ..., iterN] per frame into one video.

Normalization matches the training path (diffusion.py: inputs*2-1 before encode).
Wan2.2 VAE has temporal compression (T -> 1+(T-1)//4), so the GT clip is trimmed
to the largest (4k+1) length so each roundtrip is frame-count-preserving.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


def compose_cfg(experiment):
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config

    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[f"+experiment={experiment}", "mode=train", "wandb.activated=false"],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.root = "./DATA/DL3DV/DL3DV-960"
    OmegaConf.set_struct(cfg, True)
    return load_typed_root_config(cfg)


def get_first_std_gt(cfg):
    from src.dataset.data_module import DataModule
    from src.misc.step_tracker import StepTracker

    dm = DataModule(cfg.dataset, cfg.data_loader, StepTracker(0))
    loaders = dm.val_dataloader()
    std = loaders["standard"]
    batch = next(iter(std))
    gt = batch["target"]["latent"][0]  # (V, 3, H, W) in [0,1]
    scene = batch.get("scene", ["?"])
    return gt.float(), (scene[0] if isinstance(scene, (list, tuple)) else scene)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="custom/lagernvs_va-wan_dl3dv_recon_sceneLRnorm_shuf_extrap")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--vae", default="checkpoints/Wan2.2_VAE.pth")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out", default=str(REPO / "results/wan_vae_roundtrip_valstd"))
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print(f"[cfg] composing {args.experiment}")
    cfg = compose_cfg(args.experiment)
    gt, scene = get_first_std_gt(cfg)
    V = gt.shape[0]
    keep = 1 + ((V - 1) // 4) * 4  # largest 4k+1 <= V (Wan temporal alignment)
    gt = gt[:keep].to(dev)
    print(f"[data] scene={scene} GT frames={V} -> using {keep} (4k+1), hw={tuple(gt.shape[-2:])}")

    from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg

    print("[vae] loading Wan2.2 VAE (48ch)")
    vae = AutoencoderWan(WanKwargsCfg(in_channels=3, latent_channels=48, scaling_factor=1.0))
    vae.from_pretrained(args.vae)
    vae = vae.to(dev).eval()

    def roundtrip(video01):  # (T,3,H,W) [0,1] -> (T,3,H,W) [0,1]
        x = (video01 * 2 - 1).unsqueeze(0).to(dev)          # (1,T,3,H,W)
        z = vae.encode(x).float()
        y = vae.decode(z.to(x.dtype))
        return ((y[0].float() + 1) / 2).clamp(0, 1)

    def psnr(a, b):
        mse = (a - b).pow(2).mean().item()
        return float("inf") if mse <= 1e-12 else 10 * np.log10(1.0 / mse)

    rows = [gt.cpu()]  # iteration 0 = GT
    cur = gt
    for i in range(1, args.iters + 1):
        cur = roundtrip(cur)
        assert cur.shape[0] == keep, f"frame count drift at iter {i}: {cur.shape[0]} != {keep}"
        rows.append(cur.cpu())
        print(f"[iter {i:2d}] PSNR vs GT = {psnr(gt.cpu(), cur.cpu()):.2f} dB  |  "
              f"PSNR vs prev = {psnr(rows[-2], rows[-1]):.2f} dB")

    # height-concat: for each frame t, stack the 11 versions vertically
    stacked = torch.stack(rows, dim=0)              # (N+1, T, 3, H, W)
    grid = stacked.permute(1, 2, 0, 3, 4)           # (T, 3, N+1, H, W)
    T_, C, R, H, W = grid.shape
    grid = grid.reshape(T_, C, R * H, W)            # (T, 3, (N+1)*H, W)
    frames = (grid.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).byte().numpy()  # (T, RH, W, 3)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_tag = str(scene).replace("/", "_")[:24]
    stem = out_dir / f"{scene_tag}_roundtrip_x{args.iters}_hconcat"
    imageio.mimsave(f"{stem}.mp4", list(frames), fps=args.fps, quality=9)
    imageio.mimsave(f"{stem}.gif", list(frames), fps=args.fps)
    print(f"\n[out] rows top->bottom = GT, iter1..iter{args.iters}")
    print(f"[out] {stem}.mp4 / .gif  ({T_} frames, {R*H}x{W})")


if __name__ == "__main__":
    main()
