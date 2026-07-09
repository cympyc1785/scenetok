"""Decode the intermediate noisy latent x_t at denoising timesteps t=0,0.1,0.3,0.5
for the SceneTok decoder (va-wan_dl3dv), over the 5 viser camera poses.

Grid: rows = poses (orig, move_forward, move_back, move_left, move_right),
      cols = denoising timesteps [0, 0.1, 0.3, 0.5]  (t=0 clean, higher = noisier x_t).
Each cell is a V-frame video = the VAE decode of x_t at that noise level.

Uses generate_batch_with_scene(capture_noise_levels=..., capture_store=...) which
decodes x_t at those noise levels inside the sampler loop (rectified flow,
x_t=(1-t)x0+t·ε). Batch setup mirrors viser_regen_from_pt.regen_one.
"""
import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import imageio.v2 as imageio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.visualize.viser_server import build_model, CKPT_PRESETS, DEFAULT_EVAL_INDEX  # noqa: E402
from scripts.visualize.viser_regen_from_pt import build_scene_cache, find_batch, _to_dev  # noqa: E402

COLS = [("orig", "orig"), ("move_forward", "move_forward"), ("move_backward", "move_back"),
        ("move_left", "move_left"), ("move_right", "move_right")]
TIMESTEPS = [0.0, 0.1, 0.3, 0.5]


@torch.no_grad()
def render_captures(wrapper, batch, tgt_rel, device, precision, timesteps):
    from src.misc.batch_utils import preprocess_batch
    b = _to_dev(batch, device)
    ctx_ext = b["context"]["extrinsics"]
    ctx_lat = b["context"]["latent"]
    ctx0_abs = ctx_ext[0, 0]
    edited_rel = torch.as_tensor(np.asarray(tgt_rel), dtype=ctx_ext.dtype, device=device)
    edited_abs = ctx0_abs.unsqueeze(0) @ edited_rel
    T = edited_abs.shape[0]
    gt_tgt_int = b["target"]["intrinsics"]
    tgt_int = gt_tgt_int[0, 0].unsqueeze(0).repeat(T, 1, 1)
    b["target"] = {
        "extrinsics": edited_abs.unsqueeze(0),
        "intrinsics": tgt_int.unsqueeze(0),
        "latent": torch.zeros((1, T, ctx_lat.shape[2], ctx_lat.shape[3], ctx_lat.shape[4]),
                              device=device, dtype=ctx_lat.dtype),
        "index": torch.arange(T, device=device).unsqueeze(0),
    }
    b = preprocess_batch(b, index=0)
    cap = {}
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=precision,
                                             enabled=(precision != torch.float32)):
        wrapper.generate_batch_with_scene(b, wrapper.sampler, repeat_factor=1,
                                          capture_noise_levels=timesteps, capture_store=cap)
    # cap[t] = (1,V,3,H,W) float [0,1]
    return {t: (cap[t][0].clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).numpy())
            for t in cap}   # {t: (V,H,W,3) uint8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=str(REPO / "results/viser_generate/va-wan_dl3dv"))
    ap.add_argument("--model_ckpt", default=str(REPO / "checkpoints/va-wan_dl3dv.ckpt"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/denoise_timesteps_va-wan_dl3dv"))
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    src = Path(args.src_dir)
    prefixes = sorted({d.name[:-len("_orig")] for d in src.glob("*_orig") if d.is_dir()})
    print(f"[dt] scenes: {prefixes}")

    stem = Path(args.model_ckpt).stem
    exp, shape = CKPT_PRESETS.get(stem, ("scenetok_va-wan_shift4_dl3dv_finetuned", "256,256"))
    margs = SimpleNamespace(model_experiment=exp, model_ckpt=args.model_ckpt, model_shape=shape,
                            eval_index=str(DEFAULT_EVAL_INDEX), infer_steps=50, cfg_scale=1.0,
                            seed=0, device=f"cuda:{args.gpu}", extra_overrides=[])
    wrapper, loader, device, precision = build_model(margs)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None
    cache = build_scene_cache(loader)
    print(f"[dt] scene cache: {len(cache)}")

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        batch = find_batch(cache, prefix.split("_")[0])
        if batch is None:
            print(f"[dt] SKIP {prefix}: not in loader"); continue
        rows = []       # one row per pose; each row = [cell(t0)|cell(t01)|cell(t03)|cell(t05)]
        for label, suffix in COLS:
            pt = src / f"{prefix}_{suffix}" / "poses.pt"
            if not pt.exists():
                print(f"[dt] SKIP row {label}: no {pt}"); rows = None; break
            obj = torch.load(pt, map_location="cpu", weights_only=False)
            cap = render_captures(wrapper, batch, obj["target_c2w_edited"], device, precision, TIMESTEPS)
            cells = [cap[t] for t in TIMESTEPS]                  # each (V,H,W,3)
            Tmin = min(c.shape[0] for c in cells)
            row = np.concatenate([c[:Tmin] for c in cells], axis=2)   # (Tmin,H,4W,3)
            rows.append(row)
            print(f"[dt] {prefix} :: {label}  cells "
                  f"{[tuple(cap[t].shape) for t in TIMESTEPS]}")
        if not rows:
            continue
        Tmin = min(r.shape[0] for r in rows)
        grid = np.concatenate([r[:Tmin] for r in rows], axis=1)  # (Tmin, 5H, 4W, 3)
        sdir = out_root / prefix; sdir.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(sdir / "grid_pose5_x_t4.mp4", list(grid), fps=15, quality=8)
        imageio.mimsave(sdir / "grid_pose5_x_t4.gif", list(grid), fps=15, loop=0)
        (sdir / "layout.txt").write_text(
            "rows=poses [orig, move_forward, move_backward, move_left, move_right]\n"
            "cols=denoising timestep x_t [t=0(clean), 0.1, 0.3, 0.5(noisier)]\n")
        print(f"[dt] wrote {sdir}/grid_pose5_x_t4.mp4  {grid.shape}")
    print(f"[dt] done -> {out_root}")


if __name__ == "__main__":
    main()
