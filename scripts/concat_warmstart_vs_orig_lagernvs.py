"""Compare our SceneTok-enc + LagerNVS-dec (scenetok_enc_lagernvs_dec_warmstart)
vs the ORIGINAL LagerNVS (lagernvs_dl3dv_2-6_v_256) on the SAME viser poses.pt
move trajectories.

Rows:  warmstart (top)     = rendered here (mv_B1 compressor -> scene tokens -> LagerNVS renderer)
       orig-lagernvs (bot) = pre-rendered render.gif from lagernvs_regen_from_pt.py
Cols:  orig, move_forward, move_backward, move_left, move_right (each combo's poses.pt)

warmstart render reuses lagernvs_render() from concat_scenetok_vs_lagernvs_dec_moves
(preprocess index=0, scene_scale=1.35*max ctx‖t‖, GT target K[0], Plücker rays).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import shutil
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

from scripts.concat_scenetok_vs_lagernvs_dec_moves import lagernvs_render, resize_clip
from scripts.visualize.viser_regen_from_pt import build_scene_cache, find_batch
from scripts.visualize.viser_server import build_model, DEFAULT_EVAL_INDEX

COLS = [("orig", "orig"), ("move_forward", "move_forward"), ("move_backward", "move_back"),
        ("move_left", "move_left"), ("move_right", "move_right")]


def read_gif(p):
    return np.stack([np.asarray(f)[..., :3] for f in imageio.mimread(str(p), memtest=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=str(REPO / "results/viser_generate/va-wan_dl3dv"),
                    help="dir of <prefix>_<suffix>/poses.pt (viser trajectories)")
    ap.add_argument("--orig_lnvs_dir",
                    default=str(REPO / "results/viser_generate/lagernvs_dl3dv_2-6_v_256_cmp_256x256"),
                    help="pre-rendered original-lagernvs <prefix>_<suffix>/render.gif")
    ap.add_argument("--warmstart_ckpt",
                    default=str(REPO / "my_checkpoints/scenetok_enc_lagernvs_dec_warmstart/last.ckpt"))
    ap.add_argument("--experiment", default="custom/scenetok_enc_lagernvs_dec_warmstart")
    ap.add_argument("--shape", default="256,256")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cell_h", type=int, default=256)
    ap.add_argument("--cell_w", type=int, default=256)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "results/cmp_warmstart_vs_orig_lagernvs"))
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    src = Path(args.src_dir)
    orig_dir = Path(args.orig_lnvs_dir)
    prefixes = sorted({d.name[:-len("_orig")] for d in src.glob("*_orig") if d.is_dir()})
    print(f"[cmp2] scenes: {prefixes}")

    tmp_ckpt = Path("/tmp") / f"warmstart_snap_{args.gpu}.ckpt"
    shutil.copy(args.warmstart_ckpt, tmp_ckpt)
    margs = SimpleNamespace(model_experiment=args.experiment, model_ckpt=str(tmp_ckpt),
                            model_shape=args.shape, eval_index=str(DEFAULT_EVAL_INDEX),
                            infer_steps=1, cfg_scale=1.0, seed=0, device=f"cuda:{args.gpu}",
                            extra_overrides=[])
    wrapper, loader, device, precision = build_model(margs)
    cache = build_scene_cache(loader)
    print(f"[cmp2] scene cache: {len(cache)}")

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    cell = (args.cell_h, args.cell_w)

    for prefix in prefixes:
        batch = find_batch(cache, prefix.split("_")[0])
        if batch is None:
            print(f"[cmp2] SKIP {prefix}: not in eval loader"); continue
        top, bottom = [], []
        for label, suffix in COLS:
            pt = src / f"{prefix}_{suffix}" / "poses.pt"
            gif = orig_dir / f"{prefix}_{suffix}" / "render.gif"
            if not pt.exists() or not gif.exists():
                print(f"[cmp2] SKIP col {label}: missing {pt if not pt.exists() else gif}"); top = None; break
            obj = torch.load(pt, map_location="cpu", weights_only=False)
            ws = resize_clip(lagernvs_render(wrapper, batch, obj["target_c2w_edited"], device, precision), cell)
            og = resize_clip(read_gif(gif), cell)
            T = min(len(ws), len(og)); ws, og = ws[:T], og[:T]
            top.append(ws); bottom.append(og)
            print(f"[cmp2] {prefix} :: {label}  warmstart{ws.shape} orig{og.shape}")
        if not top:
            continue
        grid = np.concatenate([np.concatenate(top, axis=2), np.concatenate(bottom, axis=2)], axis=1)
        sdir = out_root / prefix; sdir.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(sdir / "grid_2x5.mp4", list(grid), fps=args.fps, quality=8)
        print(f"[cmp2] wrote {sdir}/grid_2x5.mp4  {grid.shape}  (top=warmstart, bottom=orig-lagernvs)")
    print(f"[cmp2] done -> {out_root}")


if __name__ == "__main__":
    main()
