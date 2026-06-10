"""Sweep context-view count on the published `checkpoints/va-wan_dl3dv.ckpt`.

For each value of `--num_context_views_list` (default 1,2,3,4,8,12,16) we slice
the first N entries from the standard eval index's 16-view context list,
render the target video with that subset, and log per-scene metrics + the
rendered mp4. Uses 256x256 (the published va-wan_dl3dv.ckpt training shape)
and the standard `scenetok_va-wan_shift4_dl3dv_finetuned` experiment composed
on the fly (no exp/.hydra dir needed).

Outputs under `--out_dir`:
  per_scene.csv                       — scene, N, psnr, ssim, lpips, num_frames
  summary.csv                         — N, mean/std of psnr/ssim/lpips
  <scene>/gt.mp4                      — ground-truth target video
  <scene>/n<N>.mp4                    — model output for context size N
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

REPO_ROOT = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.data_module import safe_collate
from src.misc.batch_utils import preprocess_batch
from src.misc.image_io import save_image_video
from src.misc.step_tracker import StepTracker
from src.model.diffusion_wrapper import DiffusionWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenetok_experiment",
        default="scenetok_va-wan_shift4_dl3dv_finetuned",
        help="Published config path under config/experiment/.",
    )
    p.add_argument(
        "--scenetok_ckpt",
        default=str(REPO_ROOT / "checkpoints" / "va-wan_dl3dv.ckpt"),
    )
    p.add_argument(
        "--evaluation_index_path",
        default=str(REPO_ROOT / "assets/evaluation_index/dl3dv_c16_37_standard.json"),
    )
    p.add_argument(
        "--num_context_views_list",
        default="1,2,3,4,8,12,16",
        help="Comma-separated list of N values to sweep.",
    )
    p.add_argument(
        "--context_sampling",
        choices=["first", "uniform"],
        default="first",
        help="How to pick N context views from the eval index's 16-view list. "
             "`first` = take the first N (default; ends clustered near scene start). "
             "`uniform` = evenly spread across all 16 (N=2 → [0, 15], N=4 → "
             "[0, 5, 10, 15], etc.) for fair coverage of the scene.",
    )
    p.add_argument("--num_scenes", type=int, default=8)
    p.add_argument("--context_shape", default="256,256")
    p.add_argument("--target_shape", default="256,256")
    p.add_argument("--num_target_views", type=int, default=10)
    p.add_argument("--scenetok_inference_steps", type=int, default=25)
    p.add_argument("--scenetok_cfg_scale", type=float, default=1.0)
    p.add_argument("--scenetok_seed", type=int, default=0)
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "results" / "eval_context_view_count_va-wan_dl3dv"),
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_shape(s):
    return [int(x) for x in s.split(",")]


def parse_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def slice_context(batch_dev: dict, n: int, mode: str = "first") -> dict:
    """Return a shallow copy of batch with context views sliced to n views.

    `mode`:
        first   — take the first n (clustered near scene start).
        uniform — `np.linspace(0, V-1, n)` rounded to int (spread across the
                  full 16-view eval index).
    """
    import numpy as np
    V = batch_dev["context"]["extrinsics"].shape[1]
    if mode == "first":
        idx = torch.arange(n)
    elif mode == "uniform":
        if n == 1:
            picks = [0]
        else:
            picks = np.linspace(0, V - 1, n).round().astype(int).tolist()
            picks = sorted(set(picks))
            # if rounding produced duplicates (e.g. n > V), fall back to first-n
            if len(picks) < n:
                picks = list(range(n))
        idx = torch.tensor(picks, dtype=torch.long)
    else:
        raise ValueError(f"unknown context_sampling mode: {mode}")
    device = batch_dev["context"]["extrinsics"].device
    idx = idx.to(device)
    out = {k: v for k, v in batch_dev.items()}
    out["context"] = {
        k: (v.index_select(1, idx) if torch.is_tensor(v) and v.ndim >= 2 else v)
        for k, v in batch_dev["context"].items()
    }
    return out


def build_wrapper_and_loader(args):
    from hydra import compose, initialize_config_dir
    print(f"[eval] hydra-compose: experiment={args.scenetok_experiment}")
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                f"+experiment={args.scenetok_experiment}",
                "dataset=dl3dv",
                "mode=test",
                "wandb.activated=false",
                "~dataset.context_root",
                "~dataset.target_root",
                "~dataset.map_dict",
                "dataset.root=./DATA/DL3DV/DL3DV-960",
            ],
        )
    OmegaConf.set_struct(cfg_dict, False)
    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.data_loader.test.num_workers = 0
    cfg_dict.data_loader.test.batch_size = 1
    cfg_dict.freeze.denoiser = True
    cfg_dict.freeze.compressor = True
    cfg_dict.freeze.autoencoder = True
    cfg_dict.dataset.smallset = False
    cfg_dict.dataset.stage_override = "train"
    cfg_dict.dataset.val_seen = True
    cfg_dict.dataset.evaluation_index_path = args.evaluation_index_path
    cfg_dict.dataset.scene_id = None
    cfg_dict.dataset.context_shape = parse_shape(args.context_shape)
    cfg_dict.dataset.target_shape = parse_shape(args.target_shape)
    # Lock the loader to the eval index's full 16 context views; we slice in-batch.
    cfg_dict.dataset.view_sampler.num_context_views = 16
    cfg_dict.dataset.view_sampler.num_target_views = args.num_target_views
    cfg_dict.model.cfg_scale = args.scenetok_cfg_scale
    cfg_dict.model.scheduler.num_inference_steps = args.scenetok_inference_steps
    if "noise_seed" in cfg_dict.model.denoiser:
        cfg_dict.model.denoiser.noise_seed = args.scenetok_seed
    OmegaConf.set_struct(cfg_dict, True)
    cfg = load_typed_root_config(cfg_dict)

    step_tracker = StepTracker(0)
    wrapper = DiffusionWrapper(
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        freeze_cfg=cfg.freeze,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        val_cfg=cfg.val,
        sampler_cfg=cfg.sampler,
        step_tracker=step_tracker,
        output_dir=None,
        batch_size=1,
        val_check_interval=cfg.trainer.val_check_interval,
        mode="test",
    )
    state = torch.load(Path(args.scenetok_ckpt), map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[eval] state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    wrapper.eval().to(args.device)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None

    ds = get_dataset(cfg.dataset, "test", step_tracker, generator=None, force_shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=safe_collate, shuffle=False)
    return wrapper, loader, cfg


def main():
    args = parse_args()
    torch.manual_seed(args.scenetok_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_list = parse_ints(args.num_context_views_list)
    print(f"[eval] sweeping N = {n_list}")
    print(f"[eval] num_scenes = {args.num_scenes}")

    wrapper, loader, cfg = build_wrapper_and_loader(args)
    precision = (
        torch.bfloat16
        if cfg.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
        else torch.float32
    )
    metric = wrapper.metric

    # ── Pull and cache first num_scenes batches; save GT video per scene ────
    cached = []  # list of (scene_name, batch_dev, gt_video)
    print(f"[eval] caching first {args.num_scenes} scenes...")
    for i, batch in enumerate(loader):
        if i >= args.num_scenes:
            break
        if batch is None:
            print(f"[eval] skip i={i}: None batch")
            continue
        scene_name = batch["scene"][0]
        batch_dev = move_to_device(batch, args.device)
        v_c = batch_dev["context"]["extrinsics"].shape[1]
        batch_dev = preprocess_batch(batch_dev, index=v_c // 2)
        # GT is the raw target latent (precomputed_latents.target=False for
        # this published config → it's actual pixels in [0,1]).
        gt = batch_dev["target"]["latent"].float().clamp(0, 1)
        scene_dir = out_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        save_image_video(
            images=gt[0],
            indices=torch.arange(0, gt.shape[1]),
            output_dir=scene_dir,
            name="gt",
            save_img=False,
            save_video=True,
            fps=args.fps,
        )
        cached.append((scene_name, batch_dev, gt))
        print(f"[eval] cached {i+1}/{args.num_scenes}: {scene_name}")

    if len(cached) < args.num_scenes:
        print(f"[eval] WARNING: only cached {len(cached)} scenes (asked for {args.num_scenes})")

    # ── Per-scene CSV header ────────────────────────────────────────────────
    per_scene_path = out_dir / "per_scene.csv"
    with per_scene_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["scene", "N", "psnr", "ssim", "lpips", "num_frames"]
        )
        writer.writeheader()

    # ── Sweep N × scenes ────────────────────────────────────────────────────
    rows = []
    for n in n_list:
        print(f"\n{'='*72}\n[N={n}] sweeping {len(cached)} scenes\n{'='*72}")
        for scene_name, batch_dev, gt in cached:
            batch_n = slice_context(batch_dev, n, mode=args.context_sampling)
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=precision, enabled=(precision != torch.float32)
            ):
                sampled, _, _ = wrapper.generate_batch_with_scene(
                    batch_n, wrapper.sampler, repeat_factor=1
                )
            sampled = sampled.float().clamp(0, 1)
            b, F_v, c, h, w = sampled.shape

            # Frame alignment: gt may have more/fewer frames depending on
            # autoencoder temporal layout. Trim to common length.
            F_gt = gt.shape[1]
            if F_gt != F_v:
                F_use = min(F_gt, F_v)
                gt_use = gt[:, :F_use]
                sampled_use = sampled[:, :F_use]
            else:
                gt_use = gt
                sampled_use = sampled

            # Finite guard.
            finite = (
                torch.isfinite(sampled_use).reshape(b, -1).all(dim=1)
                & torch.isfinite(gt_use).reshape(b, -1).all(dim=1)
            )
            if not finite.all():
                print(f"  [{scene_name} N={n}] non-finite — skip")
                continue

            pred_flat = sampled_use.flatten(0, 1)
            gt_flat = gt_use.flatten(0, 1)
            with torch.no_grad():
                psnr = metric.compute_psnr(pred_flat, gt_flat).item()
                ssim = metric.compute_ssim(pred_flat, gt_flat).item()
                lpips = metric.compute_lpips(pred_flat, gt_flat).item()

            scene_dir = out_dir / scene_name
            save_image_video(
                images=sampled_use[0],
                indices=torch.arange(0, sampled_use.shape[1]),
                output_dir=scene_dir,
                name=f"n{n}",
                save_img=False,
                save_video=True,
                fps=args.fps,
            )
            row = dict(
                scene=scene_name,
                N=n,
                psnr=psnr,
                ssim=ssim,
                lpips=lpips,
                num_frames=int(sampled_use.shape[1]),
            )
            rows.append(row)
            with per_scene_path.open("a", newline="") as fp:
                writer = csv.DictWriter(
                    fp, fieldnames=["scene", "N", "psnr", "ssim", "lpips", "num_frames"]
                )
                writer.writerow(row)
            print(f"  [{scene_name[:16]}... N={n}] psnr={psnr:.3f} ssim={ssim:.4f} lpips={lpips:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

    # ── Summary CSV (mean/std per N) ────────────────────────────────────────
    import statistics
    summary_path = out_dir / "summary.csv"
    summary_rows = []
    for n in n_list:
        sub = [r for r in rows if r["N"] == n]
        if not sub:
            continue
        psnrs = [r["psnr"] for r in sub]
        ssims = [r["ssim"] for r in sub]
        lpipses = [r["lpips"] for r in sub]
        summary_rows.append(dict(
            N=n,
            n_scenes=len(sub),
            psnr_mean=statistics.mean(psnrs),
            psnr_std=statistics.stdev(psnrs) if len(psnrs) > 1 else 0.0,
            ssim_mean=statistics.mean(ssims),
            ssim_std=statistics.stdev(ssims) if len(ssims) > 1 else 0.0,
            lpips_mean=statistics.mean(lpipses),
            lpips_std=statistics.stdev(lpipses) if len(lpipses) > 1 else 0.0,
        ))
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    for r in summary_rows:
        print(f"  N={r['N']:2d}  n={r['n_scenes']:2d}  "
              f"PSNR {r['psnr_mean']:.3f}±{r['psnr_std']:.3f}  "
              f"SSIM {r['ssim_mean']:.4f}±{r['ssim_std']:.4f}  "
              f"LPIPS {r['lpips_mean']:.4f}±{r['lpips_std']:.4f}")
    print(f"\n[eval] per_scene → {per_scene_path}")
    print(f"[eval] summary   → {summary_path}")


if __name__ == "__main__":
    main()
