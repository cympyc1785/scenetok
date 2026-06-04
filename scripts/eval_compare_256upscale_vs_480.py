"""Compare two SceneTok lightningDiT ckpts on DL3DV at 480x832:

  A. 480 ckpt rendered natively at 480x832.
  B. 256 ckpt rendered at 256x448, bilinear-upscaled to 480x832.

Uses the training-time validation indices (`assets/evaluation_index/
dl3dv_c16_37_{standard,unseen}.json`) and the same metric pipeline
(`Metric` in `src/model/metrics.py`) so the numbers are comparable to
what training prints. Writes per-scene CSVs and an aggregate JSON.

This script loads the wrappers from each exp's `.hydra/config.yaml` +
`last.ckpt` (no Hydra @main.run wrapper, similar to stage1 / fast_infer).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

REPO_ROOT = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.data_module import safe_collate
from src.misc.batch_utils import preprocess_batch
from src.misc.step_tracker import StepTracker
from src.model.diffusion_wrapper import DiffusionWrapper


# ─────────────────────────────────────────────────────────────────────────────
# config / loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp_480",
        default="scenetok_va-wan_dl3dv_480_finetune_large",
        help="Exp name for the 480x832 native model.",
    )
    p.add_argument(
        "--exp_256",
        default="scenetok_va-wan_dl3dv_256_finetune_large",
        help="Exp name for the 256x448 model (to upscale).",
    )
    p.add_argument("--ckpt_480", default=None, help="Override last.ckpt path for 480 model.")
    p.add_argument("--ckpt_256", default=None, help="Override last.ckpt path for 256 model.")
    p.add_argument(
        "--gt_shape",
        default="480,832",
        help="GT (and 256-output upscale target) HxW.",
    )
    p.add_argument(
        "--splits",
        default="standard,unseen",
        help="Comma-separated splits to evaluate (each maps to `dl3dv_c16_37_<split>.json`).",
    )
    p.add_argument(
        "--cases",
        default="480,256up",
        help="Which model cases to run. Subset of {480,256up}, comma-separated.",
    )
    p.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "results" / "eval_compare_256upscale_vs_480"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--skip_fvd_fid",
        action="store_true",
        help="If set, skip FID/FVD (per-frame metrics still computed).",
    )
    return p.parse_args()


def parse_shape(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def build_cfg_for_split(
    exp_name: str,
    split: str,
    *,
    override_target_shape: list[int] | None = None,
    override_context_shape: list[int] | None = None,
    num_inference_steps: int = 25,
    seed: int = 0,
):
    """Load the exp's .hydra/config.yaml and apply eval-style overrides.

    `override_target_shape` (and `override_context_shape`) override the model's
    native target/context shape — used to build a parallel "GT dataset" at
    480x832 for the 256-ckpt case. Leave None to keep the exp's training shape.
    """
    config_path = REPO_ROOT / "exp" / exp_name / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    cfg_dict = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg_dict, False)
    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.freeze.denoiser = True
    cfg_dict.freeze.compressor = True
    cfg_dict.freeze.autoencoder = True
    cfg_dict.dataset.smallset = False
    # Force the val/test eval-index branch in __getitem__.
    cfg_dict.dataset.stage_override = "val"
    cfg_dict.dataset.val_seen = (split == "standard")
    cfg_dict.dataset.evaluation_index_path = str(
        REPO_ROOT / "assets" / "evaluation_index" / f"dl3dv_c16_37_{split}.json"
    )
    cfg_dict.dataset.scene_id = None
    cfg_dict.data_loader.test.num_workers = 0
    cfg_dict.data_loader.test.batch_size = 1
    if override_context_shape is not None:
        cfg_dict.dataset.context_shape = list(override_context_shape)
    if override_target_shape is not None:
        cfg_dict.dataset.target_shape = list(override_target_shape)
    cfg_dict.model.cfg_scale = 1.0
    cfg_dict.model.scheduler.num_inference_steps = num_inference_steps
    if "noise_seed" in cfg_dict.model.denoiser:
        cfg_dict.model.denoiser.noise_seed = seed
    OmegaConf.set_struct(cfg_dict, True)
    return load_typed_root_config(cfg_dict)


def build_wrapper(cfg, ckpt_path: Path, device: str):
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
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[eval] {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    wrapper.eval().to(device)
    # Silence sampler's per-step logger calls (no Lightning logger attached).
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None
    return wrapper


def build_loader(cfg, *, step_tracker: StepTracker):
    ds = get_dataset(cfg.dataset, "test", step_tracker, generator=None, force_shuffle=False)
    return ds, DataLoader(ds, batch_size=1, num_workers=0, collate_fn=safe_collate, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# eval loop
# ─────────────────────────────────────────────────────────────────────────────


def run_case(
    *,
    case_label: str,                # "480" or "256up"
    exp_name: str,
    ckpt_path: Path,
    split: str,                     # "standard" or "unseen"
    gt_shape: list[int],            # [480, 832]
    num_inference_steps: int,
    seed: int,
    max_scenes: int | None,
    skip_fvd_fid: bool,
    out_dir: Path,
    device: str,
):
    print()
    print("=" * 72)
    print(f" CASE {case_label} | split={split} | exp={exp_name}")
    print("=" * 72)

    # cfg_pred: drive the model at its training shape. cfg_gt: a parallel cfg
    # that loads target pixels at gt_shape (used only for the 256up case).
    cfg_pred = build_cfg_for_split(
        exp_name, split,
        num_inference_steps=num_inference_steps, seed=seed,
    )
    pred_shape = list(cfg_pred.dataset.target_shape)

    needs_gt_loader = list(pred_shape) != list(gt_shape)
    if needs_gt_loader:
        cfg_gt = build_cfg_for_split(
            exp_name, split,
            override_target_shape=gt_shape,
            num_inference_steps=num_inference_steps, seed=seed,
        )
    else:
        cfg_gt = None

    wrapper = build_wrapper(cfg_pred, ckpt_path, device)

    # Loaders
    step_tracker = StepTracker(0)
    ds_pred, loader_pred = build_loader(cfg_pred, step_tracker=step_tracker)
    if needs_gt_loader:
        ds_gt, loader_gt = build_loader(cfg_gt, step_tracker=step_tracker)
        if len(ds_gt) != len(ds_pred):
            raise RuntimeError(
                f"pred/gt dataset length mismatch ({len(ds_pred)} vs {len(ds_gt)})"
            )
        loader_iter = zip(loader_pred, loader_gt)
    else:
        loader_iter = ((b, b) for b in loader_pred)

    n_total = len(ds_pred)
    if max_scenes is not None:
        n_total = min(n_total, max_scenes)
    print(f"[eval] iterating {n_total} scenes")

    # Use the wrapper's own Metric() so dtype/device match training-time path.
    metric = wrapper.metric
    metric.reset_fid()
    metric.reset_fvd()

    precision = (
        torch.bfloat16
        if cfg_pred.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
        else torch.float32
    )

    per_scene_rows: list[dict] = []
    pred_pool: list[torch.Tensor] = []  # CPU, (F=37, 3, H, W) per scene
    gt_pool: list[torch.Tensor] = []

    for i, (batch_pred, batch_gt) in enumerate(loader_iter):
        if max_scenes is not None and i >= max_scenes:
            break
        if batch_pred is None or batch_gt is None:
            print(f"[eval] skip i={i}: None batch")
            continue
        scene_name_pred = batch_pred["scene"][0]
        scene_name_gt = batch_gt["scene"][0]
        if scene_name_pred != scene_name_gt:
            raise RuntimeError(
                f"scene order mismatch at i={i}: {scene_name_pred} vs {scene_name_gt}"
            )

        batch_pred_dev = move_to_device(batch_pred, device)
        v_c = batch_pred_dev["context"]["extrinsics"].shape[1]
        batch_pred_dev = preprocess_batch(batch_pred_dev, index=v_c // 2)

        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=precision, enabled=(precision != torch.float32)
        ):
            sampled, _, _ = wrapper.generate_batch_with_scene(
                batch_pred_dev, wrapper.sampler, repeat_factor=1
            )
        # sampled: (1, F, 3, h_pred, w_pred) in [0,1]
        sampled = sampled.float().clamp(0.0, 1.0)
        b, F_v, c, h, w = sampled.shape
        if (h, w) != tuple(gt_shape):
            sampled_flat = sampled.flatten(0, 1)  # (F, 3, h, w)
            sampled_flat = F.interpolate(
                sampled_flat, size=tuple(gt_shape), mode="bilinear", align_corners=False
            )
            sampled = sampled_flat.view(b, F_v, c, gt_shape[0], gt_shape[1])

        # GT pixel video at gt_shape: batch_gt["target"]["latent"] from the
        # parallel dataset (or batch_pred for 480 case where shapes match).
        gt = batch_gt["target"]["latent"].to(device).float().clamp(0.0, 1.0)
        # gt: (1, F_gt, 3, H, W). Align frame count with sampled (Wan decode
        # might give 37 or 33 depending on view_sampler; both ckpts use
        # bounded + chunk_targets=False → same 37 frames as the JSON).
        F_gt = gt.shape[1]
        if F_gt != F_v:
            n = min(F_gt, F_v)
            gt = gt[:, :n]
            sampled = sampled[:, :n]
            F_v = n

        # NaN/Inf guard (mirror validation_step lines 953–968).
        finite = (
            torch.isfinite(sampled).reshape(b, -1).all(dim=1)
            & torch.isfinite(gt).reshape(b, -1).all(dim=1)
        )
        if not finite.all():
            print(f"[eval] {scene_name_pred}: drop non-finite batch")
            continue

        pred_flat = sampled.flatten(0, 1)            # (F, 3, H, W)
        gt_flat = gt.flatten(0, 1)

        # Per-scene metrics (same call signature as training's on_validation_end).
        with torch.no_grad():
            scene_psnr = metric.compute_psnr(pred_flat, gt_flat).item()
            scene_ssim = metric.compute_ssim(pred_flat, gt_flat).item()
            scene_lpips = metric.compute_lpips(pred_flat, gt_flat).item()

        per_scene_rows.append(
            dict(
                scene=scene_name_pred,
                case=case_label,
                split=split,
                num_frames=int(F_v),
                psnr=scene_psnr,
                ssim=scene_ssim,
                lpips=scene_lpips,
            )
        )
        # Accumulate for global pool (CPU to bound GPU memory).
        pred_pool.append(pred_flat.detach().cpu())
        gt_pool.append(gt_flat.detach().cpu())

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(
                f"  [{i+1}/{n_total}] {scene_name_pred} "
                f"psnr={scene_psnr:.3f} ssim={scene_ssim:.4f} lpips={scene_lpips:.4f}"
            )

    # ── Global metrics on the full pool (mirrors on_validation_end). ────────
    summary: dict = dict(case=case_label, split=split, n_scenes=len(per_scene_rows))
    if not per_scene_rows:
        print("[eval] no scenes processed — skipping global metrics")
    else:
        pred_cat = torch.cat(pred_pool, dim=0)        # (N*F, 3, H, W) CPU
        gt_cat = torch.cat(gt_pool, dim=0)
        # global PSNR/SSIM/LPIPS on GPU in chunks (compute_ssim already chunks).
        pred_gpu = pred_cat.to(device)
        gt_gpu = gt_cat.to(device)
        with torch.no_grad():
            summary["psnr"] = metric.compute_psnr(pred_gpu, gt_gpu).item()
            summary["ssim"] = metric.compute_ssim(pred_gpu, gt_gpu).item()
            summary["lpips"] = metric.compute_lpips(pred_gpu, gt_gpu).item()
        del pred_gpu, gt_gpu
        torch.cuda.empty_cache()

        # FID/FVD (CPU). num_views = frames per scene (37). The Metric class
        # rearranges by num_views and updates internal accumulators.
        if not skip_fvd_fid:
            F_v_scene = per_scene_rows[0]["num_frames"]
            try:
                with torch.no_grad():
                    fid_val = metric.compute_fid(
                        pred_cat, gt_cat, update=True, num_views=F_v_scene
                    )
                summary["fid"] = float(fid_val.item() if hasattr(fid_val, "item") else fid_val)
            except Exception as exc:
                print(f"[eval] FID failed: {exc}")
                summary["fid"] = None
            try:
                with torch.no_grad():
                    fvd_val = metric.compute_fvd(
                        pred_cat, gt_cat, update=True, num_views=F_v_scene
                    )
                summary["fvd"] = float(fvd_val.item() if hasattr(fvd_val, "item") else fvd_val)
            except Exception as exc:
                print(f"[eval] FVD failed: {exc}")
                summary["fvd"] = None
            metric.reset_fid()
            metric.reset_fvd()
        else:
            summary["fid"] = None
            summary["fvd"] = None
        summary["mean_psnr_per_scene"] = (
            sum(r["psnr"] for r in per_scene_rows) / len(per_scene_rows)
        )
        summary["mean_ssim_per_scene"] = (
            sum(r["ssim"] for r in per_scene_rows) / len(per_scene_rows)
        )
        summary["mean_lpips_per_scene"] = (
            sum(r["lpips"] for r in per_scene_rows) / len(per_scene_rows)
        )

    # ── Write per-scene CSV ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"per_scene_{case_label}_{split}.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["scene", "case", "split", "num_frames", "psnr", "ssim", "lpips"]
        )
        writer.writeheader()
        for row in per_scene_rows:
            writer.writerow(row)
    print(f"[eval] wrote {csv_path} ({len(per_scene_rows)} rows)")

    # Cleanup before next case.
    del wrapper
    del pred_pool, gt_pool
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_shape = parse_shape(args.gt_shape)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    assert all(c in {"480", "256up"} for c in cases), cases

    case_to_cfg = {
        "480": dict(exp=args.exp_480, ckpt=args.ckpt_480),
        "256up": dict(exp=args.exp_256, ckpt=args.ckpt_256),
    }

    all_summaries: list[dict] = []
    for case in cases:
        exp_name = case_to_cfg[case]["exp"]
        ckpt_override = case_to_cfg[case]["ckpt"]
        ckpt_path = (
            Path(ckpt_override) if ckpt_override
            else REPO_ROOT / "my_checkpoints" / exp_name / "last.ckpt"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

        for split in splits:
            summary = run_case(
                case_label=case,
                exp_name=exp_name,
                ckpt_path=ckpt_path,
                split=split,
                gt_shape=gt_shape,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                max_scenes=args.max_scenes,
                skip_fvd_fid=args.skip_fvd_fid,
                out_dir=out_dir,
                device=args.device,
            )
            all_summaries.append(summary)
            # Stream the summary so partial progress survives a crash.
            with (out_dir / "summary.json").open("w") as fp:
                json.dump(
                    dict(
                        gt_shape=gt_shape,
                        num_inference_steps=args.num_inference_steps,
                        seed=args.seed,
                        max_scenes=args.max_scenes,
                        cases=cases,
                        splits=splits,
                        results=all_summaries,
                    ),
                    fp,
                    indent=2,
                )

    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    for s in all_summaries:
        print(json.dumps(s, indent=2))
    print(f"[eval] summary.json → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
