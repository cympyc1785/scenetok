"""Compute val metrics (FVD/FID/PSNR/SSIM/LPIPS) directly from the locally-saved
validation videos (Sampled=generated vs Original=GT) and re-log them to wandb.

The scalar metrics were dropped at training time (val_step↔_step conflict), but
the Sampled/Original video files survived on disk, so we recompute the metrics
from them. FVD/FID are set-level (no pairing). PSNR/SSIM/LPIPS are per-scene, so
we recover the generated↔GT scene pairing by a PSNR cost matrix + Hungarian
assignment (files are hash-named, scene order lost).

Logs under `relog/<panel>/<metric>` on the `relog/val_step` axis (1..N), to the
account of the WANDB_API_KEY in env. Run with the VCAI_Vid key.
"""
import argparse

import imageio.v3 as iio
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from scripts.relog_multi_val_to_wandb import build_val_sequence, WANDB_ID, WANDB_PROJECT
from src.model.metrics import Metric

PANELS = ["standard", "unseen"]


def load_set(paths):
    """Return float tensor (N, T, 3, H, W) in [0,1] from a list of mp4 paths."""
    vids = []
    for p in paths:
        v = iio.imread(p)  # (T,H,W,3) uint8
        vids.append(torch.from_numpy(np.asarray(v)).float() / 255.0)
    T = min(v.shape[0] for v in vids)
    H = min(v.shape[1] for v in vids)
    W = min(v.shape[2] for v in vids)
    vids = [v[:T, :H, :W].permute(0, 3, 1, 2) for v in vids]  # (T,3,H,W)
    return torch.stack(vids)  # (N,T,3,H,W)


def psnr_matrix(gen, gt):
    """(N,N) mean-PSNR between each gen[i] and gt[j] over frames (for pairing)."""
    n = gen.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mse = ((gen[i] - gt[j]) ** 2).mean().item()
            M[i, j] = -10 * np.log10(max(mse, 1e-10))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    vals = build_val_sequence()
    print(f"[metrics] {len(vals)} validations")

    metric = Metric()
    metric.setup()
    metric = metric.to(args.device)
    dev = args.device

    results = []  # per val: {panel: {metric: value}}
    for vi, entry in enumerate(vals, start=1):
        per_panel = {}
        for panel in PANELS:
            if panel not in entry or "Sampled" not in entry[panel] or "Original" not in entry[panel]:
                continue
            gen = load_set(entry[panel]["Sampled"]).to(dev)  # (N,T,3,H,W)
            gt = load_set(entry[panel]["Original"]).to(dev)
            N, T = gen.shape[0], gen.shape[1]
            # pair gen↔gt by PSNR (scene order lost in hash filenames)
            M = psnr_matrix(gen.cpu(), gt.cpu())
            r, c = linear_sum_assignment(-M)
            gt = gt[c]  # reorder gt to match gen
            gen_f = gen.reshape(N * T, *gen.shape[2:])  # (N*T,3,H,W)
            gt_f = gt.reshape(N * T, *gt.shape[2:])
            m = {}
            m["psnr"] = float(metric.compute_psnr(gen_f, gt_f))
            m["ssim"] = float(metric.compute_ssim(gen_f, gt_f))
            m["lpips"] = float(metric.compute_lpips(gen_f, gt_f))
            metric.reset_fid()
            m["fid"] = float(metric.compute_fid(gen_f, gt_f, num_views=T))
            metric.reset_fvd()
            fvd = metric.compute_fvd(gen_f, gt_f, num_views=T)
            m["fvd"] = float(fvd) if fvd is not None else None
            per_panel[panel] = m
            print(f"[metrics] val {vi} {panel}: " +
                  ", ".join(f"{k}={v:.3f}" if v is not None else f"{k}=NA" for k, v in m.items()))
        results.append(per_panel)

    if args.dry_run:
        print("[metrics] dry run — not logging.")
        return

    import wandb
    wandb.init(project=WANDB_PROJECT, id=WANDB_ID, resume="allow")
    wandb.define_metric("relog/val_step")
    wandb.define_metric("relog/*", step_metric="relog/val_step")
    for vi, per_panel in enumerate(results, start=1):
        payload = {"relog/val_step": vi}
        for panel, m in per_panel.items():
            for k, v in m.items():
                if v is not None:
                    payload[f"relog/{panel}/{k}"] = v
        wandb.log(payload)
        print(f"[metrics] logged val {vi}")
    wandb.finish()
    print("[metrics] done.")


if __name__ == "__main__":
    main()
