"""Phase 2 (scenetok env): compute PSNR/SSIM/LPIPS/FVD/FID per (resolution x
context-count) combo from the pred.pt/gt.pt saved by eval_lagernvs_dl3dv_render.py.

Per combo: stack all scenes' pred/gt (uint8 → float[0,1]), flatten to (N*T,3,H,W),
and call the project Metric (set-level FVD/FID, per-frame PSNR/SSIM/LPIPS), mirroring
scripts/compute_val_metrics_from_videos.py. Writes metrics.json + a markdown table.
"""
import argparse
import glob
import json
from pathlib import Path

import torch


def load_combo(combo_dir):
    preds, gts = [], []
    for scd in sorted(glob.glob(f"{combo_dir}/*/")):
        p, g = Path(scd) / "pred.pt", Path(scd) / "gt.pt"
        if p.exists() and g.exists():
            preds.append(torch.load(p, map_location="cpu").float() / 255.0)  # (T,3,H,W)
            gts.append(torch.load(g, map_location="cpu").float() / 255.0)
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_lagernvs_dl3dv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(".").resolve()))
    from src.model.metrics import Metric
    metric = Metric().to(args.device)

    combos = sorted([Path(p).name for p in glob.glob(f"{args.root}/*_ctx*") if Path(p).is_dir()])
    results = {}
    for combo in combos:
        preds, gts = load_combo(f"{args.root}/{combo}")
        if not preds:
            print(f"[metric] {combo}: no data"); continue
        T = preds[0].shape[0]
        n = len(preds)
        pred = torch.stack(preds).to(args.device)          # (N,T,3,H,W)
        gt = torch.stack(gts).to(args.device)
        pf = pred.reshape(-1, *pred.shape[2:])              # (N*T,3,H,W)
        gf = gt.reshape(-1, *gt.shape[2:])
        m = {"scenes": n, "T": T, "res": combo.split("_ctx")[0], "ctx": int(combo.split("_ctx")[1])}
        m["psnr"] = float(metric.compute_psnr(pf, gf))
        m["ssim"] = float(metric.compute_ssim(pf, gf))
        m["lpips"] = float(metric.compute_lpips(pf, gf))
        metric.reset_fid(); m["fid"] = float(metric.compute_fid(pf, gf, num_views=T))
        metric.reset_fvd(); fvd = metric.compute_fvd(pf, gf, num_views=T)
        m["fvd"] = float(fvd) if fvd is not None else None
        results[combo] = m
        print(f"[metric] {combo}: PSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f} "
              f"LPIPS {m['lpips']:.3f} FID {m['fid']:.1f} FVD {m['fvd']}", flush=True)
        del pred, gt, pf, gf
        torch.cuda.empty_cache()

    out = Path(args.root) / "metrics.json"
    json.dump(results, open(out, "w"), indent=1)

    # markdown table (rows = context, cols = resolution)
    res_order = ["256x256", "256x448", "512x512", "480x832"]
    ctxs = sorted({v["ctx"] for v in results.values()})
    lines = ["", "| metric | ctx | " + " | ".join(res_order) + " |",
             "|---|---|" + "---|" * len(res_order)]
    for met in ["psnr", "ssim", "lpips", "fid", "fvd"]:
        for c in ctxs:
            row = [f"| {met} | {c} |"]
            for r in res_order:
                v = results.get(f"{r}_ctx{c}", {}).get(met)
                row.append(f" {v:.3f} |" if isinstance(v, float) else " - |")
            lines.append("".join(row))
    table = "\n".join(lines)
    (Path(args.root) / "metrics_table.md").write_text(table)
    print(table)
    print(f"\n[metric] saved {out} + metrics_table.md")


if __name__ == "__main__":
    main()
