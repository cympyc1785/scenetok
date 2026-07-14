"""Run a SceneTok diffusion model (va-wan_dl3dv or g1020) on an extrapolation
eval index and record per-frame PSNR/SSIM/LPIPS vs GT, tagged by extrap distance Δ.

Mirrors diffusion_wrapper.test_step: GT = get_images(target), pred =
generate_batch_with_scene, both (1,V,3,H,W) aligned frame-by-frame to
batch["target"]["index"] → Δ from the meta json. Diffusion noise is seeded
(torch.manual_seed) so the S seeds are reproducible; per-frame metrics are
averaged over seeds.

Output: <out>/<tag>_<ctx>.json  = list of {scene, delta, seed, psnr, ssim, lpips}
        (+ optional pred/gt clips under <out>/clips/ when --save_clips)

NOTE: the experiment must load RAW dl3dv (computes latents on the fly) so 11K
scenes are readable. g1020 (override /dataset: dl3dv) works directly; va-wan's
published config uses latent_dl3dv (precomputed) → use its raw-data eval variant.
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.visualize.viser_server import build_model, DEFAULT_EVAL_INDEX  # noqa: E402
from src.dataset import get_dataset  # noqa: E402
from src.dataset.data_module import safe_collate  # noqa: E402
from src.misc.step_tracker import StepTracker  # noqa: E402
from src.misc.batch_utils import preprocess_batch  # noqa: E402
from src.model.diffusion import get_images  # noqa: E402


def _to_dev(o, dev):
    if torch.is_tensor(o):
        return o.to(dev)
    if isinstance(o, dict):
        return {k: _to_dev(v, dev) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_dev(v, dev) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_index", required=True, help="extrap index json (ctx16 or ctx6)")
    ap.add_argument("--meta", required=True, help="extrap meta json (Δ per frame)")
    ap.add_argument("--shape", default="256,256")
    ap.add_argument("--tag", required=True, help="output tag, e.g. va-wan / g1020")
    ap.add_argument("--ctx", default="ctx6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/extrap_eval"))
    ap.add_argument("--save_clips", action="store_true")
    ap.add_argument("--override", action="append", default=[],
                    help="extra hydra override(s) for build_model, e.g. +model.force_incorrect=true")
    ap.add_argument("--framewise", action="store_true",
                    help="sample EACH target view independently (v_t=1 → 1 latent per frame), "
                         "no temporal coupling; concat into the clip.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    meta = json.load(open(args.meta))
    torch.cuda.set_device(args.gpu)

    margs = SimpleNamespace(model_experiment=args.experiment, model_ckpt=args.ckpt,
                            model_shape=args.shape, eval_index=str(DEFAULT_EVAL_INDEX),
                            infer_steps=25, cfg_scale=1.0, seed=0, device=f"cuda:{args.gpu}",
                            extra_overrides=args.override)
    wrapper, _, device, precision = build_model(margs)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None

    dcfg = deepcopy(wrapper.dataset_cfg)                    # dataclass (DatasetDL3DVCfg)
    dcfg.stage_override = None                              # else data_stage_override='train' → 1K-10K
    dcfg.val_seen = False                                   # test/val branch + val_seen False → 11K prefix
    dcfg.evaluation_index_path = Path(args.eval_index)
    ds = get_dataset(dcfg, "test", StepTracker(0), None, force_shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=safe_collate, shuffle=False)

    ac_t = getattr(wrapper.model_cfg.autoencoders, "target")
    rows = []
    clip_dir = Path(args.out) / "clips"
    for batch in loader:
        if batch is None:
            continue
        scene = batch["scene"][0] if isinstance(batch["scene"], (list, tuple)) else batch["scene"]
        scene = str(scene)
        if scene not in meta:
            print(f"[skip] {scene[:16]} not in meta"); continue
        batch = _to_dev(batch, device)
        v_c = batch["context"]["extrinsics"].shape[1]
        gt = get_images(autoencoder=wrapper.autoencoder, inputs=batch["target"], view_type="target",
                        precomputed_latents=wrapper.dataset_cfg.precomputed_latents,
                        autoencoder_name=ac_t.name, scaling_factor=ac_t.kwargs.scaling_factor,
                        chunk_targets=getattr(wrapper.dataset_cfg.view_sampler, "chunk_targets", True))
        tgt_idx = batch["target"]["index"][0].tolist()
        for seed in seeds:
            torch.manual_seed(seed)
            b = preprocess_batch(deepcopy(batch), index=v_c // 2)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=precision, enabled=(precision != torch.float32)):
                if args.framewise:
                    # sample each target view independently (v_t=1 → 1 latent per frame)
                    Vt = b["target"]["extrinsics"].shape[1]
                    outs = []
                    for j in range(Vt):
                        bj = dict(b)
                        bj["target"] = {k: (v[:, j:j + 1] if torch.is_tensor(v) and v.dim() > 1 else v)
                                        for k, v in b["target"].items()}
                        sj, _, _ = wrapper.generate_batch_with_scene(bj, wrapper.sampler, repeat_factor=1)
                        outs.append(sj[:, :1])
                    sampled = torch.cat(outs, dim=1)
                else:
                    sampled, _, _ = wrapper.generate_batch_with_scene(b, wrapper.sampler, repeat_factor=1)
            sampled = sampled.float().clamp(0, 1)
            V = min(sampled.shape[1], gt.shape[1], len(tgt_idx))
            for i in range(V):
                d = meta[scene]["delta"].get(str(tgt_idx[i]))
                if d is None:
                    continue
                p, g = sampled[0, i], gt[0, i].float().clamp(0, 1)
                psnr = wrapper.metric.compute_psnr(p[None], g[None]).item()
                ssim = wrapper.metric.compute_ssim(p[None], g[None]).item()
                lpips = wrapper.metric.compute_lpips(p[None], g[None]).item()
                rows.append({"scene": scene, "delta": int(d), "seed": seed,
                             "psnr": psnr, "ssim": ssim, "lpips": lpips})
            print(f"[{args.tag}/{scene[:12]}] seed={seed} V={V} "
                  f"Δrange={min(meta[scene]['delta'].values())}..{max(meta[scene]['delta'].values())}")
            if args.save_clips and seed == seeds[0]:
                from src.misc.image_io import save_image_video
                cd = clip_dir / f"{args.tag}_{args.ctx}" / scene
                save_image_video(images=sampled[0], indices=torch.arange(sampled.shape[1]),
                                 output_dir=cd, name="pred", save_img=False, save_video=True, fps=8)
                gd = clip_dir / "gt" / scene
                save_image_video(images=gt[0].float().clamp(0, 1), indices=torch.arange(gt.shape[1]),
                                 output_dir=gd, name="gt", save_img=False, save_video=True, fps=8)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    outp = Path(args.out) / f"{args.tag}_{args.ctx}.json"
    json.dump(rows, open(outp, "w"), indent=1)
    print(f"[done] {len(rows)} rows → {outp}")


if __name__ == "__main__":
    main()
