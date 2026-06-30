"""Offline batch re-generation of viser target-camera results from saved poses.pt.

Replays the *exact* target-camera trajectories saved by `viser_server.py`
(each result dir holds a `poses.pt` with `target_c2w_edited` (rel→ctx0) +
`source_bundle` + `scene`) through a DIFFERENT SceneTok ckpt/resolution.

Use case: `results/viser_generate/va-wan_dl3dv_256x448` was rendered with the
256x448 wide ckpt. This re-renders the SAME poses with `va-wan_dl3dv`
(256x256, shift4) so the two can be compared head-to-head.

Reuses `build_model` / `load_bundle` / `save_gif` / `CKPT_PRESETS` from
viser_server and the generate() core (GT target intrinsics — NOT the
focal-scaled context K — and preprocess `index=0` reference). Context views,
extrinsics and GT intrinsics come from the dataset loader (so they are at the
model's own resolution); only the target camera *poses* are taken from poses.pt.

Output: <out_dir>/<same_subdir_name>/{generated.mp4, generated.gif, poses.pt, meta.json}
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.visualize.viser_server import (  # noqa: E402
    build_model, load_bundle, save_gif, CKPT_PRESETS, DEFAULT_EVAL_INDEX,
)


def _to_dev(o, dev):
    if torch.is_tensor(o):
        return o.to(dev)
    if isinstance(o, dict):
        return {k: _to_dev(v, dev) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_dev(v, dev) for v in o]
    return o


def build_scene_cache(loader):
    """Iterate the eval loader once → {scene_name: batch}."""
    cache = {}
    for b in loader:
        if b is None:
            continue
        name = b["scene"][0] if isinstance(b.get("scene"), (list, tuple)) else b.get("scene")
        if name is None:
            continue
        cache[str(name)] = b
    return cache


def find_batch(cache, scene_hash):
    if scene_hash in cache:
        return cache[scene_hash]
    for name, b in cache.items():
        if scene_hash in name:
            return b
    return None


def regen_one(wrapper, batch, tgt_rel, device, precision, out_dir, fps, infer_steps, cfg_scale, seed):
    """Mirror viser_server.generate() core (lines ~574-628)."""
    from src.misc.batch_utils import preprocess_batch
    from src.misc.image_io import save_image_video

    b = _to_dev(batch, device)
    ctx_ext = b["context"]["extrinsics"]            # (1,V,4,4) world
    ctx_lat = b["context"]["latent"]
    ctx0_abs = ctx_ext[0, 0]                         # dataset ctx0 == bundle ctx0
    edited_rel = tgt_rel.to(dtype=ctx_ext.dtype, device=device)   # (T,4,4) rel→ctx0
    edited_abs = ctx0_abs.unsqueeze(0) @ edited_rel  # back to world; preprocess re-relativizes
    T = edited_abs.shape[0]

    # GT target intrinsics (NOT context K: config scales context focal ~15x via
    # scale_context_focal_by_256). DL3DV = one K per scene → repeat [0].
    gt_tgt_int = b["target"]["intrinsics"]
    tgt_int = gt_tgt_int[0, 0].unsqueeze(0).repeat(T, 1, 1).to(gt_tgt_int.dtype)
    b["target"] = {
        "extrinsics": edited_abs.unsqueeze(0),
        "intrinsics": tgt_int.unsqueeze(0),
        "latent": torch.zeros((1, T, ctx_lat.shape[2], ctx_lat.shape[3], ctx_lat.shape[4]),
                              device=device, dtype=ctx_lat.dtype),
        "index": torch.arange(T, device=device).unsqueeze(0),
    }
    # Reference frame = FIRST context view (index 0): target poses are defined
    # identity-at-context[0], so index 0 keeps a pattern's identity start truly
    # identity (matches viser_server).
    b = preprocess_batch(b, index=0)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=precision,
                                             enabled=(precision != torch.float32)):
        sampled, _, _ = wrapper.generate_batch_with_scene(b, wrapper.sampler, repeat_factor=1)
    sampled = sampled.float().clamp(0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_image_video(images=sampled[0], indices=torch.arange(sampled.shape[1]),
                     output_dir=out_dir, name="generated", save_img=False,
                     save_video=True, fps=fps)
    try:
        save_gif(sampled[0], out_dir / "generated.gif", fps=fps)
    except Exception as ge:
        print("  gif save failed:", ge)
    return tuple(sampled.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=str(REPO / "results/viser_generate/va-wan_dl3dv_256x448"),
                    help="dir of per-pose subdirs, each holding a poses.pt")
    ap.add_argument("--out_dir", default=str(REPO / "results/viser_generate/va-wan_dl3dv"),
                    help="output root; results saved under <out_dir>/<subdir_name>/")
    ap.add_argument("--model_ckpt", default=str(REPO / "checkpoints/va-wan_dl3dv.ckpt"))
    ap.add_argument("--model_experiment", default=None)
    ap.add_argument("--model_shape", default=None)
    ap.add_argument("--eval_index", default=str(DEFAULT_EVAL_INDEX))
    ap.add_argument("--infer_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--num_target_views", type=int, default=None,
                    help="override view_sampler.num_target_views (va-vdc needs 8; va-wan leaves None)")
    ap.add_argument("--only", default=None, help="comma-sep substring filter of subdir names")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip subdirs whose out generated.mp4 already exists")
    args = ap.parse_args()

    stem = Path(args.model_ckpt).stem
    exp_d, shape_d = CKPT_PRESETS.get(stem, ("scenetok_va-wan_shift4_dl3dv_finetuned", "256,256"))
    if args.model_experiment is None:
        args.model_experiment = exp_d
    if args.model_shape is None:
        args.model_shape = shape_d
    print(f"[regen] model: {stem} → experiment={args.model_experiment} shape={args.model_shape}")

    src = Path(args.src_dir)
    subdirs = sorted([d for d in src.iterdir()
                      if d.is_dir() and d.name != "backup" and (d / "poses.pt").exists()])
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        subdirs = [d for d in subdirs if any(k in d.name for k in keys)]
    print(f"[regen] {len(subdirs)} pose dirs from {src}")
    if not subdirs:
        print("[regen] nothing to do"); return

    margs = SimpleNamespace(
        model_experiment=args.model_experiment, model_ckpt=args.model_ckpt,
        model_shape=args.model_shape, eval_index=args.eval_index,
        infer_steps=args.infer_steps, cfg_scale=args.cfg_scale, seed=args.seed,
        device=args.device,
    )
    wrapper, loader, device, precision = build_model(margs)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None
    # Some experiments (e.g. va-vdc video models) leave view_sampler.num_target_views
    # unset (=0) in the config → the autoregressive sampler's `concurrency` is 0 and
    # `clean_targets >= concurrency` errors. Allow overriding it (va-vdc uses 8).
    if args.num_target_views:
        try:
            from omegaconf import OmegaConf
            OmegaConf.set_struct(wrapper.dataset_cfg.view_sampler, False)
        except Exception:
            pass
        wrapper.dataset_cfg.view_sampler.num_target_views = args.num_target_views
        print(f"[regen] set view_sampler.num_target_views = {args.num_target_views}")
    cache = build_scene_cache(loader)
    print(f"[regen] scene cache: {len(cache)} scenes")

    ok, fail = 0, 0
    for d in subdirs:
        out_d = Path(args.out_dir) / d.name
        if args.skip_existing and (out_d / "generated.mp4").exists():
            print(f"[regen] skip (exists): {d.name}"); continue
        try:
            obj = torch.load(d / "poses.pt", map_location="cpu", weights_only=False)
            tgt = obj.get("target_c2w_edited")
            scene_hash = str(obj.get("scene", ""))
            if tgt is None or not scene_hash:
                print(f"[regen] SKIP {d.name}: missing target_c2w_edited/scene"); fail += 1; continue
            tgt = torch.as_tensor(np.asarray(tgt), dtype=torch.float32)
            batch = find_batch(cache, scene_hash)
            if batch is None:
                print(f"[regen] SKIP {d.name}: scene {scene_hash[:12]} not in eval loader"); fail += 1; continue
            t0 = time.time()
            shape = regen_one(wrapper, batch, tgt, device, precision, out_d,
                              args.fps, args.infer_steps, args.cfg_scale, args.seed)
            # save inputs/config alongside the result (CLAUDE.md workflow)
            torch.save({"target_c2w_edited": tgt, "scene": scene_hash,
                        "source_bundle": obj.get("source_bundle")}, out_d / "poses.pt")
            (out_d / "meta.json").write_text(json.dumps({
                "model_ckpt": str(args.model_ckpt), "experiment": args.model_experiment,
                "shape": args.model_shape, "infer_steps": args.infer_steps,
                "cfg_scale": args.cfg_scale, "seed": args.seed, "fps": args.fps,
                "src": str(d), "scene": scene_hash, "out_shape": list(shape),
                "intrinsics": "GT target K (scale_context_focal_by_256 aware)",
                "reference_index": 0,
            }, indent=2))
            print(f"[regen] OK {d.name} → {shape} ({time.time()-t0:.1f}s)")
            ok += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[regen] FAIL {d.name}: {e}"); fail += 1

    print(f"[regen] done. ok={ok} fail={fail} → {args.out_dir}")


if __name__ == "__main__":
    main()
