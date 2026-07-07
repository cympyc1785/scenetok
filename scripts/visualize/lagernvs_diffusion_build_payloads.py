"""Stage 1 (scenetok env): build per-trajectory LagerNVS payloads from the DL3DV
dataset + viser poses.pt, so the LagerNVS *diffusion* decoder (separate lagernvs
env) can render them.

The original viser context bundle (`results/context_views_dl3dv_c16_37/*.pt`) is
gone, so we re-derive the scene's context (raw RGB + c2w + GT target K) straight
from the DL3DV dataset via a 1-scene eval index — the SAME context/frame the
scenetok viser_regen used (target_c2w_edited is rel→ctx0 in this dataset frame).

For each `<src>/<subdir>/poses.pt` we write `<out>/<subdir>/payload.pt` with:
  context_image_paths (PNGs), context_c2w (world c2w), target_c2w (world c2w =
  ctx0_abs @ target_c2w_edited), target_intrinsics_norm (GT). Stage 2 relativizes
  to ctx0 + scene-scale exactly like scripts/visualize/lagernvs_infer.py.

Context is emitted at the dataset's context_shape; Stage 2 square-crops/resizes to
the LagerNVS training resolution and subsamples to the training cond-view count.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def compose_dl3dv_dataset(eval_index: str, shape):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.misc.step_tracker import StepTracker

    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=["+experiment=custom/scenetok_va-wan_shift4_dl3dv_finetuned_wide",
                       "dataset=dl3dv", "mode=test", "wandb.activated=false"],
        )
    OmegaConf.set_struct(cfg_dict, False)
    for key in ("context_root", "target_root", "map_dict"):
        cfg_dict.dataset.pop(key, None)
    cfg_dict.dataset.root = "./DATA/DL3DV/DL3DV-960"
    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.dataset.smallset = False
    cfg_dict.dataset.stage_override = "train"
    cfg_dict.dataset.val_seen = True
    cfg_dict.dataset.scene_id = None
    cfg_dict.dataset.context_shape = shape
    cfg_dict.dataset.target_shape = shape
    cfg_dict.dataset.evaluation_index_path = eval_index
    cfg_dict.dataset.view_sampler = OmegaConf.create({
        "name": "evaluation", "index_path": eval_index,
        "num_context_views": 16, "num_target_views": 36})
    OmegaConf.set_struct(cfg_dict, True)
    cfg = load_typed_root_config(cfg_dict)
    ds = get_dataset(cfg.dataset, "test", StepTracker(0), generator=None, force_shuffle=False)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=str(REPO / "results/viser_generate/va-wan_dl3dv_256x448"))
    ap.add_argument("--out_dir", default=str(REPO / "results/viser_generate/_lagernvs_diffusion_payloads"))
    ap.add_argument("--eval_index", default=str(REPO / "assets/evaluation_index/dl3dv_a4c20f_only.json"))
    ap.add_argument("--shape", default="256,448")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    shape = [int(x) for x in args.shape.split(",")]
    src = Path(args.src_dir)
    subdirs = sorted([d for d in src.iterdir()
                      if d.is_dir() and d.name != "backup" and (d / "poses.pt").exists()])
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        subdirs = [d for d in subdirs if any(k in d.name for k in keys)]
    print(f"[payloads] {len(subdirs)} trajectories")

    ds = compose_dl3dv_dataset(args.eval_index, shape)
    # Build a {scene: sample} cache (one scene in the 1-scene index).
    cache = {}
    for i in range(len(ds)):
        s = ds[i]
        if s is None:
            continue
        name = s.get("scene")
        name = name[0] if isinstance(name, (list, tuple)) else name
        cache[str(name)] = s
    print(f"[payloads] dataset scenes: {list(cache)[:3]} (n={len(cache)})")

    def find(scene_hash):
        if scene_hash in cache:
            return cache[scene_hash]
        for k, v in cache.items():
            if scene_hash in k or k in scene_hash:
                return v
        return None

    ok = 0
    for d in subdirs:
        obj = torch.load(d / "poses.pt", map_location="cpu", weights_only=False)
        scene = str(obj.get("scene", ""))
        tgt_rel = torch.as_tensor(np.asarray(obj["target_c2w_edited"]), dtype=torch.float32)  # (T,4,4) rel ctx0
        s = find(scene)
        if s is None:
            print(f"[payloads] SKIP {d.name}: scene {scene[:12]} not found"); continue
        ctx_rgb = s["context"]["latent"].float()          # (Vc,3,H,W) [0,1]
        ctx_c2w = s["context"]["extrinsics"].float()       # (Vc,4,4) world c2w
        tgt_K = s["target"]["intrinsics"].float()          # (Vt,3,3) normalized
        ctx0_abs = ctx_c2w[0]
        tgt_c2w_world = ctx0_abs.unsqueeze(0) @ tgt_rel     # (T,4,4) world c2w
        K0 = tgt_K[0].unsqueeze(0).repeat(tgt_c2w_world.shape[0], 1, 1)

        out_d = Path(args.out_dir) / d.name
        img_dir = out_d / "context_imgs"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_paths = []
        for i in range(ctx_rgb.shape[0]):
            arr = (ctx_rgb[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            p = img_dir / f"ctx_{i:03d}.png"
            Image.fromarray(arr).save(p)
            img_paths.append(str(p))
        torch.save({"context_image_paths": img_paths, "context_c2w": ctx_c2w,
                    "target_c2w": tgt_c2w_world, "target_intrinsics_norm": K0,
                    "scene": scene}, out_d / "payload.pt")
        print(f"[payloads] OK {d.name}: ctx {ctx_rgb.shape[0]}v, tgt {tgt_c2w_world.shape[0]}")
        ok += 1
    print(f"[payloads] done. ok={ok} -> {args.out_dir}")


if __name__ == "__main__":
    main()
