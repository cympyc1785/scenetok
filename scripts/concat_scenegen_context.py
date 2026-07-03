"""Show which CONTEXT (conditioning) frames a SceneGen render used, next to the
generated video. Loads the scene's target frames (no model), picks the same
num_cond uniform indices viser_server_scenegen uses (linspace(0, T-1, num_cond)),
and builds a side-by-side [context strip | generated] mp4 (context static, matched
to the generated frame count).

Usage:
  python scripts/concat_scenegen_context.py --scene 004e9db3337e8206 \
      --num_cond 5 --generated results/viser_generate/scenegen_re10k/<dir>/generated.mp4
"""
import argparse, os
from pathlib import Path
import numpy as np
import torch
import imageio.v3 as iio
import imageio
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
DEFAULT_RE10K_ROOT = str((REPO / "../dataset/re10k/re10k").resolve())


def load_target_frames(scene, re10k_root, eval_index):
    """Raw target RGB frames (T,3,H,W) in [0,1] for `scene` via the re10k dataset."""
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_config
    from src.dataset import get_dataset, DatasetRE10kCfg
    from src.misc.batch_utils import batch_expand
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[
            "dataset=re10k", f"dataset.root={re10k_root}",
            "dataset/view_sampler=evaluation_video", "dataset.view_sampler.max_cond_number=3",
            "+experiment=scenegen_shift12_re10k",
            "dataset.view_sampler.num_target_views=8", "dataset.view_sampler.temporal_downsample=4",
            "dataset.view_sampler.num_context_views=12",
            f"dataset.view_sampler.index_path={eval_index}",
            "dataset.precomputed_latents.context=false", "dataset.precomputed_latents.target=false",
            "wandb.activated=false",
        ])
    ds_cfg = load_typed_config(cfg.dataset, DatasetRE10kCfg)
    ds = get_dataset(ds_cfg, stage="test", step_tracker=None)
    ds.overfit_to_scene = [scene]
    b = ds[0]
    b["target"] = batch_expand(b["target"])
    return b["target"]["latent"][0].float().clamp(0, 1)   # (T,3,H,W)


def label(img_uint8, text):
    im = Image.fromarray(img_uint8)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 7 + 4, 14], fill=(0, 0, 0))
    d.text((2, 1), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--num_cond", type=int, required=True)
    ap.add_argument("--generated", required=True, help="path to generated.mp4")
    ap.add_argument("--re10k_root", default=DEFAULT_RE10K_ROOT)
    ap.add_argument("--eval_index", default="./assets/evaluation_index/re10k_c1_192.json")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gen = iio.imread(args.generated)                       # (Tg,H,W,3) uint8
    Tg, H, W, _ = gen.shape
    tgt = load_target_frames(args.scene, args.re10k_root, args.eval_index)   # (T,3,H,W)
    T = tgt.shape[0]
    cond_idx = torch.linspace(0, T - 1, max(1, args.num_cond)).long().tolist()
    print(f"[concat] scene={args.scene} target_frames={T} num_cond={args.num_cond} cond_idx={cond_idx}")

    # context strip: cond frames as thumbnails (height H), labeled with their index,
    # tiled horizontally then resized so total height == H (static left panel).
    thumbs = []
    for i in cond_idx:
        a = (tgt[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
        thumbs.append(label(a, f"ctx f{i}"))
    strip = np.concatenate(thumbs, axis=1)                # (H, num_cond*W, 3)
    # scale strip to height H (keep width proportional)
    sw = int(round(strip.shape[1] * H / strip.shape[0]))
    strip = np.asarray(Image.fromarray(strip).resize((sw, H)))
    sep = np.full((H, 6, 3), 255, np.uint8)

    frames = []
    genl = gen.copy()
    for t in range(Tg):
        g = label(genl[t], f"gen f{t}")
        frames.append(np.concatenate([strip, sep, g], axis=1))
    frames = np.stack(frames)

    out = args.out or str(Path(args.generated).with_name(f"context_concat_nc{args.num_cond}.mp4"))
    imageio.mimsave(out, frames, fps=args.fps, quality=8)
    print(f"[concat] saved → {out}  {frames.shape}")


if __name__ == "__main__":
    main()
