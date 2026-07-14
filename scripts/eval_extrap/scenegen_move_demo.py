"""SceneGen (re10k) qualitative move demo — GT-less. Generates scene tokens
(stochastic) once, then renders the generated scene from orig + 4 move
trajectories (forward/back/left/right), concat into a 1x5 gif/mp4.

SceneGen ≠ our extrap NVS: it HALLUCINATES a scene (no GT), re10k-trained. This
is a standalone demo, not part of the DL3DV extrap-vs-GT comparison.

Reuses SceneGenEngine + _make_pattern_poses from viser_server_scenegen (unmodified).
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import imageio.v2 as im

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.visualize.viser_server_scenegen import SceneGenEngine, _make_pattern_poses, DEFAULT_RE10K_ROOT, DEFAULT_EVAL_INDEX  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="004e9db3337e8206")
    ap.add_argument("--amount", type=float, default=1.0, help="move magnitude (re10k baseline≈1)")
    ap.add_argument("--num_cond", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/extrap_eval/scenegen_re10k_moves"))
    args = ap.parse_args()

    eargs = SimpleNamespace(
        experiment="scenegen_shift12_re10k",
        model_ckpt=str(REPO / "checkpoints/scenegen_shift12_re10k.ckpt"),
        scenetok_ckpt=str(REPO / "checkpoints/va-videodc_re10k_scene.ckpt"),
        re10k_root=DEFAULT_RE10K_ROOT, eval_index=DEFAULT_EVAL_INDEX,
        num_cond=args.num_cond, guidance=args.guidance, infer_steps=None,
        device=f"cuda:{args.gpu}",
    )
    engine = SceneGenEngine(eargs)
    engine.load_scene(args.scene)
    engine.generate_tokens(num_cond=args.num_cond, guidance=args.guidance)

    base = engine.base_target_c2w                       # (T,4,4) rel->context[0]
    T = base.shape[0]
    patterns = [("orig", None), ("move_forward", "move_forward"), ("move_back", "move_back"),
                ("move_left", "move_left"), ("move_right", "move_right")]
    clips = []
    for label, pat in patterns:
        poses = base if pat is None else _make_pattern_poses(pat, T, args.amount, 0.0, base=base[0])
        frames = engine.render(poses)                   # (T,3,H,W) float[0,1]
        arr = (frames.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        clips.append(arr)
        print(f"[scenegen] {label}: {arr.shape}")
    Tmin = min(len(c) for c in clips)
    grid = np.concatenate([c[:Tmin] for c in clips], axis=2)   # 1x5 horizontal
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    im.mimwrite(out / f"{args.scene[:16]}_moves_1x5.mp4", list(grid), fps=args.fps, quality=8)
    im.mimsave(out / f"{args.scene[:16]}_moves_1x5.gif", list(grid), fps=args.fps, loop=0)
    print(f"[scenegen] wrote {out}/{args.scene[:16]}_moves_1x5.{{mp4,gif}}  (orig|fwd|back|left|right)")


if __name__ == "__main__":
    main()
