"""Compare the CAMERA distribution the model actually sees for DL3DV vs RE10K in
the mixed config (scenetok_mvB1_dl3dv_re10k_mixed_extrap). Builds each sub-dataset
with its real view_sampler/preprocessing, samples N scenes, and reports intrinsics
(fx,fy,cx,cy + FOV) and extrinsics (inter-view baseline, trajectory spread, per-step
rotation) stats — so we can see whether merging shifts the camera distribution.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import numpy as np
import torch
from omegaconf import OmegaConf


def build_subs(experiment):
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    from src.dataset import _parse_sub_cfg, get_dataset
    from src.misc.step_tracker import StepTracker

    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="main",
                      overrides=[f"+experiment={experiment}", "mode=train", "wandb.activated=false"])
    OmegaConf.set_struct(cfg, False)
    subs = {}
    for raw in cfg.dataset.datasets:
        c = _parse_sub_cfg(raw)
        subs[c.name] = get_dataset(c, "train", StepTracker(0), None, force_shuffle=False)
    return subs


def rot_angle_deg(Ra, Rb):
    R = Ra.transpose(-1, -2) @ Rb
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def scene_stats(sample):
    out = {}
    for vt in ("context", "target"):
        K = sample[vt]["intrinsics"].numpy()          # (V,3,3)
        E = sample[vt]["extrinsics"].numpy()           # (V,4,4) c2w
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        cen = E[:, :3, 3]                               # camera centers
        # inter-view baseline (consecutive), trajectory spread
        d = np.linalg.norm(np.diff(cen, axis=0), axis=1) if len(cen) > 1 else np.array([0.0])
        spread = np.linalg.norm(cen - cen.mean(0), axis=1).mean()
        rots = [rot_angle_deg(E[i, :3, :3], E[i + 1, :3, :3]) for i in range(len(E) - 1)]
        out[vt] = dict(fx=fx.mean(), fy=fy.mean(), cx=cx.mean(), cy=cy.mean(),
                       step=d.mean(), spread=spread, rot=np.mean(rots) if rots else 0.0)
    return out


def agg(name, dataset, n):
    acc = {}
    got = 0
    idx = 0
    while got < n and idx < len(dataset):
        try:
            s = scene_stats(dataset[idx])
            for vt, d in s.items():
                for k, v in d.items():
                    acc.setdefault(f"{vt}.{k}", []).append(float(v))
            got += 1
        except Exception:
            pass
        idx += 1
    print(f"\n==== {name} (n={got}) ====")
    for key in ["context.fx", "context.fy", "context.cx", "context.cy",
                "context.step", "context.spread", "context.rot",
                "target.step", "target.spread", "target.rot"]:
        a = np.array(acc.get(key, [0]))
        print(f"  {key:16s} mean={a.mean():9.3f}  std={a.std():8.3f}  "
              f"min={a.min():8.3f}  max={a.max():8.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="custom/scenetok_mvB1_dl3dv_re10k_mixed_extrap")
    ap.add_argument("-n", type=int, default=60)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    subs = build_subs(args.experiment)
    print("sub-datasets:", list(subs.keys()))
    for name, ds in subs.items():
        agg(name, ds, args.n)


if __name__ == "__main__":
    main()
