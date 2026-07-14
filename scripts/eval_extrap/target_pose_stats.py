"""Measure the DL3DV *training* relative-target-pose distribution and compare to
the viser move sweep scales, to decide up to which scale the move stays
in-distribution (not OOD).

Both measured as: ||target_cam_translation - ctx0_translation|| in raw DL3DV
(COLMAP) units, i.e. target translation relative to context view 0 (same anchor
the viser render uses with reference_index=0).

Training: iterate the DL3DV train dataset with the model's own bounded view
sampler (interp: va-wan-like, or extrap: g1020 target_extrap_range[10,20]),
re-center to ctx0, collect per-target-view ||t||. Report percentiles.

Viser: for each scale s in {0.2,0.5,0.7,1.0}, the scaled move target ||t|| (the
poses.pt target_c2w_edited is already rel->ctx0), max/mean over frames.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.visualize.viser_server import build_model, DEFAULT_EVAL_INDEX  # noqa
from src.dataset import get_dataset  # noqa
from src.dataset.data_module import safe_collate  # noqa
from src.misc.step_tracker import StepTracker  # noqa
from src.misc.batch_utils import preprocess_batch  # noqa
from torch.utils.data import DataLoader  # noqa


def train_target_norms(experiment, overrides, n_scenes, gpu, step=60000):
    margs = SimpleNamespace(model_experiment=experiment, model_ckpt=str(REPO / "checkpoints/va-wan_dl3dv.ckpt"),
                            model_shape="256,256", eval_index=str(DEFAULT_EVAL_INDEX),
                            infer_steps=25, cfg_scale=1.0, seed=0, device=f"cuda:{gpu}", extra_overrides=overrides)
    wrapper, _, device, _ = build_model(margs)
    dcfg = wrapper.dataset_cfg
    st = StepTracker(0)
    try:
        st.set_step(step)
    except Exception:
        try: st.step = step
        except Exception: pass
    ds = get_dataset(dcfg, "train", st, None, force_shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=safe_collate, shuffle=False)
    norms = []
    for i, b in enumerate(loader):
        if b is None:
            continue
        b = preprocess_batch(b, index=0)                       # rel -> ctx0
        t = b["target"]["extrinsics"][0, :, :3, 3]             # (V,3) already rel ctx0
        norms.extend(t.norm(dim=-1).tolist())
        if i + 1 >= n_scenes:
            break
    return np.array(norms)


def viser_scale_norms():
    sc = "a4c20f668ce179db_0624"
    moves = ["move_forward", "move_back", "move_left", "move_right"]
    out = {}
    for s in (0.2, 0.5, 0.7, 1.0):
        mx, alln = 0.0, []
        for mv in moves:
            o = torch.load(f"results/viser_generate/va-wan_dl3dv_25step/{sc}_{mv}/poses.pt",
                           map_location="cpu", weights_only=False)
            M = o["target_c2w_edited"].float()
            base = M[0, :3, 3]
            ts = base + s * (M[:, :3, 3] - base)               # scaled, rel->ctx0
            n = ts.norm(dim=-1)
            mx = max(mx, n.max().item()); alln.extend(n.tolist())
        out[s] = (float(np.mean(alln)), mx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    configs = [
        ("interp (va-wan-like)", "custom/scenetok_va-wan_dl3dv_finetuned_rawdata", []),
        ("extrap (g1020 [10,20])", "custom/scenetok_mvB1_va-wan_dl3dv_recon_ctx6_extrap", []),
    ]
    for name, exp, ov in configs:
        norms = train_target_norms(exp, ov, args.n_scenes, args.gpu)
        pct = {p: np.percentile(norms, p) for p in (50, 90, 95, 99)}
        print(f"\n=== TRAIN target ||t|| rel ctx0 — {name} (n={len(norms)} views) ===")
        print(f"  mean={norms.mean():.3f}  p50={pct[50]:.3f}  p90={pct[90]:.3f}  "
              f"p95={pct[95]:.3f}  p99={pct[99]:.3f}  max={norms.max():.3f}")

    print("\n=== VISER move ||t|| rel ctx0 per scale (mean / max over 4 moves) ===")
    vs = viser_scale_norms()
    for s, (mn, mx) in vs.items():
        print(f"  scale {s}:  mean={mn:.3f}  max={mx:.3f}")


if __name__ == "__main__":
    main()
