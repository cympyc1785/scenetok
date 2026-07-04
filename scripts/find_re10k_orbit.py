"""Scan re10k eval scenes and rank by how ORBIT-like their target trajectory is
(cameras looking inward toward a convergence point in front = orbit/arc, vs
parallel forward = walkthrough). No image decode — cameras only.

orbit metrics (on target c2w, OpenCV, forward=col2):
  look_in : mean fwd·normalize(center-pos)   (+1 = look inward = orbit)
  fpar    : mean pairwise cos of forward vecs (1 = parallel = forward walk)
  pca2    : 2nd principal-axis fraction of positions (higher = arc/2D spread)
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
DEFAULT_RE10K_ROOT = str((REPO / "../dataset/re10k/re10k").resolve())


def orbit_metrics(ext):
    c = ext.numpy().astype(np.float64)
    pos = c[:, :3, 3]; fwd = c[:, :3, 2]
    fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    C = fwd @ fwd.T; iu = np.triu_indices(len(fwd), 1)
    fpar = float(C[iu].mean())
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, dv in zip(pos, fwd):
        P = np.eye(3) - np.outer(dv, dv); A += P; b += P @ o
    try:
        ctr = np.linalg.solve(A + 1e-6 * np.eye(3), b)
    except Exception:
        ctr = pos.mean(0)
    to_c = ctr - pos; nn = np.linalg.norm(to_c, axis=1, keepdims=True)
    look_in = float((fwd * (to_c / np.clip(nn, 1e-8, None))).sum(1).mean())
    pc = pos - pos.mean(0); ev = np.linalg.svd(pc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    span = float(np.linalg.norm(pos[:, None] - pos[None], axis=-1).max())
    return dict(look_in=look_in, fpar=fpar, pca2=float(ev[1]), span=span, n=len(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re10k_root", default=DEFAULT_RE10K_ROOT)
    ap.add_argument("--eval_index", default="./assets/evaluation_index/re10k_c1_192.json")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from src.config import load_typed_config
    from src.dataset import get_dataset, DatasetRE10kCfg
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[
            "dataset=re10k", f"dataset.root={args.re10k_root}",
            "dataset/view_sampler=evaluation_video", "dataset.view_sampler.max_cond_number=3",
            "+experiment=scenegen_shift12_re10k",
            "dataset.view_sampler.num_target_views=8", "dataset.view_sampler.temporal_downsample=4",
            "dataset.view_sampler.num_context_views=12",
            f"dataset.view_sampler.index_path={args.eval_index}",
            "dataset.precomputed_latents.context=false", "dataset.precomputed_latents.target=false",
            "wandb.activated=false",
        ])
    ds_cfg = load_typed_config(cfg.dataset, DatasetRE10kCfg)
    ds = get_dataset(ds_cfg, stage="test", step_tracker=None)
    index = ds.view_sampler.index

    # group scenes by chunk → load each chunk once
    by_chunk = {}
    for scene, ch in zip(ds.scenes, ds.chunks):
        by_chunk.setdefault(str(ch), []).append(scene)

    rows = []
    for ci, (ch, scenes) in enumerate(by_chunk.items()):
        chunk = torch.load(ch, weights_only=True)
        lut = {x["key"]: x for x in chunk}
        for scene in scenes:
            ex = lut.get(scene)
            if ex is None:
                continue
            ext, _ = ds.convert_poses(ex["cameras"])
            entry = index[scene]
            entry = entry[0] if isinstance(entry, (list, tuple)) else entry
            tgt = entry.target
            tgt = [int(t) for t in tgt if 0 <= int(t) < ext.shape[0]]
            if len(tgt) < 4:
                continue
            m = orbit_metrics(ext[torch.tensor(tgt)])
            rows.append((scene, m))
        del chunk, lut
        print(f"  scanned chunk {ci+1}/{len(by_chunk)} ({len(rows)} scenes)", flush=True)

    # orbit-ness: look inward (look_in high) + arc spread (pca2) + less parallel
    rows.sort(key=lambda r: (r[1]["look_in"], r[1]["pca2"]), reverse=True)
    print(f"\n=== scanned {len(rows)} scenes ===")
    print(f"{'scene':18s} {'look_in':>8s} {'fpar':>7s} {'pca2':>6s} {'span':>7s} {'n':>4s}")
    print("--- TOP orbit-like (look inward) ---")
    for s, m in rows[:args.top]:
        print(f"{s:18s} {m['look_in']:+8.3f} {m['fpar']:7.3f} {m['pca2']:6.3f} {m['span']:7.2f} {m['n']:4d}")
    print("--- most forward (walkthrough) ---")
    for s, m in rows[-5:]:
        print(f"{s:18s} {m['look_in']:+8.3f} {m['fpar']:7.3f} {m['pca2']:6.3f} {m['span']:7.2f} {m['n']:4d}")
    out = REPO / "results/re10k_orbit_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump([{"scene": s, **m} for s, m in rows], open(out, "w"), indent=1)
    print(f"\nsaved ranking → {out}")


if __name__ == "__main__":
    main()
