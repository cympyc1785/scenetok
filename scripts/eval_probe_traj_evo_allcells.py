"""Compute evo trajectory metrics (ATE / RPEt / RPEr, Sim(3)-aligned) for EVERY
(layer × noise) cell from saved probe predictions, then plot per-model panels
with one line per timestep σ. Reuses probe_weights.pt — no model re-run.
"""
import argparse, json, logging
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape, evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit
from evo.core import sync
logging.getLogger("evo").setLevel(logging.ERROR)


def euler_to_rotmat_np(e):
    x, y, z = e[:, 0], e[:, 1], e[:, 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.stack([cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx,
                     sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx,
                     -sy, cy*sx, cy*cx], -1).reshape(-1, 3, 3)


def metrics_one(Rp, tp, Rg, tg):
    ts = np.arange(tp.shape[0], dtype=np.float64)
    est = PoseTrajectory3D(positions_xyz=tp.astype(np.float64),
                           orientations_quat_wxyz=Rot.from_matrix(Rp).as_quat(scalar_first=True), timestamps=ts)
    ref = PoseTrajectory3D(positions_xyz=tg.astype(np.float64),
                           orientations_quat_wxyz=Rot.from_matrix(Rg).as_quat(scalar_first=True), timestamps=ts)
    ref, est = sync.associate_trajectories(ref, est); est.align(ref, correct_scale=True)
    a = main_ape.ape(ref, est, pose_relation=PoseRelation.translation_part, align=True, correct_scale=True).stats['rmse']
    rt = main_rpe.rpe(ref, est, pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.frames,
                      all_pairs=True, align=True, correct_scale=True).stats['rmse']
    rr = main_rpe.rpe(ref, est, pose_relation=PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames,
                      all_pairs=True, align=True, correct_scale=True).stats['rmse']
    return a, rt, rr


def eval_all(out_dir, name):
    w = torch.load(f"{out_dir}/probe_weights.pt", map_location="cpu")
    F = w["num_frames"]; gt = w["gt_test"]; R_gt = gt["R"].numpy(); t_gt = gt["trans"].numpy().reshape(-1, F, 3)
    L = w["layers"]; N = w["noise_levels"]; Nte = R_gt.shape[0]
    res = {m: np.full((len(L), len(N)), np.nan) for m in ("ATE", "RPEt", "RPEr")}
    for li, lyr in enumerate(L):
        for ni, n in enumerate(N):
            pred = w["weights"][f"{lyr}_{n}"]["pred"].numpy(); fe = pred.shape[1] // 2
            A = []
            for i in range(Nte):
                try:
                    A.append(metrics_one(euler_to_rotmat_np(pred[i, :fe].reshape(F, 3)), pred[i, fe:].reshape(F, 3),
                                         R_gt[i], t_gt[i]))
                except Exception:
                    pass
            if A:
                A = np.array(A); res["ATE"][li, ni], res["RPEt"][li, ni], res["RPEr"][li, ni] = A.mean(0)
        print(f"  {name} layer {lyr} done", flush=True)
    return L, N, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan", default="results/probe_wan_re10k_shape")
    ap.add_argument("--cog", default="results/probe_cogvideox_re10k_shape")
    args = ap.parse_args()
    wL, wN, wr = eval_all(args.wan, "Wan")
    cL, cN, cr = eval_all(args.cog, "Cog")
    json.dump({"wan": {m: wr[m].tolist() for m in wr}, "cog": {m: cr[m].tolist() for m in cr},
               "wL": wL, "cL": cL, "N": wN}, open("results/probe_compare/traj_evo_allcells.json", "w"))

    mets = [("ATE", "ATE ↓"), ("RPEt", "RPEt ↓"), ("RPEr", "RPEr [deg] ↓")]
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    for col, (name, L, N, r) in enumerate([("Wan2.2 TI2V-5B", wL, wN, wr), ("CogVideoX-5b", cL, cN, cr)]):
        rel = np.array(L) / max(L)
        for row, (mk, lab) in enumerate(mets):
            ax = axes[row, col]
            for i, n in enumerate(N):
                ax.plot(rel, r[mk][:, i], "-o", ms=3, color=cmap(i/(len(N)-1)), label=f"σ={n:.2f}")
            ax.grid(alpha=0.3)
            if row == 0: ax.set_title(name, fontsize=12)
            if col == 0: ax.set_ylabel(lab)
            if row == 2: ax.set_xlabel("relative block depth")
            if row == 0 and col == 0: ax.legend(fontsize=8, ncol=2)
    # shared y per row
    for row in range(3):
        lo = min(np.nanmin(wr[mets[row][0]]), np.nanmin(cr[mets[row][0]]))
        hi = max(np.nanmax(wr[mets[row][0]]), np.nanmax(cr[mets[row][0]]))
        pad = (hi-lo)*0.06
        for col in range(2): axes[row, col].set_ylim(lo-pad, hi+pad)
    fig.suptitle("evo trajectory metrics per (block × σ) — per-model panels, one line per timestep σ\n"
                 "Sim(3)-aligned (correct_scale) on saved probe predictions; re10k 300 scenes, 60 test", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/probe_compare/per_model_evo_metrics.png", dpi=140, bbox_inches="tight")
    print("saved → results/probe_compare/per_model_evo_metrics.png")


if __name__ == "__main__":
    main()
