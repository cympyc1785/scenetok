"""Apply DynamicVerse/SINTEL-style trajectory eval (evo: Sim(3) align + ATE/RPE)
to the SAVED camera-probe predictions (probe_weights.pt) on re10k test scenes.

Mirrors submodules/DynamicVerse/dynamicBA/eval_pose.py: build evo PoseTrajectory3D
for predicted vs GT camera trajectory per scene, align with correct_scale=True,
report ATE (APE translation rmse), RPEt (Δ=1 frame translation), RPEr (Δ=1 frame
rotation deg). Averaged over test scenes, per model, at each model's best-rotation cell.
"""
import argparse, json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit
from evo.core import sync


def euler_to_rotmat_np(e):  # (N,3) XYZ → (N,3,3) = Rz@Ry@Rx (matches probe's euler_to_rotmat)
    x, y, z = e[:, 0], e[:, 1], e[:, 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
                  sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx,
                  -sy,   cy*sx,            cy*cx], axis=-1).reshape(-1, 3, 3)
    return R


def traj_metrics(R_pred, t_pred, R_gt, t_gt):
    q_pred = Rot.from_matrix(R_pred).as_quat(scalar_first=True)   # wxyz
    q_gt = Rot.from_matrix(R_gt).as_quat(scalar_first=True)
    ts = np.arange(t_pred.shape[0], dtype=np.float64)
    est = PoseTrajectory3D(positions_xyz=t_pred.astype(np.float64), orientations_quat_wxyz=q_pred, timestamps=ts)
    ref = PoseTrajectory3D(positions_xyz=t_gt.astype(np.float64), orientations_quat_wxyz=q_gt, timestamps=ts)
    ref, est = sync.associate_trajectories(ref, est)
    est.align(ref, correct_scale=True)
    ate = main_ape.ape(ref, est, pose_relation=PoseRelation.translation_part, align=True, correct_scale=True).stats['rmse']
    rpet = main_rpe.rpe(ref, est, pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.frames,
                        all_pairs=True, align=True, correct_scale=True).stats['rmse']
    rper = main_rpe.rpe(ref, est, pose_relation=PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames,
                        all_pairs=True, align=True, correct_scale=True).stats['rmse']
    return ate, rpet, rper


def best_cell(results_json):
    r = json.load(open(results_json))["results"]
    b = min(r, key=lambda x: x["rot_err_deg"])
    return f"{b['layer']}_{b['noise']}", b["layer"], b["noise"]


def eval_model(name, out_dir):
    w = torch.load(f"{out_dir}/probe_weights.pt", map_location="cpu")
    cell, L, n = best_cell(f"{out_dir}/probe_results.json")
    pred = w["weights"][cell]["pred"].numpy()                     # (Nte, F*6) = [euler|trans]
    gt = w["gt_test"]; F = w["num_frames"]; fe = pred.shape[1] // 2
    R_gt = gt["R"].numpy(); t_gt = gt["trans"].numpy().reshape(-1, F, 3)   # (Nte,F,3,3),(Nte,F,3)
    Nte = pred.shape[0]
    ates, rpets, rpers = [], [], []
    for i in range(Nte):
        eul = pred[i, :fe].reshape(F, 3); tt = pred[i, fe:].reshape(F, 3)
        try:
            a, rt, rr = traj_metrics(euler_to_rotmat_np(eul), tt, R_gt[i], t_gt[i])
            ates.append(a); rpets.append(rt); rpers.append(rr)
        except Exception as e:
            print(f"  {name} scene {i} skip: {e}")
    return dict(model=name, cell=f"block {L} / σ{n}", n=len(ates),
                ATE=float(np.mean(ates)), RPEt=float(np.mean(rpets)), RPEr=float(np.mean(rpers)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan", default="results/probe_wan_re10k_shape")
    ap.add_argument("--cog", default="results/probe_cogvideox_re10k_shape")
    args = ap.parse_args()
    rows = [eval_model("Wan2.2 TI2V-5B", args.wan), eval_model("CogVideoX-5b", args.cog)]
    print(f"\n{'Model':<18}{'best cell':<16}{'ATE↓':<12}{'RPEt↓':<12}{'RPEr(deg)↓':<12}{'n'}")
    print("-" * 78)
    for r in rows:
        print(f"{r['model']:<18}{r['cell']:<16}{r['ATE']:<12.5f}{r['RPEt']:<12.5f}{r['RPEr']:<12.4f}{r['n']}")
    json.dump(rows, open("results/probe_compare/traj_evo_table.json", "w"), indent=2)
    print("\nsaved → results/probe_compare/traj_evo_table.json")


if __name__ == "__main__":
    main()
