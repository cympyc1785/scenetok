"""Measure a single scale-transform constant α between DA3-normalized and
COLMAP-normalized cameras across DL3DV scenes.

DA3 카메라는 `<scene>/da3/exports/mini_npz/results.npz`에 이미 precompute되어 있다
(extrinsics (N,3,4) w2c, intrinsics (N,3,3), depth (N,H,W)). 따라서 DA3 재추론은
하지 않고 이 npz를 사용한다.

Per scene:
  1. COLMAP transforms.json → c2w pose 파싱, **첫 프레임 기준 reference-relative** 변환
     → 카메라 위치(ref frame 좌표)를 centroid 대비 95퍼센타일 거리로 정규화 → v_C
  2. DA3 npz → c2w (= inv(w2c)), 같은 프레임 reference-relative 변환
     → first-frame depth에서 0/비유한/하늘(상위 퍼센타일) 마스킹 후 median depth로
        카메라 위치 정규화 → v_DA3
  3. v_DA3를 v_C에 회전 정렬(Procrustes, no-scale) 후
     α_scene = Σ(v_DA3·v_C) / Σ|v_DA3|²  (scalar LS), 정렬 후 ATE 기록

전체 scene:
  - ATE 임계 초과 scene 필터
  - α median / IQR / CV
  - α vs 평균 depth 산점도 저장
  - hold-out scene에 α median 적용 후 위치 오차 검증
  - CV 기준 "단일 상수 가능/불가" 판정

Usage:
  python scripts/measure_da3_colmap_scale.py \
    --scenes DATA/DL3DV/DL3DV-960/train/1K/<id> ... \
    [--glob] [--ate_thresh 0.1] [--out results/da3_colmap_scale]
또는 --glob 만 주면 da3 npz + transforms.json 둘 다 있는 scene을 자동 수집.
"""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ───────────────────────── camera loading ─────────────────────────
def load_colmap_c2w(transforms_json: str) -> np.ndarray:
    """Return (N,4,4) c2w, ordered by file_path for determinism."""
    d = json.load(open(transforms_json))
    frames = d["frames"]
    order = np.argsort([f.get("file_path", str(i)) for i, f in enumerate(frames)])
    mats = np.stack([np.array(frames[i]["transform_matrix"], dtype=np.float64) for i in order])
    return mats


def load_da3(npz_path: str):
    """Return (c2w (N,4,4), depth0 (H,W))."""
    z = np.load(npz_path, allow_pickle=True)
    e = z["extrinsics"].astype(np.float64)          # (N,3,4) w2c [R|t]
    N = e.shape[0]
    w2c = np.tile(np.eye(4), (N, 1, 1))
    w2c[:, :3, :4] = e
    c2w = np.linalg.inv(w2c)
    depth0 = z["depth"][0].astype(np.float64) if "depth" in z else None
    return c2w, depth0


# ───────────────────────── geometry helpers ─────────────────────────
def reference_relative_centers(c2w: np.ndarray) -> np.ndarray:
    """Reference-relative (w.r.t frame 0) camera centers, in ref-camera frame."""
    ref_inv = np.linalg.inv(c2w[0])
    rel = ref_inv[None] @ c2w                        # (N,4,4)
    return rel[:, :3, 3]                             # centers in ref frame


def normalize_by_p95(centers: np.ndarray) -> tuple[np.ndarray, float]:
    cen = centers.mean(0)
    dist = np.linalg.norm(centers - cen, axis=1)
    s = np.percentile(dist, 95)
    s = max(s, 1e-9)
    return centers / s, s


def first_frame_median_depth(depth0: np.ndarray) -> float:
    d = depth0.reshape(-1)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return 1.0
    dv = d[valid]
    # sky/infinity 제거: 상위 5% (먼 배경/하늘) 컷
    hi = np.percentile(dv, 95)
    dv = dv[dv <= hi]
    if dv.size == 0:
        dv = d[valid]
    return float(np.median(dv))


def procrustes_rotation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Best rotation R (no scale) mapping X→Y (centered). Returns R (3,3)."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def match_frames(n_da3: int, n_clm: int) -> tuple[np.ndarray, np.ndarray]:
    """Assume shared capture order; subsample both to common count via linspace."""
    n = min(n_da3, n_clm)
    di = np.linspace(0, n_da3 - 1, n).round().astype(int)
    ci = np.linspace(0, n_clm - 1, n).round().astype(int)
    return di, ci


# ───────────────────────── per-scene ─────────────────────────
def process_scene(scene_dir: str):
    tj = os.path.join(scene_dir, "transforms.json")
    npz = os.path.join(scene_dir, "da3/exports/mini_npz/results.npz")
    if not (os.path.exists(tj) and os.path.exists(npz)):
        return None

    c2w_clm = load_colmap_c2w(tj)
    c2w_da3, depth0 = load_da3(npz)
    if depth0 is None:
        return None

    di, ci = match_frames(len(c2w_da3), len(c2w_clm))
    if len(di) < 8:
        return None
    c2w_da3, c2w_clm = c2w_da3[di], c2w_clm[ci]

    C_clm = reference_relative_centers(c2w_clm)
    C_da3 = reference_relative_centers(c2w_da3)

    v_C, _ = normalize_by_p95(C_clm)
    med_depth = first_frame_median_depth(depth0)
    v_DA3 = C_da3 / max(med_depth, 1e-9)

    # rotation-align DA3 → COLMAP frame (centers are reference-relative; both have
    # frame 0 at origin but their ref-camera orientations differ).
    R = procrustes_rotation(v_DA3, v_C)
    v_DA3r = v_DA3 @ R.T
    # center removal for the LS scale (translation already ~0 at frame 0; keep robust)
    num = float((v_DA3r * v_C).sum())
    den = float((v_DA3r ** 2).sum())
    alpha = num / max(den, 1e-12)

    aligned = alpha * v_DA3r
    ate = float(np.sqrt(((aligned - v_C) ** 2).sum(1).mean()))

    return dict(
        scene=scene_dir.split("/train/")[-1] if "/train/" in scene_dir else scene_dir,
        alpha=alpha,
        ate=ate,
        median_depth=med_depth,
        n_frames=int(len(di)),
        clm_p95=float(normalize_by_p95(C_clm)[1]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--glob", action="store_true",
                    help="da3 npz + transforms.json 둘 다 있는 scene 자동 수집")
    ap.add_argument("--glob_root", default="DATA/DL3DV/DL3DV-960/train")
    ap.add_argument("--max_scenes", type=int, default=60)
    ap.add_argument("--ate_thresh", type=float, default=0.15,
                    help="이 ATE 초과 scene은 α 통계에서 제외 (정렬 실패/동적 scene)")
    ap.add_argument("--holdout_frac", type=float, default=0.3)
    ap.add_argument("--out", default="results/da3_colmap_scale")
    args = ap.parse_args()

    scenes = list(args.scenes)
    if args.glob or not scenes:
        found = glob.glob(os.path.join(args.glob_root, "*/*/da3/exports/mini_npz/results.npz"))
        scenes += [p.split("/da3/")[0] for p in found]
    # dedup, keep only those with transforms.json
    seen, uniq = set(), []
    for s in scenes:
        if s in seen:
            continue
        seen.add(s)
        if os.path.exists(os.path.join(s, "transforms.json")):
            uniq.append(s)
    scenes = uniq[: args.max_scenes]
    print(f"[scale] {len(scenes)} scenes to process")

    results = []
    for s in scenes:
        try:
            r = process_scene(s)
            if r is not None:
                results.append(r)
                print(f"  {r['scene'][:42]:42s} α={r['alpha']:.4f} ATE={r['ate']:.4f} "
                      f"med_depth={r['median_depth']:.3f} N={r['n_frames']}")
        except Exception as e:
            print(f"  {s.split('/train/')[-1][:42]:42s} ERR {e}")

    if not results:
        print("[scale] no valid scenes."); return

    os.makedirs(args.out, exist_ok=True)

    alphas_all = np.array([r["alpha"] for r in results])
    ates_all = np.array([r["ate"] for r in results])
    depths_all = np.array([r["median_depth"] for r in results])

    keep = ates_all <= args.ate_thresh
    n_drop = int((~keep).sum())
    alphas = alphas_all[keep]
    depths = depths_all[keep]
    print(f"\n[scale] ATE>{args.ate_thresh} 으로 {n_drop}/{len(results)} scene 제외, "
          f"{len(alphas)} scene 사용")

    if len(alphas) < 2:
        print("[scale] 통계 낼 scene이 부족함."); return

    med = float(np.median(alphas))
    q1, q3 = np.percentile(alphas, [25, 75])
    iqr = float(q3 - q1)
    mean = float(alphas.mean())
    std = float(alphas.std())
    cv = float(std / mean) if mean != 0 else float("inf")

    # robust CV (IQR / median) — heavy-tail에 덜 민감
    cv_robust = float(iqr / med) if med != 0 else float("inf")

    # hold-out 검증: train median α를 holdout에 적용 후 위치오차(ATE-like) 재계산
    rng_idx = np.argsort([r["scene"] for r in results])  # deterministic split
    kept_results = [r for r, k in zip(results, keep) if k]
    n_hold = max(1, int(len(kept_results) * args.holdout_frac))
    holdout = kept_results[-n_hold:]
    train = kept_results[:-n_hold] if len(kept_results) > n_hold else kept_results
    alpha_train_med = float(np.median([r["alpha"] for r in train]))
    # holdout에서 train-median α 대비 자기 α 차이로 위치오차 비율 추정
    hold_pos_err = []
    for r in holdout:
        # 단일 상수 적용 시 잔차 scale 비율 = |alpha_train/alpha_scene - 1|
        ratio_err = abs(alpha_train_med / r["alpha"] - 1.0)
        hold_pos_err.append(ratio_err)
    hold_pos_err = float(np.mean(hold_pos_err)) if hold_pos_err else float("nan")

    summary = dict(
        n_scenes_total=len(results),
        n_scenes_used=int(len(alphas)),
        n_dropped_high_ate=n_drop,
        ate_thresh=args.ate_thresh,
        alpha_median=med, alpha_mean=mean, alpha_std=std,
        alpha_q1=float(q1), alpha_q3=float(q3), alpha_iqr=iqr,
        alpha_cv=cv, alpha_cv_robust=cv_robust,
        holdout_n=n_hold, alpha_train_median=alpha_train_med,
        holdout_mean_rel_pos_err=hold_pos_err,
        per_scene=results,
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # scatter α vs mean depth
    if HAS_PLT:
        plt.figure(figsize=(6, 5))
        plt.scatter(depths_all[keep], alphas_all[keep], c="tab:blue", label="used", s=28)
        if n_drop:
            plt.scatter(depths_all[~keep], alphas_all[~keep], c="tab:red", marker="x",
                        label=f"dropped (ATE>{args.ate_thresh})", s=28)
        plt.axhline(med, color="k", ls="--", lw=1, label=f"median α={med:.3f}")
        plt.fill_between([depths_all.min(), depths_all.max()], q1, q3, color="gray",
                         alpha=0.2, label=f"IQR [{q1:.3f},{q3:.3f}]")
        plt.xlabel("first-frame median depth (DA3)")
        plt.ylabel("α_scene (DA3-norm → COLMAP-norm)")
        plt.title(f"α vs depth  |  CV={cv:.1%}  robustCV(IQR/med)={cv_robust:.1%}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "alpha_vs_depth.png"), dpi=130)
        plt.close()

    # verdict
    print("\n" + "=" * 60)
    print(f"α median = {med:.4f}   IQR = [{q1:.4f}, {q3:.4f}] (IQR={iqr:.4f})")
    print(f"CV = {cv:.1%}   robust CV (IQR/median) = {cv_robust:.1%}")
    print(f"hold-out({n_hold}) mean relative position error (단일 α 적용) = {hold_pos_err:.1%}")
    SINGLE_OK = cv_robust < 0.15 and cv < 0.25
    if SINGLE_OK:
        print(f"판정: ✅ 단일 상수 α≈{med:.3f} 사용 가능 (DA3-norm × α ≈ COLMAP-norm)")
    else:
        print(f"판정: ❌ 단일 상수 불가 — α가 scene마다 편차 큼 (per-scene 정규화 필요)")
    print("=" * 60)
    print(f"saved → {args.out}/summary.json, alpha_vs_depth.png")


if __name__ == "__main__":
    main()
