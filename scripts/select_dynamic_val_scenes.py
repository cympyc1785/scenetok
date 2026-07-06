"""Score DynamicVerse scenes by foreground dynamism (instance-mask motion +
moderate area) to curate 'dynamic 뚜렷' standard-validation scenes.

score = motion_norm * area_weight, where
  motion_norm = mean consecutive foreground-centroid displacement / image diagonal
  area_weight = mean fg area fraction, gently penalized if too small/large
                (want a prominent-but-not-full-frame moving object → bg stays visible).

Prints ranked candidates; selects top-N with subdataset diversity (<=2 each).
Requires each scene to have video_input.mp4, mask/, category/category.json,
prompts.json(prompt_scene), and >=49 frames (context idx max 48).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image

ROOT = Path("WorldTraj/dynamicverse")
# training subdatasets (exclude DAVIS=unseen, logs, dynpose-100k)
SUBS = ["MOSE", "VOST", "SAV", "spring", "dynamic_replica", "MVS-Synth", "youtube_vis", "uvo"]
MIN_FRAMES = 49


def valid_scene(d: Path) -> bool:
    if not (d / "video_input.mp4").exists():
        return False
    if not (d / "mask").is_dir():
        return False
    cat = d / "category" / "category.json"
    if not cat.exists():
        return False
    pr = d / "prompts.json"
    if not pr.exists():
        return False
    try:
        c = json.load(open(cat))
        if not c.get("dynamic"):
            return False
    except Exception:
        return False
    return True


def _n_instances(rgb: np.ndarray, min_frac: float = 0.01) -> int:
    """Count distinct mask instance colors (RGB) covering >= min_frac of the image."""
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.int64)
    key = flat[:, 0] * 65536 + flat[:, 1] * 256 + flat[:, 2]
    key = key[key != 0]                       # drop black background
    if key.size == 0:
        return 0
    vals, cnts = np.unique(key, return_counts=True)
    return int((cnts >= min_frac * h * w).sum())


def _read_video_gray(path: Path, idx, size: int = 160):
    """Read subsampled, downscaled grayscale frames (T,size,size) float[0,1]."""
    try:
        v = iio.imread(path)            # (T,H,W,3)
    except Exception:
        return None
    if v.ndim != 4:
        return None
    out = []
    for i in idx:
        i = min(i, v.shape[0] - 1)
        im = Image.fromarray(v[i]).convert("L").resize((size, size))
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(out)


def score_scene(d: Path, n_frames: int = 12):
    masks = sorted((d / "mask").glob("*.png"))
    if len(masks) < MIN_FRAMES:
        return None
    idx = np.linspace(0, len(masks) - 1, n_frames).round().astype(int)
    centroids, areas, ninst = [], [], []
    H = W = None
    mask_small = []                            # (size,size) bool for inpaint check
    SIZE = 160
    for i in idx:
        a = np.asarray(Image.open(masks[i]).convert("RGB"))
        fg = a.sum(2) > 0
        H, W = fg.shape
        areas.append(fg.mean())
        ninst.append(_n_instances(a))
        m = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((SIZE, SIZE))) > 127
        mask_small.append(m)
        if fg.any():
            ys, xs = np.nonzero(fg)
            centroids.append((xs.mean() / W, ys.mean() / H))
        else:
            centroids.append(None)
    pres = [c is not None for c in centroids]
    if sum(pres) < 0.8 * len(idx):           # foreground must be present most frames
        return None
    # ── single-instance: dominant frames must have exactly 1 instance ──
    n_inst_mode = int(np.median([n for n in ninst if n > 0])) if any(ninst) else 0
    single_instance = n_inst_mode == 1

    cs = [c for c in centroids if c is not None]
    disp = [np.hypot(cs[i + 1][0] - cs[i][0], cs[i + 1][1] - cs[i][1]) for i in range(len(cs) - 1)]
    motion = float(np.mean(disp)) if disp else 0.0
    mean_area = float(np.mean(areas))
    if mean_area < 0.03:
        aw = mean_area / 0.03 * 0.3
    elif mean_area > 0.40:
        aw = max(0.1, 1.0 - (mean_area - 0.40) * 2)
    else:
        aw = 1.0
    score = motion * aw

    # ── clean-inpaint: inpaint_result mask-region must NOT move more than bg ──
    # (residual dynamic foreground left by failed removal → high ratio).
    residual_ratio = None
    inp = d / "inpaint_result.mp4"
    if inp.exists():
        g = _read_video_gray(inp, idx, size=SIZE)
        if g is not None and len(g) == len(mask_small):
            mreg, breg = [], []
            for t in range(len(g) - 1):
                diff = np.abs(g[t + 1] - g[t])
                m = mask_small[t]
                if m.any() and (~m).any():
                    mreg.append(diff[m].mean())
                    breg.append(diff[~m].mean())
            if mreg:
                residual_ratio = float(np.mean(mreg) / (np.mean(breg) + 1e-6))

    return {"motion": motion, "area": mean_area, "aw": aw, "score": score,
            "n_inst": n_inst_mode, "single_instance": single_instance,
            "residual_ratio": residual_ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_sub", type=int, default=30, help="random candidates per subdataset")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--max_per_sub", type=int, default=2)
    ap.add_argument("--max_residual", type=float, default=1.3,
                    help="max inpaint mask-region/bg motion ratio (lower=cleaner removal)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/dynamic_scene_scores.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cands = []
    for sub in SUBS:
        sd = ROOT / sub
        if not sd.is_dir():
            continue
        scenes = [d for d in sd.iterdir() if d.is_dir() and valid_scene(d)]
        rng.shuffle(scenes)
        scenes = scenes[: args.per_sub]
        for d in scenes:
            s = score_scene(d)
            if s is None:
                continue
            s.update({"sub": sub, "scene": d.name, "path": str(d)})
            cands.append(s)
        print(f"[{sub}] scored {sum(1 for c in cands if c['sub']==sub)} / sampled {len(scenes)}")

    cands.sort(key=lambda x: x["score"], reverse=True)
    print("\n=== TOP 25 by dynamism score ===")
    for c in cands[:25]:
        rr = c.get("residual_ratio")
        rr = f"{rr:.2f}" if rr is not None else "  -"
        flag = "OK " if (c["single_instance"] and rr != "  -" and float(rr) <= args.max_residual) else "   "
        print(f"  {flag}{c['score']:.4f} motion={c['motion']:.4f} area={c['area']:.3f} "
              f"ninst={c['n_inst']} resid={rr}  {c['sub']:15s} {c['scene']}")

    # FILTERS: single instance + clean inpaint (low residual motion in mask region)
    elig = [c for c in cands if c["single_instance"]
            and c.get("residual_ratio") is not None
            and c["residual_ratio"] <= args.max_residual]
    print(f"\n[filter] single-instance + residual<= {args.max_residual}: {len(elig)} eligible")

    # diverse top-N: <= max_per_sub per subdataset
    sel, per = [], {}
    for c in elig:
        if per.get(c["sub"], 0) >= args.max_per_sub:
            continue
        sel.append(c)
        per[c["sub"]] = per.get(c["sub"], 0) + 1
        if len(sel) >= args.top:
            break
    print(f"\n=== SELECTED {len(sel)} (<= {args.max_per_sub}/sub) ===")
    for c in sel:
        print(f"  {c['score']:.4f}  {c['sub']:15s} {c['scene']}  ({c['path']})")
    json.dump({"selected": sel, "all": cands}, open(args.out, "w"), indent=1)
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
