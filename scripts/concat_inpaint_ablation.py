"""Inpainting-quality ablation, 5-way (no re-inference):
[ GT | inpaint_result | controlnet(2) | inpaint_result_effecterase | effecterase_v2 ].

Each model shown next to the actual inpaint background it was conditioned on:
  - controlnet(2)        ← inpaint_result.mp4      (regular inpaint)
  - controlnet_effecterase_v2 ← inpaint_result_effecterase.mp4 (EffectErase)

Sources:
  - GT / inpaint_result / inpaint_result_effecterase : dataset mp4 sliced at the val
    TARGET indices (temporally aligned) — clean scene names from the eval index.
  - controlnet / effecterase outputs : each run's wandb "Sampled Video" (latest full
    val round), matched to the scene by content (grayscale-mean Hungarian) via the
    scene-stable wandb "Original Video" (GT) as bridge.
Output: results/cmp_inpaint_ablation/{standard,unseen}/<scene>.{mp4,gif} + grid.
"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

REPO = Path(".").resolve()
DV = REPO / "WorldTraj/dynamicverse"
RUNS = {
    "controlnet": REPO / "exp/va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora/wandb/run-20260611_201055-exp_va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora/files/media",
    "effecterase": REPO / "exp/va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_effecterase_v2_unscaledcomp/wandb/run-20260702_232628-exp_va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_effecterase_v2_unscaledcomp/files/media",
}
INDEX = {
    "unseen": REPO / "assets/evaluation_index/dynamicverse_unseen_8.json",
    "standard": REPO / "assets/evaluation_index/dynamicverse_standard.json",
}
OUT = REPO / "results/cmp_inpaint_ablation"
FPS = 8
CELL_H, CELL_W = 240, 416
COLS = ["GT", "inpaint_result", "controlnet(2)", "inpaint_result_effecterase", "effecterase_v2"]


def resize(f):
    if f.shape[:2] == (CELL_H, CELL_W):
        return f
    return np.asarray(Image.fromarray(f).convert("RGB").resize((CELL_W, CELL_H), Image.BILINEAR))


def sig(v, k=48):
    idx = np.linspace(0, len(v) - 1, min(8, len(v))).astype(int)
    return np.stack([np.asarray(Image.fromarray(v[i]).convert("L").resize((k, k)), np.float32)
                     for i in idx]).mean(0)


def hungarian(rows, cols):
    C = np.zeros((len(rows), len(cols)))
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            C[i, j] = ((a - b) ** 2).mean()
    r, c = linear_sum_assignment(C)
    return {int(a): int(b) for a, b in zip(r, c)}


def latest_full(base, split, prefix):
    d = defaultdict(list)
    for f in glob.glob(f"{base}/videos/{split}/{prefix} Video_*.mp4"):
        m = re.search(r"_(\d+)_([0-9a-f]+)\.mp4$", f)
        if m:
            d[int(m.group(1))].append((m.group(2), f))
    full = [w for w, v in d.items() if len(v) == 8]
    w = max(full) if full else max(d, key=lambda k: len(d[k]))
    return d[w]


def scene_dir(scene, split):
    if split == "unseen":
        return DV / "DAVIS" / scene
    # standard: pick the subdataset dir that has the inpaint files
    for d in glob.glob(str(DV / "*" / scene)):
        if os.path.exists(f"{d}/inpaint_result_effecterase.mp4") and os.path.exists(f"{d}/video_input.mp4"):
            return Path(d)
    return None


def load_at(path, tgt_idx):
    v = iio.imread(path)
    idx = [i for i in tgt_idx if i < len(v)]
    return v[idx]


def build(split):
    idx = json.load(open(INDEX[split]))
    # dataset-side per scene: GT / inpaint / inpaint_effecterase at target indices
    scenes, gt, inp, inpE, gsig = [], {}, {}, {}, {}
    for sc, meta in idx.items():
        d = scene_dir(sc, split)
        if d is None:
            continue
        try:
            t = meta["target"]
            g = load_at(d / "video_input.mp4", t)
            a = load_at(d / "inpaint_result.mp4", t)
            b = load_at(d / "inpaint_result_effecterase.mp4", t)
        except Exception as e:  # noqa: BLE001
            print(f"[{split}] skip dataset {sc}: {e}")
            continue
        scenes.append(sc)
        gt[sc], inp[sc], inpE[sc], gsig[sc] = g, a, b, sig(g)

    # wandb GT (Original) -> identify which dataset scene each is
    wb_gt = latest_full(RUNS["controlnet"], split, "Original")
    wb_gt_sig = [sig(iio.imread(p)) for _, p in wb_gt]
    cand_sig = [gsig[s] for s in scenes]
    m = hungarian(wb_gt_sig, cand_sig)          # wandb_gt_idx -> scene_idx
    hash2scene = {wb_gt[i][0]: scenes[j] for i, j in m.items()}

    # each model's Sampled -> wandb GT hash (content), then -> scene
    model_by_scene = {sc: {} for sc in scenes}
    for run in ("controlnet", "effecterase"):
        samp = latest_full(RUNS[run], split, "Sampled")
        svids = [iio.imread(p) for _, p in samp]
        ssig = [sig(v) for v in svids]
        mm = hungarian(wb_gt_sig, ssig)          # wandb_gt_idx -> sampled_idx
        for gi, si in mm.items():
            sc = hash2scene.get(wb_gt[gi][0])
            if sc is not None:
                model_by_scene[sc][run] = svids[si]
    return scenes, gt, inp, inpE, model_by_scene


def main():
    for split in ("standard", "unseen"):
        odir = OUT / split
        odir.mkdir(parents=True, exist_ok=True)
        scenes, gt, inp, inpE, mbs = build(split)
        rows = []
        for sc in scenes:
            md = mbs[sc]
            if "controlnet" not in md or "effecterase" not in md:
                print(f"[{split}] skip {sc}: missing model output")
                continue
            g, a, b = gt[sc], inp[sc], inpE[sc]
            cn, ce = md["controlnet"], md["effecterase"]
            T = min(len(g), len(a), len(b), len(cn), len(ce))
            frames = []
            for t in range(T):
                cells = [resize(g[t]), resize(a[t]), resize(cn[t]), resize(b[t]), resize(ce[t])]
                frames.append(np.concatenate(cells, axis=1))
            arr = np.stack(frames)
            name = f"{split}_{sc}"
            iio.imwrite(odir / f"{name}.mp4", arr, fps=FPS, codec="libx264")
            pil = [Image.fromarray(f) for f in arr]
            pil[0].save(odir / f"{name}.gif", save_all=True, append_images=pil[1:],
                        duration=int(1000 / FPS), loop=0, disposal=2)
            rows.append(arr)
            print(f"[{split}] {name}: {arr.shape}")
        if rows:
            T = min(a.shape[0] for a in rows)
            grid = np.concatenate([a[:T] for a in rows], axis=1)
            iio.imwrite(OUT / f"ALL_{split}_grid.mp4", grid, fps=FPS, codec="libx264")
            print(f"[{split}] grid -> ALL_{split}_grid.mp4 {grid.shape}")


if __name__ == "__main__":
    main()
