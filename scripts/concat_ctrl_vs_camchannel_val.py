"""4-way comparison from EXISTING validation videos (no re-inference):
[ context | GT | controlnet (layer2) | camchannel ] per val scene.

Source = wandb local media of two training runs:
  - controlnet: va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora
  - camchannel: va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_selfattnlora_unscaledcomp

Alignment:
  - GT ("Original Video") hash is scene-stable & shared across runs → anchors scenes.
  - Sampled ("Sampled Video") hashes are per-content (differ across runs) and their
    per-round list order is not recoverable from filenames (mtime has 1s ties), so each
    run's Sampled is paired to that run's GT by content (PSNR / L2 Hungarian on small
    grayscale mean-frames). Same for the static Context strip.
Output: results/cmp_ctrl_vs_camchannel_val/{standard,unseen}/<scene>.{mp4,gif} + grid.
"""
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

REPO = Path(".").resolve()
RUNS = {
    "controlnet": REPO / "exp/va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora/wandb/run-20260611_201055-exp_va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora/files/media",
    "camchannel": REPO / "exp/va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_selfattnlora_unscaledcomp/wandb/run-20260702_232628-exp_va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_selfattnlora_unscaledcomp/files/media",
}
OUT = REPO / "results/cmp_ctrl_vs_camchannel_val"
FPS = 8
CELL_H, CELL_W = 240, 416   # half of 480x832 (matches reference gif)
COLS = ["context", "GT", "controlnet", "camchannel"]


def latest_full(base, split, prefix, ext, sub):
    """Return list of (hash, path) for the latest wstep that has exactly 8 items."""
    d = defaultdict(list)
    for f in glob.glob(f"{base}/{sub}/{split}/{prefix}*_*.{ext}"):
        m = re.search(rf"_(\d+)_([0-9a-f]+)\.{ext}$", f)
        if m:
            d[int(m.group(1))].append((m.group(2), f))
    full = [w for w, v in d.items() if len(v) == 8]
    if not full:
        # fall back to the round with the most items
        w = max(d, key=lambda k: len(d[k]))
        return d[w]
    return d[max(full)]


def read_video(p):
    return iio.imread(p)  # (T,H,W,3) uint8


def read_context_strip(p):
    """Split the horizontal context strip (frames separated by ~8px white cols)
    into a list of individual context frames."""
    a = iio.imread(p)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    W = a.shape[1]
    col = a.mean(axis=(0, 2))
    white = col > 245
    # separator column groups
    seps, i = [], 0
    while i < W:
        if white[i]:
            j = i
            while j < W and white[j]:
                j += 1
            seps.append((i, j))
            i = j
        else:
            i += 1
    # frame boundaries = between separators (and strip edges)
    cuts = [0]
    for s0, s1 in seps:
        cuts.append(s0)
        cuts.append(s1)
    cuts.append(W)
    frames = []
    for k in range(0, len(cuts) - 1, 2):
        a0, a1 = cuts[k], cuts[k + 1]
        if a1 - a0 > 8:  # skip separator-only slivers
            frames.append(a[:, a0:a1])
    return frames  # list of (H,Wf,3)


def signature_video(v, k=48):
    """Small grayscale mean-frame signature for content matching."""
    idx = np.linspace(0, len(v) - 1, min(8, len(v))).astype(int)
    frs = []
    for i in idx:
        im = Image.fromarray(v[i]).convert("L").resize((k, k))
        frs.append(np.asarray(im, dtype=np.float32))
    return np.stack(frs).mean(0)


def signature_strip(frames, k=48):
    """Mean grayscale signature over the context frames (for content pairing)."""
    accs = []
    for f in frames:
        im = Image.fromarray(f).convert("L").resize((k, k))
        accs.append(np.asarray(im, dtype=np.float32))
    return np.stack(accs).mean(0)


def pair_to_gt(gt_sigs, cand_sigs):
    """Hungarian: assign each candidate to a GT index (min L2)."""
    C = np.zeros((len(gt_sigs), len(cand_sigs)), dtype=np.float64)
    for i, g in enumerate(gt_sigs):
        for j, c in enumerate(cand_sigs):
            C[i, j] = float(((g - c) ** 2).mean())
    r, c = linear_sum_assignment(C)  # gt_idx -> cand_idx
    return {int(ri): int(ci) for ri, ci in zip(r, c)}


def resize(f):
    if f.shape[:2] == (CELL_H, CELL_W):
        return f
    return np.asarray(Image.fromarray(f).resize((CELL_W, CELL_H), Image.BILINEAR))


def label(frame, text):
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, 2 + 7 * len(text) + 8, 18], fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 0))
    return np.asarray(img)


def build_split(split):
    # --- GT anchor from camchannel run (scene-stable hashes) ---
    gt_files = latest_full(RUNS["camchannel"], split, "Original Video", "mp4", "videos")
    gt_files = sorted(gt_files)  # by hash, deterministic
    gt_hashes = [h for h, _ in gt_files]
    gt_vids = [read_video(p) for _, p in gt_files]
    gt_sigs = [signature_video(v) for v in gt_vids]

    per_scene = {h: {"GT": v} for h, v in zip(gt_hashes, gt_vids)}

    # --- each run's Sampled -> GT by content ---
    for run in ("controlnet", "camchannel"):
        samp = latest_full(RUNS[run], split, "Sampled Video", "mp4", "videos")
        vids = [read_video(p) for _, p in samp]
        sigs = [signature_video(v) for v in vids]
        # gt sigs for THIS run (same scene hashes; use canonical gt_sigs since GT identical)
        m = pair_to_gt(gt_sigs, sigs)  # gt_idx -> sampled_idx
        for gi, si in m.items():
            per_scene[gt_hashes[gi]][run] = vids[si]

    # --- context strip from camchannel run -> GT by content ---
    ctx = latest_full(RUNS["camchannel"], split, "Context (full_sequence)", "png", "images")
    strips = [read_context_strip(p) for _, p in ctx]
    csigs = [signature_strip(s) for s in strips]
    mc = pair_to_gt(gt_sigs, csigs)
    for gi, ci in mc.items():
        per_scene[gt_hashes[gi]]["context"] = strips[ci]

    return gt_hashes, per_scene


def main():
    grid_all = []
    for split in ("standard", "unseen"):
        odir = OUT / split
        odir.mkdir(parents=True, exist_ok=True)
        gt_hashes, per_scene = build_split(split)
        rows = []
        for i, h in enumerate(gt_hashes):
            d = per_scene[h]
            if not all(k in d for k in ("GT", "controlnet", "camchannel", "context")):
                print(f"[{split}] skip {h}: missing {[k for k in COLS if k not in d]}")
                continue
            gt, cn, cc, ctx = d["GT"], d["controlnet"], d["camchannel"], d["context"]
            T = min(len(gt), len(cn), len(cc))
            nctx = len(ctx)
            ctx_cells = [resize(f) for f in ctx]  # each context view resized to a cell
            frames = []
            for t in range(T):
                ci = min(nctx - 1, int(t / T * nctx))  # slideshow synced to clip length
                cells = [
                    ctx_cells[ci],
                    resize(gt[t]),
                    resize(cn[t]),
                    resize(cc[t]),
                ]
                frames.append(np.concatenate(cells, axis=1))
            arr = np.stack(frames)
            name = f"{split}_{i}_{h[:8]}"
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
            grid_all.append((split, grid))


if __name__ == "__main__":
    main()
