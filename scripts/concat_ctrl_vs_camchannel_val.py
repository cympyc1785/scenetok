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
    "effecterase": REPO / "exp/va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_effecterase_v2_unscaledcomp/wandb/run-20260702_232628-exp_va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_effecterase_v2_unscaledcomp/files/media",
    "decoupled": REPO / "exp/va-wan-ti2v_scene_decoupled_controlnet_scene_camera_no_lora_unscaledcomp/wandb/run-20260702_232638-exp_va-wan-ti2v_scene_decoupled_controlnet_scene_camera_no_lora_unscaledcomp/files/media",
}
# GT + context anchor run, and the two model columns to compare.
# mode(선택): `ablation`(기본) = controlnet vs effecterase(inpaint quality),
#            `camchannel` = controlnet vs camchannel(self-attn).
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "ablation"
if MODE == "camchannel":
    ANCHOR = "controlnet"
    MODEL_KEYS = ["controlnet", "camchannel"]
    OUT = REPO / "results/cmp_ctrl_vs_camchannel_val"
elif MODE == "decoupled":
    # scene-decoupled: context=wohuman(input), GT("Original")=whuman, model=controlnet output.
    ANCHOR = "decoupled"
    MODEL_KEYS = ["decoupled"]
    OUT = REPO / "results/cmp_scene_decoupled_val"
else:
    ANCHOR = "controlnet"
    MODEL_KEYS = ["controlnet", "effecterase"]
    OUT = REPO / "results/cmp_ctrl_inpaint_ablation_val"
FPS = 8
CELL_H, CELL_W = 240, 416   # half of 480x832 (matches reference gif)
COLS = ["context", "GT"] + MODEL_KEYS


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


def bg_sig_rgb(frames, k=64):
    """RGB **median**-frame signature → removes the moving foreground object so the
    comparison is on the static background (robust for context↔GT scene pairing;
    grayscale-mean collapses water/crowd scenes together)."""
    stack = []
    for f in frames:
        im = Image.fromarray(f).convert("RGB").resize((k, k))
        stack.append(np.asarray(im, dtype=np.float32))
    return np.median(np.stack(stack), axis=0).reshape(-1)


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


import json

DECOUPLED_ROOT = REPO / "DATA/Scene-Decoupled-Video-dataset"


def load_caption_map(anchor_base):
    """{video-file hash -> caption(scene_id_traj)} from the run's wandb-summary.json
    (Original Video panels of both splits). Hash = the 20-char id in the filename."""
    summ = anchor_base.parent / "wandb-summary.json"
    out = {}
    if not summ.exists():
        return out
    d = json.load(open(summ))
    for k, v in d.items():
        if "Original Video" in k and isinstance(v, dict):
            for vid in v.get("videos", []):
                m = re.search(r"_(\d+)_([0-9a-f]+)\.mp4$", vid.get("path", ""))
                if m and vid.get("caption"):
                    out[m.group(2)] = vid["caption"]
    return out


def decoupled_text(caption):
    """caption '<scene_id>_<NN>' -> (scene_id, traj, caption_action_only text)."""
    scene_id, traj = caption.rsplit("_", 1)
    p = DECOUPLED_ROOT / "text" / "whuman" / scene_id / "action.json"
    text = None
    if p.exists():
        data = json.load(open(p))
        text = data.get("caption_action_only") or data.get("caption")
    return scene_id, traj, text


def build_split(split):
    # --- GT anchor from the anchor run (scene-stable hashes) ---
    gt_files = latest_full(RUNS[ANCHOR], split, "Original Video", "mp4", "videos")
    gt_files = sorted(gt_files)  # by hash, deterministic
    gt_hashes = [h for h, _ in gt_files]
    gt_vids = [read_video(p) for _, p in gt_files]
    gt_sigs = [signature_video(v) for v in gt_vids]

    per_scene = {h: {"GT": v} for h, v in zip(gt_hashes, gt_vids)}

    # --- each model run's Sampled -> GT by content ---
    for run in MODEL_KEYS:
        samp = latest_full(RUNS[run], split, "Sampled Video", "mp4", "videos")
        vids = [read_video(p) for _, p in samp]
        sigs = [signature_video(v) for v in vids]
        # gt sigs for THIS run (same scene hashes; use canonical gt_sigs since GT identical)
        m = pair_to_gt(gt_sigs, sigs)  # gt_idx -> sampled_idx
        for gi, si in m.items():
            per_scene[gt_hashes[gi]][run] = vids[si]

    # --- context strip from the anchor run -> GT by content ---
    # Use RGB median-frame (background) signatures so water/crowd scenes (e.g.
    # boat vs classic-car) don't get swapped by grayscale-mean collapse.
    gt_bg = [bg_sig_rgb([v[i] for i in np.linspace(0, len(v) - 1, 8).astype(int)]) for v in gt_vids]
    ctx = latest_full(RUNS[ANCHOR], split, "Context (full_sequence)", "png", "images")
    strips = [read_context_strip(p) for _, p in ctx]
    csigs = [bg_sig_rgb(s) for s in strips]
    mc = pair_to_gt(gt_bg, csigs)
    for gi, ci in mc.items():
        per_scene[gt_hashes[gi]]["context"] = strips[ci]

    return gt_hashes, per_scene


def main():
    cap_map = load_caption_map(RUNS[ANCHOR]) if MODE == "decoupled" else {}
    text_index = {}
    grid_all = []
    for split in ("standard", "unseen"):
        odir = OUT / split
        odir.mkdir(parents=True, exist_ok=True)
        gt_hashes, per_scene = build_split(split)
        rows = []
        for i, h in enumerate(gt_hashes):
            d = per_scene[h]
            need = ["GT", "context"] + MODEL_KEYS
            if not all(k in d for k in need):
                print(f"[{split}] skip {h}: missing {[k for k in need if k not in d]}")
                continue
            gt, ctx = d["GT"], d["context"]
            models = [d[k] for k in MODEL_KEYS]
            T = min([len(gt)] + [len(m) for m in models])
            nctx = len(ctx)
            ctx_cells = [resize(f) for f in ctx]  # each context view resized to a cell
            frames = []
            for t in range(T):
                ci = min(nctx - 1, int(t / T * nctx))  # slideshow synced to clip length
                cells = [ctx_cells[ci], resize(gt[t])] + [resize(m[t]) for m in models]
                frames.append(np.concatenate(cells, axis=1))
            arr = np.stack(frames)
            name = f"{split}_{i}_{h[:8]}"
            iio.imwrite(odir / f"{name}.mp4", arr, fps=FPS, codec="libx264")
            pil = [Image.fromarray(f) for f in arr]
            pil[0].save(odir / f"{name}.gif", save_all=True, append_images=pil[1:],
                        duration=int(1000 / FPS), loop=0, disposal=2)
            rows.append(arr)
            # --- text input sidecar (decoupled: caption_action_only) ---
            if cap_map:
                caption = cap_map.get(h)
                if caption:
                    scene_id, traj, text = decoupled_text(caption)
                    rec = {"scene": caption, "scene_id": scene_id, "trajectory": traj,
                           "prompt_key": "caption_action_only", "text_input": text}
                    with (odir / f"{name}.json").open("w") as fp:
                        json.dump(rec, fp, ensure_ascii=False, indent=2)
                    text_index[f"{split}/{name}"] = rec
                else:
                    print(f"[{split}] no caption for {h[:8]} in wandb-summary")
            print(f"[{split}] {name}: {arr.shape}")
        if rows:
            T = min(a.shape[0] for a in rows)
            grid = np.concatenate([a[:T] for a in rows], axis=1)
            iio.imwrite(OUT / f"ALL_{split}_grid.mp4", grid, fps=FPS, codec="libx264")
            print(f"[{split}] grid -> ALL_{split}_grid.mp4 {grid.shape}")
            grid_all.append((split, grid))
    if text_index:
        with (OUT / "text_inputs.json").open("w") as fp:
            json.dump(text_index, fp, ensure_ascii=False, indent=2)
        print(f"[text] wrote {len(text_index)} entries -> {OUT / 'text_inputs.json'}")


if __name__ == "__main__":
    main()
