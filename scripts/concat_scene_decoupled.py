"""Scene-decoupled val visualization with the wohuman INPUT in true model-input form.

[ wohuman(input, 256x448 resize&crop) | whuman(GT, 480x832) | controlnet output ]

- wohuman / whuman come from the dataset video (video/{wo,w}human/<scene_id>/<..>_<traj>_24mm.mp4),
  sliced at the val TARGET frames and processed exactly like the model input
  (`rescale_and_crop`: scale-to-fill LANCZOS + center crop) — context 256x448, target 480x832.
- Scene identity (scene_id + traj) comes from wandb-summary.json media captions (exact metadata).
- Target frame indices are recovered by matching the wandb "Original Video" (whuman GT) frames to
  the raw dataset whuman video; the same indices index the wohuman video (shared camera track).
- controlnet output = wandb "Sampled Video" paired to GT by content.
- text input (caption_action_only) saved as per-scene json + text_inputs.json.
Output: results/cmp_scene_decoupled_val/{standard,unseen}/*.{mp4,gif}
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

REPO = Path(".").resolve()
RUN = REPO / "exp/va-wan-ti2v_scene_decoupled_controlnet_scene_camera_no_lora_unscaledcomp/wandb/run-20260702_232638-exp_va-wan-ti2v_scene_decoupled_controlnet_scene_camera_no_lora_unscaledcomp/files/media"
DROOT = REPO / "DATA/Scene-Decoupled-Video-dataset"
OUT = REPO / "results/cmp_scene_decoupled_val"
FPS = 8
CELL_H, CELL_W = 240, 416
CTX_SHAPE, TGT_SHAPE = (256, 448), (480, 832)


def resize(f):
    return np.asarray(Image.fromarray(f).convert("RGB").resize((CELL_W, CELL_H), Image.BILINEAR))


def resize_crop(f, shape):
    H, W = shape
    h, w = f.shape[:2]
    sf = max(H / h, W / w)
    nh, nw = round(h * sf), round(w * sf)
    a = np.asarray(Image.fromarray(f).convert("RGB").resize((nw, nh), Image.LANCZOS))
    r, c = (nh - H) // 2, (nw - W) // 2
    return a[r:r + H, c:c + W]


def disp(f, shape):
    return resize(resize_crop(f, shape))


def sig(v, k=48):
    idx = np.linspace(0, len(v) - 1, min(8, len(v))).astype(int)
    return np.stack([np.asarray(Image.fromarray(v[i]).convert("L").resize((k, k)), np.float32)
                     for i in idx]).mean(0)


def latest_full(split, prefix):
    d = defaultdict(list)
    for f in glob.glob(f"{RUN}/videos/{split}/{prefix} Video_*.mp4"):
        m = re.search(r"_(\d+)_([0-9a-f]+)\.mp4$", f)
        if m:
            d[int(m.group(1))].append((m.group(2), f))
    full = [w for w, v in d.items() if len(v) == 8]
    w = max(full) if full else max(d, key=lambda k: len(d[k]))
    return sorted(d[w])


def caption_map():
    summ = RUN.parent / "wandb-summary.json"
    out = {}
    d = json.load(open(summ))
    for k, v in d.items():
        if "Original Video" in k and isinstance(v, dict):
            for vid in v.get("videos", []):
                m = re.search(r"_(\d+)_([0-9a-f]+)\.mp4$", vid.get("path", ""))
                if m and vid.get("caption"):
                    out[m.group(2)] = vid["caption"]
    return out


def recover_indices(gt_vid, ref_vid, k=32):
    """Nearest dataset-frame index for each wandb-GT frame (small grayscale L2)."""
    def small(v):
        return np.stack([np.asarray(Image.fromarray(f).convert("L").resize((k, k)), np.float32) for f in v])
    g, r = small(gt_vid), small(ref_vid).reshape(len(ref_vid), -1)
    return [int(((r - g[i].reshape(-1)) ** 2).mean(1).argmin()) for i in range(len(g))]


def dataset_video(cat, scene_id, traj):
    p = DROOT / "video" / cat / scene_id / f"{scene_id}_{traj}_24mm.mp4"
    return iio.imread(p) if p.exists() else None


def load_text(scene_id):
    p = DROOT / "text" / "whuman" / scene_id / "action.json"
    if p.exists():
        d = json.load(open(p))
        return d.get("caption_action_only") or d.get("caption")
    return None


def main():
    caps = caption_map()
    text_index = {}
    for split in ("standard", "unseen"):
        odir = OUT / split
        odir.mkdir(parents=True, exist_ok=True)
        orig = latest_full(split, "Original")
        gt_vids = [iio.imread(p) for _, p in orig]
        gt_sigs = [sig(v) for v in gt_vids]
        samp = latest_full(split, "Sampled")
        s_vids = [iio.imread(p) for _, p in samp]
        s_sigs = [sig(v) for v in s_vids]
        # pair Sampled -> GT (content)
        C = np.array([[((a - b) ** 2).mean() for b in s_sigs] for a in gt_sigs])
        r, c = linear_sum_assignment(C)
        gt2samp = {int(a): int(b) for a, b in zip(r, c)}

        rows = []
        for i, (h, _) in enumerate(orig):
            caption = caps.get(h)
            if not caption:
                print(f"[{split}] no caption for {h[:8]}"); continue
            scene_id, traj = caption.rsplit("_", 1)
            whu = dataset_video("whuman", scene_id, traj)
            who = dataset_video("wohuman", scene_id, traj)
            if whu is None or who is None:
                print(f"[{split}] missing dataset video {caption}"); continue
            gt_wb = gt_vids[i]
            idx = recover_indices(gt_wb, whu)          # target frames into the 81-frame clip
            whu_t, who_t = whu[idx], who[idx]
            out_wb = s_vids[gt2samp[i]]
            T = min(len(whu_t), len(who_t), len(out_wb))
            frames = [np.concatenate(
                [disp(who_t[t], CTX_SHAPE), disp(whu_t[t], TGT_SHAPE), resize(out_wb[t])], axis=1)
                for t in range(T)]
            arr = np.stack(frames)
            name = f"{split}_{scene_id[:24]}_{traj}"
            iio.imwrite(odir / f"{name}.mp4", arr, fps=FPS, codec="libx264")
            pil = [Image.fromarray(f) for f in arr]
            pil[0].save(odir / f"{name}.gif", save_all=True, append_images=pil[1:],
                        duration=int(1000 / FPS), loop=0, disposal=2)
            text = load_text(scene_id)
            rec = {"scene": caption, "scene_id": scene_id, "trajectory": traj,
                   "prompt_key": "caption_action_only", "text_input": text}
            json.dump(rec, (odir / f"{name}.json").open("w"), ensure_ascii=False, indent=2)
            text_index[f"{split}/{name}"] = rec
            rows.append(arr)
            print(f"[{split}] {name}: {arr.shape}")
        if rows:
            T = min(a.shape[0] for a in rows)
            grid = np.concatenate([a[:T] for a in rows], axis=1)
            iio.imwrite(OUT / f"ALL_{split}_grid.mp4", grid, fps=FPS, codec="libx264")
            print(f"[{split}] grid -> ALL_{split}_grid.mp4 {grid.shape}")
    if text_index:
        json.dump(text_index, (OUT / "text_inputs.json").open("w"), ensure_ascii=False, indent=2)
        print(f"[text] {len(text_index)} entries -> text_inputs.json")


if __name__ == "__main__":
    main()
