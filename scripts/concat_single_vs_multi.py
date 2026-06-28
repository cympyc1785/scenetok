"""4-way comparison concat for dynamicverse-unseen scenes:
[ inpaint (bg) | GT (video_input target) | single(dynamicverse) | multi ].

- inpaint: inpaint_result.mp4 frames at the scene's TARGET indices (aligned with
  GT/generated; bg-only).
- GT: target_gt.mp4 from fast_infer (decoded target latent).
- single / multi: fast_infer full_sequence.mp4 (cfg1.0_text-dataset) of each ckpt.

Per-scene hcat (mp4 + gif) + a stacked grid over all scenes.
Output: results/cmp_single_vs_multi/_concat/
"""
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

REPO = Path(".").resolve()
ROOT = REPO / "results/cmp_single_vs_multi"
OUT = ROOT / "_concat"
DAVIS = REPO / "WorldTraj/dynamicverse/DAVIS"
EVAL_IDX = json.load(open(REPO / "assets/evaluation_index/dynamicverse_unseen_8.json"))
COMBO = "cfg1.0_text-dataset"
FPS = 10
CELL_H, CELL_W = 256, 448  # common cell (resize all)
COLS = ["inpaint", "GT", "single", "multi"]


def vid(path):
    return iio.imread(path)  # (T,H,W,3) uint8


def resize_frame(f):
    if f.shape[:2] == (CELL_H, CELL_W):
        return f
    return np.asarray(Image.fromarray(f).resize((CELL_W, CELL_H), Image.BILINEAR))


def label(frame, text):
    img = Image.fromarray(frame.copy())
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, 2 + 8 * len(text) + 8, 20], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 0))
    return np.asarray(img)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = list(EVAL_IDX.keys())
    grid_rows = []
    for sc in scenes:
        tgt_idx = EVAL_IDX[sc]["target"]
        try:
            inp_full = vid(DAVIS / sc / "inpaint_result.mp4")
            inpaint = inp_full[[i for i in tgt_idx if i < len(inp_full)]]
            gt = vid(ROOT / "single" / sc / "input/target/target_gt.mp4")
            single = vid(ROOT / "single" / sc / COMBO / "full_sequence.mp4")
            multi = vid(ROOT / "multi" / sc / COMBO / "full_sequence.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"[concat] skip {sc}: {e}")
            continue
        T = min(len(inpaint), len(gt), len(single), len(multi))
        srcs = {"inpaint": inpaint, "GT": gt, "single": single, "multi": multi}
        frames = []
        for t in range(T):
            frames.append(np.concatenate(
                [label(resize_frame(srcs[c][t]), c) for c in COLS], axis=1))
        arr = np.stack(frames)  # (T, H, 4W, 3)
        iio.imwrite(OUT / f"{sc}.mp4", arr, fps=FPS, codec="libx264")
        pil = [Image.fromarray(f) for f in arr]
        pil[0].save(OUT / f"{sc}.gif", save_all=True, append_images=pil[1:],
                    duration=int(1000 / FPS), loop=0, disposal=2)
        grid_rows.append(arr)
        print(f"[concat] {sc}: {arr.shape}")

    if grid_rows:
        T = min(a.shape[0] for a in grid_rows)
        grid = np.concatenate([a[:T] for a in grid_rows], axis=1)  # stack scenes vertically
        iio.imwrite(OUT / "ALL_scenes_grid.mp4", grid, fps=FPS, codec="libx264")
        print(f"[concat] grid: {grid.shape} -> ALL_scenes_grid.mp4")


if __name__ == "__main__":
    main()
