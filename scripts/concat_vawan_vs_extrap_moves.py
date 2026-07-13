"""2x5 concat: viser poses.pt move trajectories rendered by
  TOP    = original va-wan_dl3dv  (results/viser_generate/va-wan_dl3dv)
  BOTTOM = extrap g1020           (results/viser_generate/scenetok_mvB1_dl3dv_extrap_g1020)
Cols = orig, move_forward, move_back, move_left, move_right.
Both rows are pre-rendered generated.mp4; trim to common frame count.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "results/viser_generate/va-wan_dl3dv_25step"
BOT = REPO / "results/viser_generate/scenetok_mvB1_dl3dv_extrap_g1020_25step"
OUT = REPO / "results/cmp_va-wan_vs_extrap_g1020_25step"
COLS = ["orig", "move_forward", "move_back", "move_left", "move_right"]
CELL = (256, 256)
FPS = 15


def read_mp4(p):
    return np.stack([np.asarray(f)[..., :3] for f in imageio.mimread(str(p), memtest=False)])


def resize_clip(clip, hw):
    from PIL import Image
    h, w = hw
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in clip])


def main():
    scenes = sorted({d.name[:-len("_orig")] for d in TOP.glob("*_orig") if d.is_dir()})
    print(f"scenes: {scenes}")
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        top_row, bot_row = [], []
        ok = True
        for c in COLS:
            tp = TOP / f"{scene}_{c}" / "generated.mp4"
            bp = BOT / f"{scene}_{c}" / "generated.mp4"
            if not tp.exists() or not bp.exists():
                print(f"SKIP col {c}: missing {tp if not tp.exists() else bp}"); ok = False; break
            t = resize_clip(read_mp4(tp), CELL)
            b = resize_clip(read_mp4(bp), CELL)
            T = min(len(t), len(b))
            top_row.append(t[:T]); bot_row.append(b[:T])
        if not ok:
            continue
        T = min(min(len(x) for x in top_row), min(len(x) for x in bot_row))
        top_row = [x[:T] for x in top_row]; bot_row = [x[:T] for x in bot_row]
        grid = np.concatenate([np.concatenate(top_row, axis=2),
                               np.concatenate(bot_row, axis=2)], axis=1)
        sdir = OUT / scene; sdir.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(sdir / "grid_2x5.gif", list(grid), fps=FPS, loop=0)
        imageio.mimwrite(sdir / "grid_2x5.mp4", list(grid), fps=FPS, quality=8)
        print(f"wrote {sdir}/grid_2x5.gif  {grid.shape}  (top=va-wan_dl3dv, bottom=extrap g1020, T={T})")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
