"""2-row model-comparison grids: SceneTok (top) vs LagerNVS (bottom) for each
camera-sequence combo. Columns = the combo's sequences (matched by folder
suffix). Each cell is resized to a common size (SceneTok 256x448 vs LagerNVS
288x512 differ), then hcat per row and vcat the two rows → 2xN grid. No text
overlay (same clean format as concat_lagernvs_videos.py).

Output: results/viser_generate/_compare_scenetok_lagernvs/<name>.{mp4,gif}
"""
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

SCENETOK_ROOT = Path("results/viser_generate/va-wan_dl3dv_256x448")
SCENETOK_PREFIX = "a4c20f668ce179db_0624_"
LAGERNVS_ROOT = Path("results/viser_generate/lagernvs_general_512")
LAGERNVS_PREFIX = "a4c20f668ce179db_0626_"
OUT = Path("results/viser_generate/_compare_scenetok_lagernvs")
FPS = 15
CELL_H, CELL_W = 256, 448  # common cell size (SceneTok native; LagerNVS downscaled)

# (output name, [folder suffixes = columns])
COMBOS = [
    ("cmp_1_orig-back-back_lot",            ["orig", "back", "back_lot"]),
    ("cmp_2_orig-forward-forward_lot",      ["orig", "forward", "forward_lot"]),
    ("cmp_3_orig-rotate_left-rotate_right", ["orig", "rotate_left", "rotate_right"]),
    ("cmp_4_move_forward-back-left-right",   ["move_forward", "move_back", "move_left", "move_right"]),
    ("cmp_5_rot_up-down-left-right",         ["rot_up", "rot_down", "rot_left", "rot_right"]),
]


def load_clip(root, prefix, suffix):
    p = root / f"{prefix}{suffix}" / "generated.mp4"
    if not p.exists():
        raise FileNotFoundError(p)
    return iio.imread(p)  # (T, H, W, 3) uint8


def resize_frame(frame):
    if frame.shape[0] == CELL_H and frame.shape[1] == CELL_W:
        return frame
    return np.asarray(Image.fromarray(frame).resize((CELL_W, CELL_H), Image.BILINEAR))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, suffixes in COMBOS:
        top = [load_clip(SCENETOK_ROOT, SCENETOK_PREFIX, s) for s in suffixes]
        bot = [load_clip(LAGERNVS_ROOT, LAGERNVS_PREFIX, s) for s in suffixes]
        T = min(c.shape[0] for c in top + bot)
        out_frames = []
        for t in range(T):
            top_row = np.concatenate([resize_frame(c[t]) for c in top], axis=1)
            bot_row = np.concatenate([resize_frame(c[t]) for c in bot], axis=1)
            out_frames.append(np.concatenate([top_row, bot_row], axis=0))  # vcat rows
        out_frames = np.stack(out_frames)  # (T, 2*CELL_H, n*CELL_W, 3)
        mp4 = OUT / f"{name}.mp4"
        gif = OUT / f"{name}.gif"
        iio.imwrite(mp4, out_frames, fps=FPS, codec="libx264")
        imgs = [Image.fromarray(f) for f in out_frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, disposal=2)
        print(f"[compare] {name}: 2x{len(suffixes)} × {T}f → {tuple(out_frames.shape)}  "
              f"{mp4.name}, {gif.name}")


if __name__ == "__main__":
    main()
