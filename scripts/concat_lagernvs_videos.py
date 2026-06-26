"""Side-by-side concat of LagerNVS per-camera-sequence renders into comparison
videos (+ GIFs). Each requested combo → one horizontal strip with a label per
clip. Clips share frame count (37) and size (288x512), so a plain hcat works.

Output: results/viser_generate/lagernvs_general_512/_concat/<name>.{mp4,gif}
"""
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("results/viser_generate/lagernvs_general_512")
OUT = ROOT / "_concat"
PREFIX = "a4c20f668ce179db_0626_"
FPS = 15

# (output name, [folder suffixes in order])
COMBOS = [
    ("1_orig-back-back_lot",                 ["orig", "back", "back_lot"]),
    ("2_orig-forward-forward_lot",           ["orig", "forward", "forward_lot"]),
    ("3_orig-rotate_left-rotate_right",      ["orig", "rotate_left", "rotate_right"]),
    ("4_move_forward-back-left-right",        ["move_forward", "move_back", "move_left", "move_right"]),
    ("5_rot_up-down-left-right",              ["rot_up", "rot_down", "rot_left", "rot_right"]),
]


def load_clip(suffix):
    p = ROOT / f"{PREFIX}{suffix}" / "generated.mp4"
    if not p.exists():
        raise FileNotFoundError(p)
    return iio.imread(p)  # (T, H, W, 3) uint8


def label(frame, text):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    # readability: dark box behind text
    tb = d.textbbox((0, 0), text, font=font)
    d.rectangle([6, 6, 6 + (tb[2] - tb[0]) + 10, 6 + (tb[3] - tb[1]) + 10], fill=(0, 0, 0))
    d.text((11, 9), text, fill=(255, 255, 0), font=font)
    return np.asarray(img)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, suffixes in COMBOS:
        clips = [load_clip(s) for s in suffixes]
        T = min(c.shape[0] for c in clips)
        clips = [c[:T] for c in clips]
        # label each clip's frames, then hcat horizontally
        out_frames = []
        for t in range(T):
            row = [label(clips[i][t], suffixes[i]) for i in range(len(clips))]
            out_frames.append(np.concatenate(row, axis=1))  # hcat along width
        out_frames = np.stack(out_frames)  # (T, H, W*n, 3)
        mp4 = OUT / f"{name}.mp4"
        gif = OUT / f"{name}.gif"
        iio.imwrite(mp4, out_frames, fps=FPS, codec="libx264")
        # gif via PIL (loop forever)
        imgs = [Image.fromarray(f) for f in out_frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, disposal=2)
        print(f"[concat] {name}: {len(suffixes)} clips × {T}f → {tuple(out_frames.shape)}  {mp4.name}, {gif.name}")


if __name__ == "__main__":
    main()
