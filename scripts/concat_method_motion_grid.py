"""Build a rows(methods) x cols(motions) comparison grid GIF from viser_generate
outputs. Each cell = <method_dir>/*_<motion>/(generated.gif|render.gif)."""
import argparse
import glob
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw

ROOT = Path("results/viser_generate")


def load_gif(method, motion, cell):
    cand = sorted(glob.glob(str(ROOT / method / f"*_{motion}")))
    if not cand:
        raise FileNotFoundError(f"{method}/*_{motion}")
    d = Path(cand[0])
    gif = d / "generated.gif"
    if not gif.exists():
        gif = d / "render.gif"
    frames = iio.imread(str(gif))  # (T,H,W,C) or (T,H,W,4)
    out = []
    for f in frames:
        im = Image.fromarray(f[..., :3]).resize((cell, cell))
        out.append(np.asarray(im))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["va-wan_dl3dv", "lagernvs_general_512", "lagernvs_dl3dv_2-6_v_256_256x256"])
    ap.add_argument("--motions", nargs="+",
                    default=["move_forward", "move_back", "move_left", "move_right"])
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="results/viser_generate/_grid_move_3x4.gif")
    ap.add_argument("--no_label", action="store_true")
    args = ap.parse_args()

    grid = {}
    T = None
    for r in args.methods:
        for c in args.motions:
            v = load_gif(r, c, args.cell)
            grid[(r, c)] = v
            T = v.shape[0] if T is None else min(T, v.shape[0])
    print(f"[grid] {len(args.methods)}x{len(args.motions)}, frames={T}, cell={args.cell}")

    lab = args.cell // 12
    frames_out = []
    for t in range(T):
        rows = []
        for r in args.methods:
            cells = []
            for c in args.motions:
                cell = grid[(r, c)][t].copy()
                if not args.no_label:
                    im = Image.fromarray(cell)
                    dr = ImageDraw.Draw(im)
                    dr.rectangle([0, 0, args.cell, 14], fill=(0, 0, 0))
                    dr.text((3, 2), f"{r.split('_')[0][:10]}|{c.replace('move_','')}", fill=(255, 255, 255))
                    cell = np.asarray(im)
                cells.append(cell)
            rows.append(np.concatenate(cells, axis=1))
        frames_out.append(np.concatenate(rows, axis=0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(f) for f in frames_out]
    pil[0].save(args.out, save_all=True, append_images=pil[1:],
                duration=int(1000 / max(args.fps, 1)), loop=0)
    print(f"[grid] saved {args.out}  shape={frames_out[0].shape}")


if __name__ == "__main__":
    main()
