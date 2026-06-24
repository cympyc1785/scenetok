"""Convert mp4(s) to gif. Accepts files and/or directories (recursed for *.mp4).

  python scripts/mp4_to_gif.py results/viser_generate            # all mp4 under dir
  python scripts/mp4_to_gif.py a.mp4 b.mp4 --fps 10 --overwrite
"""
import argparse
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image


def convert(mp4: Path, fps=None, overwrite=False):
    gif = mp4.with_suffix(".gif")
    if gif.exists() and not overwrite:
        return "skip"
    rd = imageio.get_reader(str(mp4))
    src_fps = fps or rd.get_meta_data().get("fps", 12)
    frames = [Image.fromarray(f) for f in rd]
    rd.close()
    if not frames:
        return "empty"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(src_fps, 1)), loop=0)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="mp4 files and/or dirs (recursed)")
    ap.add_argument("--fps", type=float, default=None, help="override gif fps (default: source fps)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    mp4s = []
    for r in args.roots:
        p = Path(r)
        if p.is_file() and p.suffix == ".mp4":
            mp4s.append(p)
        elif p.is_dir():
            mp4s += sorted(p.rglob("*.mp4"))
    print(f"[mp4_to_gif] {len(mp4s)} mp4 found")
    n_ok = n_skip = n_err = 0
    for m in mp4s:
        try:
            r = convert(m, args.fps, args.overwrite)
            n_ok += r == "ok"; n_skip += r == "skip"
            print(f"  {r}: {m}")
        except Exception as e:
            n_err += 1; print(f"  ERR {m}: {e}")
    print(f"[mp4_to_gif] done: ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
