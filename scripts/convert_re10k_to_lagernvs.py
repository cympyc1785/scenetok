"""Convert a SUBSET of our RE10K (.torch pixelsplat chunks) into the on-disk
format LagerNVS's original Re10kDataset expects, so lagernvs_orig can train on a
mixed dl3dv+re10k set.

LagerNVS re10k layout (under $LAGERNVS_DATA_ROOT/re10k/<split>/):
  full_list.txt              # one seq id per line
  images/<seq>/frame_*.png   # RGB frames
  metadata/<seq>.json        # {"frames": [{"fxfycxcy":[px], "w2c":[4x4]}, ...]}

Our .torch example: {"key", "images": [jpeg bytes tensor], "cameras": (N,18)}
cameras row = [fx,fy,cx,cy (NORMALIZED 0-1), 0, 0, w2c 3x4 (12)].
LagerNVS adjust_intrinsics treats fxfycxcy as PIXELS at im_hw_orig, so we
denormalize: fx_px = fx_norm*W, fy_px = fy_norm*H, cx_px=cx_norm*W, cy_px=cy_norm*H.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "DATA/re10k/re10k"
DST_ROOT = REPO / "submodules/lagernvs/data_mixed/re10k"
MIN_FRAMES = 25  # LagerNVS re10k view_sampler_range starts at 25


def convert_split(split, n_scenes, start=0):
    index = json.load(open(SRC / split / "index.json"))
    keys = list(index.keys())[start:start + n_scenes * 3]  # over-scan; some get skipped
    img_root = DST_ROOT / split / "images"
    meta_root = DST_ROOT / split / "metadata"
    img_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    done = []
    cache = {}  # chunk file -> loaded list
    for key in keys:
        if len(done) >= n_scenes:
            break
        chunk_file = index[key]
        if chunk_file not in cache:
            cache = {chunk_file: torch.load(SRC / split / chunk_file, weights_only=False)}
        ex = next((x for x in cache[chunk_file] if x["key"] == key), None)
        if ex is None or len(ex["images"]) < MIN_FRAMES:
            continue

        seq_img_dir = img_root / key
        seq_img_dir.mkdir(parents=True, exist_ok=True)
        cams = ex["cameras"]  # (N, 18)
        frames = []
        for i, raw in enumerate(ex["images"]):
            img = Image.open(io.BytesIO(raw.numpy().tobytes())).convert("RGB")
            W, H = img.size
            img.save(seq_img_dir / f"frame_{i:05d}.png")
            fx, fy, cx, cy = [float(v) for v in cams[i][:4]]
            w2c34 = np.array(cams[i][6:18]).reshape(3, 4).astype(np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :4] = w2c34
            frames.append({
                "fxfycxcy": [fx * W, fy * H, cx * W, cy * H],  # normalized -> pixels
                "w2c": w2c.tolist(),
            })
        json.dump({"frames": frames}, open(meta_root / f"{key}.json", "w"))
        done.append(key)
        if len(done) % 50 == 0:
            print(f"[{split}] {len(done)}/{n_scenes}")

    with open(DST_ROOT / split / "full_list.txt", "w") as f:
        f.write("\n".join(done) + "\n")
    print(f"[{split}] DONE: {len(done)} scenes -> {DST_ROOT/split}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_scenes", type=int, default=1000)
    ap.add_argument("--test_scenes", type=int, default=200)
    args = ap.parse_args()
    convert_split("train", args.train_scenes)
    convert_split("test", args.test_scenes)

    # symlink existing dl3dv adapter into the same mixed root so LAGERNVS_DATA_ROOT
    # can serve both dl3dv/ and re10k/.
    dl_src = REPO / "submodules/lagernvs/data_dl3dv_smallset/dl3dv"
    dl_dst = DST_ROOT.parent / "dl3dv"
    if dl_src.exists() and not dl_dst.exists():
        dl_dst.symlink_to(dl_src.resolve())
        print(f"[link] {dl_dst} -> {dl_src}")


if __name__ == "__main__":
    main()
