"""Context-view ORDER invariance test for LagerNVS, then a 2x4 grid vs SceneTok.

Columns: [context views | original | reverse | shuffle]. Rows: top=SceneTok
(reuses the existing ctx_order_va-wan_dl3dv_256x448_std renders), bottom=LagerNVS
(rendered here). For LagerNVS we keep target + context_c2w FIXED and permute ONLY
the context IMAGE order — so target rays/anchor/scene-scale are identical across
original/reverse/shuffle and the only change is the order (and first=reference)
of conditioning images. cond rays are 0 so context_c2w order is irrelevant to the
model anyway. Drives the resident lagernvs_infer.py worker (model loaded once).

Output: results/ctx_order_compare_scenetok_lagernvs/<scene>/cmp_2x4.{mp4,gif}
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(".").resolve()
SCENE = "a4c20f668ce179db"
BUNDLE = REPO / f"results/context_views_dl3dv_c16_37/{SCENE}.pt"
ST_DIR = REPO / f"results/ctx_order_va-wan_dl3dv_256x448_std/{SCENE}"   # existing SceneTok renders
LAGERNVS_PY = "/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/lagernvs/bin/python"
LAGERNVS_REPO = REPO / "submodules/lagernvs"
LAGERNVS_CKPT = LAGERNVS_REPO / "checkpoints/lagernvs_general_512/model.pt"
OUT = REPO / f"results/ctx_order_compare_scenetok_lagernvs/{SCENE}"
FPS = 15
CELL_H, CELL_W = 256, 448
SHUFFLE_SEED = 0


def make_orders(n):
    rng = np.random.RandomState(SHUFFLE_SEED)
    perm = rng.permutation(n).tolist()
    return {"original": list(range(n)), "reverse": list(range(n))[::-1], "shuffle": perm}


def resize(frame):
    if frame.shape[:2] == (CELL_H, CELL_W):
        return frame
    return np.asarray(Image.fromarray(frame).resize((CELL_W, CELL_H), Image.BILINEAR))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img_dir = OUT / "context_images"; img_dir.mkdir(exist_ok=True)
    b = torch.load(BUNDLE, map_location="cpu")
    imgs = np.asarray(b["images"])                    # (16,3,H,W) uint8
    ctx_c2w = b["c2w"].float()                         # (16,4,4) FIXED
    tgt_c2w = b["target_c2w"].float()                  # (37,4,4) FIXED
    tk = b.get("target_intrinsics", b["intrinsics"]).float()[0]
    tgt_K = tk.unsqueeze(0).repeat(tgt_c2w.shape[0], 1, 1)

    img_paths = []
    for i in range(imgs.shape[0]):
        p = img_dir / f"ctx_{i:03d}.png"
        Image.fromarray(imgs[i].transpose(1, 2, 0).astype("uint8")).save(p)
        img_paths.append(str(p))

    orders = make_orders(len(img_paths))
    print("[ctx-order] shuffle perm:", orders["shuffle"])

    # Start the resident LagerNVS worker once, render the 3 orders.
    cmd = [LAGERNVS_PY, str(REPO / "scripts/visualize/lagernvs_infer.py"),
           "--repo", str(LAGERNVS_REPO), "--ckpt", str(LAGERNVS_CKPT), "--target_size", "512"]
    proc = subprocess.Popen(cmd, cwd=str(LAGERNVS_REPO), env=dict(os.environ),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                            text=True, bufsize=1)
    for line in proc.stdout:
        if line.strip() == "READY":
            break

    lg = {}
    for name, order in orders.items():
        # permute ONLY images; context_c2w + target fixed (anchor/scale identical).
        payload = OUT / f"payload_{name}.pt"
        torch.save({"context_image_paths": [img_paths[i] for i in order],
                    "context_c2w": ctx_c2w, "target_c2w": tgt_c2w,
                    "target_intrinsics_norm": tgt_K, "scene": SCENE}, payload)
        fo = OUT / f"lagernvs_{name}.pt"
        proc.stdin.write(json.dumps({"payload": str(payload), "frames_out": str(fo)}) + "\n")
        proc.stdin.flush()
        reply = None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("DONE\t") or line.startswith("ERR\t"):
                reply = line; break
        if reply is None or reply.startswith("ERR"):
            proc.stdin.write("QUIT\n"); proc.stdin.flush()
            raise RuntimeError(f"lagernvs {name} failed: {reply}")
        lg[name] = (torch.load(fo, map_location="cpu").clamp(0, 1).numpy() * 255).astype("uint8")
        lg[name] = np.transpose(lg[name], (0, 2, 3, 1))  # (T,3,H,W)->(T,H,W,3)
        print(f"[ctx-order] lagernvs {name}: {lg[name].shape}")
    proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=10)

    # SceneTok existing renders (top row).
    st = {n: iio.imread(ST_DIR / f"{n}.mp4") for n in ["context", "original", "reverse", "shuffle"]}
    ctx_vid = iio.imread(ST_DIR / "context.mp4")  # shared context-views column

    cols = ["context", "original", "reverse", "shuffle"]
    T = min([st[c].shape[0] for c in cols] + [lg[c].shape[0] for c in cols[1:]] + [ctx_vid.shape[0]])
    out_frames = []
    for t in range(T):
        top = np.concatenate([resize(st[c][t]) for c in cols], axis=1)
        bot = np.concatenate([resize(ctx_vid[t])] + [resize(lg[c][t]) for c in cols[1:]], axis=1)
        out_frames.append(np.concatenate([top, bot], axis=0))
    out_frames = np.stack(out_frames)
    mp4 = OUT / "cmp_2x4.mp4"; gif = OUT / "cmp_2x4.gif"
    iio.imwrite(mp4, out_frames, fps=FPS, codec="libx264")
    pil = [Image.fromarray(f) for f in out_frames]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=int(1000 / FPS), loop=0, disposal=2)
    print(f"[ctx-order] 2x4 → {tuple(out_frames.shape)}  {mp4}, {gif}")


if __name__ == "__main__":
    main()
