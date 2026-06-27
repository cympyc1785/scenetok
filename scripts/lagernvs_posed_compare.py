"""LagerNVS unposed vs posed (context cameras provided) — same bundle/target.

general_512 supports both. Renders the original context order twice: cond rays=0
(unposed) and cond rays = real context Plücker rays (posed). Output a
[context | unposed | posed] strip (mp4 + gif).
"""
import json
import os
import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(".").resolve()
SCENE = "a4c20f668ce179db"
BUNDLE = REPO / f"results/context_views_dl3dv_c16_37/{SCENE}.pt"
OUT = REPO / f"results/lagernvs_posed_compare/{SCENE}"
LAGERNVS_PY = "/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/lagernvs/bin/python"
LAGERNVS_REPO = REPO / "submodules/lagernvs"
LAGERNVS_CKPT = LAGERNVS_REPO / "checkpoints/lagernvs_general_512/model.pt"
FPS = 15
CELL_H, CELL_W = 288, 512


def resize(f):
    return f if f.shape[:2] == (CELL_H, CELL_W) else np.asarray(
        Image.fromarray(f).resize((CELL_W, CELL_H), Image.BILINEAR))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bnd = torch.load(BUNDLE, map_location="cpu")
    imgs = np.asarray(bnd["images"])
    img_dir = OUT / "context_images"; img_dir.mkdir(exist_ok=True)
    img_paths = []
    for i in range(imgs.shape[0]):
        p = img_dir / f"ctx_{i:03d}.png"
        Image.fromarray(imgs[i].transpose(1, 2, 0).astype("uint8")).save(p)
        img_paths.append(str(p))
    ctx_c2w = bnd["c2w"].float()
    tgt_c2w = bnd["target_c2w"].float()
    tgt_K = bnd.get("target_intrinsics", bnd["intrinsics"]).float()[0]
    tgt_K = tgt_K.unsqueeze(0).repeat(tgt_c2w.shape[0], 1, 1)
    ctx_K = bnd["intrinsics"].float()  # (16,3,3) normalized — context intrinsics

    cmd = [LAGERNVS_PY, str(REPO / "scripts/visualize/lagernvs_infer.py"),
           "--repo", str(LAGERNVS_REPO), "--ckpt", str(LAGERNVS_CKPT), "--target_size", "512"]
    proc = subprocess.Popen(cmd, cwd=str(LAGERNVS_REPO), env=dict(os.environ),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                            text=True, bufsize=1)
    for line in proc.stdout:
        if line.strip() == "READY":
            break

    out = {}
    for name, posed in [("unposed", False), ("posed", True)]:
        payload = OUT / f"payload_{name}.pt"
        torch.save({"context_image_paths": img_paths, "context_c2w": ctx_c2w,
                    "target_c2w": tgt_c2w, "target_intrinsics_norm": tgt_K,
                    "context_intrinsics_norm": ctx_K, "posed": posed, "scene": SCENE}, payload)
        fo = OUT / f"frames_{name}.pt"
        proc.stdin.write(json.dumps({"payload": str(payload), "frames_out": str(fo)}) + "\n")
        proc.stdin.flush()
        reply = None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("DONE\t") or line.startswith("ERR\t"):
                reply = line; break
        if reply is None or reply.startswith("ERR"):
            proc.stdin.write("QUIT\n"); proc.stdin.flush()
            raise RuntimeError(f"{name}: {reply}")
        v = torch.load(fo, map_location="cpu").clamp(0, 1).numpy()
        out[name] = (np.transpose(v, (0, 2, 3, 1)) * 255).astype("uint8")
        print(f"[{name}] {out[name].shape}")
    proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=10)

    T = min(out["unposed"].shape[0], out["posed"].shape[0])
    n = imgs.shape[0]
    frames = []
    for t in range(T):
        ctx = resize(imgs[min(t * n // T, n - 1)].transpose(1, 2, 0).astype("uint8"))
        frames.append(np.concatenate([ctx, resize(out["unposed"][t]), resize(out["posed"][t])], axis=1))
    frames = np.stack(frames)
    mp4 = OUT / "cmp_unposed_vs_posed.mp4"; gif = OUT / "cmp_unposed_vs_posed.gif"
    iio.imwrite(mp4, frames, fps=FPS, codec="libx264")
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=int(1000 / FPS), loop=0, disposal=2)
    # quantify difference
    d = float(np.abs(out["unposed"][:T].astype(int) - out["posed"][:T].astype(int)).mean())
    print(f"[posed-compare] cols=[context|unposed|posed] {tuple(frames.shape)} unposed↔posed |diff|={d:.2f}")
    print(f"  -> {mp4}, {gif}")


if __name__ == "__main__":
    main()
