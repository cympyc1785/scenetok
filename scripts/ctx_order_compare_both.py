"""Re-render BOTH SceneTok and LagerNVS under IDENTICAL conditions for the
context-view ORDER invariance test, then a 2x4 grid.

Same bundle (context images + cameras), same target trajectory (FIXED), same
permutations (original/reverse/shuffle). Anchor = context[0] (index=0) for both.
Only the context FEED order is permuted (target/anchor fixed):
  - SceneTok: permute the (latent, extrinsics, intrinsics, index) context tuple
    AFTER preprocess (target already relativized → fixed; scene tokens are
    permutation-invariant so order shouldn't matter).
  - LagerNVS: permute the context IMAGE order (cond rays=0 → context_c2w/target
    fixed; first image = reference).

Columns: [context | original | reverse | shuffle]. Rows: top=SceneTok, bottom=
LagerNVS. Output: results/ctx_order_compare_both/<scene>/cmp_2x4.{mp4,gif}
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
from scripts.visualize.viser_server import build_model, CKPT_PRESETS  # noqa: E402

SCENE = "a4c20f668ce179db"
BUNDLE = REPO / f"results/context_views_dl3dv_c16_37/{SCENE}.pt"
OUT = REPO / f"results/ctx_order_compare_both/{SCENE}"
ST_CKPT = REPO / "checkpoints/va-wan_dl3dv_256x448.ckpt"
EVAL_INDEX = REPO / "assets/evaluation_index/dl3dv_c16_37_caption_standard.json"
LAGERNVS_PY = "/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/lagernvs/bin/python"
LAGERNVS_REPO = REPO / "submodules/lagernvs"
LAGERNVS_CKPT = LAGERNVS_REPO / "checkpoints/lagernvs_general_512/model.pt"
FPS = 15
CELL_H, CELL_W = 256, 448
SEED = 0


def make_orders(n):
    rng = np.random.RandomState(SEED)
    return {"original": list(range(n)), "reverse": list(range(n))[::-1],
            "shuffle": rng.permutation(n).tolist()}


def resize(f):
    return f if f.shape[:2] == (CELL_H, CELL_W) else np.asarray(
        Image.fromarray(f).resize((CELL_W, CELL_H), Image.BILINEAR))


def to_dev(o, dev):
    if torch.is_tensor(o):
        return o.to(dev)
    if isinstance(o, dict):
        return {k: to_dev(v, dev) for k, v in o.items()}
    if isinstance(o, list):
        return [to_dev(v, dev) for v in o]
    return o


def render_scenetok(orders, bnd):
    from src.misc.batch_utils import preprocess_batch
    args = types.SimpleNamespace(
        model_experiment=None, model_shape=None, model_ckpt=str(ST_CKPT),
        eval_index=str(EVAL_INDEX), infer_steps=25, cfg_scale=1.0, seed=0, device="cuda")
    exp_d, shape_d = CKPT_PRESETS[Path(args.model_ckpt).stem]
    args.model_experiment, args.model_shape = exp_d, shape_d
    wrapper, loader, dev, prec = build_model(args)

    batch = None
    for b in loader:
        if b is None:
            continue
        nm = b["scene"][0] if isinstance(b.get("scene"), (list, tuple)) else b.get("scene")
        if SCENE in str(nm):
            batch = b
            break
    assert batch is not None, "scene not found in eval loader"

    tgt_rel = bnd["target_c2w"].float()
    tk0 = bnd.get("target_intrinsics", bnd["intrinsics"]).float()[0]
    out = {}
    for name, order in orders.items():
        b = to_dev(batch, dev)
        ctx_ext = b["context"]["extrinsics"]
        ctx_lat = b["context"]["latent"]
        ctx0 = ctx_ext[0, 0]
        T = tgt_rel.shape[0]
        tgt_abs = (ctx0.unsqueeze(0) @ tgt_rel.to(dev)).unsqueeze(0)
        tgt_int = tk0.to(dev).unsqueeze(0).repeat(T, 1, 1).unsqueeze(0)
        b["target"] = {
            "extrinsics": tgt_abs, "intrinsics": tgt_int,
            "latent": torch.zeros((1, T, ctx_lat.shape[2], ctx_lat.shape[3], ctx_lat.shape[4]),
                                  device=dev, dtype=ctx_lat.dtype),
            "index": torch.arange(T, device=dev).unsqueeze(0)}
        b = preprocess_batch(b, index=0)
        idx = torch.tensor(order, device=dev)
        for k in ("latent", "extrinsics", "intrinsics", "index"):
            if k in b["context"]:
                b["context"][k] = b["context"][k][:, idx]
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=prec,
                                                 enabled=(prec != torch.float32)):
            sampled, _, _ = wrapper.generate_batch_with_scene(b, wrapper.sampler, repeat_factor=1)
        s = sampled.float().clamp(0, 1)[0]  # (T,3,H,W)
        out[name] = (s.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
        print(f"[scenetok] {name}: {out[name].shape}")
    del wrapper
    torch.cuda.empty_cache()
    return out


def render_lagernvs(orders, bnd):
    imgs = np.asarray(bnd["images"])
    img_dir = OUT / "context_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_paths = []
    for i in range(imgs.shape[0]):
        p = img_dir / f"ctx_{i:03d}.png"
        Image.fromarray(imgs[i].transpose(1, 2, 0).astype("uint8")).save(p)
        img_paths.append(str(p))
    ctx_c2w = bnd["c2w"].float()
    tgt_c2w = bnd["target_c2w"].float()
    tk = bnd.get("target_intrinsics", bnd["intrinsics"]).float()[0]
    tgt_K = tk.unsqueeze(0).repeat(tgt_c2w.shape[0], 1, 1)

    cmd = [LAGERNVS_PY, str(REPO / "scripts/visualize/lagernvs_infer.py"),
           "--repo", str(LAGERNVS_REPO), "--ckpt", str(LAGERNVS_CKPT), "--target_size", "512"]
    proc = subprocess.Popen(cmd, cwd=str(LAGERNVS_REPO), env=dict(os.environ),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                            text=True, bufsize=1)
    for line in proc.stdout:
        if line.strip() == "READY":
            break
    out = {}
    for name, order in orders.items():
        payload = OUT / f"lg_payload_{name}.pt"
        torch.save({"context_image_paths": [img_paths[i] for i in order],
                    "context_c2w": ctx_c2w, "target_c2w": tgt_c2w,
                    "target_intrinsics_norm": tgt_K, "scene": SCENE}, payload)
        fo = OUT / f"lg_{name}.pt"
        proc.stdin.write(json.dumps({"payload": str(payload), "frames_out": str(fo)}) + "\n")
        proc.stdin.flush()
        reply = None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("DONE\t") or line.startswith("ERR\t"):
                reply = line
                break
        if reply is None or reply.startswith("ERR"):
            proc.stdin.write("QUIT\n"); proc.stdin.flush()
            raise RuntimeError(f"lagernvs {name}: {reply}")
        v = torch.load(fo, map_location="cpu").clamp(0, 1).numpy()
        out[name] = (np.transpose(v, (0, 2, 3, 1)) * 255).astype("uint8")
        print(f"[lagernvs] {name}: {out[name].shape}")
    proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=10)
    return out


def context_slideshow(bnd, T):
    imgs = np.asarray(bnd["images"])  # (16,3,H,W)
    n = imgs.shape[0]
    return np.stack([resize(imgs[min(t * n // T, n - 1)].transpose(1, 2, 0).astype("uint8"))
                     for t in range(T)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bnd = torch.load(BUNDLE, map_location="cpu")
    orders = make_orders(int(bnd["c2w"].shape[0]))
    print("shuffle perm:", orders["shuffle"])
    st = render_scenetok(orders, bnd)
    lg = render_lagernvs(orders, bnd)

    cols = ["original", "reverse", "shuffle"]
    T = min([st[c].shape[0] for c in cols] + [lg[c].shape[0] for c in cols])
    ctx = context_slideshow(bnd, T)
    frames = []
    for t in range(T):
        top = np.concatenate([ctx[t]] + [resize(st[c][t]) for c in cols], axis=1)
        bot = np.concatenate([ctx[t]] + [resize(lg[c][t]) for c in cols], axis=1)
        frames.append(np.concatenate([top, bot], axis=0))
    frames = np.stack(frames)
    mp4 = OUT / "cmp_2x4.mp4"
    gif = OUT / "cmp_2x4.gif"
    iio.imwrite(mp4, frames, fps=FPS, codec="libx264")
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=int(1000 / FPS), loop=0, disposal=2)
    print(f"[2x4] {tuple(frames.shape)} → {mp4}, {gif}")


if __name__ == "__main__":
    main()
