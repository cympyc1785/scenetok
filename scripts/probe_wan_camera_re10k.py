"""AC3D-faithful camera linear probing on Wan2.2 TI2V-5B, using RealEstate10K.

Mirrors AC3D (arXiv:2411.18673) §camera probing as closely as possible on a
different base model (Wan2.2 TI2V-5B instead of their internal 11.5B VDiT):
  - dataset: RealEstate10K (pixelSplat .torch format), 49-frame clips
  - per (DiT block × noise level): GLOBAL-pool the block activation to one vector
    per video, then linear ridge regression to the ENTIRE trajectory target
    (num_frames × 6 = rotation Euler pitch/yaw/roll + translation), AC3D-style.
  - report rotation (geodesic deg) + translation (L2) test error + raw-target MSE
    (AC3D scale ≈0.025 rot / ≈0.48 trans) vs predict-train-mean baseline.
  - 3-split (train/val/test), λ tuned on val, reported on test (no leakage).

This isolates the MODEL difference from the dataset/target-formulation
differences vs scripts/probe_wan_camera_dl3dv.py.

Usage:
  python scripts/probe_wan_camera_re10k.py --num_scenes 500 --num_frames 49 \
      --noise_levels 0.3,0.5,0.7,0.9,0.96 --layers all --out results/probe_wan_re10k
"""
from __future__ import annotations
import argparse, glob, io, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/model/DiffSynth-Studio"))

from diffsynth.models.model_loader import ModelPool
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.pipelines.wan_video import model_fn_wan_video

MODEL_ROOT = ROOT / "src/model/DiffSynth-Studio/Wan2.2/Wan2.2-TI2V-5B"


# ───────────────── pose helpers (Euler XYZ, AC3D pitch/yaw/roll) ─────────────────
def rotmat_to_euler(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,3) XYZ Euler (radians)."""
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    x = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    y = np.arctan2(-R[:, 2, 0], sy)
    z = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([x, y, z], axis=-1)


def euler_to_rotmat(e: torch.Tensor) -> torch.Tensor:
    """(N,3) XYZ Euler → (N,3,3) = Rz@Ry@Rx."""
    x, y, z = e[:, 0], e[:, 1], e[:, 2]
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    R = torch.stack([
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy,     cy * sx,                cy * cx,
    ], dim=-1).reshape(-1, 3, 3)
    return R


def geodesic_deg(Rp: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
    m = torch.matmul(Rp.transpose(-1, -2), Rg)
    tr = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def re10k_scene_to_c2w(cameras: torch.Tensor) -> np.ndarray:
    """pixelSplat cameras (N,18): [fx,fy,cx,cy,0,0, 3x4 w2c] → c2w (N,4,4)."""
    ext = cameras[:, 6:18].reshape(-1, 3, 4).cpu().numpy().astype(np.float64)
    N = ext.shape[0]
    w2c = np.tile(np.eye(4), (N, 1, 1))
    w2c[:, :3, :4] = ext
    return np.linalg.inv(w2c)


def decode_re10k_image(t: torch.Tensor) -> Image.Image:
    return Image.open(io.BytesIO(t.numpy().tobytes())).convert("RGB")


def load_re10k_scene(scene: dict, num_frames: int, hw: tuple[int, int]):
    """Return (frames (V,3,H,W) in [-1,1], c2w (V,4,4)) or None if too short."""
    cams = scene["cameras"]
    n = cams.shape[0]
    if n < num_frames:
        return None
    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    H, W = hw
    c2w = re10k_scene_to_c2w(cams)[idx]
    imgs = []
    for i in idx:
        im = decode_re10k_image(scene["images"][i]).resize((W, H), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.array(im)).float() / 255.0 * 2 - 1)
    frames = torch.stack(imgs).permute(0, 3, 1, 2)
    return frames, c2w


def trajectory_target(c2w: np.ndarray):
    """c2w (F,4,4) → (euler (F,3) rel frame0, trans (F,3) rel frame0 norm, Rrel (F,3,3))."""
    rel = np.linalg.inv(c2w[0])[None] @ c2w
    R, t = rel[:, :3, :3], rel[:, :3, 3]
    s = max(np.percentile(np.linalg.norm(t, axis=1), 95), 1e-6)
    return rotmat_to_euler(R), t / s, R


def ridge_fit(X, Y, lam):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, Xb.T @ Y)


def ridge_pred(X, W):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
    return Xb @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re10k_root", default="DATA/re10k/re10k/train")
    ap.add_argument("--num_scenes", type=int, default=500)
    ap.add_argument("--num_frames", type=int, default=49)      # AC3D uses 49
    ap.add_argument("--hw", default="480,832")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--noise_levels", default="0.3,0.5,0.7,0.9,0.96")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out", default="results/probe_wan_re10k")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    H, W = [int(x) for x in args.hw.split(",")]
    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    dev = args.device

    from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg
    vae = AutoencoderWan(WanKwargsCfg(latent_channels=48)).from_pretrained(str(ROOT / "checkpoints/Wan2.2_VAE.pth")).to(dev).eval()
    dit_paths = sorted(glob.glob(str(MODEL_ROOT / "diffusion_pytorch_model*.safetensors")))
    pool = ModelPool(); pool.auto_load_model(dit_paths); dit = pool.fetch_model("wan_video_dit").to(dev).eval()
    pool2 = ModelPool(); pool2.auto_load_model(str(MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"))
    text_encoder = pool2.fetch_model("wan_video_text_encoder").to(dev).eval()
    tok = HuggingfaceTokenizer(name=str(MODEL_ROOT / "google" / "umt5-xxl"), seq_len=512, clean="whitespace")
    dtype = next(dit.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype
    n_layers = len(dit.blocks)
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"[probe-re10k] DiT layers={n_layers}, probing {layers}, noise={noise_levels}, AC3D trajectory mode")

    with torch.no_grad():
        ids, mask = tok("", return_mask=True); ctx = text_encoder(ids.to(dev), mask.to(dev))

    captured = {}
    def mk_hook(i):
        def hook(m, inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    handles = [dit.blocks[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]

    # gather scenes from .torch shards until num_scenes
    shards = sorted(glob.glob(os.path.join(args.re10k_root, "*.torch")))
    from collections import defaultdict
    feats = defaultdict(lambda: defaultdict(list))      # feats[L][noise] = list of (dim,) per scene
    euler_store, trans_store, R_store = [], [], []      # per-scene targets (aligned with scene order)
    n_used = 0
    for shard in shards:
        if n_used >= args.num_scenes:
            break
        try:
            scenes = torch.load(shard, map_location="cpu")
        except Exception as e:
            print(f"  skip shard {shard}: {e}"); continue
        for scene in scenes:
            if n_used >= args.num_scenes:
                break
            loaded = load_re10k_scene(scene, args.num_frames, (H, W))
            if loaded is None:
                continue
            frames, c2w = loaded
            eul, tr, Rrel = trajectory_target(c2w)                      # (F,3),(F,3),(F,3,3)
            with torch.no_grad():
                lat = vae.encode(frames.unsqueeze(0).to(dev, vae_dtype))
            lat = rearrange(lat, "b v c h w -> b c v h w").to(dtype)
            for t_lvl in noise_levels:
                ts = torch.full((1,), float(t_lvl) * 1000.0, device=dev, dtype=dtype)
                noisy = (1 - t_lvl) * lat + t_lvl * torch.randn_like(lat)
                captured.clear()
                with torch.no_grad():
                    model_fn_wan_video(dit=dit, latents=noisy, timestep=ts, context=ctx,
                                       vace=None, use_gradient_checkpointing=False)
                for L in layers:
                    feats[L][t_lvl].append(captured[L][0].float().mean(0).cpu())  # global pool → (dim,)
            euler_store.append(torch.from_numpy(eul).float().reshape(-1))   # (F*3,)
            trans_store.append(torch.from_numpy(tr).float().reshape(-1))
            R_store.append(torch.from_numpy(Rrel).float())                  # (F,3,3)
            n_used += 1
            if n_used % 25 == 0:
                print(f"  [{n_used}/{args.num_scenes}] scenes")
    for hd in handles:
        hd.remove()
    print(f"[probe-re10k] scenes used = {n_used}")

    F_ = R_store[0].shape[0]
    Yeul = torch.stack(euler_store)        # (N, F*3)
    Ytr = torch.stack(trans_store)         # (N, F*3)
    Rgt = torch.stack(R_store)             # (N, F, 3, 3)
    N = Yeul.shape[0]
    n_te = max(1, int(N * args.test_frac)); n_va = max(1, int(N * args.val_frac)); n_tr = N - n_va - n_te
    assert n_tr > 0
    print(f"[probe-re10k] train/val/test = {n_tr}/{n_va}/{n_te}, frames={F_}")

    def split(t):
        return t[:n_tr].to(dev), t[n_tr:n_tr + n_va].to(dev), t[n_tr + n_va:].to(dev)
    Yeul_tr, Yeul_va, Yeul_te = split(Yeul.double())
    Ytr_tr, Ytr_va, Ytr_te = split(Ytr.double())
    Rgt_te = Rgt[n_tr + n_va:].to(dev)

    def traj_err(pred_eul, pred_tr):
        Rp = euler_to_rotmat(pred_eul.reshape(-1, 3).float())
        Rg = Rgt_te.reshape(-1, 3, 3)
        rot = geodesic_deg(Rp, Rg).mean().item()
        trans = (pred_tr.reshape(-1, 3).float() - Ytr_te.reshape(-1, 3).float()).norm(dim=1).mean().item()
        return rot, trans

    os.makedirs(args.out, exist_ok=True)
    results = []
    # predict-train-mean baseline (test)
    base_eul = Yeul_tr.mean(0, keepdim=True).repeat(Yeul_te.shape[0], 1)
    base_tr = Ytr_tr.mean(0, keepdim=True).repeat(Ytr_te.shape[0], 1)
    base_rot, base_trans = traj_err(base_eul, base_tr)
    print(f"[probe-re10k] baseline: rot {base_rot:.2f}°  trans {base_trans:.3f}")

    for L in layers:
        for t_lvl in noise_levels:
            Xs = torch.stack(feats[L][t_lvl]).double()       # (N, dim)
            Xtr, Xva, Xte = split(Xs)
            mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
            Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
            Ycat_tr = torch.cat([Yeul_tr, Ytr_tr], 1)        # joint trajectory (F*6)
            best = None
            for lam in (1e-1, 1e0, 1e1, 1e2, 1e3):
                W_ = ridge_fit(Xtr, Ycat_tr, lam)
                pv = ridge_pred(Xva, W_)
                fe = pv.shape[1] // 2
                # val rotation error (λ selection criterion)
                Rp = euler_to_rotmat(pv[:, :fe].reshape(-1, 3).float())
                Rg = Rgt[n_tr:n_tr + n_va].to(dev).reshape(-1, 3, 3)
                vrot = geodesic_deg(Rp, Rg).mean().item()
                if best is None or vrot < best[0]:
                    best = (vrot, W_, lam)
            W_ = best[1]
            pt = ridge_pred(Xte, W_); fe = pt.shape[1] // 2
            te_rot, te_trans = traj_err(pt[:, :fe], pt[:, fe:])
            raw_mse = (pt - torch.cat([Yeul_te, Ytr_te], 1)).pow(2).mean().item()
            results.append(dict(layer=L, noise=t_lvl, rot_err_deg=te_rot, trans_err=te_trans,
                                raw_mse=raw_mse, lam=best[2]))
            print(f"  layer {L:2d} noise {t_lvl:.3f}: rot {te_rot:6.2f}° trans {te_trans:.3f} "
                  f"raw_mse {raw_mse:.4f} (base rot {base_rot:.1f} trans {base_trans:.2f}) λ={best[2]:g}")

    with open(os.path.join(args.out, "probe_results.json"), "w") as f:
        json.dump(dict(layers=layers, noise_levels=noise_levels, results=results, n_scenes=n_used,
                       num_frames=args.num_frames, baseline_rot=base_rot, baseline_trans=base_trans,
                       mode="ac3d_trajectory_re10k"), f, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for metric in ["rot_err_deg", "trans_err"]:
            M = np.full((len(layers), len(noise_levels)), np.nan)
            for r in results:
                M[layers.index(r["layer"]), noise_levels.index(r["noise"])] = r[metric]
            plt.figure(figsize=(7, max(4, len(layers) * 0.3)))
            plt.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
            plt.colorbar(label=metric); plt.xlabel("noise"); plt.ylabel("DiT layer")
            plt.xticks(range(len(noise_levels)), [f"{n:.2f}" for n in noise_levels])
            plt.yticks(range(len(layers)), layers)
            plt.title(f"Wan TI2V camera probing — RealEstate10K AC3D ({metric})")
            plt.tight_layout(); plt.savefig(os.path.join(args.out, f"heatmap_{metric}.png"), dpi=130); plt.close()
    except Exception as e:
        print("plot skip:", e)
    print(f"[probe-re10k] saved → {args.out}")


if __name__ == "__main__":
    main()
