"""AC3D-faithful camera linear probing on CogVideoX-5b (the AC3D backbone family), RealEstate10K.

Same probe as scripts/probe_wan_camera_re10k.py but on CogVideoX-5b instead of
Wan2.2 TI2V-5B → apples-to-apples MODEL comparison (same dataset re10k, same
target = joint 49×6 trajectory, same global-pool→ridge readout, frozen backbone).

CogVideoX specifics: diffusers CogVideoXPipeline (THUDM/CogVideoX-5b, 42 blocks,
dim 3072, RoPE, native 49 frames, 16ch VAE /8 spatial /4 temporal, DDIM scheduler).
Blocks return (video_hidden, encoder_hidden) → hook grabs video tokens.

Usage:
  python scripts/probe_cogvideox_camera_re10k.py --num_scenes 300 --num_frames 49 \
      --noise_levels 0.3,0.5,0.7,0.9,0.96 --layers all --out results/probe_cogvideox_re10k
"""
from __future__ import annotations
import argparse, glob, io, json, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# cuDNN SDPA backend errors ("No valid execution plans built") on CogVideoX attention
# at these seq lengths → disable it, fall back to flash/mem-efficient/math.
torch.backends.cuda.enable_cudnn_sdp(False)
from diffusers import CogVideoXPipeline

MODEL = "THUDM/CogVideoX-5b"


# ── pose helpers (identical to probe_wan_camera_re10k.py) ──
def rotmat_to_euler(R):
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    return np.stack([np.arctan2(R[:, 2, 1], R[:, 2, 2]),
                     np.arctan2(-R[:, 2, 0], sy),
                     np.arctan2(R[:, 1, 0], R[:, 0, 0])], axis=-1)


def euler_to_rotmat(e):
    x, y, z = e[:, 0], e[:, 1], e[:, 2]
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    return torch.stack([
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx], dim=-1).reshape(-1, 3, 3)


def geodesic_deg(Rp, Rg):
    m = torch.matmul(Rp.transpose(-1, -2), Rg)
    tr = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def re10k_c2w(cameras):
    ext = cameras[:, 6:18].reshape(-1, 3, 4).cpu().numpy().astype(np.float64)
    N = ext.shape[0]; w2c = np.tile(np.eye(4), (N, 1, 1)); w2c[:, :3, :4] = ext
    return np.linalg.inv(w2c)


def load_re10k_scene(scene, num_frames, hw):
    cams = scene["cameras"]; n = cams.shape[0]
    if n < num_frames:
        return None
    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    H, W = hw; c2w = re10k_c2w(cams)[idx]; imgs = []
    for i in idx:
        im = Image.open(io.BytesIO(scene["images"][i].numpy().tobytes())).convert("RGB").resize((W, H), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.array(im)).float() / 255.0 * 2 - 1)
    return torch.stack(imgs).permute(0, 3, 1, 2), c2w           # (V,3,H,W), (V,4,4)


def trajectory_target(c2w):
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
    ap.add_argument("--num_scenes", type=int, default=300)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--hw", default="480,720")              # CogVideoX-5b native
    ap.add_argument("--layers", default="all")
    ap.add_argument("--noise_levels", default="0.3,0.5,0.7,0.9,0.96")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out", default="results/probe_cogvideox_re10k")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    H, W = [int(x) for x in args.hw.split(",")]
    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    dev = args.device

    pipe = CogVideoXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to(dev)
    pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
    tr, vae, sched = pipe.transformer.eval(), pipe.vae.eval(), pipe.scheduler
    dtype = next(tr.parameters()).dtype
    sf = vae.config.scaling_factor
    n_layers = len(tr.transformer_blocks)
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"[probe-cog] CogVideoX-5b blocks={n_layers}, probing {layers}, noise={noise_levels}")

    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt("", negative_prompt=None, do_classifier_free_guidance=False,
                                              num_videos_per_prompt=1, device=dev, dtype=dtype)

    captured = {}
    def mk_hook(i):
        def hook(m, inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()   # video hidden (B,seq,dim)
        return hook
    handles = [tr.transformer_blocks[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]

    shards = sorted(glob.glob(os.path.join(args.re10k_root, "*.torch")))
    from collections import defaultdict
    feats = defaultdict(lambda: defaultdict(list))
    euler_store, trans_store, R_store = [], [], []
    n_used = 0
    for shard in shards:
        if n_used >= args.num_scenes:
            break
        try:
            scenes = torch.load(shard, map_location="cpu")
        except Exception as e:
            print(f"  skip {shard}: {e}"); continue
        for scene in scenes:
            if n_used >= args.num_scenes:
                break
            loaded = load_re10k_scene(scene, args.num_frames, (H, W))
            if loaded is None:
                continue
            frames, c2w = loaded
            eul, t_, Rrel = trajectory_target(c2w)
            video = frames.permute(1, 0, 2, 3).unsqueeze(0).to(dev, dtype)         # (1,3,V,H,W)
            with torch.no_grad():
                lat = vae.encode(video).latent_dist.sample() * sf                 # (1,16,T,h,w)
            lat = lat.permute(0, 2, 1, 3, 4)                                       # (1,T,16,h,w) for transformer
            T_lat = lat.shape[1]
            rope = pipe._prepare_rotary_positional_embeddings(H, W, T_lat, dev) \
                if tr.config.use_rotary_positional_embeddings else None
            for s in noise_levels:
                t_int = int(round(s * (sched.config.num_train_timesteps - 1)))
                ts_idx = torch.tensor([t_int], device=dev)
                noisy = sched.add_noise(lat, torch.randn_like(lat), ts_idx)
                captured.clear()
                with torch.no_grad():
                    tr(hidden_states=noisy, encoder_hidden_states=prompt_embeds,
                       timestep=ts_idx.to(dtype), image_rotary_emb=rope, return_dict=False)
                for L in layers:
                    feats[L][s].append(captured[L][0].float().mean(0).cpu())      # global pool → (dim,)
            euler_store.append(torch.from_numpy(eul).float().reshape(-1))
            trans_store.append(torch.from_numpy(t_).float().reshape(-1))
            R_store.append(torch.from_numpy(Rrel).float())
            n_used += 1
            if n_used % 20 == 0:
                print(f"  [{n_used}/{args.num_scenes}] scenes")
    for hd in handles:
        hd.remove()
    print(f"[probe-cog] scenes used = {n_used}")

    Yeul = torch.stack(euler_store); Ytr = torch.stack(trans_store); Rgt = torch.stack(R_store)
    N = Yeul.shape[0]; F_ = Rgt.shape[1]
    n_te = max(1, int(N * args.test_frac)); n_va = max(1, int(N * args.val_frac)); n_tr = N - n_va - n_te
    assert n_tr > 0
    print(f"[probe-cog] train/val/test = {n_tr}/{n_va}/{n_te}, frames={F_}")

    def split(t): return t[:n_tr].to(dev), t[n_tr:n_tr + n_va].to(dev), t[n_tr + n_va:].to(dev)
    Yeul_tr, Yeul_va, Yeul_te = split(Yeul.double())
    Ytr_tr, Ytr_va, Ytr_te = split(Ytr.double())
    Rgt_va = Rgt[n_tr:n_tr + n_va].to(dev); Rgt_te = Rgt[n_tr + n_va:].to(dev)

    def err(pred_eul, pred_tr, Rg, Yt):
        rot = geodesic_deg(euler_to_rotmat(pred_eul.reshape(-1, 3).float()), Rg.reshape(-1, 3, 3)).mean().item()
        trans = (pred_tr.reshape(-1, 3).float() - Yt.reshape(-1, 3).float()).norm(dim=1).mean().item()
        return rot, trans

    os.makedirs(args.out, exist_ok=True); results = []
    base_rot, base_trans = err(Yeul_tr.mean(0, keepdim=True).repeat(Yeul_te.shape[0], 1),
                               Ytr_tr.mean(0, keepdim=True).repeat(Ytr_te.shape[0], 1), Rgt_te, Ytr_te)
    print(f"[probe-cog] baseline: rot {base_rot:.2f}°  trans {base_trans:.3f}")
    for L in layers:
        for s in noise_levels:
            Xtr, Xva, Xte = split(torch.stack(feats[L][s]).double())
            mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
            Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
            Ycat_tr = torch.cat([Yeul_tr, Ytr_tr], 1)
            best = None
            for lam in (1e-1, 1e0, 1e1, 1e2, 1e3):
                W_ = ridge_fit(Xtr, Ycat_tr, lam); pv = ridge_pred(Xva, W_); fe = pv.shape[1] // 2
                vrot, _ = err(pv[:, :fe], pv[:, fe:], Rgt_va, Ytr_va)
                if best is None or vrot < best[0]:
                    best = (vrot, W_, lam)
            W_ = best[1]; pt = ridge_pred(Xte, W_); fe = pt.shape[1] // 2
            te_rot, te_trans = err(pt[:, :fe], pt[:, fe:], Rgt_te, Ytr_te)
            raw_mse = (pt - torch.cat([Yeul_te, Ytr_te], 1)).pow(2).mean().item()
            results.append(dict(layer=L, noise=s, rot_err_deg=te_rot, trans_err=te_trans, raw_mse=raw_mse, lam=best[2]))
            print(f"  layer {L:2d} noise {s:.3f}: rot {te_rot:6.2f}° trans {te_trans:.3f} raw_mse {raw_mse:.4f} "
                  f"(base rot {base_rot:.1f} trans {base_trans:.2f}) λ={best[2]:g}")

    with open(os.path.join(args.out, "probe_results.json"), "w") as f:
        json.dump(dict(layers=layers, noise_levels=noise_levels, results=results, n_scenes=n_used,
                       num_frames=args.num_frames, baseline_rot=base_rot, baseline_trans=base_trans,
                       model="CogVideoX-5b", mode="ac3d_trajectory_re10k"), f, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for metric in ["rot_err_deg", "trans_err"]:
            M = np.full((len(layers), len(noise_levels)), np.nan)
            for r in results:
                M[layers.index(r["layer"]), noise_levels.index(r["noise"])] = r[metric]
            plt.figure(figsize=(7, max(4, len(layers) * 0.25)))
            plt.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
            plt.colorbar(label=metric); plt.xlabel("noise"); plt.ylabel("CogVideoX block")
            plt.xticks(range(len(noise_levels)), [f"{n:.2f}" for n in noise_levels])
            plt.yticks(range(len(layers)), layers)
            plt.title(f"CogVideoX-5b camera probing — re10k ({metric})")
            plt.tight_layout(); plt.savefig(os.path.join(args.out, f"heatmap_{metric}.png"), dpi=130); plt.close()
    except Exception as e:
        print("plot skip:", e)
    print(f"[probe-cog] saved → {args.out}")


if __name__ == "__main__":
    main()
