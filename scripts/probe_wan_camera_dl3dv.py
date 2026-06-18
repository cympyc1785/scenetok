"""AC3D-style camera linear probing on the Wan TI2V 5B video DiT, using DL3DV.

각 DiT block × 각 noise level(timestep)에서 hidden state를 추출 → frozen feature에
linear probe(closed-form ridge)를 fit → per-frame camera pose(rotation/translation)를
회귀 → geodesic rotation error + translation error 측정. (AC3D Fig. 재현: "camera 정보가
어느 layer / 어느 noise level에 사는가".)

입력은 **video** (DL3DV 프레임 시퀀스를 latent으로 인코딩). image 아님.
backbone(Wan DiT)은 frozen, probe(Linear)만 closed-form으로 fit.

Usage:
  python scripts/probe_wan_camera_dl3dv.py --num_scenes 40 --num_frames 33 \
      --layers all --noise_levels 0.1,0.3,0.5,0.7,0.9 --out results/probe_wan_dl3dv
"""
from __future__ import annotations
import argparse, glob, json, os, sys
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


# ───────────────── pose helpers ─────────────────
def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,6) = [col0(3) | col1(3)], matching sixd_to_rot's a1,a2 split."""
    return R[:, :, :2].transpose(0, 2, 1).reshape(R.shape[0], 6)


def sixd_to_rot(x: torch.Tensor) -> torch.Tensor:
    """(N,6) → (N,3,3) via Gram-Schmidt."""
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def geodesic_deg(Rp: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
    m = torch.matmul(Rp.transpose(-1, -2), Rg)
    tr = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos))


def load_scene(scene_dir: Path, num_frames: int, hw: tuple[int, int]):
    """Return (frames (V,3,H,W) in [-1,1], poses_rel9 (V,9), Rrel (V,3,3))."""
    tj = json.load(open(scene_dir / "transforms.json"))
    frames = tj["frames"]
    order = np.argsort([f["file_path"] for f in frames])
    frames = [frames[i] for i in order]
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    H, W = hw
    # DL3DV stores downsampled frames; transforms.json file_path says "images/..".
    img_dir = next((d for d in ["images_4", "images_8", "images"]
                    if (scene_dir / d).is_dir()), "images")
    imgs, c2w = [], []
    for i in idx:
        fr = frames[i]
        p = scene_dir / img_dir / os.path.basename(fr["file_path"])
        im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.array(im)).float() / 255.0 * 2 - 1)
        c2w.append(np.array(fr["transform_matrix"], dtype=np.float64))
    frames_t = torch.stack(imgs).permute(0, 3, 1, 2)        # (V,3,H,W)
    c2w = np.stack(c2w)                                      # (V,4,4)
    ref_inv = np.linalg.inv(c2w[0])
    rel = ref_inv[None] @ c2w                                # reference-relative
    R = rel[:, :3, :3]
    t = rel[:, :3, 3]
    s = max(np.percentile(np.linalg.norm(t, axis=1), 95), 1e-6)
    t = t / s
    target9 = np.concatenate([rot_to_6d(R), t], axis=1)      # (V,9)
    return frames_t, torch.from_numpy(target9).float(), torch.from_numpy(R).float()


def load_prompt(scene_dir: Path) -> str:
    """DL3DV per-scene caption (prompts_*.json → ['0']['prompt_scene_simple']['detail']). '' if absent."""
    for pj in sorted(scene_dir.glob("prompts_*.json")):
        try:
            d = json.load(open(pj))
            return d["0"]["prompt_scene_simple"]["detail"]
        except Exception:
            continue
    return ""


def ridge_fit(X, Y, lam=1e-2):
    """closed-form ridge with bias. X (N,D), Y (N,K) → W (D+1,K)."""
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, Xb.T @ Y)


def ridge_pred(X, W):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
    return Xb @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob_root", default="DATA/DL3DV/DL3DV-960/train")
    ap.add_argument("--num_scenes", type=int, default=40)
    ap.add_argument("--num_frames", type=int, default=33)   # 4N+1 → clean latent temporal
    ap.add_argument("--hw", default="480,832")
    ap.add_argument("--layers", default="all")              # "all" or "0,5,10,..."
    ap.add_argument("--noise_levels", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--n_pc", type=int, default=256)        # PCA dims for the linear probe
    ap.add_argument("--pool_grid", type=int, default=4)     # G×G spatial grid pooling (1 = global mean)
    ap.add_argument("--in_distribution", action="store_true")  # native TI2V: clean 1st frame + real prompt + fused embedding
    ap.add_argument("--out", default="results/probe_wan_dl3dv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    H, W = [int(x) for x in args.hw.split(",")]
    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    dev = args.device

    # ── load Wan2.2 VAE(48ch) + DiT + text encoder ──
    from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg
    vae = AutoencoderWan(WanKwargsCfg(latent_channels=48)).from_pretrained(str(ROOT / "checkpoints/Wan2.2_VAE.pth")).to(dev).eval()
    dit_paths = sorted(glob.glob(str(MODEL_ROOT / "diffusion_pytorch_model*.safetensors")))
    pool = ModelPool(); pool.auto_load_model(dit_paths); dit = pool.fetch_model("wan_video_dit").to(dev).eval()
    pool2 = ModelPool(); pool2.auto_load_model(str(MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"))
    text_encoder = pool2.fetch_model("wan_video_text_encoder").to(dev).eval()
    tok = HuggingfaceTokenizer(name=str(MODEL_ROOT / "google" / "umt5-xxl"), seq_len=512, clean="whitespace")
    dtype = next(dit.parameters()).dtype
    n_layers = len(dit.blocks)
    p_h, p_w = dit.patch_size[1], dit.patch_size[2]            # spatial patch stride
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"[probe] DiT layers={n_layers}, probing {layers}, noise={noise_levels}")

    def encode_text(prompt: str):
        with torch.no_grad():
            ids, mask = tok(prompt, return_mask=True); ids, mask = ids.to(dev), mask.to(dev)
            return text_encoder(ids, mask)
    ctx = encode_text("")  # empty text (off-distribution default)
    print(f"[probe] in_distribution={args.in_distribution} (clean 1st frame + real prompt + fused embedding)"
          if args.in_distribution else "[probe] off-distribution (empty text, all frames noised, no fused embedding)")

    # forward hooks → capture each block output
    captured = {}
    def mk_hook(i):
        def hook(m, inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    handles = [dit.blocks[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]

    scenes = sorted(p.rsplit("/transforms.json", 1)[0]
                    for p in glob.glob(os.path.join(args.glob_root, "*/*/transforms.json")))[: args.num_scenes]
    print(f"[probe] {len(scenes)} scenes")

    # feats[layer][noise] = list of (V, dim); targets aligned
    from collections import defaultdict
    feats = defaultdict(lambda: defaultdict(list))
    targets = defaultdict(lambda: defaultdict(list))
    Rgt_store = defaultdict(lambda: defaultdict(list))

    for si, sc in enumerate(scenes):
        try:
            frames, tgt9, Rrel = load_scene(Path(sc), args.num_frames, (H, W))
        except Exception as e:
            print(f"  skip {sc}: {e}"); continue
        vae_dtype = next(vae.parameters()).dtype
        frames = frames.unsqueeze(0).to(dev, vae_dtype)           # (1,V,3,H,W)
        with torch.no_grad():
            lat = vae.encode(frames)                              # (1,Vlat,48,h,w)
        lat = rearrange(lat, "b v c h w -> b c v h w").to(dtype)
        b, c, f, h, w = lat.shape
        # align GT poses (V pixel) → Vlat latent frames via linspace
        pix_idx = np.linspace(0, args.num_frames - 1, f).round().astype(int)
        tgt = tgt9[pix_idx].to(dev)                                # (f,9)
        Rg = Rrel[pix_idx].to(dev)
        scene_ctx = encode_text(load_prompt(Path(sc))) if args.in_distribution else ctx
        for t_lvl in noise_levels:
            # diffsynth/base-Wan convention: sigma=t_lvl ∈ [0,1] but model timestep = sigma*1000.
            ts = torch.full((1,), float(t_lvl) * 1000.0, device=dev, dtype=dtype)
            noise = torch.randn_like(lat)
            noisy = (1 - t_lvl) * lat + t_lvl * noise              # rectified-flow interpolation
            if args.in_distribution:
                noisy[:, :, 0] = lat[:, :, 0]                      # frame 0 = clean conditioning (native TI2V)
            captured.clear()
            with torch.no_grad():
                _ = model_fn_wan_video(dit=dit, latents=noisy, timestep=ts, context=scene_ctx,
                                       vace=None, use_gradient_checkpointing=False,
                                       fuse_vae_embedding_in_latents=args.in_distribution)
            Hp, Wp = h // p_h, w // p_w
            for L in layers:
                tok_feat = captured[L][0]                          # (f*Hp*Wp, dim)
                fmap = tok_feat.reshape(f, Hp, Wp, -1).permute(0, 3, 1, 2).float()  # (f,dim,Hp,Wp)
                if args.pool_grid <= 1:
                    pooled = fmap.mean((2, 3))                     # (f, dim) global mean
                else:
                    # coarse G×G grid pooling — preserve spatial layout (where content sits)
                    g = F.adaptive_avg_pool2d(fmap, (args.pool_grid, args.pool_grid))
                    pooled = g.reshape(f, -1)                      # (f, dim*G*G)
                feats[L][t_lvl].append(pooled.cpu())
                targets[L][t_lvl].append(tgt.cpu())
                Rgt_store[L][t_lvl].append(Rg.cpu())
        if (si + 1) % 5 == 0:
            print(f"  [{si+1}/{len(scenes)}] processed")

    for hd in handles:
        hd.remove()

    # ── fit probe per (layer, noise) + eval ──
    os.makedirs(args.out, exist_ok=True)
    results = []
    n_scenes_used = len(feats[layers[0]][noise_levels[0]])
    n_val = max(1, int(n_scenes_used * args.val_frac))
    print(f"[probe] scenes used={n_scenes_used}, val={n_val}")
    def rot_trans_err(pred, Rgt, Ygt):
        Rp = sixd_to_rot(pred[:, :6])
        return (geodesic_deg(Rp, Rgt).mean().item(),
                (pred[:, 6:] - Ygt[:, 6:]).norm(dim=1).mean().item())

    for L in layers:
        for t_lvl in noise_levels:
            Xs = feats[L][t_lvl]; Ys = targets[L][t_lvl]; Rs = Rgt_store[L][t_lvl]
            # split by scene
            Xtr = torch.cat(Xs[:-n_val]).double().to(dev); Ytr = torch.cat(Ys[:-n_val]).double().to(dev)
            Xva = torch.cat(Xs[-n_val:]).double().to(dev); Yva = torch.cat(Ys[-n_val:]).double().to(dev)
            Rtr = torch.cat(Rs[:-n_val]).to(dev); Rva = torch.cat(Rs[-n_val:]).to(dev)
            # z-score standardize features (train stats) — DiT activations vary wildly in scale
            mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
            Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
            # PCA reduction (train-fit) — D(3072) >> N(~500) → raw ridge overfits badly
            n_pc = min(args.n_pc, Xtr.shape[0] - 1, Xtr.shape[1])
            U, S, Vh = torch.linalg.svd(Xtr, full_matrices=False)
            P = Vh[:n_pc].T                                        # (D, n_pc)
            Xtr = Xtr @ P; Xva = Xva @ P
            # ridge λ search on a small holdout from train (last 20% of train scenes)
            best = None
            for lam in (1e-3, 1e-1, 1e0, 1e1, 1e2):
                W_ = ridge_fit(Xtr, Ytr, lam=lam)
                ve, _ = rot_trans_err(ridge_pred(Xva, W_).float(), Rva, Yva)
                if best is None or ve < best[0]:
                    best = (ve, W_, lam)
            W_ = best[1]
            tr_rot, tr_tr = rot_trans_err(ridge_pred(Xtr, W_).float(), Rtr, Ytr)
            va_rot, va_tr = rot_trans_err(ridge_pred(Xva, W_).float(), Rva, Yva)
            # predict-train-mean baseline (val): how much does the probe beat constant prediction?
            base = Ytr.mean(0, keepdim=True).repeat(Yva.shape[0], 1).float()
            base_rot, base_tr = rot_trans_err(base, Rva, Yva)
            results.append(dict(layer=L, noise=t_lvl, rot_err_deg=va_rot, trans_err=va_tr,
                                train_rot_err_deg=tr_rot, train_trans_err=tr_tr,
                                baseline_rot_err_deg=base_rot, baseline_trans_err=base_tr, lam=best[2]))
            print(f"  layer {L:2d} noise {t_lvl:.2f}: rot {va_rot:6.2f}° (tr {tr_rot:6.2f}, base {base_rot:6.2f}) "
                  f"trans {va_tr:.3f} (tr {tr_tr:.3f}, base {base_tr:.3f}) λ={best[2]:g}")

    with open(os.path.join(args.out, "probe_results.json"), "w") as fjson:
        json.dump(dict(layers=layers, noise_levels=noise_levels, results=results,
                       n_scenes=n_scenes_used, num_frames=args.num_frames), fjson, indent=2)

    # heatmap
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for metric in ["rot_err_deg", "trans_err"]:
            M = np.full((len(layers), len(noise_levels)), np.nan)
            for r in results:
                M[layers.index(r["layer"]), noise_levels.index(r["noise"])] = r[metric]
            plt.figure(figsize=(7, max(4, len(layers) * 0.3)))
            plt.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
            plt.colorbar(label=metric); plt.xlabel("noise level"); plt.ylabel("DiT layer")
            plt.xticks(range(len(noise_levels)), [f"{n:.2f}" for n in noise_levels])
            plt.yticks(range(len(layers)), layers)
            plt.title(f"Wan TI2V camera probing ({metric})")
            plt.tight_layout(); plt.savefig(os.path.join(args.out, f"heatmap_{metric}.png"), dpi=130); plt.close()
    except Exception as e:
        print("plot skip:", e)
    print(f"[probe] saved → {args.out}")


if __name__ == "__main__":
    main()
