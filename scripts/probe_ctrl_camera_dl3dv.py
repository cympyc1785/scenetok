"""Propagation probe of the TRAINED camera-controlnet model (DL3DV).

Does the trained controlnet inject linearly-decodable camera pose into the FROZEN
main Wan DiT stream? Reuse fast_infer's proven cfg/wrapper build, hook the MAIN
dit.blocks, run generation with few denoising steps (= noise levels), capture
per-(block,step) token-pooled features, ridge-probe the clip's last-frame
relative camera pose. Compare error vs predict-mean baseline (and vs base-Wan null).
"""
from __future__ import annotations
import argparse, json, os, sys, types
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src/model/DiffSynth-Studio"))
import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False; _wvd.FLASH_ATTN_2_AVAILABLE = False; _wvd.SAGE_ATTN_AVAILABLE = False

from torch.utils.data import DataLoader
from scripts.fast_infer_t2v_swap_dataset import build_cfg, move_to_device
from src.dataset.data_module import safe_collate
from src.dataset import get_dataset
from src.misc.step_tracker import StepTracker
from src.misc.batch_utils import preprocess_batch
from src.model.t2v_wrapper import T2VWrapper


def rotmat_to_euler(R):
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    return np.stack([np.arctan2(R[:, 2, 1], R[:, 2, 2]), np.arctan2(-R[:, 2, 0], sy),
                     np.arctan2(R[:, 1, 0], R[:, 0, 0])], -1)

def euler_to_rotmat(e):
    x, y, z = e[:, 0], e[:, 1], e[:, 2]
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    return torch.stack([cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx,
                        sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx,
                        -sy, cy*sx, cy*cx], -1).reshape(-1, 3, 3)

def geodesic_deg(Rp, Rg):
    m = torch.matmul(Rp.transpose(-1, -2), Rg)
    return torch.rad2deg(torch.arccos(((m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))

def ridge_fit(X, Y, lam):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
    return torch.linalg.solve(Xb.T @ Xb + lam * torch.eye(Xb.shape[1], device=X.device, dtype=X.dtype), Xb.T @ Y)

def ridge_pred(X, W):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--num_scenes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ns = types.SimpleNamespace(
        experiment="custom/scenetok_va-wan-ti2v_dynamicverse", dataset="dl3dv",
        dataset_root="./DATA/DL3DV/DL3DV-960",
        scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97",
        context_shape="256,448", target_shape="480,832", num_context_views=12, num_target_views=37,
        load_prompts=False, cfg_scale=1.0, num_inference_steps=args.steps, noise_seed=0,
        lora_disabled=True, prompt_style=None, scene_input_type="controlnet",
        camera_input_type="controlnet", prompt="", ood_prompt="", negative_prompt=None,
        static_target_camera=False, controlnet_ablation=False, single_cfg=True, text_modes="empty",
        ckpt=None, output_dir=args.out, exp_name=args.exp_name, device=dev)
    cfg = build_cfg(ns)
    try:
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass                              # cfg가 RootCfg 객체면 불필요
    cfg.dataset.scene_id = None          # None → split의 여러 scene 순회
    cfg.model.cfg_scale = 1.0
    cfg.model.scheduler.num_inference_steps = args.steps

    st = StepTracker(0)
    wrapper = T2VWrapper(model_cfg=cfg.model, dataset_cfg=cfg.dataset, freeze_cfg=cfg.freeze,
                         optimizer_cfg=cfg.optimizer, test_cfg=cfg.test, train_cfg=cfg.train,
                         val_cfg=cfg.val, sampler_cfg=cfg.sampler, step_tracker=st,
                         output_dir=Path(args.out), batch_size=1, val_check_interval=1, mode="test")
    ckpt = str(REPO / f"my_checkpoints/{args.exp_name}/last.ckpt")
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    miss, unexp = wrapper.load_state_dict(sd, strict=False)
    print(f"[probe-ctrl] ckpt load: missing={len(miss)} unexpected={len(unexp)}")
    wrapper.eval().to(dev).to(torch.bfloat16)   # get_latents가 inputs.bfloat16() 하드코딩 → 전체 bf16 일관
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None

    dit = wrapper.denoiser.model
    n_layers = len(dit.blocks)
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"[probe-ctrl] main blocks={n_layers}, probing {layers}, steps={args.steps}")
    captured = {i: [] for i in range(n_layers)}
    def mk(i):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[i].append(o.detach().float().mean(1).cpu()[0])   # (dim,)
        return hook
    handles = [dit.blocks[i].register_forward_hook(mk(i)) for i in range(n_layers)]

    dataset = get_dataset(cfg.dataset, "test", st, generator=None, force_shuffle=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=safe_collate, shuffle=False)

    from collections import defaultdict
    feats = defaultdict(list); Ys = []; Rs = []; n_used = 0
    for batch in loader:
        if batch is None or n_used >= args.num_scenes:
            continue
        batch = move_to_device(batch, dev)
        v_c = batch["context"]["extrinsics"].shape[1]
        for k in captured: captured[k].clear()
        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                wrapper.generate_batch_with_scene(batch, wrapper.sampler)   # bf16 wrapper + autocast (stray fp32 matmul 처리)
        except Exception as e:
            import traceback as _tb
            if n_used == 0:
                _tb.print_exc()
            print(f"  skip scene: {e}"); continue
        nst = min(len(captured[layers[0]]), args.steps)
        if nst == 0:
            print("  WARN 0 captures"); continue
        bb = preprocess_batch(batch, index=v_c // 2)
        ext = bb["target"]["extrinsics"][0].cpu().numpy()        # (V,4,4) relative
        R = ext[:, :3, :3]; t = ext[:, :3, 3]
        s = max(np.percentile(np.linalg.norm(t, axis=1), 95), 1e-6)
        Ys.append(np.concatenate([rotmat_to_euler(R)[-1], (t / s)[-1]]))   # last-frame pose (6,)
        Rs.append(R[-1])
        for L in layers:
            for si in range(nst):
                feats[(L, si)].append(captured[L][si])
        n_used += 1
        if n_used % 5 == 0:
            print(f"  [{n_used}/{args.num_scenes}] scenes (steps={nst})", flush=True)
    for h in handles: h.remove()
    print(f"[probe-ctrl] scenes used={n_used}")

    Y = torch.tensor(np.stack(Ys), dtype=torch.float64, device=dev)
    Rlast = torch.tensor(np.stack(Rs), dtype=torch.float32, device=dev)
    N = Y.shape[0]
    n_te = max(1, int(N*args.test_frac)); n_va = max(1, int(N*args.val_frac)); n_tr = N-n_va-n_te
    assert n_tr > 0, f"too few scenes {N}"
    Ytr, Yva, Yte = Y[:n_tr], Y[n_tr:n_tr+n_va], Y[n_tr+n_va:]
    Rva, Rte = Rlast[n_tr:n_tr+n_va], Rlast[n_tr+n_va:]
    def err(p, Rg, Yt):
        return (geodesic_deg(euler_to_rotmat(p[:, :3].float()), Rg).mean().item(),
                (p[:, 3:].float() - Yt[:, 3:].float()).norm(dim=1).mean().item())
    base_rot, base_tr = err(Ytr.mean(0, keepdim=True).repeat(Yte.shape[0], 1), Rte, Yte)
    print(f"[probe-ctrl] N={N} train/val/test={n_tr}/{n_va}/{n_te}  baseline rot {base_rot:.2f} trans {base_tr:.3f}")

    os.makedirs(args.out, exist_ok=True); results = []
    nst_all = max(s for (_, s) in feats.keys()) + 1
    for L in layers:
        for si in range(nst_all):
            if (L, si) not in feats or len(feats[(L, si)]) < N:
                continue
            X = torch.stack(feats[(L, si)]).double().to(dev)
            Xtr, Xva, Xte = X[:n_tr], X[n_tr:n_tr+n_va], X[n_tr+n_va:]
            mu, sd_ = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
            Xtr, Xva, Xte = (Xtr-mu)/sd_, (Xva-mu)/sd_, (Xte-mu)/sd_
            best = None
            for lam in (1e0, 1e1, 1e2, 1e3, 1e4):
                W = ridge_fit(Xtr, Ytr, lam); vr, _ = err(ridge_pred(Xva, W), Rva, Yva)
                if best is None or vr < best[0]: best = (vr, W, lam)
            tr, tt = err(ridge_pred(Xte, best[1]), Rte, Yte)
            results.append(dict(layer=L, step=si, rot_err_deg=tr, trans_err=tt, lam=best[2]))
            print(f"  layer {L:2d} step {si}: rot {tr:6.2f}° trans {tt:.3f} (base {base_rot:.1f}/{base_tr:.2f})")
    json.dump(dict(results=results, n_scenes=N, baseline_rot=base_rot, baseline_trans=base_tr,
                   exp=args.exp_name, steps=args.steps), open(os.path.join(args.out, "probe_results.json"), "w"), indent=2)
    print(f"[probe-ctrl] saved → {args.out}")


if __name__ == "__main__":
    main()
