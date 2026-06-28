"""Standalone fast inference for SceneTok TI2V / T2V models.

Bypasses Hydra config composition and Lightning Trainer setup. Loads the saved
hydra dump (`exp/<exp_name>/.hydra/config.yaml`) and `my_checkpoints/<exp_name>/last.ckpt`
directly, builds T2VWrapper, samples one batch, and writes only the video.

Saves ~15-20 sec startup vs `python -m src.main mode=test ...`:
  - no hydra config composition
  - no Lightning Trainer init / sanity val
  - no wandb / no checkpoint callback
  - no val dataloaders, no metric module forward
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.data_module import safe_collate
from src.misc.batch_utils import preprocess_batch
from src.misc.image_io import save_image_video
from src.misc.step_tracker import StepTracker
from src.model.t2v_wrapper import T2VWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", required=True, help="Used to resolve config + ckpt paths.")
    p.add_argument("--ckpt", default=None, help="Override path to last.ckpt.")
    p.add_argument("--config", default=None, help="Override path to .hydra/config.yaml.")
    p.add_argument("--scene_id", default=None, help="DL3DV scene id (e.g. '1K/abc...').")
    p.add_argument(
        "--stage_override",
        default="train",
        choices=["train", "val", "test"],
        help="Pool to draw scene from (train→1K, val/test→11K with val_seen=False). "
             "Dataset itself is always built with stage='test' so evaluation_index is used.",
    )
    p.add_argument(
        "--evaluation_index_path",
        default=None,
        help="Explicit path to dl3dv_c16_{34,37}_{standard,unseen}.json. "
             "If omitted, dataset auto-resolves from (chunk_targets, val_seen).",
    )
    p.add_argument("--prompt", default=None,
                   help="User-supplied prompt (in-distribution baseline).")
    p.add_argument("--ood_prompt", default=None,
                   help="Out-of-distribution prompt (e.g. a prompt from a different "
                        "dataset) for ood_text combo.")
    p.add_argument("--negative_prompt", default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--repeat_factor", type=int, default=1,
                   help="MC samples per combo. >1 enables variance.mp4 (per-pixel std "
                        "across N samples) + mean-video as full_sequence.mp4. Costs N× sampling.")
    p.add_argument("--controlnet_ablation", action="store_true",
                   help="추가로 ac3d_projectors 가중치를 0으로 만든 채 한 번 더 sample → "
                        "controlnet OFF mp4 + |on - off| diff mp4. Costs 2× sampling per combo.")
    p.add_argument("--val_seen", default="true", choices=["true", "false"],
                   help="dataset.val_seen 토글. unseen 풀(DAVIS 등) scene 쓰려면 false.")
    p.add_argument("--target_shape", default=None, help="H,W (e.g. 480,832)")
    p.add_argument("--context_shape", default=None, help="H,W (e.g. 256,256)")
    p.add_argument("--num_context_views", type=int, default=None)
    p.add_argument("--num_target_views", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_scenes", type=int, default=1,
                   help="How many scenes (batches) from the eval index to process. "
                        "Default 1 = single scene (backward compat).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_shape(s):
    return [int(x) for x in s.split(",")] if s else None


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def main():
    args = parse_args()

    config_path = (
        Path(args.config) if args.config else REPO_ROOT / "exp" / args.exp_name / ".hydra" / "config.yaml"
    )
    ckpt_path = (
        Path(args.ckpt) if args.ckpt else REPO_ROOT / "my_checkpoints" / args.exp_name / "last.ckpt"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / f"fast_infer_{args.exp_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fast_infer] config: {config_path}")
    print(f"[fast_infer] ckpt:   {ckpt_path}")
    print(f"[fast_infer] output: {output_dir}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg_dict = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg_dict, False)

    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.data_loader.test.num_workers = 0
    cfg_dict.data_loader.test.batch_size = 1
    cfg_dict.freeze.denoiser = True
    cfg_dict.freeze.compressor = True
    cfg_dict.freeze.autoencoder = True

    cfg_dict.dataset.smallset = True
    cfg_dict.dataset.stage_override = args.stage_override
    cfg_dict.dataset.val_seen = (args.val_seen == "true")
    if args.evaluation_index_path is not None:
        cfg_dict.dataset.evaluation_index_path = args.evaluation_index_path
    if args.scene_id is not None:
        cfg_dict.dataset.scene_id = args.scene_id
    if parse_shape(args.context_shape):
        cfg_dict.dataset.context_shape = parse_shape(args.context_shape)
    if parse_shape(args.target_shape):
        cfg_dict.dataset.target_shape = parse_shape(args.target_shape)
    if args.num_context_views:
        cfg_dict.dataset.view_sampler.num_context_views = args.num_context_views
    if args.num_target_views:
        cfg_dict.dataset.view_sampler.num_target_views = args.num_target_views

    if args.cfg_scale is not None:
        cfg_dict.model.cfg_scale = args.cfg_scale
    if args.num_inference_steps is not None:
        cfg_dict.model.scheduler.num_inference_steps = args.num_inference_steps
    if args.prompt:
        cfg_dict.test.prompt = args.prompt
    if args.negative_prompt:
        cfg_dict.test.negative_prompt = args.negative_prompt
    if "noise_seed" in cfg_dict.model.denoiser:
        cfg_dict.model.denoiser.noise_seed = args.noise_seed

    OmegaConf.set_struct(cfg_dict, True)
    cfg = load_typed_root_config(cfg_dict)

    step_tracker = StepTracker(0)

    print("[fast_infer] Building T2V wrapper...")
    wrapper = T2VWrapper(
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        freeze_cfg=cfg.freeze,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        val_cfg=cfg.val,
        sampler_cfg=cfg.sampler,
        step_tracker=step_tracker,
        output_dir=output_dir,
        batch_size=1,
        val_check_interval=cfg.trainer.val_check_interval,
        mode="test",
    )

    print(f"[fast_infer] Loading state_dict from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[fast_infer] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"[fast_infer] first missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"[fast_infer] first unexpected: {unexpected[:3]}")

    device = torch.device(args.device)
    wrapper.eval()
    wrapper.to(device)

    # Sampler.log_vis tries to hit `self.logger.log_image(...)` inside
    # generate_batch_with_scene. Trainer 없이 instantiate한 LightningModule은
    # self.logger==None이라 AttributeError. inference에선 vis log가 필요없으니 no-op.
    wrapper.sampler.log_vis = lambda *a, **kw: None

    print("[fast_infer] Building dataset (single scene, stage='test' so evaluation_index applies)...")
    dataset = get_dataset(
        cfg.dataset,
        "test",
        step_tracker,
        generator=None,
        force_shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=safe_collate,
        shuffle=False,
    )

    print("[fast_infer] Sampling...")
    fps = getattr(cfg.dataset, "fps", 24)
    precision = (
        torch.bfloat16
        if cfg.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
        else torch.float16
        if cfg.trainer.precision in ("16-mixed", "16-true", "fp16")
        else torch.float32
    )

    # Build the 8-combo grid: cfg ∈ {user_cfg, 1.0} × text ∈ {empty, user, ood, dataset}.
    # `dataset` text는 첫 batch에서 `batch["text"][0]`를 읽어 결정.
    # `ood` text는 `--ood_prompt`가 주어졌을 때만 활성.
    user_cfg = cfg.model.cfg_scale
    user_prompt = args.prompt or ""
    ood_prompt = args.ood_prompt or ""
    combos_template = [
        ("empty",    ""),
        ("user",     user_prompt),
        ("ood_text", ood_prompt),
        ("dataset",  None),  # filled at runtime
    ]
    cfg_scales = [user_cfg, 1.0]
    if abs(user_cfg - 1.0) < 1e-6:
        cfg_scales = [user_cfg]

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            batch = move_to_device(batch, device)
            if args.negative_prompt:
                batch["negative_prompt"] = args.negative_prompt

            v_c = batch["context"]["extrinsics"].shape[1]
            v_t = batch["target"]["extrinsics"].shape[1]
            print(f"[fast_infer] scene={batch['scene']}, context={v_c}, target={v_t}")

            # dataset가 제공하는 prompt 추출 (없으면 빈 문자열 → 그 combo skip)
            dataset_text_raw = batch.get("text", None)
            if dataset_text_raw is not None and len(dataset_text_raw) > 0:
                dataset_text = dataset_text_raw[0] if isinstance(dataset_text_raw, (list, tuple)) else str(dataset_text_raw)
            else:
                dataset_text = ""

            # Build combos with dataset text filled in
            combos = []
            for cfg_s in cfg_scales:
                for text_name, text_val in combos_template:
                    if text_name == "dataset":
                        text_val = dataset_text
                        if not text_val:
                            continue  # skip dataset combo if no dataset prompt available
                    elif text_name == "ood_text":
                        if not text_val:
                            continue  # skip ood combo if --ood_prompt not given
                    tag = f"cfg{cfg_s:.1f}_text-{text_name}"
                    combos.append((cfg_s, text_val, tag))
            print(f"[fast_infer] {len(combos)} combo(s):")
            for cfg_s, p, tag in combos:
                print(f"  - {tag}: cfg={cfg_s}, prompt={p[:60]!r}{'...' if len(p) > 60 else ''}")

            # ── Dump model input for sanity (before preprocess_batch mutates) ──
            scene_root = output_dir / batch["scene"][0]
            input_dir = scene_root / "input"
            ctx_dir = input_dir / "context"
            tgt_dir = input_dir / "target"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            tgt_dir.mkdir(parents=True, exist_ok=True)
            ctx_pixels = batch["context"]["latent"][0].float().clamp(0, 1)
            tgt_pixels = batch["target"]["latent"][0].float().clamp(0, 1)
            save_image_video(
                images=ctx_pixels,
                indices=torch.arange(ctx_pixels.shape[0]),
                output_dir=ctx_dir,
                name="context",
                save_img=True,
                save_video=True,
                fps=fps,
            )
            save_image_video(
                images=tgt_pixels,
                indices=torch.arange(tgt_pixels.shape[0]),
                output_dir=tgt_dir,
                name="target_gt",
                save_img=False,
                save_video=True,
                fps=fps,
            )
            import json
            meta = {
                "scene": batch["scene"][0],
                "exp_name": args.exp_name,
                "ckpt": str(ckpt_path),
                "user_prompt": args.prompt,
                "dataset_prompt": dataset_text,
                "context_views": v_c,
                "target_views_raw": v_t,
                "context_shape": list(ctx_pixels.shape[-2:]),
                "target_shape": list(tgt_pixels.shape[-2:]),
                "context_indices": batch["context"]["index"][0].tolist() if "index" in batch["context"] else None,
                "target_indices": batch["target"]["index"][0].tolist() if "index" in batch["target"] else None,
            }
            with (input_dir / "meta.json").open("w") as fp:
                json.dump(meta, fp, indent=2, default=str)
            print(f"[fast_infer] saved inputs → {input_dir}")

            # preprocess_batch mutates extrinsics → call ONCE then reuse.
            batch = preprocess_batch(batch, index=v_c // 2)

            for combo_idx, (cfg_scale_i, prompt_i, tag) in enumerate(combos):
                cfg.model.cfg_scale = cfg_scale_i
                wrapper.model_cfg.cfg_scale = cfg_scale_i
                batch["text"] = [prompt_i] * len(batch["scene"])
                print(f"\n[fast_infer] === combo {combo_idx+1}/{len(combos)}: {tag} ===")
                print(f"  cfg_scale={cfg_scale_i}, prompt={prompt_i!r}")

                def _sample_raw():
                    """N samples 그대로 (N*B, F, C, H, W) 반환. variance 계산용."""
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=precision,
                        enabled=(device.type == "cuda" and precision != torch.float32),
                    ):
                        s, u, _ = wrapper.generate_batch_with_scene(
                            batch, wrapper.sampler, repeat_factor=args.repeat_factor,
                        )
                    return s, u

                def _reduce(s_raw):
                    """(N*B, F, C, H, W) → (B, F, C, H, W) MC mean + variance."""
                    if args.repeat_factor > 1:
                        N = args.repeat_factor
                        NB, F_v, C, H_v, W_v = s_raw.shape
                        B = NB // N
                        s_nbf = s_raw.view(N, B, F_v, C, H_v, W_v).float().clamp(0, 1)
                        return s_nbf.mean(dim=0), s_nbf.std(dim=0).mean(dim=2)
                    return s_raw.float().clamp(0, 1), None

                # ON: ControlNet active
                torch.manual_seed(args.noise_seed)
                sampled_raw, uncertainty_maps = _sample_raw()
                sampled_views, variance_maps = _reduce(sampled_raw)

                # OFF: ac3d_projectors weights를 0으로 강제 후 같은 seed로 재샘플.
                # diff = ON - OFF → 순수 ControlNet residual의 픽셀 공간 영향.
                ablation_off = None
                if args.controlnet_ablation:
                    projs = getattr(wrapper.denoiser.model, "ac3d_projectors", None)
                    if projs is not None:
                        saved = []
                        for proj in projs:
                            saved.append((
                                proj.weight.data.clone(),
                                proj.bias.data.clone() if proj.bias is not None else None,
                            ))
                            proj.weight.data.zero_()
                            if proj.bias is not None:
                                proj.bias.data.zero_()
                        try:
                            torch.manual_seed(args.noise_seed)
                            off_raw, _ = _sample_raw()
                            ablation_off, _ = _reduce(off_raw)
                        finally:
                            for proj, (w, b) in zip(projs, saved):
                                proj.weight.data.copy_(w)
                                if b is not None and proj.bias is not None:
                                    proj.bias.data.copy_(b)
                    else:
                        print("  [ablation] no ac3d_projectors found — skipping OFF pass.")

                for j in range(sampled_views.shape[0]):
                    scene_name = batch["scene"][j]
                    out_subdir = output_dir / scene_name / tag
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    save_image_video(
                        images=sampled_views[j].float().clamp(0, 1),
                        indices=torch.arange(0, sampled_views[j].shape[0]),
                        output_dir=out_subdir,
                        name=wrapper.sampler.cfg.name,
                        save_img=False,
                        save_video=True,
                        fps=fps,
                    )
                    # combo의 실제 text 입력 저장
                    with (out_subdir / "prompt.txt").open("w") as fp:
                        fp.write(f"cfg_scale={cfg_scale_i}\ntext={prompt_i!r}\n")
                    # ControlNet ablation: OFF mp4 + diff visualization
                    if ablation_off is not None:
                        from src.misc.image_io import colorize
                        off_j = ablation_off[j].float().clamp(0, 1)        # (F, C, H, W)
                        save_image_video(
                            images=off_j,
                            indices=torch.arange(0, off_j.shape[0]),
                            output_dir=out_subdir,
                            name="controlnet_off",
                            save_img=False,
                            save_video=True,
                            fps=fps,
                        )
                        diff = (sampled_views[j].float() - off_j).abs().mean(dim=1).detach().cpu()  # (F, H, W)
                        torch.save(diff, out_subdir / "controlnet_diff.pt")
                        d_min = diff.amin()
                        d_max = diff.amax()
                        d_norm = (diff - d_min) / (d_max - d_min + 1e-8)
                        diff_vis = []
                        for fr in d_norm:
                            diff_vis.append(
                                torch.from_numpy(colorize(fr.numpy())).permute(2, 0, 1).float() / 255.0
                            )
                        diff_rgb = torch.stack(diff_vis)
                        save_image_video(
                            images=diff_rgb,
                            indices=torch.arange(0, diff_rgb.shape[0]),
                            output_dir=out_subdir,
                            name="controlnet_diff",
                            save_img=False,
                            save_video=True,
                            fps=fps,
                        )
                    # uncertainty map 저장 (viridis + raw .pt). uncertainty_maps shape:
                    # (b, v_t, h, w) — diffusion sample()의 pred_conditional.norm(dim=2).
                    if uncertainty_maps is not None:
                        # SceneTok 원조 visualization 패턴 (image_io.colorize + temporal
                        # repeat 방식)에 맞춤:
                        #   - cmap = magma_r (낮음=어두움, 높음=노랑)
                        #   - per-frame min/max normalize
                        #   - 공간 bilinear upsample, 시간축은 step-wise repeat (또는 F_pix
                        #     가 V_lat의 배수가 아니면 linspace nearest-index)
                        import torch.nn.functional as F_t
                        from einops import rearrange, repeat as einops_repeat
                        from src.misc.image_io import colorize

                        # raw .pt: latent res 그대로 저장
                        u_raw = uncertainty_maps[j].detach().cpu().float()  # (V_lat, H_lat, W_lat)
                        torch.save(u_raw, out_subdir / "uncertainty.pt")

                        u = u_raw.unsqueeze(0)              # (1, V_lat, H_lat, W_lat)
                        V_lat = u.shape[1]
                        F_pix, _, H_pix, W_pix = sampled_views[j].shape

                        # Per-frame normalize
                        u_min = u.reshape(1, V_lat, -1).min(dim=-1).values  # (1, V_lat)
                        u_max = u.reshape(1, V_lat, -1).max(dim=-1).values
                        u_norm = (u - u_min[..., None, None]) / (
                            u_max[..., None, None] - u_min[..., None, None] + 1e-6
                        )                                   # (1, V_lat, H_lat, W_lat)

                        # Spatial bilinear upsample
                        u_norm = u_norm.unsqueeze(2)        # (1, V_lat, 1, H_lat, W_lat)
                        u_norm = F_t.interpolate(
                            rearrange(u_norm, "b v c h w -> (b v) c h w"),
                            size=(H_pix, W_pix), mode="bilinear", align_corners=False,
                        )
                        u_norm = rearrange(
                            u_norm, "(b v) c h w -> b v c h w", b=1, v=V_lat,
                        )                                   # (1, V_lat, 1, H_pix, W_pix)

                        # Temporal expansion (step-wise repeat vs linspace fallback)
                        if V_lat != F_pix:
                            if F_pix % V_lat == 0:
                                r = F_pix // V_lat
                                u_norm = einops_repeat(
                                    u_norm, "b v c h w -> b (v r) c h w", r=r,
                                )
                            else:
                                time_idx = torch.linspace(0, V_lat - 1, steps=F_pix).round().long()
                                u_norm = u_norm[:, time_idx]

                        # Colorize per frame with magma_r
                        u_vis = []
                        for frame in u_norm[0]:             # (1, H_pix, W_pix)
                            colored = colorize(frame[0].numpy())  # (H, W, 3) uint8
                            u_vis.append(torch.from_numpy(colored).permute(2, 0, 1).float() / 255.0)
                        u_rgb = torch.stack(u_vis)          # (F_pix, 3, H_pix, W_pix)
                        save_image_video(
                            images=u_rgb,
                            indices=torch.arange(0, u_rgb.shape[0]),
                            output_dir=out_subdir,
                            name="uncertainty",
                            save_img=False,
                            save_video=True,
                            fps=fps,
                        )
                        # Grayscale (per-frame normalized scalar)
                        u_gray = u_norm[0].expand(-1, 3, -1, -1).float()  # (F_pix, 3, H_pix, W_pix)
                        save_image_video(
                            images=u_gray,
                            indices=torch.arange(0, u_gray.shape[0]),
                            output_dir=out_subdir,
                            name="uncertainty_gray",
                            save_img=False,
                            save_video=True,
                            fps=fps,
                        )
                    # MC variance map (pixel-space): repeat_factor>1일 때만.
                    # (F_pix, H_pix, W_pix) 스칼라 필드 → magma_r + grayscale + raw .pt.
                    if variance_maps is not None:
                        from src.misc.image_io import colorize
                        var_map = variance_maps[j].detach().cpu().float()   # (F_pix, H_pix, W_pix)
                        torch.save(var_map, out_subdir / "variance.pt")
                        vm_min = var_map.amin()
                        vm_max = var_map.amax()
                        var_norm = (var_map - vm_min) / (vm_max - vm_min + 1e-8)
                        var_vis = []
                        for frame in var_norm:
                            colored = colorize(frame.numpy())
                            var_vis.append(torch.from_numpy(colored).permute(2, 0, 1).float() / 255.0)
                        var_rgb = torch.stack(var_vis)
                        save_image_video(
                            images=var_rgb,
                            indices=torch.arange(0, var_rgb.shape[0]),
                            output_dir=out_subdir,
                            name="variance",
                            save_img=False,
                            save_video=True,
                            fps=fps,
                        )
                        var_gray = var_norm.unsqueeze(1).expand(-1, 3, -1, -1).float()
                        save_image_video(
                            images=var_gray,
                            indices=torch.arange(0, var_gray.shape[0]),
                            output_dir=out_subdir,
                            name="variance_gray",
                            save_img=False,
                            save_video=True,
                            fps=fps,
                        )
                    print(f"  saved: {out_subdir}/{wrapper.sampler.cfg.name}.mp4")
            if batch_idx + 1 >= args.max_scenes:
                break  # default 1 = single scene (backward compat); set --max_scenes N for more

    print(f"[fast_infer] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
