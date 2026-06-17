"""Fast inference variant of fast_infer_t2v.py that runs a dynamicverse-trained
TI2V ckpt on a DL3DV (or other) dataset by Hydra-composing the experiment from
scratch and overriding `dataset=...`.

Why a separate entrypoint: the original `fast_infer_t2v.py` loads the saved
`.hydra/config.yaml` verbatim. That config has the trained dataset baked in
(e.g. dynamicverse with its own `evaluation_index_path`, scene-name format,
prompt-file lookup), so we can't simply set `cfg.dataset.scene_id` to a
DL3DV-style key. Here we instead re-compose the experiment's model/denoiser
blocks but swap the dataset entirely.

Caveats:
- Re-composed config may not be byte-identical to the trained run if extra
  shell overrides were used. We mirror the ones from the training shells.
- State-dict load is `strict=False` — minor key mismatches OK (e.g. dataset
  buffers).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

REPO_ROOT = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
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
    p.add_argument("--exp_name", required=True,
                   help="exp dir under exp/<exp_name>/ — used only to find ckpt + tags.")
    p.add_argument("--experiment",
                   default="custom/scenetok_va-wan-ti2v_dynamicverse",
                   help="Hydra `+experiment=<path>` for model arch. Defaults to dynamicverse.")
    p.add_argument("--dataset", default="dl3dv",
                   help="Dataset module name (matches config/dataset/<name>.yaml).")
    p.add_argument("--dataset_root", default="./DATA/DL3DV/DL3DV-960",
                   help="Raw dataset root (replaces latent paths if present).")
    p.add_argument("--scene_id", required=True,
                   help="Dataset-native scene id (e.g. '1K/a4c2...' for DL3DV).")
    p.add_argument("--ckpt", default=None,
                   help="Override `my_checkpoints/<exp_name>/last.ckpt`.")
    p.add_argument("--prompt", default="",
                   help="User-supplied prompt (in-distribution baseline).")
    p.add_argument("--ood_prompt", default="",
                   help="OOD prompt (from another dataset) for ood_text combo.")
    p.add_argument("--negative_prompt", default=None)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--context_shape", default="256,448")
    p.add_argument("--target_shape", default="480,832")
    p.add_argument("--num_context_views", type=int, default=12)
    p.add_argument("--num_target_views", type=int, default=10)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--load_prompts", action="store_true",
                   help="Enable dataset.load_prompts=True (DL3DV needs this for "
                        "the dataset prompt combo to fire). Other datasets ignore.")
    p.add_argument("--controlnet_ablation", action="store_true",
                   help="ac3d_projectors weights를 0으로 강제한 두번째 sample → "
                        "controlnet OFF mp4 + |on - off| diff mp4. 2x sampling per combo.")
    p.add_argument("--scene_input_type", default=None,
                   help="model.denoiser.scene_input_type override (yaml default가 다르면 명시).")
    p.add_argument("--camera_input_type", default=None,
                   help="model.denoiser.camera_input_type override (yaml default가 다르면 명시).")
    p.add_argument("--static_target_camera", action="store_true",
                   help="target extrinsics/intrinsics를 context[:, 0]으로 모두 덮어써서 "
                        "정지 카메라 시점(첫 context view 고정)에서 dynamic foreground만 "
                        "움직이게 만든다. context view 1개 추론에 주로 사용.")
    p.add_argument("--lora_disabled", action="store_true",
                   help="model.denoiser.lora.enabled=False로 강제. yaml default가 true라 "
                        "no_lora 학습 ckpt 추론 시 key mismatch 발생 → 이 옵션으로 fix.")
    p.add_argument("--prompt_style", default=None,
                   help="dataset.prompt_style override (e.g. category_first for dynamicverse "
                        "models trained with category.json dynamic-object descriptions).")
    return p.parse_args()


def parse_shape(s):
    return [int(x) for x in s.split(",")]


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def build_cfg(args):
    from hydra import compose, initialize_config_dir
    overrides = [
        f"+experiment={args.experiment}",
        f"dataset={args.dataset}",
        "mode=test",
        "wandb.activated=false",
    ]
    print(f"[infer] hydra-compose: {overrides}")
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    OmegaConf.set_struct(cfg_dict, False)
    # Drop latent-style fields if any
    for key in ("context_root", "target_root", "map_dict"):
        if key in cfg_dict.dataset:
            del cfg_dict.dataset[key]
    cfg_dict.dataset.root = args.dataset_root
    cfg_dict.dataset.smallset = True
    cfg_dict.dataset.stage_override = "train"
    cfg_dict.dataset.val_seen = True
    cfg_dict.dataset.scene_id = args.scene_id
    # Drop dynamicverse-specific evaluation_index_path (lives in dynamicverse cfg only)
    if "evaluation_index_path" in cfg_dict.dataset:
        cfg_dict.dataset.evaluation_index_path = None
    cfg_dict.dataset.context_shape = parse_shape(args.context_shape)
    cfg_dict.dataset.target_shape = parse_shape(args.target_shape)
    cfg_dict.dataset.view_sampler.num_context_views = args.num_context_views
    cfg_dict.dataset.view_sampler.num_target_views = args.num_target_views
    if args.load_prompts:
        cfg_dict.dataset.load_prompts = True
        # DL3DV는 target frame count(37 또는 49)에 따른 prompts_<N>.json 파일을 사용.
        # target=37 가정 (4N+1 layout). 다르면 prompts_filename 추가 override 필요.
        if "prompts_filename" not in cfg_dict.dataset:
            cfg_dict.dataset.prompts_filename = "prompts_37.json"

    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.data_loader.test.num_workers = 0
    cfg_dict.data_loader.test.batch_size = 1
    cfg_dict.freeze.denoiser = True
    cfg_dict.freeze.compressor = True
    cfg_dict.freeze.autoencoder = True
    cfg_dict.model.cfg_scale = args.cfg_scale
    cfg_dict.model.scheduler.num_inference_steps = args.num_inference_steps
    if "noise_seed" in cfg_dict.model.denoiser:
        cfg_dict.model.denoiser.noise_seed = args.noise_seed
    if args.lora_disabled:
        cfg_dict.model.denoiser.lora.enabled = False
    if args.prompt_style is not None:
        cfg_dict.dataset.prompt_style = args.prompt_style
    if args.scene_input_type is not None:
        cfg_dict.model.denoiser.scene_input_type = args.scene_input_type
    if args.camera_input_type is not None:
        cfg_dict.model.denoiser.camera_input_type = args.camera_input_type
    if args.prompt:
        cfg_dict.test.prompt = args.prompt
    if args.negative_prompt:
        cfg_dict.test.negative_prompt = args.negative_prompt
    OmegaConf.set_struct(cfg_dict, True)
    return load_typed_root_config(cfg_dict)


def main():
    args = parse_args()
    torch.manual_seed(args.noise_seed)

    ckpt_path = (
        Path(args.ckpt) if args.ckpt
        else REPO_ROOT / "my_checkpoints" / args.exp_name / "last.ckpt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else REPO_ROOT / "results" / f"fast_infer_{args.exp_name}_swap-{args.dataset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)

    step_tracker = StepTracker(0)
    print("[infer] Building T2VWrapper...")
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

    print(f"[infer] Loading ckpt from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[infer] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"[infer]   first missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"[infer]   first unexpected: {unexpected[:3]}")

    device = torch.device(args.device)
    wrapper.eval().to(device)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None

    print("[infer] Building dataset...")
    dataset = get_dataset(cfg.dataset, "test", step_tracker, generator=None, force_shuffle=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=safe_collate, shuffle=False)

    fps = getattr(cfg.dataset, "fps", 24)
    precision = (
        torch.bfloat16 if cfg.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
        else torch.float16 if cfg.trainer.precision in ("16-mixed", "16-true", "fp16")
        else torch.float32
    )

    # 8-combo grid: cfg ∈ {user, 1.0} × text ∈ {empty, user, ood, dataset}
    user_cfg = cfg.model.cfg_scale
    user_prompt = args.prompt or ""
    ood_prompt = args.ood_prompt or ""
    combos_template = [
        ("empty",    ""),
        ("user",     user_prompt),
        ("ood_text", ood_prompt),
        ("dataset",  None),
    ]
    cfg_scales = [user_cfg, 1.0] if abs(user_cfg - 1.0) > 1e-6 else [user_cfg]

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            batch = move_to_device(batch, device)
            if args.negative_prompt:
                batch["negative_prompt"] = args.negative_prompt
            v_c = batch["context"]["extrinsics"].shape[1]
            v_t = batch["target"]["extrinsics"].shape[1]
            scene = batch["scene"][0]
            print(f"[infer] scene={scene}, context={v_c}, target={v_t}")
            if args.static_target_camera:
                ctx_ext0 = batch["context"]["extrinsics"][:, 0:1]
                ctx_int0 = batch["context"]["intrinsics"][:, 0:1]
                batch["target"]["extrinsics"] = ctx_ext0.expand(-1, v_t, -1, -1).clone()
                batch["target"]["intrinsics"] = ctx_int0.expand(-1, v_t, -1, -1).clone()
                print(f"[static_target_camera] target extrinsics/intrinsics overwritten with context[:, 0] for all {v_t} views")

            # dataset prompt extraction (없으면 빈 문자열 → dataset combo skip)
            dataset_text_raw = batch.get("text", None)
            if dataset_text_raw is not None and len(dataset_text_raw) > 0:
                dataset_text = dataset_text_raw[0] if isinstance(dataset_text_raw, (list, tuple)) else str(dataset_text_raw)
            else:
                dataset_text = ""

            combos = []
            for cfg_s in cfg_scales:
                for text_name, text_val in combos_template:
                    if text_name == "dataset":
                        text_val = dataset_text
                        if not text_val:
                            continue
                    elif text_name == "ood_text":
                        if not text_val:
                            continue
                    tag = f"cfg{cfg_s:.1f}_text-{text_name}"
                    combos.append((cfg_s, text_val, tag))
            print(f"[infer] {len(combos)} combo(s):")
            for cfg_s, p, tag in combos:
                print(f"  - {tag}: cfg={cfg_s}, prompt={p[:60]!r}{'...' if len(p) > 60 else ''}")

            # ── Dump model input to disk for sanity ────────────────────────
            #   context/  : V_c PNGs + context.mp4    (input compressor가 보는 뷰들)
            #   target/   : target.mp4 (raw GT pixels) + indices.json + extrinsics/intrinsics
            scene_root = output_dir / scene.replace("/", "__")
            input_dir = scene_root / "input"
            ctx_dir = input_dir / "context"
            tgt_dir = input_dir / "target"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            tgt_dir.mkdir(parents=True, exist_ok=True)

            ctx_pixels = batch["context"]["latent"][0].float().clamp(0, 1)  # (V_c, 3, H, W)
            tgt_pixels = batch["target"]["latent"][0].float().clamp(0, 1)   # (V_t, 3, H, W)
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
                "scene": scene,
                "dataset": args.dataset,
                "dataset_root": args.dataset_root,
                "scene_id": args.scene_id,
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
            print(f"[infer] saved inputs → {input_dir}")

            batch = preprocess_batch(batch, index=v_c // 2)

            for cfg_scale_i, prompt_i, tag in combos:
                cfg.model.cfg_scale = cfg_scale_i
                wrapper.model_cfg.cfg_scale = cfg_scale_i
                batch_local = dict(batch)
                batch_local["text"] = [prompt_i] * len(batch_local["scene"])
                print(f"\n[infer] === {tag}: cfg={cfg_scale_i}, prompt={prompt_i[:50]!r} ===")

                def _sample():
                    with torch.amp.autocast(
                        device_type="cuda", dtype=precision,
                        enabled=(precision != torch.float32)
                    ):
                        s, u, _ = wrapper.generate_batch_with_scene(
                            batch_local, wrapper.sampler, repeat_factor=1
                        )
                    return s.float().clamp(0, 1), u

                torch.manual_seed(args.noise_seed)
                sampled, uncertainty_maps = _sample()

                # ControlNet ablation: ac3d_projectors zero → re-sample with same seed
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
                            ablation_off, _ = _sample()
                        finally:
                            for proj, (w, b) in zip(projs, saved):
                                proj.weight.data.copy_(w)
                                if b is not None and proj.bias is not None:
                                    proj.bias.data.copy_(b)

                save_dir = scene_root / tag
                save_dir.mkdir(parents=True, exist_ok=True)
                with (save_dir / "prompt.txt").open("w") as fp:
                    fp.write(f"cfg_scale={cfg_scale_i}\ntext={prompt_i!r}\n")
                if ablation_off is not None:
                    from src.misc.image_io import colorize as _colorize_diff
                    off_v = ablation_off[0]
                    save_image_video(
                        images=off_v,
                        indices=torch.arange(0, off_v.shape[0]),
                        output_dir=save_dir,
                        name="controlnet_off",
                        save_img=False,
                        save_video=True,
                        fps=fps,
                    )
                    diff = (sampled[0].float() - off_v.float()).abs().mean(dim=1).detach().cpu()
                    torch.save(diff, save_dir / "controlnet_diff.pt")
                    d_min, d_max = diff.amin(), diff.amax()
                    d_norm = (diff - d_min) / (d_max - d_min + 1e-8)
                    diff_vis = []
                    for fr in d_norm:
                        diff_vis.append(torch.from_numpy(_colorize_diff(fr.numpy())).permute(2, 0, 1).float() / 255.0)
                    diff_rgb = torch.stack(diff_vis)
                    save_image_video(
                        images=diff_rgb,
                        indices=torch.arange(0, diff_rgb.shape[0]),
                        output_dir=save_dir,
                        name="controlnet_diff",
                        save_img=False,
                        save_video=True,
                        fps=fps,
                    )
                if uncertainty_maps is not None:
                    # SceneTok 원조 visualization 패턴 (image_io.colorize, magma_r,
                    # per-frame normalize, bilinear spatial + step-wise temporal).
                    import torch.nn.functional as F_t
                    from einops import rearrange, repeat as einops_repeat
                    from src.misc.image_io import colorize

                    u_raw = uncertainty_maps[0].detach().cpu().float()  # (V_lat, H_lat, W_lat)
                    torch.save(u_raw, save_dir / "uncertainty.pt")

                    u = u_raw.unsqueeze(0)
                    V_lat = u.shape[1]
                    F_pix, _, H_pix, W_pix = sampled[0].shape

                    u_min = u.reshape(1, V_lat, -1).min(dim=-1).values
                    u_max = u.reshape(1, V_lat, -1).max(dim=-1).values
                    u_norm = (u - u_min[..., None, None]) / (
                        u_max[..., None, None] - u_min[..., None, None] + 1e-6
                    )

                    u_norm = u_norm.unsqueeze(2)
                    u_norm = F_t.interpolate(
                        rearrange(u_norm, "b v c h w -> (b v) c h w"),
                        size=(H_pix, W_pix), mode="bilinear", align_corners=False,
                    )
                    u_norm = rearrange(u_norm, "(b v) c h w -> b v c h w", b=1, v=V_lat)

                    if V_lat != F_pix:
                        if F_pix % V_lat == 0:
                            r = F_pix // V_lat
                            u_norm = einops_repeat(u_norm, "b v c h w -> b (v r) c h w", r=r)
                        else:
                            time_idx = torch.linspace(0, V_lat - 1, steps=F_pix).round().long()
                            u_norm = u_norm[:, time_idx]

                    u_vis = []
                    for frame in u_norm[0]:
                        colored = colorize(frame[0].numpy())
                        u_vis.append(torch.from_numpy(colored).permute(2, 0, 1).float() / 255.0)
                    u_rgb = torch.stack(u_vis)
                    save_image_video(
                        images=u_rgb,
                        indices=torch.arange(0, u_rgb.shape[0]),
                        output_dir=save_dir,
                        name="uncertainty",
                        save_img=False,
                        save_video=True,
                        fps=fps,
                    )
                    u_gray = u_norm[0].expand(-1, 3, -1, -1).float()
                    save_image_video(
                        images=u_gray,
                        indices=torch.arange(0, u_gray.shape[0]),
                        output_dir=save_dir,
                        name="uncertainty_gray",
                        save_img=False,
                        save_video=True,
                        fps=fps,
                    )
                save_image_video(
                    images=sampled[0],
                    indices=torch.arange(0, sampled.shape[1]),
                    output_dir=save_dir,
                    name="full_sequence",
                    save_img=False,
                    save_video=True,
                    fps=fps,
                )
                print(f"  saved: {save_dir / 'full_sequence.mp4'}")
            break  # only first scene
    print(f"\n[infer] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
