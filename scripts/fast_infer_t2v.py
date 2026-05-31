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
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative_prompt", default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--target_shape", default=None, help="H,W (e.g. 480,832)")
    p.add_argument("--context_shape", default=None, help="H,W (e.g. 256,256)")
    p.add_argument("--num_context_views", type=int, default=None)
    p.add_argument("--num_target_views", type=int, default=None)
    p.add_argument("--output_dir", default=None)
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
    cfg_dict.dataset.val_seen = True
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

    # Build the 4-combo grid: {user_cfg, 1.0} × {user_prompt, ""}.
    user_cfg = cfg.model.cfg_scale
    user_prompt = args.prompt or ""
    combos = [
        (user_cfg, user_prompt, f"cfg{user_cfg:.1f}_prompt-full"),
        (user_cfg, "",          f"cfg{user_cfg:.1f}_prompt-empty"),
        (1.0,      user_prompt, "cfg1.0_prompt-full"),
        (1.0,      "",          "cfg1.0_prompt-empty"),
    ]
    # dedup if user_cfg already == 1.0
    seen = set()
    combos = [c for c in combos if not (c[2] in seen or seen.add(c[2]))]
    print(f"[fast_infer] {len(combos)} combo(s):")
    for cfg_s, p, tag in combos:
        print(f"  - {tag}: cfg={cfg_s}, prompt={p!r}")

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

            # preprocess_batch mutates extrinsics → call ONCE then reuse.
            batch = preprocess_batch(batch, index=v_c // 2)

            for combo_idx, (cfg_scale_i, prompt_i, tag) in enumerate(combos):
                cfg.model.cfg_scale = cfg_scale_i
                wrapper.model_cfg.cfg_scale = cfg_scale_i
                batch["text"] = [prompt_i] * len(batch["scene"])
                print(f"\n[fast_infer] === combo {combo_idx+1}/{len(combos)}: {tag} ===")
                print(f"  cfg_scale={cfg_scale_i}, prompt={prompt_i!r}")

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=precision,
                    enabled=(device.type == "cuda" and precision != torch.float32),
                ):
                    sampled_views, _, _ = wrapper.generate_batch_with_scene(
                        batch,
                        wrapper.sampler,
                        repeat_factor=1,
                    )

                for j in range(sampled_views.shape[0]):
                    scene_name = batch["scene"][j]
                    out_subdir = output_dir / scene_name / tag
                    save_image_video(
                        images=sampled_views[j].float().clamp(0, 1),
                        indices=torch.arange(0, sampled_views[j].shape[0]),
                        output_dir=out_subdir,
                        name=wrapper.sampler.cfg.name,
                        save_img=False,
                        save_video=True,
                        fps=fps,
                    )
                    print(f"  saved: {out_subdir}/{wrapper.sampler.cfg.name}.mp4")
            break  # single batch only

    print(f"[fast_infer] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
