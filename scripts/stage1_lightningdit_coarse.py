"""Stage 1 of lightningDiT → ReCo two-stage inference.

Loads a SceneTok DiffusionWrapper (`denoiser=lightningdit_L_1` etc.) from an
exp dir + last.ckpt, runs sampling on a single scene, decodes through the
target Wan VAE (configured in saved yaml), and writes the coarse pixel video
to disk as an mp4.

Stage 2 (`stage2_reco_edit.py`) loads that mp4 and runs ReCo. Splitting into
two processes avoids the `diffsynth` package name collision between our main
`src/model/DiffSynth-Studio` (Wan 2.2 / TI2V) and ReCo's vendored
`src/model/ReCo/DiffSynth-Studio` (Wan 2.1 / VACE).
"""
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
from src.model.diffusion_wrapper import DiffusionWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenetok_exp",
        default=None,
        help="Name of an exp/<name>/.hydra dump (own training run). "
             "Mutually exclusive with --scenetok_experiment.",
    )
    p.add_argument(
        "--scenetok_experiment",
        default=None,
        help="Name of a published `config/experiment/<name>.yaml` to Hydra-compose "
             "on the fly. Use this for upstream/published ckpts that have no exp dir. "
             "Pair with explicit --scenetok_ckpt.",
    )
    p.add_argument("--scenetok_ckpt", default=None)
    p.add_argument("--scenetok_config", default=None)
    p.add_argument("--scene_id", required=True)
    p.add_argument("--stage_override", default="train", choices=["train", "val", "test"])
    p.add_argument("--context_shape", default="256,448")
    p.add_argument("--target_shape", default="480,832")
    p.add_argument("--num_context_views", type=int, default=16)
    p.add_argument("--num_target_views", type=int, default=10)
    p.add_argument("--scenetok_inference_steps", type=int, default=25)
    p.add_argument("--scenetok_cfg_scale", type=float, default=1.0)
    p.add_argument("--scenetok_seed", type=int, default=0)
    p.add_argument("--evaluation_index_path", default=None)
    p.add_argument("--out_dir", required=True, help="where to save coarse mp4 + scene_name")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
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


def main():
    args = parse_args()

    if (args.scenetok_exp is None) == (args.scenetok_experiment is None):
        raise ValueError("Provide exactly one of --scenetok_exp / --scenetok_experiment.")

    if args.scenetok_experiment is not None:
        # Hydra-compose path: published config without a saved exp dir.
        from hydra import compose, initialize_config_dir

        if args.scenetok_ckpt is None:
            raise ValueError("--scenetok_ckpt is required when using --scenetok_experiment.")
        ckpt_path = Path(args.scenetok_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SceneTok ckpt not found: {ckpt_path}")
        print(f"[stage1] hydra-compose experiment={args.scenetok_experiment}")
        print(f"[stage1] ckpt: {ckpt_path}")
        with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
            cfg_dict = compose(
                config_name="main",
                overrides=[
                    f"+experiment={args.scenetok_experiment}",
                    "mode=test",
                    "wandb.activated=false",
                    # Some published experiments set precomputed-latent paths that
                    # only apply to the `latent` dataset (not `dl3dv`). Strip them
                    # so dacite doesn't choke on unknown fields.
                    "~dataset.context_root",
                    "~dataset.target_root",
                    "~dataset.map_dict",
                    # `dl3dv.yaml` defaults `root: null`; point it at the actual
                    # raw dataset so dataset init can find frames.
                    "dataset.root=./DATA/DL3DV/DL3DV-960",
                ],
            )
    else:
        config_path = (
            Path(args.scenetok_config)
            if args.scenetok_config
            else REPO_ROOT / "exp" / args.scenetok_exp / ".hydra" / "config.yaml"
        )
        ckpt_path = (
            Path(args.scenetok_ckpt)
            if args.scenetok_ckpt
            else REPO_ROOT / "my_checkpoints" / args.scenetok_exp / "last.ckpt"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"SceneTok config not found: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SceneTok ckpt not found: {ckpt_path}")

        print(f"[stage1] config: {config_path}")
        print(f"[stage1] ckpt:   {ckpt_path}")

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
    cfg_dict.dataset.scene_id = args.scene_id
    cfg_dict.dataset.context_shape = parse_shape(args.context_shape)
    cfg_dict.dataset.target_shape = parse_shape(args.target_shape)
    cfg_dict.dataset.view_sampler.num_context_views = args.num_context_views
    cfg_dict.dataset.view_sampler.num_target_views = args.num_target_views
    if args.evaluation_index_path:
        cfg_dict.dataset.evaluation_index_path = args.evaluation_index_path
    cfg_dict.model.cfg_scale = args.scenetok_cfg_scale
    cfg_dict.model.scheduler.num_inference_steps = args.scenetok_inference_steps
    if "noise_seed" in cfg_dict.model.denoiser:
        cfg_dict.model.denoiser.noise_seed = args.scenetok_seed
    OmegaConf.set_struct(cfg_dict, True)
    cfg = load_typed_root_config(cfg_dict)

    step_tracker = StepTracker(0)
    print("[stage1] building DiffusionWrapper...")
    wrapper = DiffusionWrapper(
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        freeze_cfg=cfg.freeze,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        val_cfg=cfg.val,
        sampler_cfg=cfg.sampler,
        step_tracker=step_tracker,
        output_dir=None,
        batch_size=1,
        val_check_interval=cfg.trainer.val_check_interval,
        mode="test",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[stage1] state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    wrapper.eval().to(args.device)
    wrapper.sampler.log_vis = lambda *a, **kw: None

    print("[stage1] dataset + sampling...")
    dataset = get_dataset(cfg.dataset, "test", step_tracker, generator=None, force_shuffle=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=safe_collate, shuffle=False)
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("Empty batch — check scene_id / evaluation_index_path")
    batch = move_to_device(batch, args.device)
    v_c = batch["context"]["extrinsics"].shape[1]
    batch = preprocess_batch(batch, index=v_c // 2)

    precision = (
        torch.bfloat16
        if cfg.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
        else torch.float32
    )
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=precision, enabled=(precision != torch.float32)
    ):
        sampled, _, _ = wrapper.generate_batch_with_scene(batch, wrapper.sampler, repeat_factor=1)
    # `sampled`: (1, F, 3, H, W) in [0, 1] (post Wan 2.2 VAE decode).
    print(f"[stage1] coarse video shape: {tuple(sampled.shape)}, "
          f"range [{sampled.min():.3f}, {sampled.max():.3f}]")

    scene_name = batch["scene"][0]
    out_dir = Path(args.out_dir) / scene_name / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image_video(
        images=sampled[0].float().clamp(0, 1),
        indices=torch.arange(0, sampled.shape[1]),
        output_dir=out_dir,
        name="coarse",
        save_img=False,
        save_video=True,
        fps=args.fps,
    )
    coarse_path = out_dir / "coarse.mp4"
    print(f"[stage1] saved coarse video → {coarse_path}")

    # Stage 2 metadata
    meta_path = out_dir / "scene.txt"
    meta_path.write_text(scene_name + "\n")


if __name__ == "__main__":
    main()
