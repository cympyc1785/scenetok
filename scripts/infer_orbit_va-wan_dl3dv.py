"""Single-view 360° orbit synthesis with `checkpoints/va-wan_dl3dv.ckpt`.

Takes one DL3DV scene's first frame as the only context view, synthesizes a
target camera trajectory orbiting around a look-at point in front of the
context camera, and runs the SceneTok pipeline (compressor → wan_ti2v denoiser
→ Wan VAE) to render the 360° video.

Camera convention: DL3DV's `convert_poses` returns c2w (`w2c.inverse()`) in
OpenCV convention (+X right, +Y down, +Z forward). The orbit camera is built
by:
  1. Place look-at point at `ref_pos + ref_forward * --orbit_radius`.
  2. At angle θ, rotate the (ref_pos − look_at) vector around `--orbit_axis`
     in WORLD frame (default Y) — i.e., the camera circles around the
     vertical axis through the look-at point at the same radius/height as
     the context view.
  3. Build c2w with look-at orientation (forward = look_at - cam_pos,
     up_world = --up_world).

After `preprocess_batch(batch, index=0)`, all extrinsics become relative to
the single context view. At θ=0 the target pose equals the context pose (so
the first generated frame should match the input frame).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

REPO_ROOT = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
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
        "--scenetok_experiment",
        default="scenetok_va-wan_shift4_dl3dv_finetuned",
    )
    p.add_argument(
        "--scenetok_ckpt",
        default=str(REPO_ROOT / "checkpoints" / "va-wan_dl3dv.ckpt"),
    )
    p.add_argument(
        "--evaluation_index_path",
        default=str(REPO_ROOT / "assets/evaluation_index/dl3dv_c16_37_standard.json"),
        help="Used only to pick which scene to draw a single context view from.",
    )
    p.add_argument(
        "--scene_index",
        type=int,
        default=0,
        help="Which scene in the eval index iteration order to use.",
    )
    p.add_argument(
        "--orbit_radius",
        type=float,
        default=1.0,
        help="Distance from context camera to the orbit center (look-at point), "
             "in DL3DV world units. DL3DV poses are NeRF-normalized so ~1.0 is "
             "typically inside the scene.",
    )
    p.add_argument(
        "--orbit_axis",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="y",
        help="World axis the camera orbits around (passes through the look-at "
             "point). Default y. Try -y or z if the orbit tilts.",
    )
    p.add_argument(
        "--up_world",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="-y",
        help="World up axis used to construct the camera's up vector (OpenCV "
             "convention is +Y down, so world up is typically -y).",
    )
    p.add_argument(
        "--full_angle_deg",
        type=float,
        default=360.0,
        help="Total sweep angle. Set to e.g. 180 to do a half-orbit.",
    )
    p.add_argument(
        "--start_angle_deg",
        type=float,
        default=0.0,
        help="Angle of the first frame. 0 means start at the context pose.",
    )
    p.add_argument(
        "--num_target_views",
        type=int,
        default=37,
        help="Number of target frames. Default 37 to match Wan 4N+1 layout used "
             "by the trained ckpt.",
    )
    p.add_argument("--scenetok_inference_steps", type=int, default=25)
    p.add_argument("--scenetok_cfg_scale", type=float, default=1.0)
    p.add_argument("--scenetok_seed", type=int, default=0)
    p.add_argument("--context_shape", default="256,256")
    p.add_argument("--target_shape", default="256,256")
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "results" / "infer_orbit_va-wan_dl3dv"),
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_shape(s):
    return [int(x) for x in s.split(",")]


def axis_vec(name: str) -> torch.Tensor:
    sign = -1.0 if name.startswith("-") else 1.0
    letter = name[-1]
    v = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[letter]
    return torch.tensor([sign * v[0], sign * v[1], sign * v[2]], dtype=torch.float32)


def rotation_matrix_about_axis(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rodrigues — rotate by `angle_rad` around unit `axis`."""
    a = axis / (axis.norm() + 1e-12)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    K = torch.tensor([
        [0.0, -a[2].item(), a[1].item()],
        [a[2].item(), 0.0, -a[0].item()],
        [-a[1].item(), a[0].item(), 0.0],
    ], dtype=torch.float32)
    return torch.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def look_at_c2w(cam_pos: torch.Tensor, look_at: torch.Tensor, up_world: torch.Tensor) -> torch.Tensor:
    """Build OpenCV c2w: forward = look_at − cam_pos, x = forward × up, y = forward × x."""
    forward = look_at - cam_pos
    forward = forward / (forward.norm() + 1e-12)
    # OpenCV: camera +Y points DOWN in image. So image-y axis in world = -up_world.
    # Pick right (x) perpendicular to forward and world-up, then re-derive up.
    right = torch.cross(up_world, forward, dim=-1)
    if right.norm() < 1e-6:
        # forward parallel to up — fall back to alternate up.
        alt = torch.tensor([1.0, 0.0, 0.0]) if abs(up_world[0].item()) < 0.9 else torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(alt, forward, dim=-1)
    right = right / (right.norm() + 1e-12)
    down = torch.cross(forward, right, dim=-1)  # camera +Y is down in OpenCV → world-down direction
    down = down / (down.norm() + 1e-12)
    rot = torch.stack([right, down, forward], dim=1)  # (3, 3) columns = camera basis in world
    c2w = torch.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = cam_pos
    return c2w


def make_orbit(
    ref_c2w: torch.Tensor,
    num_views: int,
    radius: float,
    orbit_axis: torch.Tensor,
    up_world: torch.Tensor,
    full_angle_deg: float,
    start_angle_deg: float,
) -> torch.Tensor:
    """Return (V, 4, 4) c2w trajectory orbiting around a look-at point."""
    ref_c2w = ref_c2w.float().cpu()
    ref_pos = ref_c2w[:3, 3]
    ref_forward = ref_c2w[:3, 2]
    look_at = ref_pos + ref_forward * radius

    rel_initial = ref_pos - look_at

    angles = torch.linspace(
        math.radians(start_angle_deg),
        math.radians(start_angle_deg + full_angle_deg),
        num_views,
    )

    poses = []
    for theta in angles:
        R = rotation_matrix_about_axis(orbit_axis, float(theta.item()))
        cam_pos = look_at + R @ rel_initial
        c2w = look_at_c2w(cam_pos, look_at, up_world)
        poses.append(c2w)
    return torch.stack(poses, dim=0)


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def build_wrapper_and_loader(args):
    from hydra import compose, initialize_config_dir
    print(f"[infer] hydra-compose: {args.scenetok_experiment}")
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                f"+experiment={args.scenetok_experiment}",
                "dataset=dl3dv",
                "mode=test",
                "wandb.activated=false",
            ],
        )
    OmegaConf.set_struct(cfg_dict, False)
    for key in ("context_root", "target_root", "map_dict"):
        if key in cfg_dict.dataset:
            del cfg_dict.dataset[key]
    cfg_dict.dataset.root = "./DATA/DL3DV/DL3DV-960"
    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.data_loader.test.num_workers = 0
    cfg_dict.data_loader.test.batch_size = 1
    cfg_dict.freeze.denoiser = True
    cfg_dict.freeze.compressor = True
    cfg_dict.freeze.autoencoder = True
    cfg_dict.dataset.smallset = False
    cfg_dict.dataset.stage_override = "train"
    cfg_dict.dataset.val_seen = True
    cfg_dict.dataset.evaluation_index_path = args.evaluation_index_path
    cfg_dict.dataset.scene_id = None
    cfg_dict.dataset.context_shape = parse_shape(args.context_shape)
    cfg_dict.dataset.target_shape = parse_shape(args.target_shape)
    cfg_dict.dataset.view_sampler.num_context_views = 16
    cfg_dict.dataset.view_sampler.num_target_views = args.num_target_views
    cfg_dict.model.cfg_scale = args.scenetok_cfg_scale
    cfg_dict.model.scheduler.num_inference_steps = args.scenetok_inference_steps
    if "noise_seed" in cfg_dict.model.denoiser:
        cfg_dict.model.denoiser.noise_seed = args.scenetok_seed
    OmegaConf.set_struct(cfg_dict, True)
    cfg = load_typed_root_config(cfg_dict)

    step_tracker = StepTracker(0)
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
    state = torch.load(Path(args.scenetok_ckpt), map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[infer] state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    wrapper.eval().to(args.device)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None

    ds = get_dataset(cfg.dataset, "test", step_tracker, generator=None, force_shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=safe_collate, shuffle=False)
    return wrapper, loader, cfg


def main():
    args = parse_args()
    torch.manual_seed(args.scenetok_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper, loader, cfg = build_wrapper_and_loader(args)
    precision = (
        torch.bfloat16
        if cfg.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
        else torch.float32
    )

    print(f"[infer] picking scene index {args.scene_index} from loader iter...")
    batch = None
    for i, b in enumerate(loader):
        if b is None:
            continue
        if i == args.scene_index:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"Could not obtain scene at index {args.scene_index}")
    scene_name = batch["scene"][0]
    print(f"[infer] scene = {scene_name}")

    batch_dev = move_to_device(batch, args.device)

    ctx_ext = batch_dev["context"]["extrinsics"][:, :1]
    ctx_int = batch_dev["context"]["intrinsics"][:, :1]
    ctx_lat = batch_dev["context"]["latent"][:, :1]
    ctx_idx = batch_dev["context"]["index"][:, :1] if "index" in batch_dev["context"] else None

    ref_c2w = ctx_ext[0, 0].clone()
    orbit_axis = axis_vec(args.orbit_axis)
    up_world = axis_vec(args.up_world)
    print(f"[infer] orbit_axis={args.orbit_axis} ({orbit_axis.tolist()}), "
          f"up_world={args.up_world} ({up_world.tolist()}), "
          f"radius={args.orbit_radius}, "
          f"angle=[{args.start_angle_deg}, {args.start_angle_deg + args.full_angle_deg}] deg, "
          f"frames={args.num_target_views}")
    orbit_c2w = make_orbit(
        ref_c2w=ref_c2w,
        num_views=args.num_target_views,
        radius=args.orbit_radius,
        orbit_axis=orbit_axis,
        up_world=up_world,
        full_angle_deg=args.full_angle_deg,
        start_angle_deg=args.start_angle_deg,
    ).to(args.device).to(ctx_ext.dtype)

    target_intrinsics = ctx_int[0, 0].unsqueeze(0).repeat(args.num_target_views, 1, 1)

    batch_dev["context"] = {
        "extrinsics": ctx_ext,
        "intrinsics": ctx_int,
        "latent": ctx_lat,
    }
    if ctx_idx is not None:
        batch_dev["context"]["index"] = ctx_idx
    batch_dev["target"] = {
        "extrinsics": orbit_c2w.unsqueeze(0),
        "intrinsics": target_intrinsics.unsqueeze(0).to(ctx_int.dtype),
        "latent": torch.zeros(
            (1, args.num_target_views, ctx_lat.shape[2], ctx_lat.shape[3], ctx_lat.shape[4]),
            device=args.device, dtype=ctx_lat.dtype,
        ),
        "index": torch.arange(args.num_target_views, device=args.device).unsqueeze(0),
    }

    batch_dev = preprocess_batch(batch_dev, index=0)

    save_image_video(
        images=ctx_lat[0].float().clamp(0, 1),
        indices=torch.arange(0, ctx_lat.shape[1]),
        output_dir=out_dir,
        name=f"context_{scene_name[:16]}",
        save_img=True,
        save_video=False,
        fps=args.fps,
    )

    print("[infer] sampling 360° orbit...")
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=precision, enabled=(precision != torch.float32)
    ):
        sampled, _, _ = wrapper.generate_batch_with_scene(
            batch_dev, wrapper.sampler, repeat_factor=1
        )
    sampled = sampled.float().clamp(0, 1)
    print(f"[infer] sampled shape: {tuple(sampled.shape)}")

    save_image_video(
        images=sampled[0],
        indices=torch.arange(0, sampled.shape[1]),
        output_dir=out_dir,
        name=f"orbit_{scene_name[:16]}_r{args.orbit_radius}_a{int(args.full_angle_deg)}_{args.orbit_axis}",
        save_img=False,
        save_video=True,
        fps=args.fps,
    )
    print(f"[infer] saved → {out_dir}")


if __name__ == "__main__":
    main()
