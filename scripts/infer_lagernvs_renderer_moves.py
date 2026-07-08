"""Render camera-move trajectories from a feed-forward SceneTok-enc + LagerNVS-dec
checkpoint (denoiser.name == "lagernvs_renderer").

For each scene: FROZEN SceneTok compressor(context) -> scene tokens; then render
    - orig            : GT target trajectory (recon quality)
    - move_left/right : base @ local ∓X ramp
    - move_forward/back: base @ local ±Z ramp
Base = first GT target camera. Move convention mirrors viser_server._make_pattern_poses
(OpenCV local axes +X right, +Y down, +Z forward). Everything happens in the
scene-scale-normalized, ctx0-relative frame exactly as _lagernvs_renderer_step.

Feed-forward: each target ray renders independently, so novel cameras are free.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submodules" / "lagernvs"))

MOVE = {"move_forward": (0, 0, 1), "move_back": (0, 0, -1),
        "move_right": (1, 0, 0), "move_left": (-1, 0, 0)}
VARIANTS = ["orig", "move_left", "move_right", "move_forward", "move_back"]


def compose_cfg(exp, shape):
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg_dict = compose(config_name="main",
                           overrides=[f"+experiment={exp}", "dataset=dl3dv",
                                      "mode=test", "wandb.activated=false"])
    OmegaConf.set_struct(cfg_dict, False)
    for key in ("context_root", "target_root", "map_dict"):
        cfg_dict.dataset.pop(key, None)
    cfg_dict.dataset.root = "./DATA/DL3DV/DL3DV-960"
    cfg_dict.mode = "test"
    cfg_dict.wandb.activated = False
    cfg_dict.data_loader.test.num_workers = 0
    cfg_dict.data_loader.test.batch_size = 1
    cfg_dict.freeze.denoiser = True
    cfg_dict.freeze.compressor = True
    cfg_dict.freeze.autoencoder = True
    cfg_dict.dataset.smallset = True          # same pool the model trained on
    cfg_dict.dataset.context_shape = shape
    cfg_dict.dataset.target_shape = shape
    return load_typed_root_config(cfg_dict), cfg_dict


def save_gif(frames_hw3_uint8, path, fps=8):
    imageio.mimsave(path, list(frames_hw3_uint8), fps=fps, loop=0)


def to_uint8(rgb):  # (V,3,H,W) float [0,1] -> (V,H,W,3) uint8
    x = rgb.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return x


@torch.no_grad()
def render_variant(wrapper, scene_tokens, tgt_ext_scaled, K, HW, device):
    """tgt_ext_scaled (V,4,4) c2w already /scene_scale; K (V,3,3) normalized."""
    from vis import compute_plucker_coordinates
    H, W = HW
    fxfycxcy = torch.stack([K[:, 0, 0] * W, K[:, 1, 1] * H,
                            K[:, 0, 2] * W, K[:, 1, 2] * H], dim=-1)  # (V,4)
    rays = compute_plucker_coordinates(tgt_ext_scaled[None].float(),
                                       fxfycxcy[None].float(), (H, W)).to(device)  # (1,V,6,H,W)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb = wrapper.denoiser.render(scene_tokens, rays).float()[0]  # (V,3,H,W)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_scenes", type=int, default=6)
    ap.add_argument("--amount", type=float, default=0.6,
                    help="move distance in scene-scale-normalized units (ramped 0->amount)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", default="256,256")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    shape = [int(x) for x in args.shape.split(",")]

    from src.dataset import get_dataset
    from src.dataset.data_module import safe_collate
    from src.misc.step_tracker import StepTracker
    from src.misc.batch_utils import preprocess_batch
    from src.model.diffusion_wrapper import DiffusionWrapper
    from src.model.types import CameraInputs, CompressorInputs
    from torch.utils.data import DataLoader

    cfg, _ = compose_cfg(args.exp, shape)
    step_tracker = StepTracker(0)
    kwargs = dict(model_cfg=cfg.model, dataset_cfg=cfg.dataset, freeze_cfg=cfg.freeze,
                  optimizer_cfg=cfg.optimizer, test_cfg=cfg.test, train_cfg=cfg.train,
                  val_cfg=cfg.val, sampler_cfg=cfg.sampler, step_tracker=step_tracker,
                  output_dir=Path(args.out), batch_size=1,
                  val_check_interval=cfg.trainer.val_check_interval, mode="test")
    wrapper = DiffusionWrapper(**kwargs)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[load] {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    wrapper = wrapper.to(device).eval()

    ds = get_dataset(cfg.dataset, "val", step_tracker, torch.Generator(), force_shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=safe_collate, shuffle=False)

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()} \
            if not isinstance(batch, dict) else batch
        batch = preprocess_batch(batch, index=0)
        scene = batch["scene"][0] if isinstance(batch["scene"], (list, tuple)) else str(batch["scene"])
        ctx_ext = batch["context"]["extrinsics"].to(device)
        scene_scale = (1.35 * ctx_ext[:, :, :3, 3].norm(dim=-1).amax(dim=1)).clamp(min=1e-6)  # (1,)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ctx_inputs = CompressorInputs(
                view=wrapper._compressor_context_view(batch),
                pose=CameraInputs(intrinsics=batch["context"]["intrinsics"].to(device), extrinsics=ctx_ext),
                mask=None)
            tokens, *_ = wrapper.compressor(inputs=ctx_inputs)
        scene_tokens = tokens.sample() if cfg.model.compressor.scene_token_projection == "kl" else tokens

        tgt = batch["target"]
        H, W = tgt["latent"].shape[-2], tgt["latent"].shape[-1]
        tgt_ext = tgt["extrinsics"].clone().float().to(device)[0]              # (V,4,4)
        tgt_ext[..., :3, 3] /= scene_scale[0]
        K = tgt["intrinsics"].float().to(device)[0]                            # (V,3,3)
        V = tgt_ext.shape[0]
        base = tgt_ext[0].clone()                                             # (4,4) scaled c2w
        K0 = K[0:1].repeat(V, 1, 1)

        variants = {}
        for name in VARIANTS:
            if name == "orig":
                variants[name] = render_variant(wrapper, scene_tokens, tgt_ext, K, (H, W), device)
                continue
            poses = []
            for i in range(V):
                f = (i / (V - 1)) if V > 1 else 1.0
                delta = torch.eye(4, device=device, dtype=torch.float32)
                delta[:3, 3] = f * args.amount * torch.tensor(MOVE[name], device=device, dtype=torch.float32)
                poses.append(base @ delta)
            variants[name] = render_variant(wrapper, scene_tokens, torch.stack(poses), K0, (H, W), device)

        sdir = out_root / scene.replace("/", "_")
        sdir.mkdir(parents=True, exist_ok=True)
        cols = []
        for name in VARIANTS:
            u8 = to_uint8(variants[name])
            save_gif(u8, sdir / f"{name}.gif")
            cols.append(u8)
        # side-by-side concat [orig | left | right | forward | back]
        concat = np.concatenate(cols, axis=2)  # (V,H,5W,3)
        save_gif(concat, sdir / "concat.gif")
        imageio.mimwrite(sdir / "concat.mp4", list(concat), fps=8, quality=8)
        print(f"[{done+1}/{args.n_scenes}] {scene}  V={V} HxW={H}x{W}  scene_scale={scene_scale.item():.3f}")
        done += 1
        if done >= args.n_scenes:
            break
    print(f"done -> {out_root}")


if __name__ == "__main__":
    main()
