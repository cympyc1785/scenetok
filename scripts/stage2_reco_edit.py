"""Stage 2 of lightningDiT → ReCo two-stage inference.

Loads ReCo (Wan2.1-VACE-1.3B + LoRA) and runs editing on a coarse mp4 produced
by `stage1_lightningdit_coarse.py`. Side-by-side layout: left = coarse video
(preserved), right = zeros (filled by ReCo according to `--prompt`).

Run in a separate process from Stage 1 so the two `diffsynth` packages
(`src/model/DiffSynth-Studio` vs `src/model/ReCo/DiffSynth-Studio`) don't
collide on package name.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

REPO_ROOT = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
RECO_ROOT = REPO_ROOT / "src/model/ReCo"
RECO_DIFFSYNTH = RECO_ROOT / "DiffSynth-Studio"
for p in (RECO_DIFFSYNTH, RECO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from diffsynth import ModelManager, WanVideoPipeline, save_video
from diffsynth.models.utils import load_state_dict as diffsynth_load_state_dict
from peft import LoraConfig, inject_adapter_in_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coarse_mp4", required=True, help="Stage 1 output mp4 path")
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--negative_prompt",
        default=(
            "Bright tones, overexposed, static, blurred details, subtitles, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, "
            "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
            "deformed, disfigured, misshapen limbs, fused fingers, still picture, "
            "messy background, three legs, many people in the background, walking backwards"
        ),
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scene_name", default=None)
    p.add_argument(
        "--reco_wan_root",
        default=str(RECO_ROOT / "checkpoints/Wan2.1-VACE-1.3B"),
    )
    p.add_argument(
        "--reco_lora_ckpt",
        default=str(RECO_ROOT / "checkpoints/ReCo/2026_01_16_v1_release.ckpt"),
    )
    p.add_argument("--reco_lora_rank", type=int, default=128)
    p.add_argument("--reco_lora_alpha", type=int, default=128)
    p.add_argument("--reco_lora_targets", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--reco_num_inference_steps", type=int, default=50)
    p.add_argument("--reco_seed", type=int, default=1)
    p.add_argument("--reco_num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument(
        "--upscale",
        action="store_true",
        help="If set, bilinear-upscale coarse mp4 to (height, width) before ReCo. "
             "Otherwise ReCo runs at the coarse mp4's native resolution and "
             "--height/--width are ignored.",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def add_lora_to_model(model, lora_rank, lora_alpha, lora_targets, lora_ckpt):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=lora_targets.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    if lora_ckpt is not None:
        sd = diffsynth_load_state_dict(lora_ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  LoRA load: {len(missing)} missing, {len(unexpected)} unexpected from {lora_ckpt}")


def read_mp4_frames(path, num_frames, h, w, upscale):
    """Read mp4 → (F, 3, H, W) tensor in [-1, 1]. Pad with last frame to num_frames.

    If `upscale` is True, bilinear-resize to (h, w). Otherwise keep native size
    and ignore h/w. Returns (tensor, actual_h, actual_w).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"empty video: {path}")
    arr = np.stack(frames, axis=0)  # (F, H, W, 3)
    arr = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (F, 3, H, W) [0,1]
    if arr.shape[0] < num_frames:
        last = arr[-1:].expand(num_frames - arr.shape[0], -1, -1, -1)
        arr = torch.cat([arr, last], dim=0)
    elif arr.shape[0] > num_frames:
        arr = arr[:num_frames]
    if upscale:
        if arr.shape[2:] != (h, w):
            print(f"[stage2] upscale: {tuple(arr.shape[2:])} → ({h}, {w}) (bilinear)")
            arr = F.interpolate(arr, size=(h, w), mode="bilinear", align_corners=False)
    else:
        print(f"[stage2] no upscale; using native coarse size {tuple(arr.shape[2:])}")
    arr = arr * 2.0 - 1.0  # [0,1] → [-1,1]
    return arr, arr.shape[2], arr.shape[3]  # (F, 3, h, w), h, w


def build_reco_inputs(coarse_video_m11: torch.Tensor, h: int, w: int):
    """coarse_video_m11: (F, 3, H, W) in [-1, 1].
    Returns side-by-side packed tensors ready for ReCo.
    """
    f = coarse_video_m11.shape[0]
    v = coarse_video_m11.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, F, H, W)
    right = torch.zeros_like(v)
    tar_video_key = torch.cat([v, right], dim=-1)  # (1, 3, F, H, W*2)
    mask = torch.zeros_like(tar_video_key)
    mask[..., w:] = 1.0
    zeros = torch.zeros_like(tar_video_key)
    return {
        "tar_video_key": tar_video_key,
        "tar_video_key_mask": mask,
        "ref_video": zeros,
        "tar_video": zeros,
    }


def main():
    args = parse_args()
    h, w = args.height, args.width

    out_dir = Path(args.out_dir)
    if args.scene_name:
        out_dir = out_dir / args.scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage2] loading Wan2.1-VACE from {args.reco_wan_root}")
    ckpts = [
        f"{args.reco_wan_root}/diffusion_pytorch_model.safetensors",
        f"{args.reco_wan_root}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{args.reco_wan_root}/Wan2.1_VAE.pth",
    ]
    mm = ModelManager(device="cpu")
    mm.load_models(ckpts, torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(
        mm, torch_dtype=torch.bfloat16, device=args.device
    )

    print(f"[stage2] applying ReCo LoRA: {args.reco_lora_ckpt}")
    add_lora_to_model(
        pipe.vace,
        lora_rank=args.reco_lora_rank,
        lora_alpha=args.reco_lora_alpha,
        lora_targets=args.reco_lora_targets,
        lora_ckpt=args.reco_lora_ckpt,
    )
    add_lora_to_model(
        pipe.denoising_model(),
        lora_rank=args.reco_lora_rank,
        lora_alpha=args.reco_lora_alpha,
        lora_targets=args.reco_lora_targets,
        lora_ckpt=args.reco_lora_ckpt,
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.eval()

    print(f"[stage2] loading coarse video: {args.coarse_mp4}")
    coarse, h, w = read_mp4_frames(args.coarse_mp4, args.reco_num_frames, h, w, args.upscale)
    inputs = build_reco_inputs(coarse, h, w)

    print(f"[stage2] running ReCo editing... prompt={args.prompt!r}")
    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.reco_num_inference_steps,
            height=h,
            width=w * 2,
            num_frames=args.reco_num_frames,
            seed=args.reco_seed,
            tiled=False,
            vace_video=inputs["tar_video_key"].to(dtype=pipe.torch_dtype, device=pipe.device),
            vace_video_ref=inputs["ref_video"].to(dtype=pipe.torch_dtype, device=pipe.device),
            vace_mask=inputs["tar_video_key_mask"].to(dtype=pipe.torch_dtype, device=pipe.device),
            tar_video=inputs["tar_video"].to(dtype=pipe.torch_dtype, device=pipe.device),
            ref_img_pil=None,
            inference=True,
        )

    full_path = out_dir / "stage2_full_grid.mp4"
    save_video(video, str(full_path), fps=args.fps, quality=5)
    print(f"[stage2] saved full visualization grid → {full_path}")

    # ReCo's pipeline returns a (2 rows × 4 cols) visualization grid; only the
    # bottom row's first two cells carry the meaningful content:
    #   row 2 col 1 = input coarse video (re-rendered by pipeline)
    #   row 2 col 2 = edited result
    # We crop:
    #   * `stage2_input_output.mp4`  : both cells concatenated → 2w × h
    #   * `stage2_edit.mp4`           : col 2 only (edit result) → w × h
    try:
        in_out_frames, edit_frames = [], []
        for frame in video:
            arr = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
            fw, fh = arr.size
            cell_w = fw // 4  # 4 columns
            cell_h = fh // 2  # 2 rows
            top_of_row2 = cell_h
            in_out_frames.append(arr.crop((0, top_of_row2, 2 * cell_w, top_of_row2 + cell_h)))
            edit_frames.append(arr.crop((cell_w, top_of_row2, 2 * cell_w, top_of_row2 + cell_h)))

        io_path = out_dir / "stage2_input_output.mp4"
        save_video(in_out_frames, str(io_path), fps=args.fps, quality=5)
        print(f"[stage2] saved input|output side-by-side → {io_path}")

        edit_path = out_dir / "stage2_edit.mp4"
        save_video(edit_frames, str(edit_path), fps=args.fps, quality=5)
        print(f"[stage2] saved edit-only → {edit_path}")
    except Exception as exc:
        print(f"[stage2] (warn) crop save failed: {exc}")


if __name__ == "__main__":
    main()
