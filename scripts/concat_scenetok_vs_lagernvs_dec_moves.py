"""2x5 concat comparing SceneTok's native decoder vs the LagerNVS decoder on the
SAME SceneTok scene, over the viser camera-move trajectories.

Rows:  scenetok_dec (top)  = existing results/viser_generate/va-wan_dl3dv_256x448/<combo>/generated.mp4
                             (rendered by the va-wan_dl3dv_256x448 SceneTok LightningDiT decoder)
       lagernvs_dec (bottom)= rendered here with scenetok_enc_lagernvs_dec_warmstart
                             (frozen va-wan_dl3dv compressor -> scene tokens -> LagerNVS renderer)
Cols:  orig, move_forward, move_backward, move_left, move_right  (each combo's poses.pt)

Same poses.pt target trajectory feeds both. lagernvs render mirrors _lagernvs_renderer_step
(preprocess index=0, scene_scale=1.35*max ctx‖t‖, GT target K[0], Plücker rays) and
viser_regen_from_pt (edited_abs = ctx0_abs @ edited_rel; context from the model's own
256x256 eval loader).
"""
import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import imageio.v2 as imageio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submodules" / "lagernvs"))

from scripts.visualize.viser_server import build_model, CKPT_PRESETS, DEFAULT_EVAL_INDEX  # noqa: E402
from scripts.visualize.viser_regen_from_pt import build_scene_cache, find_batch, _to_dev  # noqa: E402

# column label -> viser subdir suffix
COLS = [("orig", "orig"), ("move_forward", "move_forward"), ("move_backward", "move_back"),
        ("move_left", "move_left"), ("move_right", "move_right")]


@torch.no_grad()
def lagernvs_render(wrapper, batch, tgt_rel, device, precision, hw=(256, 256)):
    from src.misc.batch_utils import preprocess_batch
    from src.model.types import CameraInputs, CompressorInputs
    from vis import compute_plucker_coordinates
    H, W = hw
    b = _to_dev(batch, device)
    ctx_ext = b["context"]["extrinsics"]
    ctx0_abs = ctx_ext[0, 0]
    edited_rel = torch.as_tensor(np.asarray(tgt_rel), dtype=ctx_ext.dtype, device=device)
    edited_abs = ctx0_abs.unsqueeze(0) @ edited_rel           # (T,4,4) world
    T = edited_abs.shape[0]
    gt_tgt_int = b["target"]["intrinsics"]
    tgt_int = gt_tgt_int[0, 0].unsqueeze(0).repeat(T, 1, 1)
    b["target"] = {
        "extrinsics": edited_abs.unsqueeze(0),
        "intrinsics": tgt_int.unsqueeze(0),
        "latent": torch.zeros((1, T, 3, H, W), device=device, dtype=gt_tgt_int.dtype),
        "index": torch.arange(T, device=device).unsqueeze(0),
    }
    b = preprocess_batch(b, index=0)                          # ctx0-relative (== training)

    ctx_ext = b["context"]["extrinsics"]
    scene_scale = (1.35 * ctx_ext[:, :, :3, 3].norm(dim=-1).amax(dim=1)).clamp(min=1e-6)
    with torch.autocast("cuda", dtype=precision if precision != torch.float32 else torch.bfloat16):
        ctx_inputs = CompressorInputs(
            view=wrapper._compressor_context_view(b),
            pose=CameraInputs(intrinsics=b["context"]["intrinsics"], extrinsics=ctx_ext), mask=None)
        tokens, *_ = wrapper.compressor(inputs=ctx_inputs)
    kl = wrapper.model_cfg.compressor.scene_token_projection == "kl"
    scene_tokens = tokens.sample() if kl else tokens

    tgt_ext = b["target"]["extrinsics"][0].float().clone()   # (T,4,4)
    tgt_ext[..., :3, 3] /= scene_scale[0]
    K = b["target"]["intrinsics"][0].float()                 # (T,3,3)
    fxfycxcy = torch.stack([K[:, 0, 0] * W, K[:, 1, 1] * H, K[:, 0, 2] * W, K[:, 1, 2] * H], dim=-1)
    rays = compute_plucker_coordinates(tgt_ext[None], fxfycxcy[None], (H, W)).to(device)  # (1,T,6,H,W)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb = wrapper.denoiser.render(scene_tokens, rays).float()[0].clamp(0, 1)          # (T,3,H,W)
    return (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)          # (T,H,W,3)


def read_mp4(path):
    return np.stack(imageio.mimread(path, memtest=False))     # (T,H,W,3) uint8


def resize_clip(clip, hw):
    from PIL import Image
    H, W = hw
    return np.stack([np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR)) for f in clip])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=str(REPO / "results/viser_generate/va-wan_dl3dv_256x448"))
    ap.add_argument("--warmstart_ckpt",
                    default=str(REPO / "my_checkpoints/scenetok_enc_lagernvs_dec_warmstart/last.ckpt"))
    ap.add_argument("--experiment", default="custom/scenetok_enc_lagernvs_dec_warmstart")
    ap.add_argument("--shape", default="256,256")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cell_h", type=int, default=256)
    ap.add_argument("--cell_w", type=int, default=448)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=str(REPO / "results/cmp_scenetok_vs_lagernvs_dec"))
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    src = Path(args.src_dir)
    # discover scene prefixes from *_orig dirs
    prefixes = sorted({d.name[:-len("_orig")] for d in src.glob("*_orig") if d.is_dir()})
    print(f"[cmp] scenes: {prefixes}")

    # copy ckpt (avoid racing the live training writer)
    tmp_ckpt = Path("/tmp") / f"warmstart_snapshot_{args.gpu}.ckpt"
    shutil.copy(args.warmstart_ckpt, tmp_ckpt)

    margs = SimpleNamespace(model_experiment=args.experiment, model_ckpt=str(tmp_ckpt),
                            model_shape=args.shape, eval_index=str(DEFAULT_EVAL_INDEX),
                            infer_steps=1, cfg_scale=1.0, seed=0, device=f"cuda:{args.gpu}",
                            extra_overrides=[])
    wrapper, loader, device, precision = build_model(margs)
    cache = build_scene_cache(loader)
    print(f"[cmp] scene cache: {len(cache)}")

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    cell = (args.cell_h, args.cell_w)

    for prefix in prefixes:
        # find the scene batch (prefix like a4c20f668ce179db_0624 → hash prefix a4c20f668ce179db)
        scene_key = prefix.split("_")[0]
        batch = find_batch(cache, scene_key)
        if batch is None:
            print(f"[cmp] SKIP {prefix}: scene {scene_key} not in eval loader"); continue

        top, bottom = [], []
        for label, suffix in COLS:
            combo = src / f"{prefix}_{suffix}"
            st_mp4 = combo / "generated.mp4"
            pt = combo / "poses.pt"
            if not st_mp4.exists() or not pt.exists():
                print(f"[cmp] SKIP col {label}: missing {combo}"); top = None; break
            # scenetok_dec (existing render)
            st = resize_clip(read_mp4(st_mp4), cell)
            # lagernvs_dec (render here from same poses.pt)
            obj = torch.load(pt, map_location="cpu", weights_only=False)
            lg = lagernvs_render(wrapper, batch, obj["target_c2w_edited"], device, precision)
            lg = resize_clip(lg, cell)
            # match frame count
            T = min(len(st), len(lg)); st, lg = st[:T], lg[:T]
            top.append(st); bottom.append(lg)
            print(f"[cmp] {prefix} :: {label}  scenetok{st.shape} lagernvs{lg.shape}")
        if not top:
            continue
        row0 = np.concatenate(top, axis=2)      # (T, H, 5W, 3)
        row1 = np.concatenate(bottom, axis=2)
        grid = np.concatenate([row0, row1], axis=1)  # (T, 2H, 5W, 3)
        sdir = out_root / prefix
        sdir.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(sdir / "grid_2x5.mp4", list(grid), fps=args.fps, quality=8)
        imageio.mimsave(sdir / "grid_2x5.gif", list(grid), fps=args.fps, loop=0)
        print(f"[cmp] wrote {sdir}/grid_2x5.mp4  {grid.shape}")
    print(f"[cmp] done -> {out_root}")


if __name__ == "__main__":
    main()
