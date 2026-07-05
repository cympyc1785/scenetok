"""Camera-following eval for trained DynamicVerse camera-control models.

For each model, generate a video with the dataset's GT target-camera trajectory,
estimate the camera back from the generated frames with VGGT, and compute evo
trajectory error (ATE / RPEt / RPEr, Sim3-aligned) vs the GT target camera.
Saves the generated mp4 + per-scene poses. Runs 3 models × chosen dataset.

Models compared: controlnet / camchannel / camchannel_selfattnlora (all unscaledcomp).
Metric: lower = follows camera better. evo align(correct_scale=True) makes it
invariant to global frame/scale (VGGT absolute vs GT relative is handled).

Usage (scenetok env, VGGT weight cached locally):
  CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 python scripts/eval_camera_following.py \
    --dataset dl3dv --num_scenes 8 --out results/cam_follow_dl3dv
  ... --dataset dynamicverse ...
"""
from __future__ import annotations
import argparse, glob, json, os, sys, types
from pathlib import Path
import numpy as np
import torch

REPO = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src/model/DiffSynth-Studio"))
sys.path.insert(0, str(REPO / "submodules/lagernvs"))
import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False; _wvd.FLASH_ATTN_2_AVAILABLE = False; _wvd.SAGE_ATTN_AVAILABLE = False
# flash attn is disabled above → SDPA falls back to the cuDNN attention backend,
# which errors on B200 ("cuDNN Frontend: No valid execution plans"). Route SDPA to
# the mem-efficient/math backend instead.
torch.backends.cuda.enable_cudnn_sdp(False)
try:
    torch.backends.cuda.enable_mem_efficient_sdp(True); torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from scripts.fast_infer_t2v_swap_dataset import build_cfg, move_to_device
from scripts.eval_probe_traj_evo import traj_metrics
from src.dataset.data_module import safe_collate
from src.dataset import get_dataset
from src.misc.step_tracker import StepTracker
from src.misc.batch_utils import preprocess_batch
from src.misc.image_io import save_image_video
from src.model.t2v_wrapper import T2VWrapper

VGGT_W = glob.glob("/home/korea_kh63/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.safetensors") \
       + glob.glob("/NHNHOME/WORKSPACE/0226010013_A/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.safetensors")

MODELS = [
    dict(name="controlnet",
         exp="va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_effecterase_v2_unscaledcomp",
         ckpt="epoch=90-step=45000.ckpt", scene="controlnet", camera="controlnet",
         lora=False, dv_video="inpaint_result_effecterase.mp4"),
    dict(name="camchannel",
         exp="va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_no_lora_unscaledcomp",
         ckpt="epoch=77-step=40000.ckpt", scene="new_cross_attention", camera="channel_concat",
         lora=False, dv_video="inpaint_result.mp4"),
    dict(name="camchannel_selfattnlora",
         exp="va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_selfattnlora_unscaledcomp",
         ckpt="epoch=77-step=40000.ckpt", scene="new_cross_attention", camera="channel_concat",
         lora=True, dv_video="inpaint_result.mp4"),
]


def load_vggt(device):
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    m = VGGT(pred_cameras=True)
    sd = load_file(VGGT_W[0])
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[vggt] load missing={len(miss)} unexpected={len(unexp)} from {VGGT_W[0].split('snapshots/')[-1]}")
    return m.to(device).eval()


@torch.no_grad()
def vggt_poses(vggt, frames_uint8, device):
    """frames_uint8: (F,H,W,3) uint8 → per-frame c2w (F,3,3),(F,3) via VGGT."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from PIL import Image
    import torchvision.transforms.functional as TF
    imgs = []
    for a in frames_uint8:
        im = Image.fromarray(a).resize((518, 518))
        imgs.append(TF.to_tensor(im))
    x = torch.stack(imgs).unsqueeze(0).to(device)               # (1,F,3,518,518)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pe = vggt(x)
    pe = pe["pose_enc"] if isinstance(pe, dict) else pe
    if pe.dim() == 3 and pe.shape[-1] != 9:  # some versions return (1,F,9) already
        pe = pe
    extr, _ = pose_encoding_to_extri_intri(pe.float(), image_size_hw=(518, 518),
                                           pose_encoding_type="absT_quaR_FoV", build_intrinsics=True)
    w2c = extr[0].cpu().numpy()                                  # (F,3,4)
    R_w2c = w2c[:, :3, :3]; t_w2c = w2c[:, :3, 3]
    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    t_c2w = -np.einsum("nij,nj->ni", R_c2w, t_w2c)
    return R_c2w, t_c2w


def interp_traj(R, t, n_out):
    """Resample (V,3,3),(V,3) → (n_out,...) via slerp+lerp."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    V = R.shape[0]
    if V == n_out:
        return R, t
    src = np.linspace(0, 1, V); dst = np.linspace(0, 1, n_out)
    t_i = np.stack([np.interp(dst, src, t[:, k]) for k in range(3)], 1)
    R_i = Slerp(src, Rot.from_matrix(R))(dst).as_matrix()
    return R_i, t_i


def build_model(spec, dataset, dataset_root, out, device):
    ns = types.SimpleNamespace(
        experiment="custom/scenetok_va-wan-ti2v_dynamicverse", dataset=dataset,
        dataset_root=dataset_root,
        scene_id="x", context_shape="256,448", target_shape="480,832",
        num_context_views=12, num_target_views=10,   # match training (10); 37 = OOD → cuDNN plan fail
        load_prompts=False, cfg_scale=1.0, num_inference_steps=25, noise_seed=0,
        lora_disabled=not spec["lora"], prompt_style=None,
        scene_input_type=spec["scene"], camera_input_type=spec["camera"],
        prompt="", ood_prompt="", negative_prompt=None, static_target_camera=False,
        controlnet_ablation=False, single_cfg=True, text_modes="empty",
        ckpt=None, output_dir=out, exp_name=spec["exp"], device=device)
    cfg = build_cfg(ns)
    try: OmegaConf.set_struct(cfg, False)
    except Exception: pass
    cfg.dataset.scene_id = None
    cfg.model.cfg_scale = 1.0
    cfg.model.scheduler.num_inference_steps = 25
    if dataset == "dynamicverse":
        cfg.dataset.video_name = spec["dv_video"]
        cfg.dataset.target_video_name = "video_input.mp4"
        cfg.dataset.prompt_style = "category_first"
    if spec["lora"]:   # camchannel_selfattnlora: self-attn LoRA + allow_with_new_ca
        cfg.model.denoiser.lora.enabled = True
        cfg.model.denoiser.lora.rank = 32; cfg.model.denoiser.lora.alpha = 32
        cfg.model.denoiser.lora.target_modules = "self_attn.q,self_attn.k,self_attn.v,self_attn.o"
        cfg.model.denoiser.lora.allow_with_new_ca = True
    st = StepTracker(0)
    w = T2VWrapper(model_cfg=cfg.model, dataset_cfg=cfg.dataset, freeze_cfg=cfg.freeze,
                   optimizer_cfg=cfg.optimizer, test_cfg=cfg.test, train_cfg=cfg.train,
                   val_cfg=cfg.val, sampler_cfg=cfg.sampler, step_tracker=st,
                   output_dir=Path(out), batch_size=1, val_check_interval=1, mode="test")
    sd = torch.load(REPO / f"my_checkpoints/{spec['exp']}/{spec['ckpt']}", map_location="cpu")
    sd = sd.get("state_dict", sd)
    miss, unexp = w.load_state_dict(sd, strict=False)
    print(f"[{spec['name']}] ckpt {spec['ckpt']} load: missing={len(miss)} unexpected={len(unexp)}")
    w.eval().to(device).to(torch.bfloat16)
    if hasattr(w, "sampler") and hasattr(w.sampler, "log_vis"):
        w.sampler.log_vis = lambda *a, **kw: None
    return cfg, w, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dl3dv", "dynamicverse"], required=True)
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--num_scenes", type=int, default=8)
    ap.add_argument("--models", default="controlnet,camchannel,camchannel_selfattnlora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = args.dataset_root or ("./DATA/DL3DV/DL3DV-960" if args.dataset == "dl3dv" else "./WorldTraj/dynamicverse")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sel = args.models.split(",")
    assert VGGT_W, "VGGT weight not found in HF cache"
    vggt = load_vggt(args.device)

    summary = {}
    for spec in [m for m in MODELS if m["name"] in sel]:
        cfg, w, st = build_model(spec, args.dataset, root, args.out, args.device)
        ds = get_dataset(cfg.dataset, "test", st, generator=None, force_shuffle=False)
        loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=safe_collate, shuffle=False)
        rows = []; n = 0
        for batch in loader:
            if batch is None or n >= args.num_scenes:
                continue
            scene = batch["scene"][0] if isinstance(batch.get("scene"), (list, tuple)) else batch.get("scene")
            batch = move_to_device(batch, args.device)
            v_c = batch["context"]["extrinsics"].shape[1]
            try:
                batch = preprocess_batch(batch, index=v_c // 2)          # relative-to-center; target = generation cameras
                gt_ext = batch["target"]["extrinsics"][0].float().cpu().numpy()   # (V,4,4) c2w rel
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sampled, _, _ = w.generate_batch_with_scene(batch, w.sampler)
                sampled = sampled[0].float().clamp(0, 1)                 # (F,3,H,W)
            except Exception as e:
                import traceback
                if n == 0: traceback.print_exc()
                print(f"  [{spec['name']}] skip {str(scene)[:16]}: {e}"); continue
            sdir = Path(args.out) / spec["name"] / str(scene)[:24]
            sdir.mkdir(parents=True, exist_ok=True)
            save_image_video(images=sampled, indices=torch.arange(sampled.shape[0]),
                             output_dir=sdir, name="generated", save_img=False, save_video=True, fps=8)
            frames = (sampled.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")   # (F,H,W,3)
            try:
                Rp, tp = vggt_poses(vggt, frames, args.device)          # (F,3,3),(F,3) c2w
                Rg, tg = gt_ext[:, :3, :3], gt_ext[:, :3, 3]
                Rg, tg = interp_traj(Rg, tg, Rp.shape[0])               # match F
                ate, rpet, rper = traj_metrics(Rp, tp, Rg, tg)
                rows.append(dict(scene=str(scene)[:24], ATE=float(ate), RPEt=float(rpet), RPEr=float(rper)))
                np.savez(sdir / "poses.npz", Rp=Rp, tp=tp, Rg=Rg, tg=tg)
                print(f"  [{spec['name']}] {n+1}/{args.num_scenes} {str(scene)[:16]} F={frames.shape[0]} ATE={ate:.4f} RPEt={rpet:.4f} RPEr={rper:.3f}")
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"  metric fail: {e}")
            n += 1
        if rows:
            m = lambda k: float(np.mean([r[k] for r in rows]))
            summary[spec["name"]] = dict(n=len(rows), ATE=m("ATE"), RPEt=m("RPEt"), RPEr=m("RPEr"), per_scene=rows)
            print(f"=== {spec['name']}: n={len(rows)} ATE={m('ATE'):.4f} RPEt={m('RPEt'):.4f} RPEr={m('RPEr'):.3f} ===")
        del w; torch.cuda.empty_cache()

    json.dump(summary, open(Path(args.out) / "camera_following.json", "w"), indent=2)
    print("\n===== CAMERA FOLLOWING (%s) — lower=better =====" % args.dataset)
    print(f"{'model':<26}{'n':>4}{'ATE':>10}{'RPEt':>10}{'RPEr(deg)':>12}")
    for nm, s in summary.items():
        print(f"{nm:<26}{s['n']:>4}{s['ATE']:>10.4f}{s['RPEt']:>10.4f}{s['RPEr']:>12.3f}")
    print("saved →", Path(args.out) / "camera_following.json")


if __name__ == "__main__":
    main()
