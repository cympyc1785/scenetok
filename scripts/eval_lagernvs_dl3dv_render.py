"""Phase 1 (lagernvs env): render LagerNVS general_512 on DL3DV val across a
resolution x context-count sweep, saving pred/gt per combo/scene + an index JSON.

Sweep: resolutions x context counts. Target = 37 consecutive frames (from the
SceneTok dl3dv eval index); context = `ctx_count` frames evenly spaced across the
target window (interpolation). Camera intrinsics/structure auto-adjust per
resolution (center-crop to target aspect + resize, intrinsics adjusted).

general_512 is UNPOSED (context cameras ignored): cond rays = 0, cam_tokens =
zeros + [9]=camera_scale; targets are posed via Plucker rays.

Outputs per (combo, scene): pred.pt, gt.pt (T,3,H,W float[0,1]) + a preview mp4.
Index JSON records the exact frame indices for reuse.
Metrics are computed separately in scenetok env (eval_lagernvs_dl3dv_metric.py).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(".").resolve()
EVAL_INDEX = "assets/evaluation_index/dl3dv_c16_37_caption_standard.json"
DL3DV_ROOT = "DATA/DL3DV/DL3DV-960/train"
RESOLUTIONS = [(256, 256), (256, 448), (512, 512), (480, 832)]
CONTEXT_COUNTS = [6, 10, 16]


def find_scene_dir(scene_hash):
    g = glob.glob(f"{DL3DV_ROOT}/*/{scene_hash}")
    return g[0] if g else None


def load_scene(scene_hash):
    d = find_scene_dir(scene_hash)
    if d is None:
        return None
    tj = json.load(open(f"{d}/transforms.json"))
    w, h = tj["w"], tj["h"]
    Kn = np.array([[tj["fl_x"] / w, 0, tj["cx"] / w],
                   [0, tj["fl_y"] / h, tj["cy"] / h],
                   [0, 0, 1]], dtype=np.float32)                    # normalized (orig aspect)
    # preprocessed/transforms.npz['extrinsics'] is ALREADY c2w (verified: matches
    # the SceneTok bundle c2w to 1e-6 with NO inversion; the "w2c" comment is wrong).
    c2w = np.load(f"{d}/preprocessed/transforms.npz")["extrinsics"].astype(np.float32)  # (N,4,4) c2w
    pngs = sorted(glob.glob(f"{d}/images/*.png"))
    return {"c2w": torch.from_numpy(c2w), "Kn": torch.from_numpy(Kn), "pngs": pngs}


def crop_resize_load(png_paths, idxs, crop_hw, H, W):
    """Load PNGs at idxs, center-crop to crop_hw, BICUBIC resize to (H,W) → (n,3,H,W) [0,1]."""
    ch, cw = crop_hw
    out = []
    for i in idxs:
        im = Image.open(png_paths[i]).convert("RGB")
        w0, h0 = im.size
        top, left = (h0 - ch) // 2, (w0 - cw) // 2
        im = im.crop((left, top, left + cw, top + ch)).resize((W, H), Image.Resampling.BICUBIC)
        out.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="submodules/lagernvs")
    ap.add_argument("--ckpt", default="submodules/lagernvs/checkpoints/lagernvs_general_512/model.pt")
    ap.add_argument("--n_scenes", type=int, default=50)
    ap.add_argument("--out_dir", default="results/eval_lagernvs_dl3dv")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 scene only (render check)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from models.encoder_decoder import EncDec_VitB8
    from vis import render_chunked, compute_plucker_coordinates
    from data.camera_utils import (
        get_full_res_crop_dims_constant_ar, adjust_intrinsics_for_crop_and_resize)

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(device).eval()
    print(f"[eval-render] model loaded ({args.ckpt})", flush=True)

    idx = json.load(open(EVAL_INDEX))
    scene_hashes = list(idx.keys())[: (1 if args.smoke else args.n_scenes)]
    out_root = Path(args.out_dir)
    index_record = {"resolutions": RESOLUTIONS, "context_counts": CONTEXT_COUNTS, "scenes": {}}

    def adj_ff(Kn, h0, w0, crop_hw, H, W):
        fx = Kn[0, 0] * w0; fy = Kn[1, 1] * h0; cx = Kn[0, 2] * w0; cy = Kn[1, 2] * h0
        fx, fy, cx, cy = adjust_intrinsics_for_crop_and_resize((fx, fy, cx, cy), (h0, w0), crop_hw, (H, W))
        return torch.tensor([fx, fy, cx, cy]).view(1, 1, 4)

    ok = fail = 0
    for si, sh in enumerate(scene_hashes):
        sc = load_scene(sh)
        if sc is None:
            print(f"[eval-render] SKIP {sh[:12]}: no data"); fail += 1; continue
        tgt_idx = idx[sh]["target"]                       # 37 consecutive frame indices
        Kn = sc["Kn"]; c2w_all = sc["c2w"]; pngs = sc["pngs"]
        w0, h0 = Image.open(pngs[0]).size                 # (W,H)
        rec = {"target": tgt_idx, "context": {}}
        for cc in CONTEXT_COUNTS:
            ctx_idx = np.unique(np.linspace(tgt_idx[0], tgt_idx[-1], cc).round().astype(int)).tolist()
            rec["context"][str(cc)] = ctx_idx
            for (H, W) in RESOLUTIONS:
                combo = f"{H}x{W}_ctx{cc}"
                out_d = out_root / combo / sh[:16]
                try:
                    crop_h, crop_w = get_full_res_crop_dims_constant_ar((h0, w0), (H, W))
                    crop_hw = (crop_h, crop_w)
                    ctx_imgs = crop_resize_load(pngs, ctx_idx, crop_hw, H, W)          # (Vc,3,H,W)
                    tgt_imgs = crop_resize_load(pngs, tgt_idx, crop_hw, H, W)          # (Vt,3,H,W) = GT
                    Vc, Vt = len(ctx_idx), len(tgt_idx)
                    ctx_c2w = c2w_all[ctx_idx].clone()
                    tgt_c2w = c2w_all[tgt_idx].clone()
                    first_inv = torch.linalg.inv(ctx_c2w[0:1])
                    ctx_c2w = first_inv @ ctx_c2w; tgt_c2w = first_inv @ tgt_c2w
                    scale = torch.clamp(1.35 * torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)), min=1e-6)
                    ctx_c2w[:, :3, 3] /= scale; tgt_c2w[:, :3, 3] /= scale
                    camera_scale = torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)).item()
                    tgt_ff = adj_ff(Kn, h0, w0, crop_hw, H, W).repeat(1, Vt, 1)
                    target_rays = compute_plucker_coordinates(tgt_c2w.unsqueeze(0), tgt_ff, (H, W))
                    cond_rays = torch.zeros(1, Vc, 6, H, W)                            # UNPOSED
                    rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
                    cam_tokens = torch.zeros(1, Vc + Vt, 11, device=device); cam_tokens[:, :, 9] = camera_scale
                    images = ctx_imgs.to(device).unsqueeze(0)
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                        pred = render_chunked(model, (images, rays, cam_tokens), num_cond_views=Vc)
                    pred = pred[0].detach().cpu().float().clamp(0, 1)                  # (Vt,3,H,W)
                    out_d.mkdir(parents=True, exist_ok=True)
                    # save uint8 (lossless, 1/4 of float32) for the metric phase.
                    torch.save((pred * 255).round().to(torch.uint8), out_d / "pred.pt")
                    torch.save((tgt_imgs * 255).round().to(torch.uint8), out_d / "gt.pt")
                    # preview mp4 (pred|gt side by side)
                    sbs = torch.cat([pred, tgt_imgs], dim=3)
                    frames = (sbs.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
                    pil = [Image.fromarray(f) for f in frames]
                    pil[0].save(str(out_d / "pred_gt.gif"), save_all=True, append_images=pil[1:],
                                duration=int(1000 / max(args.fps, 1)), loop=0)
                    print(f"[eval-render] OK {sh[:10]} {combo} pred{tuple(pred.shape)}", flush=True)
                    ok += 1
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"[eval-render] FAIL {sh[:10]} {combo}: {e}", flush=True); fail += 1
        index_record["scenes"][sh] = rec

    idx_out = out_root / ("index_smoke.json" if args.smoke else "index.json")
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(index_record, open(idx_out, "w"), indent=1)
    print(f"[eval-render] done. ok={ok} fail={fail} | index → {idx_out}", flush=True)


if __name__ == "__main__":
    main()
