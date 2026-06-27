"""LagerNVS persistent inference server for viser_server.py.

Runs in the `lagernvs` conda env as a LONG-LIVED subprocess: loads the
general_512 model ONCE at startup, then serves render requests over stdin/stdout
so repeated viser "Generate (LagerNVS)" clicks reuse the resident model (no
per-click reload). viser_server runs in the `scenetok` env and cannot import
LagerNVS in-process, hence the subprocess.

Protocol (line-based text on stdin/stdout; all diagnostics go to stderr):
  startup  → prints a line "READY" once the model is loaded.
  request  ← one JSON object per line: {"payload": "...", "frames_out": "..."}
  response → "DONE\t<frames_out>\t<shape>"  on success
             "ERR\t<message>"               on failure
  "QUIT" line or EOF on stdin → clean exit.
viser scans stdout for lines starting with READY/DONE/ERR (library prints, which
also land on stdout, are ignored).

Renders the viser-EDITED target cameras (not LagerNVS's auto VGGT path): builds
Plücker rays + cam_tokens from the payload's context images + (context/target)
c2w + target intrinsics, mirroring the multi-view normalization in
`vis.create_target_camera_path` (scene_scale=1.35·max‖ctx t‖, cam_tokens[9]=
camera_scale/[10]=0, cond rays=0). Feeding our cameras skips the VGGT download.
Returns raw frames (Vt,3,H,W float[0,1]) in `frames_out`; viser saves the mp4/gif.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="LagerNVS persistent render server")
    ap.add_argument("--repo", required=True, help="submodules/lagernvs path (added to sys.path)")
    ap.add_argument("--ckpt", required=True, help="general_512 model.pt path")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--mode", default="resize", choices=["resize", "square_crop"])
    ap.add_argument("--attention_type", default="bidirectional_cross_attention",
                    choices=["bidirectional_cross_attention", "full_attention"])
    args = ap.parse_args()

    # All diagnostics → stderr so stdout stays a clean protocol channel.
    def log(*a):
        print("[lagernvs-server]", *a, file=sys.stderr, flush=True)

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from models.encoder_decoder import EncDec_VitB8
    from vggt.utils.load_fn import load_and_preprocess_images
    from vis import render_chunked, compute_plucker_coordinates

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16
             if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    log(f"loading {args.ckpt} on {device} ({dtype}) ...")
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type=args.attention_type)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(device).eval()
    log("model loaded.")

    def render(payload_path, frames_out):
        pl = torch.load(payload_path, map_location="cpu")
        image_names = list(pl["context_image_paths"])
        num_cond = len(image_names)
        images = load_and_preprocess_images(
            image_names, mode=args.mode, target_size=args.target_size, patch_size=8
        ).to(device).unsqueeze(0)  # (1, Vc, 3, H, W)
        H, W = images.shape[-2], images.shape[-1]

        ctx_c2w = pl["context_c2w"].float()             # (Vc,4,4) OpenCV, rel ctx0
        tgt_c2w = pl["target_c2w"].float()              # (Vt,4,4)
        tgt_K = pl["target_intrinsics_norm"].float()    # (Vt,3,3) normalized

        first_inv = torch.linalg.inv(ctx_c2w[0:1])
        ctx_c2w = first_inv @ ctx_c2w
        tgt_c2w = first_inv @ tgt_c2w

        scene_scale = 1.35 * torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1))
        scene_scale = torch.clamp(scene_scale, min=1e-6)
        ctx_c2w[:, :3, 3] /= scene_scale
        tgt_c2w[:, :3, 3] /= scene_scale
        camera_scale = torch.max(torch.norm(ctx_c2w[:, :3, 3], dim=-1)).item()

        Vt = tgt_c2w.shape[0]
        cam_tokens = torch.zeros(1, num_cond + Vt, 11)
        cam_tokens[:, :, 9] = camera_scale
        cam_tokens[:, :, 10] = 0.0

        fx = tgt_K[:, 0, 0] * W
        fy = tgt_K[:, 1, 1] * H
        cx = tgt_K[:, 0, 2] * W
        cy = tgt_K[:, 1, 2] * H
        tgt_fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).unsqueeze(0)

        target_rays = compute_plucker_coordinates(
            tgt_c2w.unsqueeze(0), tgt_fxfycxcy, (H, W))
        # cond rays: zeros = UNPOSED (default). If `posed` + context intrinsics are
        # given, build real context Plücker rays (general_512 supports posed too).
        ctx_K = pl.get("context_intrinsics_norm", None)
        if pl.get("posed", False) and ctx_K is not None:
            ctx_K = ctx_K.float()
            cfx = ctx_K[:, 0, 0] * W; cfy = ctx_K[:, 1, 1] * H
            ccx = ctx_K[:, 0, 2] * W; ccy = ctx_K[:, 1, 2] * H
            ctx_fxfycxcy = torch.stack([cfx, cfy, ccx, ccy], dim=-1).unsqueeze(0)
            cond_rays = compute_plucker_coordinates(ctx_c2w.unsqueeze(0), ctx_fxfycxcy, (H, W))
        else:
            cond_rays = torch.zeros(1, num_cond, 6, H, W)
        rays = torch.cat([cond_rays, target_rays], dim=1).to(device)
        cam_tokens = cam_tokens.to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                video_out = render_chunked(
                    model, (images, rays, cam_tokens), num_cond_views=num_cond)
        out = Path(frames_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(video_out[0].detach().cpu().float().clamp(0, 1), str(out))
        return tuple(video_out.shape)

    print("READY", flush=True)
    log("READY — waiting for requests on stdin.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            log("QUIT received, exiting.")
            break
        try:
            req = json.loads(line)
            shape = render(req["payload"], req["frames_out"])
            print(f"DONE\t{req['frames_out']}\t{shape}", flush=True)
            log(f"rendered {shape} -> {req['frames_out']}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"ERR\t{e}", flush=True)


if __name__ == "__main__":
    main()
