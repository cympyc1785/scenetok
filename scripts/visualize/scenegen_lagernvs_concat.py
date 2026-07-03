"""Render the SAME re10k scene + target trajectory that SceneGen used, but with
LagerNVS general_512 (512x512), and concat side-by-side with the SceneGen video.

Same setting as viser_server_scenegen: scene loaded via re10k dataset, poses
relative to context[0] (preprocess_batch index=0), conditioning frames = the
num_cond uniform target indices (linspace). LagerNVS general_512 runs unposed
(context cameras only used for scale normalization) at target_size=512.

scenetok env prepares data + payload + concat; the LagerNVS worker runs in the
lagernvs conda env as a subprocess (lagernvs_infer.py protocol).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/visualize/scenegen_lagernvs_concat.py \
    --scene 004e9db3337e8206 --num_cond 5 \
    --scenegen_mp4 results/viser_generate/scenegen_re10k/<dir>/generated.mp4
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import imageio.v3 as iio
import imageio
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEFAULT_RE10K_ROOT = str((REPO / "../dataset/re10k/re10k").resolve())
LAGERNVS_PY = "/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/lagernvs/bin/python"
LAGERNVS_REPO = REPO / "submodules/lagernvs"
LAGERNVS_CKPT = LAGERNVS_REPO / "checkpoints/lagernvs_general_512/model.pt"


def load_scene_batch(scene, re10k_root, eval_index):
    """target frames/extrinsics/intrinsics (rel ctx0) for `scene`, no model."""
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_config
    from src.dataset import get_dataset, DatasetRE10kCfg
    from src.misc.batch_utils import preprocess_batch, batch_expand
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[
            "dataset=re10k", f"dataset.root={re10k_root}",
            "dataset/view_sampler=evaluation_video", "dataset.view_sampler.max_cond_number=3",
            "+experiment=scenegen_shift12_re10k",
            "dataset.view_sampler.num_target_views=8", "dataset.view_sampler.temporal_downsample=4",
            "dataset.view_sampler.num_context_views=12",
            f"dataset.view_sampler.index_path={eval_index}",
            "dataset.precomputed_latents.context=false", "dataset.precomputed_latents.target=false",
            "wandb.activated=false",
        ])
    ds_cfg = load_typed_config(cfg.dataset, DatasetRE10kCfg)
    ds = get_dataset(ds_cfg, stage="test", step_tracker=None)
    ds.overfit_to_scene = [scene]
    b = ds[0]
    b["context"] = batch_expand(b["context"]); b["target"] = batch_expand(b["target"])
    b = preprocess_batch(b, index=0)                       # rel to context[0]
    t = b["target"]
    return (t["latent"][0].float().clamp(0, 1),            # (T,3,H,W) images
            t["extrinsics"][0].float(),                    # (T,4,4) rel ctx0
            t["intrinsics"][0].float())                    # (T,3,3) normalized


def run_lagernvs(payload_path, frames_out, gpu, target_size=512):
    cmd = [LAGERNVS_PY, str(REPO / "scripts/visualize/lagernvs_infer.py"),
           "--repo", str(LAGERNVS_REPO), "--ckpt", str(LAGERNVS_CKPT),
           "--target_size", str(target_size)]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen(cmd, cwd=str(LAGERNVS_REPO), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
    for line in proc.stdout:                                # wait for READY
        if line.strip() == "READY":
            break
        if proc.poll() is not None:
            raise RuntimeError("LagerNVS worker died before READY")
    proc.stdin.write(json.dumps({"payload": str(payload_path), "frames_out": str(frames_out)}) + "\n")
    proc.stdin.flush()
    reply = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("DONE\t") or line.startswith("ERR\t"):
            reply = line; break
        if proc.poll() is not None:
            break
    try:
        proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=5)
    except Exception:
        proc.kill()
    if reply is None or reply.startswith("ERR\t"):
        raise RuntimeError(f"LagerNVS render failed: {reply}")
    return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--num_cond", type=int, default=5, help="context views for LagerNVS")
    ap.add_argument("--scenegen_mp4", required=True)
    ap.add_argument("--poses_pt", default=None,
                    help="viser poses.pt → use its target_c2w_rel(edited) as target trajectory")
    ap.add_argument("--re10k_root", default=DEFAULT_RE10K_ROOT)
    ap.add_argument("--eval_index", default="./assets/evaluation_index/re10k_c1_192.json")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for LagerNVS worker")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # target trajectory: from viser poses.pt (edited) if given, else the GT trajectory.
    tgt_c2w = None; scene = args.scene
    if args.poses_pt:
        pp = torch.load(args.poses_pt, map_location="cpu", weights_only=False)
        traj = pp.get("target_c2w_rel", pp.get("target_c2w_edited", pp.get("target_c2w")))
        tgt_c2w = torch.as_tensor(np.asarray(traj), dtype=torch.float32)   # (Vt,4,4) rel ctx0
        scene = pp.get("scene", scene)
    assert scene is not None, "need --scene or a poses.pt with 'scene'"

    imgs, ext, K = load_scene_batch(scene, args.re10k_root, args.eval_index)  # context imgs + K
    if tgt_c2w is None:
        tgt_c2w = ext
    Vt = tgt_c2w.shape[0]
    Kt = K[0].unsqueeze(0).repeat(Vt, 1, 1) if K.shape[0] != Vt else K       # normalized K per target
    cond_idx = torch.linspace(0, ext.shape[0] - 1, max(1, args.num_cond)).long()
    print(f"[sg-vs-lager] scene={scene} target_frames={Vt} (poses_pt={bool(args.poses_pt)}) "
          f"num_cond={args.num_cond} cond_idx={cond_idx.tolist()}")

    # absolute paths: the LagerNVS worker runs with cwd=lagernvs repo, so payload /
    # context image / frames paths must be absolute.
    work = Path(args.scenegen_mp4).resolve().parent / f"lagernvs_general_512_nc{args.num_cond}"
    ctx_dir = work / "context_images"; ctx_dir.mkdir(parents=True, exist_ok=True)
    ctx_paths = []
    for i in cond_idx.tolist():
        p = ctx_dir / f"ctx_{i:03d}.png"
        Image.fromarray((imgs[i].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(p)
        ctx_paths.append(str(p))

    payload = work / "payload.pt"
    torch.save({"context_image_paths": ctx_paths,
                "context_c2w": ext[cond_idx],              # (Vc,4,4) rel ctx0
                "target_c2w": tgt_c2w,                      # (Vt,4,4) rel ctx0 (poses.pt edited or GT)
                "target_intrinsics_norm": Kt,               # (Vt,3,3) normalized
                "scene": scene}, payload)
    frames_out = work / "frames.pt"
    print(f"[sg-vs-lager] running LagerNVS general_512 @ {args.target_size}...")
    print("  ", run_lagernvs(payload, frames_out, args.gpu, args.target_size))
    lager = torch.load(frames_out, map_location="cpu").float().clamp(0, 1)   # (T,3,H,W)
    lager = (lager.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")        # (T,H,W,3)

    sg = iio.imread(args.scenegen_mp4)                                        # (T,256,256,3)
    Tn = min(len(sg), len(lager))
    S = args.target_size
    def rs(a):  # resize frame to SxS
        return np.asarray(Image.fromarray(a).resize((S, S)))
    sep = np.full((S, 6, 3), 255, np.uint8)
    frames = np.stack([np.concatenate([rs(sg[t]), sep, rs(lager[t])], axis=1) for t in range(Tn)])
    out = args.out or str(work / f"concat_scenegen_vs_lagernvs512_nc{args.num_cond}.mp4")
    imageio.mimsave(out, frames, fps=args.fps, quality=8)
    print(f"[sg-vs-lager] saved → {out}  {frames.shape}  (left=SceneGen, right=LagerNVS general_512)")


if __name__ == "__main__":
    main()
