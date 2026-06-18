"""Batch Wan2.2 TI2V-5B pure-T2V generation over camera-motion test prompts.

Loads the fp32 pipeline ONCE (cf. diffsynth_pure_t2v_fp32.py) and loops over
prompts/camera_motion_t2v.json. Each output is saved together with its text
input (prompt.txt) so the result folder is self-contained. Purpose: check
whether Wan T2V actually renders the camera move described by the text.

Usage:
  python scripts/gen_camera_motion_t2v.py --steps 40 --num_frames 49
  python scripts/gen_camera_motion_t2v.py --ids 06_orbit_around,09_drone_flyover
"""
import argparse, glob, json, os, sys, torch

DIFF_ROOT = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/src/model/DiffSynth-Studio"
sys.path.insert(0, DIFF_ROOT)
import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False
_wvd.FLASH_ATTN_2_AVAILABLE = False
_wvd.SAGE_ATTN_AVAILABLE = False
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

ROOT = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok"

p = argparse.ArgumentParser()
p.add_argument("--prompts_json", default=f"{ROOT}/prompts/camera_motion_t2v.json")
p.add_argument("--out_dir", default=f"{ROOT}/results/camera_motion_t2v")
p.add_argument("--ids", default="")          # comma-sep subset of prompt ids; empty = all
p.add_argument("--num_frames", type=int, default=49)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--cfg", type=float, default=5.0)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--shift", type=float, default=8.0)
p.add_argument("--height", type=int, default=480)
p.add_argument("--width", type=int, default=832)
p.add_argument("--fps", type=int, default=10)
args = p.parse_args()

spec = json.load(open(args.prompts_json))
neg = spec.get("negative_prompt", "")
items = spec["prompts"]
if args.ids:
    want = set(args.ids.split(","))
    items = [it for it in items if it["id"] in want]
print(f"[gen] {len(items)} prompts, steps={args.steps}, frames={args.num_frames}, cfg={args.cfg}")

model_root = f"{DIFF_ROOT}/Wan2.2/Wan2.2-TI2V-5B"
vram_config = {
    "offload_dtype": torch.float32, "offload_device": "cpu",
    "onload_dtype": torch.float32, "onload_device": "cuda",
    "preparing_dtype": torch.float32, "preparing_device": "cuda",
    "computation_dtype": torch.float32, "computation_device": "cuda",
}
dit_paths = sorted(glob.glob(f"{model_root}/diffusion_pytorch_model*.safetensors"))
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.float32, device="cuda",
    model_configs=[
        ModelConfig(path=dit_paths, **vram_config),
        ModelConfig(path=f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(path=f"{model_root}/Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(path=f"{model_root}/google/umt5-xxl"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

for i, it in enumerate(items):
    out_d = os.path.join(args.out_dir, it["id"])
    os.makedirs(out_d, exist_ok=True)
    video = pipe(
        prompt=it["prompt"], negative_prompt=neg, input_image=None,
        height=args.height, width=args.width, num_frames=args.num_frames,
        cfg_scale=args.cfg, num_inference_steps=args.steps, sigma_shift=args.shift,
        seed=args.seed, rand_device="cpu", tiled=False,
    )
    mp4 = os.path.join(out_d, f"{it['id']}.mp4")
    save_video(video, mp4, fps=args.fps, quality=5)
    # save text input alongside the result
    with open(os.path.join(out_d, "prompt.txt"), "w") as f:
        f.write(f"id: {it['id']}\ncamera_move: {it['camera_move']}\n\nprompt:\n{it['prompt']}\n\nnegative_prompt:\n{neg}\n")
    with open(os.path.join(out_d, "gen_config.json"), "w") as f:
        json.dump({k: getattr(args, k) for k in
                   ["num_frames", "seed", "cfg", "steps", "shift", "height", "width", "fps"]}, f, indent=2)
    print(f"  [{i+1}/{len(items)}] {it['id']} ({it['camera_move']}) → {mp4}")

print(f"[gen] done → {args.out_dir}")
