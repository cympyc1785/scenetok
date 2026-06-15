"""DiffSynth pure T2V in fp32 (everything: weights, activations, noise gen)."""
import argparse, glob, os, sys, torch

DIFF_ROOT = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/src/model/DiffSynth-Studio"
sys.path.insert(0, DIFF_ROOT)
# Disable FlashAttention (fp16/bf16 only) so SDPA falls back for fp32 attention.
import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False
_wvd.FLASH_ATTN_2_AVAILABLE = False
_wvd.SAGE_ATTN_AVAILABLE = False
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

DEFAULT_NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

p = argparse.ArgumentParser()
p.add_argument("--prompt", required=True)
p.add_argument("--negative_prompt", default=DEFAULT_NEG)
p.add_argument("--num_frames", type=int, default=37)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--cfg", type=float, default=5.0)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--shift", type=float, default=8.0)
p.add_argument("--height", type=int, default=480)
p.add_argument("--width", type=int, default=832)
p.add_argument("--out", required=True)
p.add_argument("--fps", type=int, default=10)
args = p.parse_args()

model_root = f"{DIFF_ROOT}/Wan2.2/Wan2.2-TI2V-5B"
# Everything in fp32
vram_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.float32,
    "onload_device": "cuda",
    "preparing_dtype": torch.float32,
    "preparing_device": "cuda",
    "computation_dtype": torch.float32,
    "computation_device": "cuda",
}
dit_paths = sorted(glob.glob(f"{model_root}/diffusion_pytorch_model*.safetensors"))
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.float32,
    device="cuda",
    model_configs=[
        ModelConfig(path=dit_paths, **vram_config),
        ModelConfig(path=f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(path=f"{model_root}/Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(path=f"{model_root}/google/umt5-xxl"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
video = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    input_image=None,
    height=args.height, width=args.width,
    num_frames=args.num_frames,
    cfg_scale=args.cfg,
    num_inference_steps=args.steps,
    sigma_shift=args.shift,
    seed=args.seed,
    rand_device="cpu",
    tiled=False,
)
save_video(video, args.out, fps=args.fps, quality=5)
print(f"[diffsynth_pure_t2v_fp32] saved → {args.out}")
