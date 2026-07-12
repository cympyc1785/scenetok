"""Ultra-slim the controlnet Lightning ckpts: keep ONLY trained params.

The wan_ti2v denoiser reconstructs the frozen base (Wan TI2V DiT + T5 text encoder)
from pretrained paths at init, THEN loads the Lightning ckpt with strict=False. So
every frozen param in the ckpt is redundant — it equals the base and is re-filled at
load. We drop:
  - denoiser.text_encoder.*   (T5, always frozen → drop entirely)
  - denoiser.model.<k>        (drop iff byte-identical to base DiT weight `k`)
Everything else (trained controlnet layers in denoiser.model, compressor.*,
pose_embed, cnd_proj, null_tokens, ...) is kept.

⚠️ The result MUST be loaded with load_strict=False (missing base keys are refilled
from the pretrained base at model construction). Verified: every dropped key equals
the base, so base(strict base-load) + slim(strict=False) == original state_dict.
"""
import glob
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
BASE_ROOT = REPO / "src/model/DiffSynth-Studio/Wan2.2/Wan2.2-TI2V-5B"
CKPTS = [
    "va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora",
    "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_no_lora",
]
SRC = REPO / "my_checkpoints/backup"
OUT = REPO / "my_checkpoints/backup_slim"


def gb(x):
    return x / 1e9


print("loading base Wan TI2V DiT ...")
base_dit = {}
for shard in sorted(glob.glob(str(BASE_ROOT / "diffusion_pytorch_model*.safetensors"))):
    base_dit.update(load_file(shard))
print(f"  base DiT keys: {len(base_dit)}")

for name in CKPTS:
    f = SRC / name / "last.ckpt"
    ck = torch.load(f, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    kept, dropped_te, dropped_base, kept_bytes = {}, 0, 0, 0.0
    for k, v in sd.items():
        if k.startswith("denoiser.text_encoder."):
            dropped_te += 1
            continue
        if k.startswith("denoiser.model."):
            bk = k[len("denoiser.model."):]
            b = base_dit.get(bk)
            if b is not None and b.shape == v.shape and torch.equal(b.to(v.dtype), v):
                dropped_base += 1
                continue
        kept[k] = v
        kept_bytes += v.numel() * v.element_size() if hasattr(v, "numel") else 0

    slim = {kk: ck[kk] for kk in ("epoch", "global_step", "pytorch-lightning_version") if kk in ck}
    slim["state_dict"] = kept
    dst = OUT / name / "last.ckpt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    print(f"\n{name}")
    print(f"  dropped text_encoder tensors: {dropped_te}")
    print(f"  dropped base-identical model tensors: {dropped_base}")
    print(f"  kept (trained) tensors: {len(kept)}  ({gb(kept_bytes):.2f} GB)")
    print(f"  file: {gb(f.stat().st_size):.2f} GB -> {gb(dst.stat().st_size):.2f} GB")
