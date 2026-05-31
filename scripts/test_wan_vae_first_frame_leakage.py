"""Test whether Wan VAE decoder leaks noise from neighboring latent frames
into the first pixel frame — to verify if hard-replacement at V_lat=0 yields
a clean first pixel frame even when V_lat>=1 latents are random.

Setup:
- Encode a known image as a 4-frame block → first latent (V_lat=0)
- Build a multi-latent sequence: V_lat=0 = clean, V_lat=1..N-1 = noise
- Decode and check first pixel frame quality vs the "decode V_lat=0 alone" baseline.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import io, json
import numpy as np
import torch
import torchvision.transforms as TT
from PIL import Image

from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg

DEVICE = "cuda"
H, W = 480, 832
OUT = REPO_ROOT / "tmp" / "wan_vae_temporal_leak_test"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)

# Load Wan VAE
vae = AutoencoderWan(WanKwargsCfg(latent_channels=48)).from_pretrained(
    "checkpoints/Wan2.2_VAE.pth"
).to(DEVICE).eval()

# Load any DL3DV image as a clean first frame
img_path = "DATA/DL3DV/DL3DV-960/train/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/images/frame_00021.png"
img = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
x01 = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
x_m11 = (x01 * 2 - 1).unsqueeze(0).unsqueeze(0).to(DEVICE)        # (1, 1, 3, H, W)

# Encode as 4-frame block (1 + 3 zero pad) → 1 latent
pad = torch.zeros(1, 3, 3, H, W, device=DEVICE, dtype=x_m11.dtype)
ff_video = torch.cat([x_m11, pad], dim=1)                          # (1, 4, 3, H, W)
ff_latent = vae.encode(ff_video)[:, :1]                            # (1, 1, 48, h, w)
print(f"first latent: {tuple(ff_latent.shape)}, std={ff_latent.float().std().item():.3f}")

# Baseline: decode just the single latent (V_lat=1) — first pixel frame baseline
dec_alone = vae.decode(ff_latent)                                   # (1, V_pix_alone, 3, H, W)
print(f"decode 1-latent → {tuple(dec_alone.shape)}")

# Build (1, 10, 48, h, w) — V_lat=0 = clean, V_lat=1..9 = noise scaled like real latents
V_lat = 10
hL, wL = ff_latent.shape[-2:]
noise_scale = float(ff_latent.float().std().item()) * 1.0           # match magnitude
noisy_seq = torch.randn(1, V_lat, 48, hL, wL, device=DEVICE) * noise_scale
noisy_seq[:, :1] = ff_latent
print(f"sequence: {tuple(noisy_seq.shape)}, V_lat=0=clean, V_lat=1..9=noise (std≈{noise_scale:.3f})")

dec_seq = vae.decode(noisy_seq)                                     # (1, V_pix, 3, H, W)
print(f"decode {V_lat}-latent → {tuple(dec_seq.shape)}")

# Save first pixel frames from each
def save(t, path):
    img = ((t.detach().cpu().float() + 1) / 2).clamp(0, 1)
    TT.functional.to_pil_image(img).save(path)

save(dec_alone[0, 0], OUT / "00_decode_alone_v_pix_0.png")
save(dec_seq[0, 0], OUT / "01_decode_seq_v_pix_0.png")
save(x_m11[0, 0], OUT / "02_gt.png")

# L1 vs GT on first pixel frame
gt = x_m11[0, 0].cpu()
alone = dec_alone[0, 0].cpu()
seq = dec_seq[0, 0].cpu()
print(f"L1 GT vs decode_alone[0]: {(gt - alone).abs().mean().item():.4f}")
print(f"L1 GT vs decode_seq[0]:   {(gt - seq).abs().mean().item():.4f}")
print(f"L1 decode_alone[0] vs decode_seq[0]: {(alone - seq).abs().mean().item():.4f}")
print(f"(if temporal leakage exists, the latter is non-trivial.)")
print(f"\nFiles in {OUT}")
