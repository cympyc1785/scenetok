"""Encode + decode 10 re10k test scenes through Wan2.2 VAE at two resolutions
and dump per-scene PNGs under tmp/wan_vae_recon/ for visual comparison.

Uses the same [-1,1] normalization the training path applies
(`src/model/diffusion.py:35` does `inputs * 2 - 1` before encode).
"""

import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms as T
from PIL import Image

from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg

N_SCENES = 10
RESOLUTIONS = [(256, 256), (256, 448), (480, 832)]
OUT_DIR = REPO_ROOT / "tmp" / "wan_vae_recon"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading Wan2.2 VAE  (device={DEVICE})")
vae = AutoencoderWan(WanKwargsCfg(in_channels=3, latent_channels=48, scaling_factor=1.0))
vae.from_pretrained("checkpoints/Wan2.2_VAE.pth")
vae = vae.to(DEVICE).eval()
torch.set_grad_enabled(False)

ROOT = Path("DATA/re10k/re10k/test")
with open(ROOT / "index.json") as f:
    index = json.load(f)
scene_keys = list(index.keys())[:N_SCENES]


def make_resize_crop(hw):
    return T.Compose([T.Resize(max(hw)), T.CenterCrop(hw), T.ToTensor()])


def to_pil(x01: torch.Tensor) -> Image.Image:
    """x01: (3, H, W) in [0,1]"""
    return T.functional.to_pil_image(x01.clamp(0, 1).cpu())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse <= 1e-12:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


print(f"\nScenes: {N_SCENES}, Resolutions: {RESOLUTIONS}")
print(f"Out dir: {OUT_DIR}\n")

rows = []
for i, scene in enumerate(scene_keys):
    chunk = torch.load(ROOT / index[scene], weights_only=False)
    example = next(x for x in chunk if x["key"] == scene)
    raw = example["images"][len(example["images"]) // 2]
    img = Image.open(io.BytesIO(raw.numpy().tobytes())).convert("RGB")

    row = {"scene": scene}
    for hw in RESOLUTIONS:
        tag = f"{hw[0]}x{hw[1]}"
        x01 = make_resize_crop(hw)(img)                  # (3, H, W) in [0,1]
        x = (x01 * 2 - 1).unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1, 1, 3, H, W)
        z = vae.encode(x).float()                         # (1, 1, 48, h, w)
        y = vae.decode(z.to(x.dtype))                     # (1, 1, 3, H, W)
        y01 = ((y[0, 0].float() + 1) / 2).clamp(0, 1).cpu()

        scene_short = scene[:16]
        to_pil(x01).save(OUT_DIR / f"scene{i:02d}_{scene_short}_{tag}_orig.png")
        to_pil(y01).save(OUT_DIR / f"scene{i:02d}_{scene_short}_{tag}_recon.png")

        row[f"psnr_{tag}"] = psnr(x01, y01)
        row[f"l1_{tag}"] = (x01 - y01).abs().mean().item()
        row[f"z_std_{tag}"] = z.std().item()
        row[f"z_absmax_{tag}"] = z.abs().max().item()

    rows.append(row)
    print(
        f"scene{i:02d} {scene[:16]}  "
        + "  ".join(
            f"{hw[0]}x{hw[1]}: psnr={row[f'psnr_{hw[0]}x{hw[1]}']:.2f}dB "
            f"l1={row[f'l1_{hw[0]}x{hw[1]}']:.4f} "
            f"z_std={row[f'z_std_{hw[0]}x{hw[1]}']:.3f} "
            f"z|max|={row[f'z_absmax_{hw[0]}x{hw[1]}']:.2f}"
            for hw in RESOLUTIONS
        )
    )

print("\n=== aggregate ===")
for hw in RESOLUTIONS:
    tag = f"{hw[0]}x{hw[1]}"
    psnrs = torch.tensor([r[f"psnr_{tag}"] for r in rows])
    l1s = torch.tensor([r[f"l1_{tag}"] for r in rows])
    zstds = torch.tensor([r[f"z_std_{tag}"] for r in rows])
    zmaxs = torch.tensor([r[f"z_absmax_{tag}"] for r in rows])
    print(
        f"[{tag}]  PSNR mean={psnrs.mean():.2f}dB (min={psnrs.min():.2f})  "
        f"L1 mean={l1s.mean():.4f}  z_std mean={zstds.mean():.3f}  "
        f"z|max| mean={zmaxs.mean():.2f} (max={zmaxs.max():.2f})"
    )
print(f"\nFiles written to: {OUT_DIR}")
