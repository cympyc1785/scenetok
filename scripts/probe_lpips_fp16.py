"""Reproduce the 'LPIPS returns same value' symptom.

Hypothesis: src/model/metrics.py loads `LPIPS(net='vgg').eval().to(torch.float16)`
and casts pred/gt to fp16 before forwarding. VGG features in fp16 often saturate /
underflow so LPIPS collapses toward a constant regardless of input.

This script builds 6 random image pairs with increasing per-pixel L1 difference
and prints LPIPS for both fp32 and fp16 variants. If fp16 is buggy we'll see the
fp16 column nearly constant while fp32 grows monotonically with the perturbation.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from lpips import LPIPS

torch.manual_seed(0)
device = "cuda"

lpips_fp32 = LPIPS(net="vgg").eval().to(device)
lpips_fp16 = LPIPS(net="vgg").eval().to(torch.float16).to(device)

# Build a fixed clean image and 6 progressively noisier versions.
# Use a "natural-image-like" base (low-pass random noise) rather than uniform
# pixel noise — uniform noise has very high LPIPS even at small magnitudes,
# while natural images at small perturbations are the realistic operating range.
H, W = 256, 448
base = torch.rand(1, 3, H // 16, W // 16, device=device)
clean = torch.nn.functional.interpolate(base, size=(H, W), mode="bilinear", align_corners=False)
noise_levels = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

print(f"{'noise':>8s}  {'L1':>8s}  {'lpips_fp32':>10s}  {'lpips_fp16':>10s}")
for nl in noise_levels:
    perturbed = (clean + nl * torch.randn_like(clean)).clamp(0, 1)
    l1 = (clean - perturbed).abs().mean().item()
    with torch.no_grad():
        d32 = lpips_fp32(clean.float(), perturbed.float(), normalize=True).item()
        d16 = lpips_fp16(
            clean.to(torch.float16), perturbed.to(torch.float16), normalize=True
        ).float().item()
    print(f"{nl:8.3f}  {l1:8.4f}  {d32:10.4f}  {d16:10.4f}")

# Also: two totally different images
print("\n--- two unrelated random images ---")
a = torch.rand(1, 3, H, W, device=device)
b = torch.rand(1, 3, H, W, device=device)
with torch.no_grad():
    d32 = lpips_fp32(a.float(), b.float(), normalize=True).item()
    d16 = lpips_fp16(a.to(torch.float16), b.to(torch.float16), normalize=True).float().item()
print(f"  fp32={d32:.4f}   fp16={d16:.4f}")

# --- Reproduce the exact metrics.py Metric.compute_lpips path ---
print("\n--- Metric.compute_lpips() with bf16 inputs (real validation dtype) ---")
from src.model.metrics import Metric
from src.misc.torch_utils import convert_to_buffer

metric = Metric().eval()
# Match the wrapper's exact setup: freeze() + convert_to_buffer()
for param in metric.parameters():
    param.requires_grad = False
convert_to_buffer(metric, persistent=False)
metric = metric.to(device)

# Now exercise the exact path metrics.py uses on real re10k Wan-recon images:
# (orig, recon) at 256x448 — small but nonzero differences typical of val.
print("\n--- LPIPS on real re10k recon pair (256x448 Wan VAE) ---")
import json, io
from PIL import Image
import torchvision.transforms as T

ROOT = Path("DATA/re10k/re10k/test")
with open(ROOT / "index.json") as f:
    index = json.load(f)
keys = list(index.keys())[:10]

recon_dir = REPO_ROOT / "tmp" / "wan_vae_recon"
if not recon_dir.exists():
    print(f"  (no recon dump at {recon_dir} — run scripts/dump_wan_vae_recon.py first)")
    sys.exit(0)

to_t = T.ToTensor()
preds, gts = [], []
for i, scene in enumerate(keys):
    orig_p = recon_dir / f"scene{i:02d}_{scene[:16]}_256x448_orig.png"
    recon_p = recon_dir / f"scene{i:02d}_{scene[:16]}_256x448_recon.png"
    if not orig_p.exists() or not recon_p.exists():
        continue
    gts.append(to_t(Image.open(orig_p)))
    preds.append(to_t(Image.open(recon_p)))
preds = torch.stack(preds).to(device)
gts = torch.stack(gts).to(device)

# Call compute_lpips one image at a time so we can verify per-image variation.
print(f"{'scene':>16s}  {'Metric.compute_lpips':>22s}")
for i, scene in enumerate(keys):
    if i >= len(preds):
        break
    # Match validation-time dtype: gathered_sampled comes through bf16 autocast
    p_bf = preds[i:i+1].to(torch.bfloat16)
    g_bf = gts[i:i+1].to(torch.bfloat16)
    val = metric.compute_lpips(p_bf, g_bf).item()
    print(f"  {scene[:14]:>14s}  {val:22.6f}")

# Aggregate call (the actual validation pattern: all N images at once).
print("\n--- Aggregate Metric.compute_lpips over all 10 pairs (bf16) ---")
agg = metric.compute_lpips(preds.to(torch.bfloat16), gts.to(torch.bfloat16)).item()
print(f"  Metric.compute_lpips(10-image bf16 batch) = {agg:.6f}")
# Sanity: same as mean of fp32 per-image LPIPS?
fp32_each = []
for i in range(len(preds)):
    with torch.no_grad():
        d = lpips_fp32(preds[i:i+1].float(), gts[i:i+1].float(), normalize=True).item()
    fp32_each.append(d)
print(f"  expected (fp32, mean per-image)         = {sum(fp32_each)/len(fp32_each):.6f}")
