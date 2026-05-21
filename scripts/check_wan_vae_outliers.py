"""Quick Wan2.2 VAE anomaly probe over re10k scenes.

Loads N test scenes, encodes each through the Wan VAE used as `wan_single` context
encoder (single-frame mode, matching training), and reports any scene whose latent
shows outlier statistics (NaN/Inf, large mean, large std, extreme min/max, heavy
tails). Use this to check whether specific scenes are responsible for unstable
gradient signals into the compressor.

Usage:
    python scripts/check_wan_vae_outliers.py [num_scenes] [resolution]

    num_scenes  : how many scenes to sample (default 64)
    resolution  : 'h,w' input shape (default '256,448')
"""

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make `import src.*` work no matter where the script is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms as T
from PIL import Image

from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg

# ---- args ----
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
HW = tuple(int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "256,448").split(","))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Outlier thresholds (per latent tensor over one frame)
THRESH = {
    "mean_abs": 0.5,    # |E[z]| should be near 0
    "std_max": 1.5,     # std(z) should be near 1
    "std_min": 0.5,
    "abs_max": 6.0,     # max|z| (assuming N(0,1)-ish)
    "kurt": 6.0,        # >6 = heavy tails (normal=3)
}

# ---- load Wan VAE (context: wan_single) ----
print(f"Loading Wan2.2 VAE  (input HW={HW},  N={N} scenes,  device={DEVICE})")
vae = AutoencoderWan(WanKwargsCfg(in_channels=3, latent_channels=48, scaling_factor=1.0))
vae.from_pretrained("checkpoints/Wan2.2_VAE.pth")
vae = vae.to(DEVICE).eval()
torch.set_grad_enabled(False)

resize_crop = T.Compose([T.Resize(max(HW)), T.CenterCrop(HW), T.ToTensor()])

# ---- iterate re10k test scenes ----
ROOT = Path("DATA/re10k/re10k/test")
with open(ROOT / "index.json") as f:
    index = json.load(f)
scene_keys = list(index.keys())[:N]

violations = defaultdict(list)
per_channel_std_log = []

for i, scene in enumerate(scene_keys):
    chunk = torch.load(ROOT / index[scene], weights_only=False)
    example = next(x for x in chunk if x["key"] == scene)
    # Pick the middle frame
    images_raw = example["images"]
    if len(images_raw) < 1:
        continue
    raw = images_raw[len(images_raw) // 2]
    img = Image.open(io.BytesIO(raw.numpy().tobytes())).convert("RGB")
    x = resize_crop(img).unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1, 1, 3, H, W)
    try:
        z = vae.encode(x).float()                                # (1, 1, 48, h, w)
    except Exception as e:
        violations["encode_error"].append((scene, str(e)))
        continue

    if not torch.isfinite(z).all():
        violations["nan_or_inf"].append((scene, "encode produced NaN/Inf"))
        continue

    mean = z.mean().item()
    std = z.std().item()
    amax = z.abs().max().item()
    centered = z - z.mean()
    kurt = (centered.pow(4).mean() / centered.pow(2).mean().pow(2)).item()
    per_ch_std = z.view(48, -1).std(dim=1).cpu()
    per_channel_std_log.append(per_ch_std)

    flagged = []
    if abs(mean) > THRESH["mean_abs"]:
        flagged.append(f"|mean|={abs(mean):.2f}")
    if std > THRESH["std_max"] or std < THRESH["std_min"]:
        flagged.append(f"std={std:.2f}")
    if amax > THRESH["abs_max"]:
        flagged.append(f"max|z|={amax:.2f}")
    if kurt > THRESH["kurt"]:
        flagged.append(f"kurtosis={kurt:.2f}")
    if flagged:
        violations["soft_outlier"].append((scene, ", ".join(flagged)))

    if (i + 1) % 16 == 0:
        print(f"  scanned {i+1}/{len(scene_keys)}")

# ---- summary ----
print("\n=== summary ===")
print(f"scenes scanned: {len(scene_keys)}")
for cat, lst in violations.items():
    print(f"\n[{cat}] {len(lst)} hits")
    for scene, why in lst[:8]:
        print(f"  {scene}  ({why})")
    if len(lst) > 8:
        print(f"  ... and {len(lst)-8} more")

if per_channel_std_log:
    ch = torch.stack(per_channel_std_log)              # (N, 48)
    print("\nper-channel std (across scenes):")
    print(f"  mean        = {ch.mean(dim=0).mean().item():.3f}")
    print(f"  worst ch std (max over scenes & channels) = {ch.max().item():.3f}")
    print(f"  channels w/ mean>1.5 across scenes        = {(ch.mean(dim=0) > 1.5).sum().item()}/48")
    print(f"  channels w/ mean<0.5 across scenes        = {(ch.mean(dim=0) < 0.5).sum().item()}/48")
