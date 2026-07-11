"""Generate a LagerNVS FixedViewSelector eval json for the converted RE10K test
subset (submodules/lagernvs/data_mixed/re10k/test). Per scene: context = 2 views
spanning the clip, target = evenly-spaced interior frames. Keyed by seq id.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEST = REPO / "submodules/lagernvs/data_mixed/re10k/test"
OUT = REPO / "submodules/lagernvs/assets/re10k_2v_mixed.json"
N_TARGET = 6

d = {}
for meta in sorted((TEST / "metadata").glob("*.json")):
    seq = meta.stem
    n = len(json.load(open(meta))["frames"])
    if n < 8:
        continue
    ctx = [0, n - 1]
    step = n / (N_TARGET + 1)
    tgt = sorted({int(round(step * (i + 1))) for i in range(N_TARGET)} - set(ctx))
    tgt = [t for t in tgt if 0 < t < n - 1][:N_TARGET]
    d[seq] = {"context": ctx, "target": tgt}

json.dump(d, open(OUT, "w"))
print(f"wrote {OUT}: {len(d)} scenes")
