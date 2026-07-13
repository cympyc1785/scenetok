"""Build extrapolation-evaluation indices for the 3-model comparison
(va-wan_dl3dv vs g1020 vs lagernvs_dl3dv_2-6_v_256).

Common held-out pool = DL3DV 11K (all 498 lagernvs eval scenes exist under our
DATA/DL3DV/DL3DV-960/train/11K). 11K is held out for va-wan/g1020 (1K-trained)
and IS the lagernvs eval pool -> fair for all three.

The SceneTok diffusion decoder (wan target) is a VIDEO model (td=4 latent
chunking) -> targets must be a temporally-COHERENT contiguous clip, not scattered
Delta samples. So each eval sample = one contiguous clip starting at the window
edge and extending CLIP_LEN frames:
    dir=fwd: target = [b, b+1, ..., b+CLIP_LEN-1]   (Delta = frame - b   = 0..L-1)
    dir=bwd: target = [a, a-1, ..., a-(CLIP_LEN-1)]  (Delta = a - frame  = 0..L-1)
Delta=0 is the window edge (reference), Delta>0 = extrapolation depth. CLIP_LEN is
4m+1 so wan chunking (1+(T-1)//4 latents) is exact (no truncation). lagernvs
(feed-forward) renders the same per-frame set; per-frame metrics bin by Delta, and
the contiguous clip also enables FVD.

Per scene (N sorted frames == transforms cameras == frame index):
  * central CONTEXT WINDOW [a,b] (span WINDOW), same for every model so Delta compares.
  * ctx16 = 16 frames evenly in [a,b]  (va-wan native)
  * ctx6  =  6 frames evenly in [a,b]  (g1020 native, and matched-context for all)

Outputs under assets/evaluation_index/:
  extrap_{tag}_{dir}_ctx16.json {scene: {context:[16], target:[CLIP_LEN]}}
  extrap_{tag}_{dir}_ctx6.json  {scene: {context:[6],  target:[CLIP_LEN]}}
  extrap_{tag}_{dir}_meta.json  {scene: {"N":N,"window":[a,b],"dir":..,"delta":{frame:Delta}}}

Loaded via stage=test/val + val_seen=false (-> 11K prefix) +
evaluation_index_path=<json>. Keys = scene hash (== chunk.name)."""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL_JSON = REPO / "submodules/lagernvs/assets/dl3dv_6v_11k.json"
DATA_11K = REPO / "DATA/DL3DV/DL3DV-960/train/11K"
OUT_DIR = REPO / "assets/evaluation_index"

WINDOW = 60           # central context window span (frames)
CLIP_LEN = 33         # contiguous target clip length (4m+1 for exact wan chunking); Delta 0..32


def n_frames(scene_hash: str) -> int | None:
    tj = DATA_11K / scene_hash / "transforms.json"
    if not tj.exists():
        return None
    try:
        return len(json.load(open(tj)).get("frames", []))
    except Exception:
        return None


def evenly(a: int, b: int, k: int) -> list[int]:
    """k evenly spaced unique integer indices in [a,b] inclusive."""
    if k == 1:
        return [(a + b) // 2]
    step = (b - a) / (k - 1)
    out = []
    for i in range(k):
        v = int(round(a + i * step))
        while v in out:
            v += 1
        out.append(v)
    return sorted(out)


def build_scene(N: int, direction: str):
    c = N // 2
    a, b = c - WINDOW // 2, c + WINDOW // 2
    ctx16 = evenly(a, b, 16)
    ctx6 = evenly(a, b, 6)
    if direction == "fwd":
        target = [b + d for d in range(CLIP_LEN) if b + d < N]
        delta = {f: f - b for f in target}
    else:  # bwd
        target = [a - d for d in range(CLIP_LEN) if a - d >= 0]
        delta = {f: a - f for f in target}
    return dict(a=a, b=b, ctx16=ctx16, ctx6=ctx6, target=target, delta=delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_scenes", type=int, default=4)
    ap.add_argument("--min_frames", type=int, default=200)
    ap.add_argument("--dir", choices=["fwd", "bwd"], default="fwd")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    hashes = [k.split("/")[-1] for k in json.load(open(POOL_JSON)).keys()]
    picked = []
    for h in hashes:
        N = n_frames(h)
        if N is not None and N >= args.min_frames:
            picked.append((h, N))
        if len(picked) >= args.num_scenes:
            break
    if not picked:
        raise SystemExit("no eligible scenes")
    print(f"[extrap-index] dir={args.dir} picked {len(picked)} scenes (min_frames={args.min_frames})")

    ctx16_idx, ctx6_idx, meta = {}, {}, {}
    for h, N in picked:
        s = build_scene(N, args.dir)
        ctx16_idx[h] = {"context": s["ctx16"], "target": s["target"]}
        ctx6_idx[h] = {"context": s["ctx6"], "target": s["target"]}
        meta[h] = {"N": N, "window": [s["a"], s["b"]], "dir": args.dir,
                   "delta": {str(f): s["delta"][f] for f in s["target"]}}
        print(f"  {h[:16]} N={N} win=[{s['a']},{s['b']}] "
              f"ctx16={len(s['ctx16'])} ctx6={len(s['ctx6'])} target={len(s['target'])} "
              f"(Delta {min(s['delta'].values())}..{max(s['delta'].values())})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in [(f"extrap_{args.tag}_{args.dir}_ctx16", ctx16_idx),
                      (f"extrap_{args.tag}_{args.dir}_ctx6", ctx6_idx),
                      (f"extrap_{args.tag}_{args.dir}_meta", meta)]:
        p = OUT_DIR / f"{name}.json"
        json.dump(obj, open(p, "w"), indent=1)
        print(f"[extrap-index] wrote {p}")


if __name__ == "__main__":
    main()
