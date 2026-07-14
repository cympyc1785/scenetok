"""Build extrapolation-evaluation indices for the 3-model comparison
(va-wan_dl3dv vs g1020 vs lagernvs_dl3dv_2-6_v_256).

Common held-out pool = DL3DV 11K (all 498 lagernvs eval scenes exist under our
DATA/DL3DV/DL3DV-960/train/11K). 11K is held out for va-wan/g1020 (1K-trained)
and IS the lagernvs eval pool -> fair for all three.

PER-CLIP Δ (lagernvs dl3dv setting): Δ is the offset of the WHOLE contiguous
target clip relative to the context window [a,b] -> each Δ is a separate eval:
    Δ=0  (interpolation): clip fully INSIDE [a,b] (between context views)
    Δ=k>0 (extrapolation): clip = [b+k, ..., b+k+L-1]  (entire clip k frames beyond b)
Each clip is contiguous (video-model friendly) and CLIP_LEN is 4m+1 for exact
wan chunking. All frames of a Δ-clip are tagged with that regime Δ, so the
aggregator reports one metric per regime (Δ = 0 / 10 / 20).

Per scene (N sorted frames == transforms cameras == frame index):
  * central CONTEXT WINDOW [a,b] (span WINDOW), same for every model.
  * ctx16 = 16 frames evenly in [a,b]  (va-wan native)
  * ctx6  =  6 frames evenly in [a,b]  (g1020 native, and matched-context for all)

Outputs under assets/evaluation_index/ (one set PER Δ):
  extrap_{tag}_d{Δ}_ctx16.json {scene: {context:[16], target:[L]}}
  extrap_{tag}_d{Δ}_ctx6.json  {scene: {context:[6],  target:[L]}}
  extrap_{tag}_d{Δ}_meta.json  {scene: {"N":N,"window":[a,b],"regime":Δ,"delta":{frame:Δ}}}

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
CLIP_LEN = 9          # contiguous target clip length (4m+1 for exact wan chunking)
                      # clip = [b+Δ-L+1 .. b+Δ] → FARTHEST view is exactly Δ beyond the window edge


def n_frames(scene_hash: str) -> int | None:
    tj = DATA_11K / scene_hash / "transforms.json"
    if not tj.exists():
        return None
    try:
        return len(json.load(open(tj)).get("frames", []))
    except Exception:
        return None


def evenly(a: int, b: int, k: int) -> list[int]:
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


def clip_for(N: int, a: int, b: int, delta: int, L: int):
    """Contiguous length-L clip whose FARTHEST view is exactly delta beyond the
    window edge b:  clip = [b+delta-L+1, ..., b+delta].
      delta=0  -> ends at b, extends back into [a,b]  (interpolation)
      delta=k  -> farthest view is k frames beyond b   (extrapolation, capped at k)
    None if it doesn't fit."""
    end = b + delta
    start = end - L + 1
    if start < 0 or end >= N:
        return None
    return list(range(start, end + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_scenes", type=int, default=4)
    ap.add_argument("--min_frames", type=int, default=200)
    ap.add_argument("--deltas", default="0,10,20")
    ap.add_argument("--clip_len", type=int, default=CLIP_LEN)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    deltas = [int(d) for d in args.deltas.split(",")]

    hashes = [k.split("/")[-1] for k in json.load(open(POOL_JSON)).keys()]
    picked = []
    for h in hashes:
        Nf = n_frames(h)
        if Nf is not None and Nf >= args.min_frames:
            picked.append((h, Nf))
        if len(picked) >= args.num_scenes:
            break
    if not picked:
        raise SystemExit("no eligible scenes")
    print(f"[extrap-index] picked {len(picked)} scenes, deltas={deltas}, L={args.clip_len}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for delta in deltas:
        ctx16_idx, ctx6_idx, meta = {}, {}, {}
        for h, Nf in picked:
            c = Nf // 2
            a, b = c - WINDOW // 2, c + WINDOW // 2
            clip = clip_for(Nf, a, b, delta, args.clip_len)
            if clip is None:
                print(f"  [skip Δ={delta}] {h[:12]} clip doesn't fit (N={Nf})")
                continue
            ctx16_idx[h] = {"context": evenly(a, b, 16), "target": clip}
            ctx6_idx[h] = {"context": evenly(a, b, 6), "target": clip}
            meta[h] = {"N": Nf, "window": [a, b], "regime": delta,
                       "delta": {str(f): delta for f in clip}}
        for name, obj in [(f"extrap_{args.tag}_d{delta}_ctx16", ctx16_idx),
                          (f"extrap_{args.tag}_d{delta}_ctx6", ctx6_idx),
                          (f"extrap_{args.tag}_d{delta}_meta", meta)]:
            json.dump(obj, open(OUT_DIR / f"{name}.json", "w"), indent=1)
        print(f"[extrap-index] Δ={delta}: {len(meta)} scenes -> "
              f"extrap_{args.tag}_d{delta}_{{ctx16,ctx6,meta}}.json "
              f"(target e.g. {list(meta.values())[0]['delta'].keys().__iter__().__next__()}..)")


if __name__ == "__main__":
    main()
