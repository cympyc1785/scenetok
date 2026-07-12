"""Strip Lightning checkpoints down to WEIGHTS ONLY (drop optimizer_states,
lr_schedulers, loops, callbacks) so they're small and easy to move.

Keeps: state_dict + light metadata (epoch, global_step, pytorch-lightning_version).
Result loads fine for inference / warm-start (load_weights reads 'state_dict'),
but NOT for resuming training (no optimizer state) — that's the point.

Originals are left untouched; slim copies go to <out_root>/<name>/last.ckpt.
"""
import argparse
import io
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
KEEP = {"state_dict", "epoch", "global_step", "pytorch-lightning_version"}


def gb(p):
    return Path(p).stat().st_size / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "my_checkpoints/backup"))
    ap.add_argument("--out", default=str(REPO / "my_checkpoints/backup_slim"))
    args = ap.parse_args()

    ckpts = sorted(Path(args.src).rglob("*.ckpt"))
    print(f"found {len(ckpts)} ckpt(s) under {args.src}\n")
    tot_in = tot_out = 0.0
    for f in ckpts:
        rel = f.relative_to(args.src)
        dst = Path(args.out) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        ck = torch.load(f, map_location="cpu", weights_only=False)
        slim = {k: ck[k] for k in ck if k in KEEP}
        torch.save(slim, dst)
        i, o = gb(f), gb(dst)
        tot_in += i
        tot_out += o
        dropped = [k for k in ck if k not in KEEP]
        print(f"{rel}\n  {i:.2f}GB -> {o:.2f}GB  (dropped: {dropped})")
    print(f"\nTOTAL: {tot_in:.1f}GB -> {tot_out:.1f}GB  (saved {tot_in - tot_out:.1f}GB)")
    print(f"slim copies at: {args.out}  (originals untouched)")


if __name__ == "__main__":
    main()
