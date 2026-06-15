"""Pre-generate DiffSynth-style noise for axis-permuted injection into my T2VWrapper."""
import argparse, torch
p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=1)
p.add_argument("--C", type=int, default=48)
p.add_argument("--T", type=int, default=10)
p.add_argument("--H", type=int, default=30)
p.add_argument("--W", type=int, default=52)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
p.add_argument("--out", required=True)
args = p.parse_args()
dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
g = torch.Generator("cpu").manual_seed(int(args.seed))
n = torch.randn((args.B, args.C, args.T, args.H, args.W), generator=g, device="cpu", dtype=torch.float32).to(dtype=dtype)
# Permute to my wrapper's axis: (B,C,T,H,W) → (B,T,C,H,W)
n2 = n.permute(0, 2, 1, 3, 4).contiguous()
print(f"shape={tuple(n2.shape)}  sum={n2.float().sum().item():.4f}  dtype={n2.dtype}")
torch.save(n2, args.out)
