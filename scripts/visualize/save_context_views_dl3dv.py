"""Save DL3DV context-view bundles (c2w + normalized intrinsics + images) per
scene, indexed by an evaluation index json ({scene: {context: [...], target}}).
Bundle format matches scripts/viser_server.py --data (viser frustum viz).

Usage:
  python scripts/save_context_views_dl3dv.py \
     --index assets/evaluation_index/dl3dv_c16_37_caption_standard.json \
     --out results/context_views_dl3dv_c16_37 --img_width 320
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np, torch
from PIL import Image

REPO = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok")
DL3DV_ROOT = REPO / "DATA/DL3DV/DL3DV-960/train"
GL2CV = np.diag([1, -1, -1, 1]).astype(np.float64)   # OpenGL(NeRF) c2w → OpenCV


def find_scene_dir(scene_hash):
    hits = glob.glob(str(DL3DV_ROOT / f"*/{scene_hash}"))
    return Path(hits[0]) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="assets/evaluation_index/dl3dv_c16_37_caption_standard.json")
    ap.add_argument("--out", default="results/context_views_dl3dv_c16_37")
    ap.add_argument("--img_width", type=int, default=320)
    ap.add_argument("--relative", action="store_true", default=True,
                    help="poses relative to first context cam (clean origin)")
    ap.add_argument("--target_images", action="store_true", default=False,
                    help="also store target-view images (heavier; default poses only)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = json.load(open(args.index))
    print(f"[ctx] {len(index)} scenes in index")

    saved = missing = 0
    for scene_hash, spec in index.items():
        sc = find_scene_dir(scene_hash)
        if sc is None or not (sc / "transforms.json").exists():
            missing += 1; continue
        d = json.load(open(sc / "transforms.json"))
        W, H = d["w"], d["h"]
        K = np.array([[d["fl_x"] / W, 0, d["cx"] / W],
                      [0, d["fl_y"] / H, d["cy"] / H], [0, 0, 1]], dtype=np.float64)
        frames = sorted(d["frames"], key=lambda f: f["file_path"])
        ctx = spec["context"]; tgt = spec.get("target", [])
        img_dir = next((x for x in ["images", "images_4", "images_8"] if (sc / x).is_dir()), "images")
        TW = args.img_width; TH = int(round(TW * H / W))

        def grab(indices, want_images):
            c, im = [], []
            for fi in indices:
                if fi >= len(frames):
                    return None, None
                fr = frames[fi]
                c.append(np.array(fr["transform_matrix"], dtype=np.float64) @ GL2CV)
                if want_images:
                    p = sc / img_dir / os.path.basename(fr["file_path"])
                    im.append(np.asarray(Image.open(p).convert("RGB").resize((TW, TH), Image.BILINEAR)))
            return np.stack(c), (np.stack(im) if want_images else None)

        c2w, imgs = grab(ctx, True)
        if c2w is None:
            missing += 1; continue
        # 첫 context cam 기준 relative (context·target 동일 변환으로 정합 유지)
        ref_inv = np.linalg.inv(c2w[0]) if args.relative else np.eye(4)
        c2w = ref_inv[None] @ c2w
        tgt_c2w, tgt_imgs = grab(tgt, args.target_images) if tgt else (None, None)
        bundle = {
            "c2w": torch.tensor(c2w, dtype=torch.float32),
            "intrinsics": torch.tensor(np.stack([K] * len(ctx)), dtype=torch.float32),
            "images": torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).contiguous(),  # (N,3,H,W) uint8
            "scene": scene_hash, "context_indices": ctx,
        }
        if tgt_c2w is not None:
            bundle["target_c2w"] = torch.tensor(ref_inv[None] @ tgt_c2w, dtype=torch.float32)
            bundle["target_intrinsics"] = torch.tensor(np.stack([K] * len(tgt)), dtype=torch.float32)
            bundle["target_indices"] = tgt
            if tgt_imgs is not None:
                bundle["target_images"] = torch.tensor(np.stack(tgt_imgs)).permute(0, 3, 1, 2).contiguous()
        torch.save(bundle, os.path.join(args.out, f"{scene_hash[:16]}.pt"))
        saved += 1
        if saved % 20 == 0:
            print(f"  saved {saved} ...", flush=True)
    print(f"[ctx] done: saved {saved}, missing/skipped {missing} → {args.out}")


if __name__ == "__main__":
    main()
