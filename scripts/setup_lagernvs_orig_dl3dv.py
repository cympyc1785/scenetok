#!/usr/bin/env python3
"""Build a LagerNVS-original-code data root for DL3DV 1K(train)/11K(eval) smallset.

LagerNVS's Dl3dvDataset expects, under $LAGERNVS_DATA_ROOT/dl3dv/:
  - full_list_{train,test}.txt   (lines "<batch>/<seq>")
  - <batch>/<seq>/transforms.json
  - <batch>/<seq>/images_4/*.png

Our real data uses "images/" (960x540 = 4x down of the 3840x2160 transforms res,
exactly LagerNVS's images_4 convention) so we symlink images_4 -> images and
transforms.json -> transforms.json per scene. Original code stays untouched.

Eval uses FixedViewSelector json (context/target frame indices within 0..49),
replicated per 11K scene from the canonical LagerNVS dl3dv_{2,4,6}v pattern.
"""
import json
import os

REPO = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok"
REAL = os.path.join(REPO, "DATA/DL3DV/DL3DV-960/train")
ROOT = os.path.join(REPO, "submodules/lagernvs/data_dl3dv_smallset/dl3dv")
ASSETS = os.path.join(REPO, "submodules/lagernvs/assets")

TRAIN_BATCH = "1K"
EVAL_BATCH = "11K"

# Canonical context patterns (frames within 0..49), target = the rest of 0..49.
CONTEXT = {
    2: [0, 49],
    4: [0, 19, 29, 49],
    6: [0, 19, 29, 33, 39, 49],
}


def _img_dirname(batch, seq):
    """Real image dir is 'images' (1K) or 'images_4' (11K); pick whichever exists."""
    d = os.path.join(REAL, batch, seq)
    for name in ("images", "images_4"):
        if os.path.isdir(os.path.join(d, name)):
            return name
    return None


def valid_scene(batch, seq):
    d = os.path.join(REAL, batch, seq)
    return os.path.isfile(os.path.join(d, "transforms.json")) and (
        _img_dirname(batch, seq) is not None
    )


def link_scene(batch, seq):
    dst = os.path.join(ROOT, batch, seq)
    os.makedirs(dst, exist_ok=True)
    src = os.path.join(REAL, batch, seq)
    img_src = _img_dirname(batch, seq)
    for name, target in [("transforms.json", "transforms.json"), ("images_4", img_src)]:
        lp = os.path.join(dst, name)
        tp = os.path.join(src, target)
        if os.path.islink(lp) or os.path.exists(lp):
            os.remove(lp)
        os.symlink(tp, lp)


def build_list(batch):
    seqs = sorted(
        s for s in os.listdir(os.path.join(REAL, batch)) if valid_scene(batch, s)
    )
    for s in seqs:
        link_scene(batch, s)
    return seqs


def main():
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)

    train_seqs = build_list(TRAIN_BATCH)
    eval_seqs = build_list(EVAL_BATCH)
    print(f"train ({TRAIN_BATCH}): {len(train_seqs)} scenes")
    print(f"eval  ({EVAL_BATCH}): {len(eval_seqs)} scenes")

    with open(os.path.join(ROOT, "full_list_train.txt"), "w") as f:
        f.write("\n".join(f"{TRAIN_BATCH}/{s}" for s in train_seqs) + "\n")
    with open(os.path.join(ROOT, "full_list_test.txt"), "w") as f:
        f.write("\n".join(f"{EVAL_BATCH}/{s}" for s in eval_seqs) + "\n")

    for n, ctx in CONTEXT.items():
        tgt = [i for i in range(50) if i not in ctx]
        d = {f"{EVAL_BATCH}/{s}": {"context": ctx, "target": tgt} for s in eval_seqs}
        out = os.path.join(ASSETS, f"dl3dv_{n}v_11k.json")
        with open(out, "w") as f:
            json.dump(d, f)
        print(f"wrote {out}: {len(d)} scenes (ctx={ctx}, {len(tgt)} tgt)")


if __name__ == "__main__":
    main()
