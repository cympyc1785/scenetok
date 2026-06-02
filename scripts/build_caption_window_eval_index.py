#!/usr/bin/env python
"""caption_window용 고정 evaluation index를 생성한다.

기존 `assets/evaluation_index/dl3dv_c16_37_{standard,unseen}.json`(bounded용)과 같은
포맷 `{scene_hash: {"context": [...], "target": [...]}}`이지만, target을 prompts_37.json
의 한 caption window로 잡고 context를 그 window 안/주변에서 뽑는다 (caption_window
sampler 그대로). val/test에서 이 파일을 로드하면 매번 동일한 caption-정렬 index가 쓰인다.

scene 목록은 기존 index에서 그대로 가져와(같은 scene 평가) hash로 디렉토리를 찾고,
scene별 고정 seed로 sampler를 호출해 재현 가능하게 만든다.

Usage:
  python scripts/build_caption_window_eval_index.py --split standard
  python scripts/build_caption_window_eval_index.py --split unseen
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from src.dataset.dataset_dl3dv import get_dl3dv_data_dir
from src.dataset.view_sampler.view_sampler_caption_window import (
    ViewSamplerCaptionWindow, ViewSamplerCaptionWindowCfg)
from src.misc.camera_utils import convert_poses
from src.misc.dl3dv_utils import load_metadata


def find_scene_dir(train_root: Path, scene_hash: str) -> Path | None:
    direct = next(train_root.glob(f"*/{scene_hash}"), None)
    return direct if (direct and direct.is_dir()) else None


def seed_for(scene_hash: str) -> int:
    return int(hashlib.sha1(scene_hash.encode()).hexdigest(), 16) % (2**32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["standard", "unseen"], default="standard")
    ap.add_argument("--root", default="./DATA/DL3DV/DL3DV-960")
    ap.add_argument("--size", default="37", help="frame-count tag in filename (matches V_pixel)")
    ap.add_argument("--scene_list", default=None,
                    help="기존 index json (scene 목록 출처). 기본=dl3dv_c16_<size>_<split>.json")
    ap.add_argument("--out", default=None,
                    help="출력 경로. 기본=dl3dv_c16_<size>_caption_<split>.json")
    # caption_window sampler 파라미터 (training config와 일치시킬 것)
    ap.add_argument("--num_context", type=int, default=16)
    ap.add_argument("--num_target", type=int, default=10)
    ap.add_argument("--temporal_downsample", type=int, default=4)
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--chunk_targets", action="store_true")
    ap.add_argument("--context_sampling", choices=["farthest_point", "uniform"], default="farthest_point")
    ap.add_argument("--prompts_filename", default="prompts_37.json")
    ap.add_argument("--context_window_margin", type=int, default=0)
    args = ap.parse_args()

    idx_dir = Path("assets/evaluation_index")
    scene_list_path = Path(args.scene_list) if args.scene_list else \
        idx_dir / f"dl3dv_c16_{args.size}_{args.split}.json"
    out_path = Path(args.out) if args.out else \
        idx_dir / f"dl3dv_c16_{args.size}_caption_{args.split}.json"

    scenes = list(json.load(scene_list_path.open()).keys())
    print(f"[scene list] {scene_list_path}  ({len(scenes)} scenes)")

    cfg = ViewSamplerCaptionWindowCfg(
        name="caption_window",
        num_context_views=args.num_context,
        num_target_views=args.num_target,
        context_sampling=args.context_sampling,
        prompts_filename=args.prompts_filename,
        temporal_downsample=args.temporal_downsample,
        offset=args.offset,
        chunk_targets=args.chunk_targets,
        context_window_margin=args.context_window_margin,
    )
    # step_tracker=None → global_step=0 → effective_margin = context_window_margin (고정)
    sampler = ViewSamplerCaptionWindow(cfg, "test", False, None, None)

    train_root = Path(args.root) / "train"
    out: dict[str, dict[str, list[int]]] = {}
    skipped = {"no_dir": 0, "no_poses": 0, "sampler_error": 0}

    for h in scenes:
        scene_dir = find_scene_dir(train_root, h)
        if scene_dir is None:
            skipped["no_dir"] += 1
            continue
        data_dir = get_dl3dv_data_dir(scene_dir)  # nerfstudio/ if present
        try:
            example = load_metadata(data_dir / "transforms.json")
            extrinsics, _ = convert_poses(example["cameras"])
        except Exception:
            skipped["no_poses"] += 1
            continue
        num_views = extrinsics.shape[0]
        random.seed(seed_for(h))  # 재현 가능한 window 선택
        try:
            vi, _ = sampler.sample(
                num_views=num_views, num_latents=num_views, stage="test",
                extrinsics=extrinsics, scene=h, scene_dir=scene_dir)
        except Exception:
            skipped["sampler_error"] += 1
            continue
        out[h] = {
            "context": vi.context.tolist(),
            "target": vi.target.tolist(),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"[written] {out_path}  ({len(out)}/{len(scenes)} scenes)")
    print(f"[skipped] {skipped}")
    if out:
        ex_h = next(iter(out))
        ex = out[ex_h]
        print(f"[example] {ex_h}\n  context({len(ex['context'])})={ex['context']}\n"
              f"  target ({len(ex['target'])})=[{ex['target'][0]}..{ex['target'][-1]}]")


if __name__ == "__main__":
    main()
