#!/usr/bin/env python
"""scenetok의 pixelSplat RealEstate10K `.torch` chunk에서 **scene 하나**를 골라
CameraCtrl / AC3D 데이터셋 포맷으로 변환·저장한다.

pixelSplat chunk는 scene별로 `images`(jpeg bytes), `cameras`(N,18 = fx fy cx cy +
2 reserved + 3x4 w2c), `timestamps`(N,), `url`, `key`를 이미 들고 있어 YouTube
다운로드가 필요 없다. 출력 구조 (CameraCtrl README와 동일):

    <out_root>/
      annotations/test.json        # list[{clip_name, clip_path, pose_file, caption}]
      pose_files/<key>.txt         # line0=url, 이후 프레임별 "ts fx fy cx cy 0 0 <w2c 12>"
      video_clips/<key>.mp4

AC3D 추론은 `video_root_dir=<out_root>`, `annotation_json=annotations/test.json`으로 사용.

Usage:
  python scripts/re10k_to_cameractrl_scene.py \
      --chunk DATA/re10k/re10k/test/000000.torch \
      --out_root src/model/ac3d/data/re10k_oneshot
"""
import argparse
import io
import json
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True, help="path to a re10k .torch chunk")
    ap.add_argument("--out_root", required=True, help="CameraCtrl-format dataset root to write")
    ap.add_argument("--key", default=None,
                    help="scene key; default = first scene with >= min_frames")
    ap.add_argument("--min_frames", type=int, default=49,
                    help="AC3D sample_n_frames; scene must have at least this many frames")
    ap.add_argument("--fps", type=int, default=8, help="mp4 fps (AC3D exports at 8)")
    ap.add_argument("--caption", default="a camera moving through a scene",
                    help="annotation caption (inference uses CLI --prompt, so this is a placeholder)")
    args = ap.parse_args()

    chunk = torch.load(args.chunk, weights_only=False)  # list[scene dict]

    scene = None
    if args.key is not None:
        scene = next((ex for ex in chunk if ex["key"] == args.key), None)
        if scene is None:
            raise SystemExit(f"key {args.key!r} not found in {args.chunk}")
    else:
        scene = next((ex for ex in chunk if len(ex["images"]) >= args.min_frames), None)
        if scene is None:
            raise SystemExit(f"no scene with >= {args.min_frames} frames in {args.chunk}")

    key = scene["key"]
    images = scene["images"]
    cameras = scene["cameras"].numpy()          # (N, 18)
    timestamps = scene["timestamps"].numpy()    # (N,)
    url = scene.get("url") or "https://www.youtube.com/watch?v=UNKNOWN"
    n = len(images)
    assert cameras.shape[0] == n == len(timestamps), \
        f"count mismatch: images={n} cameras={cameras.shape[0]} ts={len(timestamps)}"
    if n < args.min_frames:
        raise SystemExit(f"scene {key} has only {n} frames (< {args.min_frames})")

    out_root = Path(args.out_root)
    for sub in ("pose_files", "video_clips", "annotations"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # 1) decode jpeg frames (scenetok dataset_re10k.convert_images와 동일 방식)
    frames = [np.asarray(Image.open(io.BytesIO(b.numpy().tobytes())).convert("RGB"))
              for b in images]

    # 2) write mp4 (macro_block_size=1 → 원본 해상도 보존, intrinsic rescale 정확도 유지)
    clip_rel = f"video_clips/{key}.mp4"
    clip_path = out_root / clip_rel
    imageio.mimsave(str(clip_path), frames, fps=args.fps, quality=9, macro_block_size=1)

    # 3) decord로 프레임 수 검증 — AC3D get_batch가 decord로 읽고 cam 수만큼 인덱싱하므로
    #    mp4 프레임 수 == pose 라인 수 == cam 수 여야 무한 retry(OOB)를 피한다.
    try:
        from decord import VideoReader
        n_read = len(VideoReader(str(clip_path)))
    except Exception as e:  # decord 미설치 등
        n_read = n
        print(f"[warn] decord 검증 스킵 ({e}); 프레임 수 동일하다고 가정")

    n_use = min(n, n_read)
    if n_read != n:
        print(f"[warn] mp4 frame count {n_read} != source {n} → pose/annotation을 {n_use}로 맞춤")
    if n_use < args.min_frames:
        raise SystemExit(f"usable frames {n_use} < min_frames {args.min_frames}")

    # 4) pose txt: line0 = url, 이후 "ts fx fy cx cy 0 0 <w2c 12>" (Camera 파서 entry[1:5],[7:] 매핑)
    pose_rel = f"pose_files/{key}.txt"
    lines = [url + "\n"]
    for i in range(n_use):
        vals = " ".join(f"{v:.9f}" for v in cameras[i])
        lines.append(f"{int(timestamps[i])} {vals}\n")
    (out_root / pose_rel).write_text("".join(lines))

    # 5) annotation test.json (list-of-dicts — generate_realestate_json.py와 동일 스키마)
    ann = [{"clip_name": key, "clip_path": clip_rel,
            "pose_file": pose_rel, "caption": args.caption}]
    (out_root / "annotations" / "test.json").write_text(json.dumps(ann))

    print("=== done ===")
    print(f"scene key   : {key}")
    print(f"frames      : source={n}, mp4(decord)={n_read}, used={n_use}")
    print(f"video_clip  : {clip_path}")
    print(f"pose_file   : {out_root / pose_rel}")
    print(f"annotation  : {out_root / 'annotations' / 'test.json'}")
    print(f"\nAC3D 추론 시: video_root_dir={out_root}  annotation_json=annotations/test.json")


if __name__ == "__main__":
    main()
