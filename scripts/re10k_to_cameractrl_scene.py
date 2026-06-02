#!/usr/bin/env python
"""scenetok의 pixelSplat RealEstate10K `.torch` chunk에서 scene을 골라
CameraCtrl / AC3D 데이터셋 포맷으로 변환·저장한다 (1개 또는 여러 개).

pixelSplat chunk는 scene별로 `images`(jpeg bytes), `cameras`(N,18 = fx fy cx cy +
2 reserved + 3x4 w2c), `timestamps`(N,), `url`, `key`를 이미 들고 있어 YouTube
다운로드가 필요 없다. 출력 구조 (CameraCtrl README와 동일):

    <out_root>/
      annotations/test.json        # list[{clip_name, clip_path, pose_file, caption}]
      pose_files/<key>.txt         # line0=url, 이후 프레임별 "ts fx fy cx cy 0 0 <w2c 12>"
      video_clips/<key>.mp4

AC3D 추론은 `video_root_dir=<out_root>`, `annotation_json=annotations/test.json`으로
사용하고, scene 개수만큼 `--start_camera_idx 0 --end_camera_idx <N>`으로 순회한다.

Usage:
  # scene 1개 (기본)
  python scripts/re10k_to_cameractrl_scene.py \
      --chunk DATA/re10k/re10k/test/000000.torch --out_root src/model/ac3d/data/re10k_oneshot
  # scene 10개
  python scripts/re10k_to_cameractrl_scene.py \
      --chunk DATA/re10k/re10k/test/000000.torch --out_root src/model/ac3d/data/re10k_multi --limit 10
"""
import argparse
import io
import json
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


def convert_scene(scene, out_root: Path, fps: int, caption: str, min_frames: int):
    """한 scene을 mp4 + pose txt로 쓰고 annotation entry(dict)를 반환. 실패 시 None."""
    key = scene["key"]
    images = scene["images"]
    cameras = scene["cameras"].numpy()          # (N, 18)
    timestamps = scene["timestamps"].numpy()    # (N,)
    url = scene.get("url") or "https://www.youtube.com/watch?v=UNKNOWN"
    n = len(images)
    if not (cameras.shape[0] == n == len(timestamps)):
        print(f"[skip {key}] count mismatch images={n} cameras={cameras.shape[0]} ts={len(timestamps)}")
        return None
    if n < min_frames:
        print(f"[skip {key}] only {n} frames (< {min_frames})")
        return None

    # 1) decode jpeg frames (scenetok dataset_re10k.convert_images와 동일 방식)
    frames = [np.asarray(Image.open(io.BytesIO(b.numpy().tobytes())).convert("RGB"))
              for b in images]

    # 2) write mp4 (macro_block_size=1 → 원본 해상도 보존)
    clip_rel = f"video_clips/{key}.mp4"
    clip_path = out_root / clip_rel
    imageio.mimsave(str(clip_path), frames, fps=fps, quality=9, macro_block_size=1)

    # 3) decord로 프레임 수 검증 (AC3D get_batch가 cam 수만큼 인덱싱 → 무한 retry/OOB 방지)
    try:
        from decord import VideoReader
        n_read = len(VideoReader(str(clip_path)))
    except Exception as e:
        n_read = n
        print(f"[warn] decord 검증 스킵 ({e}); 프레임 수 동일 가정")
    n_use = min(n, n_read)
    if n_read != n:
        print(f"[warn {key}] mp4 frames {n_read} != source {n} → {n_use}로 맞춤")
    if n_use < min_frames:
        print(f"[skip {key}] usable frames {n_use} < {min_frames}")
        return None

    # 4) pose txt (Camera 파서 entry[1:5],[7:] 매핑)
    pose_rel = f"pose_files/{key}.txt"
    lines = [url + "\n"]
    for i in range(n_use):
        vals = " ".join(f"{v:.9f}" for v in cameras[i])
        lines.append(f"{int(timestamps[i])} {vals}\n")
    (out_root / pose_rel).write_text("".join(lines))

    print(f"[ok] {key}: {n_use} frames")
    return {"clip_name": key, "clip_path": clip_rel, "pose_file": pose_rel, "caption": caption}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True, help="path to a re10k .torch chunk")
    ap.add_argument("--out_root", required=True, help="CameraCtrl-format dataset root to write")
    ap.add_argument("--key", default=None,
                    help="특정 scene key (지정 시 --limit 무시하고 그 scene만)")
    ap.add_argument("--limit", type=int, default=1,
                    help="변환할 scene 개수 (>= min_frames인 scene을 chunk 순서대로 선택)")
    ap.add_argument("--min_frames", type=int, default=49,
                    help="AC3D sample_n_frames; scene이 최소 이만큼 프레임을 가져야 함")
    ap.add_argument("--fps", type=int, default=8, help="mp4 fps (AC3D는 8)")
    ap.add_argument("--caption", default="a camera moving through a scene",
                    help="annotation caption (추론은 CLI --prompt를 쓰므로 placeholder)")
    args = ap.parse_args()

    chunk = torch.load(args.chunk, weights_only=False)  # list[scene dict]

    if args.key is not None:
        sel = [ex for ex in chunk if ex["key"] == args.key]
        if not sel:
            raise SystemExit(f"key {args.key!r} not found in {args.chunk}")
    else:
        sel = [ex for ex in chunk if len(ex["images"]) >= args.min_frames]
        if not sel:
            raise SystemExit(f"no scene with >= {args.min_frames} frames in {args.chunk}")

    out_root = Path(args.out_root)
    for sub in ("pose_files", "video_clips", "annotations"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    entries = []
    for scene in sel:
        if len(entries) >= args.limit:
            break
        entry = convert_scene(scene, out_root, args.fps, args.caption, args.min_frames)
        if entry is not None:
            entries.append(entry)

    if not entries:
        raise SystemExit("변환된 scene이 없음")

    (out_root / "annotations" / "test.json").write_text(json.dumps(entries))

    print("\n=== done ===")
    print(f"scenes written : {len(entries)}")
    print(f"out_root       : {out_root}")
    print(f"annotation     : {out_root / 'annotations' / 'test.json'}")
    print(f"\nAC3D 추론: video_root_dir={out_root}  annotation_json=annotations/test.json"
          f"  --start_camera_idx 0 --end_camera_idx {len(entries)}")


if __name__ == "__main__":
    main()
