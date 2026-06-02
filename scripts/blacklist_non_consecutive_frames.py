"""Scan DL3DV train pool for scenes whose `images/frame_{05d}.png` files have
gaps in the frame index, and append them to `blacklist.csv` with
`reason=non_consecutive_frames`.

A scene is considered "non-consecutive" if `max(index) - min(index) + 1 != count`,
i.e. there's at least one missing index in the contiguous range. Detail string
records `min..max(count)` and a sample of up to 5 missing indices.

Run from repo root:
    python scripts/blacklist_non_consecutive_frames.py
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset_dl3dv import append_blacklist, load_blacklist


TRAIN_ROOT = REPO_ROOT / "DATA/DL3DV/DL3DV-960/train"
BLACKLIST_PATH = TRAIN_ROOT / "blacklist.csv"
FRAME_RE = re.compile(r"^frame_(\d+)\.png$")
MAX_WORKERS = 32


def find_image_dir(scene_dir: Path) -> Path | None:
    """DL3DV scenes may have `images/` directly or under `nerfstudio/images/`."""
    for sub in (scene_dir / "images", scene_dir / "nerfstudio" / "images"):
        if sub.is_dir():
            return sub
    return None


def scan_scene(scene_dir: Path) -> dict | None:
    img_dir = find_image_dir(scene_dir)
    if img_dir is None:
        return None
    indices: list[int] = []
    try:
        with os.scandir(img_dir) as it:
            for entry in it:
                m = FRAME_RE.match(entry.name)
                if m:
                    indices.append(int(m.group(1)))
    except OSError:
        return None
    if not indices:
        return None
    indices.sort()
    lo, hi = indices[0], indices[-1]
    n = len(indices)
    expected = hi - lo + 1
    if expected == n:
        return None  # consecutive
    missing = sorted(set(range(lo, hi + 1)) - set(indices))
    sample_missing = ",".join(str(m) for m in missing[:5])
    if len(missing) > 5:
        sample_missing += f"+{len(missing)-5}more"
    return {
        "scene": scene_dir.name,
        "reason": "non_consecutive_frames",
        "step": "",
        "loss": "",
        "detail": f"range={lo}..{hi} count={n} missing={sample_missing}",
    }


def main() -> int:
    if not TRAIN_ROOT.is_dir():
        print(f"[error] not a directory: {TRAIN_ROOT}", file=sys.stderr)
        return 1
    print(f"[scan] root: {TRAIN_ROOT}")

    existing = load_blacklist(BLACKLIST_PATH)
    print(f"[blacklist] already-blacklisted: {len(existing)}")

    scene_dirs: list[Path] = []
    for shard in sorted(TRAIN_ROOT.iterdir()):
        if not shard.is_dir():
            continue
        for scene in shard.iterdir():
            if scene.is_dir():
                scene_dirs.append(scene)
    print(f"[scan] total scenes: {len(scene_dirs)}")

    bad: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scan_scene, sd): sd for sd in scene_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is not None:
                bad.append(res)
            if i % 500 == 0:
                print(f"[scan] {i}/{len(scene_dirs)} scanned, {len(bad)} bad so far")

    print(f"[scan] done: {len(bad)} non-consecutive scene(s)")
    fresh = [e for e in bad if e["scene"] not in existing]
    print(f"[blacklist] new (not already blacklisted): {len(fresh)}")
    if not fresh:
        return 0

    n_new = append_blacklist(BLACKLIST_PATH, fresh)
    print(f"[blacklist] appended {n_new} new row(s) to {BLACKLIST_PATH}")

    print("\n[sample] up to 10 newly blacklisted scenes:")
    for e in fresh[:10]:
        print(f"  {e['scene']}  | {e['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
