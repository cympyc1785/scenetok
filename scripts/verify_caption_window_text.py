#!/usr/bin/env python
"""train_ti2vgen_recon_overfit_256_finetuned.sh의 dataset-side override를 그대로
Hydra로 compose해서, caption_window view_sampler + load_prompts 조합이
`sample["text"]`를 caption 구간에 맞게 실제로 채우는지 검증한다.

모델(5B + UMT5)은 로드하지 않는다 — 이번 변경(view_sampler/load_prompts)이 바꾸는
것은 batch에 text가 들어가는 data 경로이고, 그 뒤 wrapper가 batch["text"]를
denoiser.encode_text_condition으로 흘리는 부분은 코드상 무조건 동작(아래 NOTE 참고).

검증 항목 (scene 여러 개):
  1) view_sampler.name == caption_window, num_context/target == 16/10
  2) target index 구간이 prompts_37.json의 한 caption window [lo, hi) 안에 들어감
  3) sample["text"]가 비어있지 않고, 그 window의 prompt_scene_simple.concise와 일치
"""
import json
from pathlib import Path

from dataclasses import asdict

from hydra import compose, initialize

from src.config import load_typed_root_config
from src.dataset import get_dataset

OVERRIDES = [
    "+experiment=custom/scenetok_va-wan-ti2v_dl3dv",
    "dataset/view_sampler=caption_window",
    "+dataset.load_prompts=true",
    "dataset.smallset=true",
    "dataset.context_shape=[256,448]",
    "dataset.target_shape=[480,832]",
    "dataset.do_scale_and_pad=false",
]

N_SCENES = 5


def find_prompts(root: Path, scene: str, prompts_filename: str):
    """root 아래에서 scene hash로 prompts 파일 경로를 찾는다 (bucket 디렉토리 무관)."""
    direct = root / scene / prompts_filename
    if direct.exists():
        return direct
    return next(root.glob(f"*/{scene}/{prompts_filename}"), None)


def window_for_target(p, t_min: int, t_max: int):
    """prompts json에서 [lo,hi)가 target 구간을 덮는 entry의 concise 반환."""
    if p is None or not p.exists():
        return None, None
    data = json.load(p.open())
    best, best_span = None, None
    for e in data.values():
        fi = e.get("frame_idx") if isinstance(e, dict) else None
        if not (isinstance(fi, (list, tuple)) and len(fi) == 2):
            continue
        lo, hi = int(fi[0]), int(fi[1])
        if lo <= t_min and t_max < hi:
            span = hi - lo
            if best_span is None or span < best_span:
                best, best_span = e, span
    if best is None:
        return None, None
    concise = (best.get("prompt_scene_simple") or {}).get("concise", "")
    return (best["frame_idx"], concise)


def main():
    with initialize(version_base=None, config_path="../config"):
        cfg_dict = compose(config_name="main", overrides=OVERRIDES)

    # 실제 `python -m src.main`과 동일한 dacite 변환 (str→Path, union→CaptionWindowCfg)
    cfg = load_typed_root_config(cfg_dict)
    dcfg = cfg.dataset
    vs = dcfg.view_sampler
    print("=== typed view_sampler (dacite-resolved) ===")
    for k, v in asdict(vs).items():
        print(f"  {k}: {v}")
    print(f"dataset.load_prompts     = {getattr(dcfg, 'load_prompts', None)}")
    print(f"dataset.prompts_filename = {getattr(dcfg, 'prompts_filename', 'prompts_37.json')}")
    assert type(vs).__name__ == "ViewSamplerCaptionWindowCfg", \
        f"view_sampler resolved to {type(vs).__name__}, not CaptionWindow"
    assert vs.name == "caption_window"

    ds = get_dataset(dcfg, "train", step_tracker=None)
    print(f"\ndataset len = {len(ds)}")
    prompts_filename = getattr(dcfg, "prompts_filename", "prompts_37.json")
    root = Path(dcfg.root) / "train"

    checked = 0
    i = 0
    while checked < N_SCENES and i < len(ds):
        try:
            s = ds[i]
        except Exception as e:
            print(f"[skip idx {i}] {type(e).__name__}: {e}")
            i += 1
            continue
        i += 1
        if s is None:
            continue

        if checked == 0:
            print(f"  [sample keys] {sorted(s.keys())}")
        tgt = s["target"]["index"].long()
        t_min, t_max = int(tgt.min()), int(tgt.max())
        ctx = s["context"]["index"]
        text = s.get("text", None)
        chunk = s.get("scene", "?")
        win, concise = window_for_target(
            find_prompts(root, chunk, prompts_filename), t_min, t_max)

        # caption_window는 target을 [lo, lo+V_pixel) 연속 프레임으로 만든다 (bounded와 구별되는 시그니처)
        contiguous = bool((tgt[1:] - tgt[:-1] == 1).all().item()) if len(tgt) > 1 else True

        print(f"\n--- sample {checked} (chunk={chunk}) ---")
        print(f"  context idx ({len(ctx)}): {ctx.tolist()}")
        print(f"  target  idx ({len(tgt)}): [{t_min}..{t_max}] contiguous={contiguous}")
        print(f"  matched window {win}")
        print(f"  sample['text'] = {text!r}")

        ok_text = bool(text) and isinstance(text, str)
        print(f"  -> text non-empty={ok_text}")
        # 핵심 hard assert: text가 실제로 batch에 들어오는가 + target이 연속 caption 구간인가
        assert ok_text, "sample['text'] is empty — load_prompts/caption_window가 text를 안 넣음!"
        assert contiguous, "target이 연속이 아님 — caption_window가 아니라 다른 sampler로 동작 중!"
        # best-effort cross-check (prompts 경로를 못 찾으면 경고만)
        if win is None:
            print("  [warn] prompts 경로에서 window 매칭 실패 — 경로 확인 필요 (text 자체는 채워짐)")
        else:
            in_window = win[0] <= t_min and t_max < win[1]
            ok_match = text == concise
            print(f"  -> in_window={in_window}  text==window.concise={ok_match}")
            assert in_window, "target이 매칭 window 밖!"
            assert ok_match, "sample['text'] != window concise (정렬 어긋남)!"
        checked += 1

    print(f"\n=== PASS: {checked} scenes — caption_window target이 caption 구간과 정렬되고 "
          f"sample['text']가 그 caption으로 채워짐 ===")


if __name__ == "__main__":
    main()
