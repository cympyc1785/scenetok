"""LagerNVS-style view sampler (ExpandedLinearViewSelector 재현).

LagerNVS 원본(`submodules/lagernvs/data/view_selector.py`)의 conditioning/target
선택 로직을 SceneTok dataset 계약(ViewIndex = 프레임 인덱스)에 맞춰 재현:
  - context: per-gap delta_t ∈ [view_range_min, view_range_max]로 등간격 배치
    (num_frames//(num_cond-1)로 상한) + ±temp_jitter_prop 지터, 정렬 후 shuffle.
  - target: cond 범위를 ±expansion_factor·delta_t 만큼 **extrapolation**한 구간에서
    num_target개 샘플. target_has_input_p 확률로 target을 (cond+tgt) 전체에서 재추출
    (target에 cond 뷰 포함 가능).

⚠️ SceneTok 배칭(고정 shape)을 위해 num_cond = num_context_views로 **고정**
(LagerNVS의 per-sample 2-6 랜덤 aug는 배치 내 통일이 필요해 제외). 나머지(sparse
간격·extrapolation·shuffle)는 그대로. 두 번째 반환값(dataset에서 미사용)은 None.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .view_sampler import ViewSampler, ViewSamplerCfg, ViewIndex


@dataclass
class ViewSamplerExpandedLinearCfg(ViewSamplerCfg):
    name: Literal["expanded_linear"]
    view_range_min: int = 10           # per-gap delta_t 하한
    view_range_max: int = 20           # per-gap delta_t 상한
    expansion_factor: float = 1.0      # target extrapolation: ±expansion·delta_t
    temp_jitter_prop: float = 0.1      # cond 타임스탬프 지터 비율
    target_has_input_p: float = 1.0    # 이 확률로 target을 (cond+tgt)에서 재추출
    shuffle_context: bool = True       # train에서 context 순서 shuffle
    # 아래는 wrapper/dataset이 직접 접근하는 필드(이 sampler 로직은 raw 프레임을
    # 반환하므로 미사용, AttributeError 방지용 안전 기본값).
    temporal_downsample: int = 1
    offset: int = 0
    chunk_index_gap: int = 1
    num_target_split: int = 1
    chunk_targets: bool = False


class ViewSamplerExpandedLinear(ViewSampler[ViewSamplerExpandedLinearCfg]):

    def _rng(self, scene, stage):
        # train: 매번 랜덤(aug). val/test: scene별 결정론(모니터링 안정).
        if stage == "train":
            return np.random.default_rng()
        return np.random.default_rng(abs(hash(("explin", str(scene)))) % (2**31))

    def sample(self, num_views, scene=None, num_latents=None, stage=None,
               extrinsics=None, scene_dir=None, device: torch.device = torch.device("cpu"),
               **kwargs):
        stage = stage or self.stage
        cfg = self.cfg
        n = int(num_views)
        num_cond = cfg.num_context_views
        num_tgt = cfg.num_target_views
        rng = self._rng(scene, stage)
        vmin, vmax = cfg.view_range_min, cfg.view_range_max

        # ── delta_t & start (get_delta_t_and_start_idx) ──
        if num_cond <= 1:
            delta_t = min(int(rng.integers(vmin, vmax + 1)), n - 1)
            start = int(rng.integers(0, max(1, n - delta_t)))
        else:
            max_dt = min(n // (num_cond - 1), vmax)
            lo = min(max_dt, vmin)
            delta_t = int(rng.integers(lo, max_dt + 1)) if max_dt > lo else max_dt
            delta_t = max(delta_t, 1)
            max_start = max(0, n - delta_t * (num_cond - 1))
            start = int(rng.integers(0, max_start + 1))

        # ── cond timestamps (jitter → sort → shuffle) ──
        tj = int(delta_t * cfg.temp_jitter_prop)
        jitter = rng.integers(0, max(1, 2 * tj), size=num_cond) - tj
        cond = sorted(min(max(0, start + i * delta_t + int(jitter[i])), n - 1) for i in range(num_cond))
        cond = np.array(cond, dtype=np.int64)

        exp = int(delta_t * cfg.expansion_factor)
        sampling_end = min(n - 1, int(cond[-1]) + exp)
        sampling_start = max(0, start - exp)

        if cfg.shuffle_context and stage == "train":
            cond = rng.permutation(cond)

        # ── target: extrapolated range, exclude cond ──
        cond_set = set(cond.tolist())
        options = [t for t in range(sampling_start, max(sampling_start + 1, sampling_end)) if t not in cond_set]
        if len(options) == 0:
            options = list(range(sampling_start, max(sampling_start + 1, sampling_end + 1)))
        replace = num_tgt > len(options)
        tgt = rng.choice(np.array(options, dtype=np.int64), size=num_tgt, replace=replace)

        # target_has_input_p: target을 (cond+tgt) 전체에서 재추출(cond 포함 허용)
        all_frames = np.concatenate([cond, tgt])
        if rng.uniform() < cfg.target_has_input_p and len(all_frames) >= num_tgt:
            tgt = rng.choice(all_frames, size=num_tgt, replace=False)
        tgt = np.sort(tgt)   # 렌더 영상 시간 정렬

        ctx_index = torch.from_numpy(cond.astype(np.int64))
        tgt_index = torch.from_numpy(tgt.astype(np.int64))
        return ViewIndex(ctx_index, tgt_index), None

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
