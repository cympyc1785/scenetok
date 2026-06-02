"""Caption-aligned view sampler.

Picks a random caption window from `<scene>/prompts_<N>.json` and uses it as
the target window. Context views are sampled from inside (or just around)
that same window so the prompt actually matches the rendered segment.

Drop-in replacement for `bounded` when training/eval with caption supervision.
Same `sample(...) → (ViewIndex, raw_target)` return shape.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.misc.camera_utils import fps_from_pose

from ..dtypes import Stage
from .view_sampler import ViewIndex, ViewSampler, ViewSamplerCfg


@dataclass
class ViewSamplerCaptionWindowCfg(ViewSamplerCfg):
    name: Literal["caption_window"]
    prompts_filename: str = "prompts_37.json"
    temporal_downsample: int = 4
    offset: int = 1
    chunk_targets: bool = False
    # Frames added on either side of the chosen caption window when picking
    # context views (0 → context only from within the target window itself).
    # This is the *final* margin; see warm-up fields below for ramping it up.
    context_window_margin: int = 0
    # bounded-style warm-up for the margin: over the first
    # `context_window_margin_warm_up_steps` global steps the effective margin
    # ramps linearly from `initial_context_window_margin` → `context_window_margin`.
    # 0 steps → no schedule (always use `context_window_margin`). At test stage the
    # full (final) margin is always used.
    initial_context_window_margin: int = 0
    context_window_margin_warm_up_steps: int = 0
    # If the picked window is shorter than V_pixel required by the V_lat/temporal
    # config, fall back to using all `window_len` frames (last chunk in
    # `prompts_37.json` is usually short).
    allow_short_last_window: bool = True


class ViewSamplerCaptionWindow(ViewSampler[ViewSamplerCaptionWindowCfg]):
    def _schedule(self, initial: int, final: int, steps: int) -> int:
        # Linear ramp initial → final over `steps` global steps (bounded-style).
        fraction = self.global_step / steps
        return min(initial + int((final - initial) * fraction), final)

    def _effective_margin(self) -> int:
        final = self.cfg.context_window_margin
        steps = self.cfg.context_window_margin_warm_up_steps
        if self.stage == "test" or steps <= 0:
            return final
        return self._schedule(self.cfg.initial_context_window_margin, final, steps)

    def _target_pixel_count(self) -> int:
        n_lat = self.cfg.num_target_views
        td = max(self.cfg.temporal_downsample, 1)
        if self.cfg.chunk_targets:
            # Caller's responsibility; default to 1 + (n - 1) * td which works
            # for most Wan setups too.
            return 1 + (n_lat - 1) * td
        if self.cfg.offset != 0:
            return 1 + (n_lat - 1) * td
        return n_lat * td

    def _pick_window(self, prompts_dict: dict, num_views: int) -> tuple[int, int]:
        windows: list[tuple[int, int]] = []
        for entry in prompts_dict.values():
            fi = entry.get("frame_idx") if isinstance(entry, dict) else None
            if not (isinstance(fi, (list, tuple)) and len(fi) == 2):
                continue
            lo, hi = int(fi[0]), int(fi[1])
            if not (0 <= lo < hi <= num_views):
                continue
            windows.append((lo, hi))
        if not windows:
            raise ValueError("no usable caption windows fit inside num_views")
        V_pixel = self._target_pixel_count()
        # Prefer full-length windows; allow short last bucket only if configured.
        full = [w for w in windows if (w[1] - w[0]) >= V_pixel]
        if full:
            return random.choice(full)
        if self.cfg.allow_short_last_window:
            return random.choice(windows)
        raise ValueError(
            f"no caption window large enough for V_pixel={V_pixel}; "
            f"windows={windows} (set allow_short_last_window=true to accept short ones)"
        )

    def _sample_context(
        self,
        c_lo: int,
        c_hi: int,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        n_ctx = self.cfg.num_context_views
        span = c_hi - c_lo
        if span <= n_ctx:
            return torch.arange(c_lo, c_lo + span, dtype=torch.long)
        mode = getattr(self.cfg, "context_sampling", "uniform")
        if mode == "uniform":
            return torch.linspace(c_lo, c_hi - 1, steps=n_ctx).long()
        if mode == "farthest_point":
            sub = extrinsics[c_lo:c_hi].float()
            fps_idx = fps_from_pose(sub, n_samples=n_ctx).tolist()
            return torch.tensor([c_lo + i for i in fps_idx], dtype=torch.long)
        # fallback
        return torch.linspace(c_lo, c_hi - 1, steps=n_ctx).long()

    def sample(
        self,
        num_views: int,
        num_latents: int,
        stage: Stage,
        extrinsics: torch.Tensor,
        scene: str | None = None,
        scene_dir: Path | str | None = None,
        **kwargs,
    ) -> tuple[ViewIndex, torch.Tensor]:
        if scene_dir is None:
            raise ValueError(
                "ViewSamplerCaptionWindow requires `scene_dir` kwarg from the dataset "
                "(point to the scene's filesystem directory)."
            )
        prompts_path = Path(scene_dir) / self.cfg.prompts_filename
        if not prompts_path.exists():
            raise ValueError(f"prompts file missing: {prompts_path}")
        with prompts_path.open() as f:
            prompts_dict = json.load(f)

        lo, hi = self._pick_window(prompts_dict, num_views=num_views)

        V_pixel = min(self._target_pixel_count(), hi - lo)
        target_start = lo
        target_end = target_start + V_pixel  # half-open
        target_indices = torch.arange(target_start, target_end, dtype=torch.long)

        margin = self._effective_margin()
        c_lo = max(0, lo - margin)
        c_hi = min(num_views, hi + margin)
        context_indices = self._sample_context(c_lo, c_hi, extrinsics)
        context_indices = context_indices.sort().values

        view_index = ViewIndex(
            context=context_indices.contiguous(),
            target=target_indices.contiguous(),
        )
        return view_index, target_indices

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
