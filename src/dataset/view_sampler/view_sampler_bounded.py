import torch
import numpy as np

from typing import Literal
from dataclasses import dataclass

from src.misc.camera_utils import fps_from_pose
from .view_sampler import ViewIndex, ViewSampler, ViewSamplerCfg
from ..dtypes import Stage

@dataclass
class ViewSamplerBoundedCfg(ViewSamplerCfg):
    name: Literal["bounded"]

    min_distance_between_context_views: int = 0
    max_distance_between_context_views: int | None = None
    max_distance_to_context_views: int = 0
    context_gap_warm_up_steps: int = 0
    target_gap_warm_up_steps: int = 0
    initial_min_distance_between_context_views: int = 0
    initial_max_distance_between_context_views: int | None = None
    initial_max_distance_to_context_views: int = 0
    # LagerNVS-style: per-example target extrapolation margin sampled U[min,max]
    # (frames), no warm-up. If set, overrides max_distance_to_context_views + warmup.
    target_extrap_range: list | None = None

    # LagerNVS-COUPLED extrapolation (ExpandedLinearViewSelector semantics): sample
    # per-adjacent context spacing delta_t ∈ [view_range_min, view_range_max]
    # (clamped to fit the clip), set context window = delta_t*(num_context-1), and
    # couple the target extrapolation margin = round(expansion_factor * delta_t).
    # Wan-compatible (target stays a contiguous clip). Overrides context-gap and
    # target-gap logic. Default off (existing behavior unchanged).
    lagernvs_extrap: bool = False
    view_range_min: int = 0
    view_range_max: int = 0
    expansion_factor: float = 1.0

    num_target_split: int=1
    chunk_index_gap: int=1
    target_split_prob: float=0.0
    temporal_downsample: int=1
    offset: int=0
    chunk_targets: bool=True
    override_context_gap: int | None = None
    # Randomly permute the (temporally-sorted) context view order at train time.
    # Matches LagerNVS's own pretraining, which shuffles conditioning views
    # (ExpandedLinearViewSelector: np.random.choice(cond, replace=False)); the
    # scene encoder is permutation-equivariant so this only affects ordering, and
    # (image, pose) stay paired since the whole index tensor is permuted together.
    # Default False = existing sorted (temporal) behavior. Target order untouched.
    shuffle_context: bool = False


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(
        self, 
        initial: int, 
        final: int,
        steps: int
    ) -> int:
        fraction = self.global_step / steps

        
        return min(initial + int((final - initial) * fraction), final)
    
    def latent_to_original_index(self, latent_idx):
        if not self.cfg.chunk_targets and self.cfg.offset != 0:
            # Wan without chunking
            if isinstance(latent_idx, int):
                if latent_idx == 0:
                    return 0
                return self.cfg.temporal_downsample * latent_idx - (self.cfg.temporal_downsample - 1)
            return torch.where(
                latent_idx == 0,
                torch.zeros_like(latent_idx),
                self.cfg.temporal_downsample * latent_idx - (self.cfg.temporal_downsample - 1),
            )

        if latent_idx % self.cfg.chunk_index_gap==0:
            idx = self.cfg.temporal_downsample  * latent_idx - (self.cfg.temporal_downsample-1)*(latent_idx//self.cfg.chunk_index_gap)
        else:
            idx = self.cfg.temporal_downsample  * latent_idx - (self.cfg.temporal_downsample-1)*(latent_idx//self.cfg.chunk_index_gap + self.cfg.offset)

        return idx

    def original_to_latent_index(self, idx):
        if not self.cfg.chunk_targets and self.cfg.offset != 0:
            # Wan without chunking
            if idx == 0:
                return 0
            return (idx - 1) // self.cfg.temporal_downsample + 1

        frame_index_gap = self.cfg.temporal_downsample * (self.cfg.chunk_index_gap-1) + 1
        k = idx // frame_index_gap
        r = idx % frame_index_gap
        return self.cfg.chunk_index_gap * k + (r + self.cfg.temporal_downsample-1) // self.cfg.temporal_downsample
    
    def sample(
        self,
        num_views: int,
        num_latents: int,
        stage: Stage,
        extrinsics: torch.Tensor,
        **kwargs
    ) -> list[ViewIndex]:
        offset = self.cfg.offset

        temporal_downsample = self.cfg.temporal_downsample
        if self.cfg.num_target_views > 0:
            if offset == 0:
                required_num_views = self.cfg.num_target_views * temporal_downsample
                available_num_latents = num_views // temporal_downsample
            else:
                required_num_views = self.latent_to_original_index(self.cfg.num_target_views)
                available_num_latents = num_latents

            if num_views < required_num_views:
                raise ValueError(
                    f"Example does not have enough frames for target views: "
                    f"num_views={num_views}, required_num_views={required_num_views}, "
                    f"num_target_views={self.cfg.num_target_views}, "
                    f"temporal_downsample={temporal_downsample}, "
                    f"chunk_index_gap={self.cfg.chunk_index_gap}, offset={offset}"
                )

            if available_num_latents < self.cfg.num_target_views:
                raise ValueError(
                    f"Example does not have enough latents for target views: "
                    f"num_latents={available_num_latents}, "
                    f"num_target_views={self.cfg.num_target_views}, "
                    f"num_views={num_views}, temporal_downsample={temporal_downsample}, "
                    f"chunk_index_gap={self.cfg.chunk_index_gap}, offset={offset}"
                )

        max_distance_between_context_views = \
            self.cfg.max_distance_between_context_views or num_views
        initial_max_distance_between_context_views = \
            self.cfg.initial_max_distance_between_context_views or num_views
        min_distance_between_context_views = \
            self.cfg.min_distance_between_context_views or (
                self.latent_to_original_index(self.cfg.num_target_views) - self.latent_to_original_index(1)
                if self.cfg.num_target_views > 0
                else 0
            )
        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_context_gap = max_distance_between_context_views
            min_context_gap = max_distance_between_context_views
        elif self.cfg.context_gap_warm_up_steps > 0:
            max_context_gap = self.schedule(
                initial_max_distance_between_context_views,
                max_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps
            )
        

            min_context_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps
            )
        else:
            max_context_gap = min(max_distance_between_context_views, num_views)
            min_context_gap = min(min_distance_between_context_views, num_views)

        if not self.cameras_are_circular:
            max_context_gap = min(max_context_gap, num_views - 1)
            min_context_gap = min(min_context_gap, num_views - 1)
        
        min_context_gap = max(min_context_gap, self.latent_to_original_index(self.cfg.num_target_views))

        # if not self.cameras_are_circular:
        #     max_context_gap = min(num_views - 1, max_context_gap)   # NOTE fixed former bug here

        # Compute the margin from context window to target window.
        # LagerNVS-style: sample the target extrapolation amount per example from
        # [min,max] (no warm-up) — mirrors ExpandedLinearViewSelector's delta_t·
        # expansion(1.0). `target_extrap_range` overrides the fixed+warmup path.
        if self.cfg.target_extrap_range is not None:
            lo, hi = int(self.cfg.target_extrap_range[0]), int(self.cfg.target_extrap_range[1])
            if self.stage == "test":
                max_target_gap = hi
            elif hi > lo:
                max_target_gap = torch.randint(lo, hi + 1, size=tuple()).item()
            else:
                max_target_gap = lo
        elif self.stage != "test" and self.cfg.target_gap_warm_up_steps > 0:
            max_target_gap = self.schedule(
                self.cfg.initial_max_distance_to_context_views,
                self.cfg.max_distance_to_context_views,
                self.cfg.target_gap_warm_up_steps
            )
        else:
            max_target_gap = self.cfg.max_distance_to_context_views
        # LagerNVS-COUPLED extrapolation: derive context window + target margin from a
        # per-adjacent delta_t (ExpandedLinearViewSelector). Target stays a contiguous
        # clip so it decodes to a video (wan-compatible).
        lagernvs_override = False
        if self.cfg.lagernvs_extrap:
            nc = self.cfg.num_context_views
            span_needed = self.latent_to_original_index(self.cfg.num_target_views) if self.cfg.num_target_views > 0 else 0
            max_dt_fit = (num_views - 1) // max(1, nc - 1)
            dt_hi = min(self.cfg.view_range_max, max_dt_fit)
            dt_lo = min(self.cfg.view_range_min, dt_hi)
            if self.stage == "test":
                delta_t = dt_hi
            elif dt_hi > dt_lo:
                delta_t = torch.randint(dt_lo, dt_hi + 1, size=tuple()).item()
            else:
                delta_t = dt_hi
            delta_t = max(int(delta_t), 1)
            # window must fit the contiguous target clip; clamp within the sequence.
            context_gap = min(max(delta_t * (nc - 1), span_needed), num_views - 1)
            max_target_gap = int(round(self.cfg.expansion_factor * delta_t))
            lagernvs_override = True

        # Pick the gap between the context views.
        if lagernvs_override:
            pass  # context_gap / max_target_gap already set above
        elif max_context_gap < min_context_gap:
            raise ValueError(f"Example does not have enough frames! {max_context_gap} <= f <= {min_context_gap}, and num views: {num_views}")
        elif max_context_gap == min_context_gap:
            context_gap = max_context_gap
        else:
            context_gap = torch.randint(
                min_context_gap,
                max_context_gap + 1,
                size=tuple()
            ).item()

        if self.cfg.override_context_gap:
            context_gap = self.cfg.override_context_gap

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            low=0,
            high=num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple()
        ).item()

        index_context_right = index_context_left + context_gap

        index_unrolled = None
        num_target_split = self.cfg.num_target_split if stage == "train" and self.cfg.chunk_targets else 1
        # Compute target indices
        if self.cfg.num_target_views > 0:
            index_target_left = index_context_left - max_target_gap
            index_target_right = index_context_right + max_target_gap

            if not self.cameras_are_circular:
                index_target_left = max(0, index_target_left)
                index_target_right = min(num_views-1, index_target_right)
            

            # Pick the target view indices.
            num_target_views = self.cfg.num_target_views
            chunk_index_gap = self.cfg.chunk_index_gap if self.cfg.chunk_targets else 1

            if offset == 0:
                num_latents = num_views // temporal_downsample
                start = ((index_target_left // temporal_downsample) // chunk_index_gap) * chunk_index_gap
                end = ((index_target_right // temporal_downsample) // chunk_index_gap) * chunk_index_gap
            else:
                start = self.original_to_latent_index(index_target_left)
                end = self.original_to_latent_index(index_target_right)
                start = (start // chunk_index_gap) * chunk_index_gap
                end = (end // chunk_index_gap) * chunk_index_gap
            try:
                starting_indices = torch.arange(start, end - num_target_views + 2, chunk_index_gap)
            except Exception as err:
                raise ValueError(
                    f"Error in generating starting indices: start={start}, end={end}, to={end - num_target_views + 2}\n"
                    f"num_target_views={num_target_views}, chunk_index_gap={chunk_index_gap}, context_gap={context_gap}\n"
                    f"num_views={num_views}, num_latents={num_latents}\n"
                    f"extrinsics={extrinsics.shape}"
                ) from err
            if len(starting_indices) == 0:
                raise ValueError(
                    f"No valid target start indices: start={start}, end={end},\n"
                    f"num_target_views={num_target_views}, chunk_index_gap={chunk_index_gap},\n"
                    f"num_views={num_views}, num_latents={num_latents}\n, context_gap={context_gap}"
                    f"extrinsics={extrinsics.shape}"
                )
            num_target_split = min(len(starting_indices), num_target_split)
            index_target = torch.arange(0, num_latents).long()
            if np.random.choice([True, False], size=1, p=[1 - self.cfg.target_split_prob, self.cfg.target_split_prob]):
                num_target_split = 1
            
            idxs = torch.multinomial(torch.ones_like(starting_indices).float(), num_target_split, replacement=False)
            index_targets = []
            index_unrolled = []
            for idx in idxs:
                starting_index = starting_indices[idx]
                target = index_target[starting_index:starting_index + num_target_views // num_target_split]
                index_targets.append(target)

                if offset == 0:
                    idx_unrolled = torch.arange(target[0]*temporal_downsample, target[-1]*temporal_downsample+temporal_downsample)
                else:
                    try:

                        start = self.latent_to_original_index(target[0])
                    except IndexError as err:
                        
                        print("Error: ", err)
                        return self.sample(num_views=num_views, num_latents=num_latents)


                    try:

                        end = self.latent_to_original_index(target[-1]+1)
                    except IndexError as err:

                        print("Error: ", err)
                        return self.sample(num_views=num_views, num_latents=num_latents)

                    if not self.cfg.chunk_targets:
                        end = start + 1 + (target.shape[0] - 1) * temporal_downsample

                    idx_unrolled = torch.arange(start, end)
                
                index_unrolled.append(idx_unrolled)
            index_target = torch.concat(index_targets)
            index_unrolled = torch.concat(index_unrolled)

        else:
            index_target = None

        indices = []
        if self.cfg.num_context_views > 2:
            if self.cfg.context_sampling == "uniform":
                context_indices = torch.linspace(index_context_left, index_context_right, steps=self.cfg.num_context_views).long()
                indices = context_indices[1:-1].tolist()
                index_context_left = context_indices[0].item()
                index_context_right = context_indices[-1].item()

            elif self.cfg.context_sampling == "farthest_point":
                context_indices = torch.arange(0, extrinsics.shape[0]).long()
                fps_indices = fps_from_pose(extrinsics[index_context_left:index_context_right+1].float(), n_samples=self.cfg.num_context_views).tolist()
                indices = context_indices[index_context_left:index_context_right+1][fps_indices[1:-1]].tolist()

            else:
                raise ValueError(f"Unknown context sampling strategy: {self.cfg.context_sampling}")
        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            if index_target is not None:
                index_target %= num_views
            index_context_right %= extrinsics.shape[0]
        context_index = torch.tensor(sorted([index_context_left, *indices, index_context_right]))
        if self.cfg.shuffle_context and self.stage == "train":
            # Permute context order (train only). Keeps target ordering intact.
            context_index = context_index[torch.randperm(context_index.numel())]
        return ViewIndex(context_index, index_unrolled), index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
