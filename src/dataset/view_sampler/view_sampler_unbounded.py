from dataclasses import dataclass
from typing import Literal

import torch
import math
import numpy as np

from src.misc.camera_utils import fps_from_pose
from .view_sampler import ViewIndex, ViewSampler, ViewSamplerCfg

@dataclass
class ViewSamplerUnboundedCfg(ViewSamplerCfg):
    name: Literal["unbounded"]

    temporal_downsample: int=4
    temporal_tile_size: int=16
    chunk_index_gap: int=4
    offset: int=0
    chunk_targets: bool=False
    min_context_views: int=1
    num_target_split: int=1
    target_split_prob: float=0.5
    sample_cond_views: bool=False
    max_cond_number: int=1
class ViewSamplerUnbounded(ViewSampler[ViewSamplerUnboundedCfg]):
    
    def schedule(
        self, 
        initial: int, 
        final: int,
        steps: int
    ) -> int:
        fraction = self.global_step / steps
        return min(initial + int((final - initial) * fraction), final)

    def latent_to_original_index(self, latent_idx):
        # Maps a latent index to its starting raw-frame index under the Wan VAE 4N+1
        # convention: latent 0 is the temporal prefix frame (raw 0), and each subsequent
        # latent advances temporal_downsample raw frames. This mirrors the Wan branch of
        # ViewSamplerBounded.latent_to_original_index. Wrapper-side chunking (chunk_targets)
        # affects only the downstream raw->latent math, not this per-chunk layout.
        td = self.cfg.temporal_downsample
        if isinstance(latent_idx, int):
            if latent_idx == 0:
                return 0
            return td * latent_idx - (td - 1)
        return torch.where(
            latent_idx == 0,
            torch.zeros_like(latent_idx),
            td * latent_idx - (td - 1),
        )

    def sample(
        self,
        num_views: int,
        num_latents: int,
        stage: str,
        extrinsics: torch.Tensor,
        **kwargs
    ) -> tuple[ViewIndex, torch.Tensor]:
        nsamples = max(self.num_context_views, self.num_target_views * self.cfg.temporal_downsample)
        if num_latents < self.num_target_views:
            raise ValueError(f"Example has less number of frames --> {num_views} < {nsamples} and {num_latents} < {self.num_target_views}!")

        if stage == "train":
            chunk_index_gap = self.cfg.chunk_index_gap
        else:
            chunk_index_gap = 8


        num_target_views = self.num_target_views
        num_context_views = self.num_context_views
        num_target_split = self.cfg.num_target_split if stage == "train" else 1
        td = self.cfg.temporal_downsample

        # In Wan mode (offset != 0) each chunk uses 4N-3 raw frames for N latents (the 4N+1
        # convention with a single temporal prefix frame). chunk_targets only governs the
        # downstream wrapper's raw->latent math (see diffusion_wrapper.py:490-495), not this
        # per-chunk layout. In the default (offset == 0) mode we keep the legacy 4N-frame
        # window for backward compat.
        wan_mode = self.cfg.offset != 0
        if wan_mode:
            num_latents_available = 1 + (num_views - 1) // td
        else:
            num_latents_available = num_latents
        if num_latents_available < num_target_views:
            raise ValueError(
                f"Example has less latents than target views: "
                f"num_latents_available={num_latents_available}, "
                f"num_target_views={num_target_views}, num_views={num_views}, "
                f"temporal_downsample={td}, offset={self.cfg.offset}, "
                f"chunk_targets={self.cfg.chunk_targets}"
            )
        index_target = torch.arange(0, num_latents_available).long()

        starting_indices = torch.arange(0, num_latents_available - num_target_views + 1, chunk_index_gap)

        if len(starting_indices) == 0:
            raise ValueError(
                f"No valid starting index produces in-bounds unrolled target: "
                f"num_views={num_views}, num_latents_available={num_latents_available}, "
                f"num_target_views={num_target_views}, td={td}, wan_mode={wan_mode}, "
                f"chunk_index_gap={chunk_index_gap}"
            )

        num_target_split = min(len(starting_indices), num_target_split)

        if np.random.choice([True, False], size=1, p=[1 - self.cfg.target_split_prob, self.cfg.target_split_prob]):
            num_target_split = 1

        idxs = torch.multinomial(torch.ones_like(starting_indices).float(), num_target_split, replacement=False)
        index_targets = []
        index_unrolled = []
        for idx in idxs:
            starting_index = starting_indices[idx]
            target = index_target[starting_index:starting_index + num_target_views // num_target_split]
            index_targets.append(target)
            if wan_mode:
                start = self.latent_to_original_index(int(target[0].item()))
                end = start + 1 + (target.shape[0] - 1) * td  # 4N-3 raw frames per N latents
                index_unrolled.append(torch.arange(start, end))
            else:
                index_unrolled.append(torch.arange(target[0]*td, target[-1]*td + td))
        index_targets = torch.concat(index_targets)
        index_unrolled = torch.concat(index_unrolled)
        if self.cfg.context_sampling == "uniform":
            context_indices = torch.linspace(0, extrinsics.shape[0] - 1, steps=num_context_views).long()

        elif self.cfg.context_sampling == "farthest_point":
            context_indices = fps_from_pose(extrinsics.float(), num_context_views)

        else:
            raise ValueError(f"Unknown context sampling strategy: {self.cfg.context_sampling}")
        

        if self.cfg.sample_cond_views:

            ref_idx = torch.randint(0, len(context_indices), size=(1, ))
            weights = torch.ones((extrinsics.shape[0], )).float()
            weights[context_indices[ref_idx]] = 0.0 # Ensure we don't sample the reference view again
            cond_indices = torch.multinomial(weights, self.cfg.max_cond_number - 1, replacement=False)
            cond_indices = torch.concat([context_indices[ref_idx], cond_indices])

        else:
            cond_indices = None

        # Sanity: refuse to emit indices that would index past the image / extrinsics arrays.
        # In wan_mode this is enforced by num_latents_available = 1 + (num_views-1)//td upfront,
        # but in offset=0 mode num_latents is taken from the dataset and could in principle
        # produce index_unrolled.max() >= num_views. Also context_indices is computed from
        # extrinsics.shape[0] which is assumed equal to num_views; this check catches mismatches.
        if int(index_unrolled.max()) >= num_views or int(context_indices.max()) >= extrinsics.shape[0]:
            raise ValueError(
                f"Unbounded sampler produced OOB indices: "
                f"index_unrolled.max()={int(index_unrolled.max())}, num_views={num_views}, "
                f"context_indices.max()={int(context_indices.max())}, "
                f"extrinsics.shape[0]={extrinsics.shape[0]}, "
                f"wan_mode={wan_mode}, num_latents_available={num_latents_available}, "
                f"num_target_views={num_target_views}, num_target_split={num_target_split}"
            )

        return ViewIndex(context_indices, index_unrolled, cond=cond_indices), index_targets
        
 

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views

    @property
    def min_context_views(self) -> int:
        return self.cfg.min_context_views
