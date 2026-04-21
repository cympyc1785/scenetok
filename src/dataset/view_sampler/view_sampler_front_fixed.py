import torch

from dataclasses import dataclass
from typing import Literal

from ..dtypes import Stage
from .view_sampler import ViewIndex, ViewSampler, ViewSamplerCfg
from src.model.diffusion import latent_to_original_index, original_to_latent_index


@dataclass
class ViewSamplerFrontFixedCfg(ViewSamplerCfg):
    name: Literal["front_fixed"]
    min_context_views: int = 1
    num_target_split: int = 1
    chunk_index_gap: int = 1
    temporal_downsample: int = 4
    offset: int = 0


class ViewSamplerFrontFixed(ViewSampler[ViewSamplerFrontFixedCfg]):
    def latent_to_original_index(self, latent_idx: int) -> int:
        if latent_idx % self.cfg.chunk_index_gap == 0:
            return (
                self.cfg.temporal_downsample * latent_idx
                - (self.cfg.temporal_downsample - 1)
                * (latent_idx // self.cfg.chunk_index_gap)
            )
        return (
            self.cfg.temporal_downsample * latent_idx
            - (self.cfg.temporal_downsample - 1)
            * (latent_idx // self.cfg.chunk_index_gap + self.cfg.offset)
        )

    def sample(
        self,
        num_views: int,
        num_latents: int,
        stage: Stage,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[ViewIndex, torch.Tensor]:
        target_count = self.cfg.num_target_views
        if target_count <= 0:
            raise ValueError("front_fixed sampler requires num_target_views > 0")
        if self.cfg.offset == 0:
            target_frame_count = target_count * self.cfg.temporal_downsample
        else:
            target_frame_count = self.latent_to_original_index(target_count)

        if num_views < target_frame_count:
            raise ValueError(
                f"front_fixed sampler requires at least {target_frame_count} views, "
                f"but got {num_views}."
            )

        target_indices = torch.arange(
            0, target_frame_count, dtype=torch.int64, device=device
        )
        latent_indices = torch.arange(
            0, target_count, dtype=torch.int64, device=device
        )

        if self.cfg.num_context_views <= 0:
            context_indices = torch.empty(0, dtype=torch.int64, device=device)
        elif self.cfg.num_context_views == 1:
            context_indices = torch.zeros(1, dtype=torch.int64, device=device)
        else:
            context_indices = torch.linspace(
                0,
                target_frame_count - 1,
                steps=self.cfg.num_context_views,
                device=device,
            ).long()

        # print(self.cfg.num_context_views)
        # print(self.cfg.num_target_views)
        # print(ViewIndex(context_indices, target_indices), target_indices)
        # exit()
        return ViewIndex(context_indices, target_indices), latent_indices

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
