from dataclasses import dataclass, field

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    shape: list[int]
    view_sampler: ViewSamplerCfg
    augment: bool
    precomputed_latents: dict[str, bool]
    cameras_are_circular: bool
    fps: int = field(default=24, kw_only=True)
