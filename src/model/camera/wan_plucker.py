
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import torch
from torch import Tensor
from jaxtyping import Float

from .camera import Camera
from ..types import CameraInputs
from ...misc.camera_utils import generate_image_rays


@dataclass
class WanPluckerCfg:
    name: Literal["wan_plucker"]
    input_shape: Union[list[int], tuple[int, int]]
    scale: Union[list[float], tuple[float, float]] = field(default_factory=lambda: [1.0, 1.0])
    normalize: bool = True
    chunk_size: int = 17  # 1 + 4·4 video-frame chunk, matches plucker.py behavior


class WanPlucker(Camera[WanPluckerCfg]):
    """Compute Wan-aligned Plucker embeddings from real (K, T).

    Produces the raw `control_camera_latents_input` tensor that FunCameraControl's
    `dit.control_adapter` expects: shape `[B, 6·td, F_lat, H, W]`, channel order
    `[moment, direction]`, with the first frame repeated `td` times before being
    folded into 4-frame channel groups (so 1 + 4·N video frames → 1 + N latent
    frames with `c·td` channels per group).
    """

    cfg: WanPluckerCfg

    def __init__(
        self,
        cfg: WanPluckerCfg,
        temporal_downsample: int = 4,
        **_: object,
    ) -> None:
        super().__init__(cfg)
        self.temporal_downsample = temporal_downsample

    def load_weights(self, path: Path | str, **kwargs) -> None:
        return

    def _plucker(self, inputs: CameraInputs) -> Float[Tensor, "b v 6 h w"]:
        intrinsics = inputs.intrinsics.clone()
        extrinsics = inputs.extrinsics
        intrinsics[..., 0, 0] *= self.cfg.scale[0]
        intrinsics[..., 1, 1] *= self.cfg.scale[1]
        _, origins, directions = generate_image_rays(
            tuple(self.cfg.input_shape),
            extrinsics.float(),
            intrinsics.float(),
            self.cfg.normalize,
        )
        moments = torch.cross(origins, directions, dim=2)
        return torch.cat([moments, directions], dim=2)

    def _pack_chunk(self, plucker: Float[Tensor, "b v 6 h w"]) -> Float[Tensor, "b c v_lat h w"]:
        td = self.temporal_downsample
        first = plucker[:, 0:1]
        rest = plucker[:, 1:]
        first_repeated = first.repeat_interleave(td, dim=1)
        padded = torch.cat([first_repeated, rest], dim=1)
        b, f, c, h, w = padded.shape
        if f % td != 0:
            raise ValueError(
                f"Padded plucker frame count {f} is not divisible by temporal_downsample {td}. "
                f"Expected an input view count of the form 1 + {td}·N."
            )
        packed = padded.view(b, f // td, td, c, h, w).transpose(2, 3).contiguous()
        packed = packed.view(b, f // td, c * td, h, w).transpose(1, 2).contiguous()
        return packed

    def forward(
        self,
        inputs: CameraInputs,
        temporal_downsample: int = 1,
        chunk_targets: bool = True,
    ) -> Float[Tensor, "b c v_lat h w"]:
        orig_dtype = inputs.extrinsics.dtype
        plucker = self._plucker(inputs)

        if (
            chunk_targets
            and plucker.shape[1] % self.cfg.chunk_size == 0
            and plucker.shape[1] > self.cfg.chunk_size
        ):
            n = plucker.shape[1] // self.cfg.chunk_size
            chunks = torch.chunk(plucker, n, dim=1)
        else:
            chunks = (plucker,)

        packed = torch.cat([self._pack_chunk(chunk) for chunk in chunks], dim=2)
        return packed.to(orig_dtype)
