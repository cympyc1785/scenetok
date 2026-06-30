"""LagerNVS reconstructor as a frozen scene encoder for the SceneTok pipeline.

Stage A (dense-direct): the LagerNVS reconstructor (VGGT + geo_feature_connector) is
kept frozen (paper recipe) and its dense `rec_tokens` (b, v_input*p, 768) are fed
straight to the lightningdit denoiser's scene cross-attention — no compression. The
denoiser's `cnd_proj` (cond_dim=768 -> inner) handles the channel projection, and the
cross-attention (flash_attn) handles the large KV count.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import einops
from torch import nn, Tensor
from jaxtyping import Float

from .compressor import Compressor
from ..types import CompressorInputs

# LagerNVS lives in a sibling repo (symlinked at submodules/lagernvs). Add to path so
# `models.encoder_decoder` imports. xformers is optional there (SDPA fallback patched).
_LAGERNVS_ROOT = "/data1/cympyc1785/lagernvs"
if _LAGERNVS_ROOT not in sys.path:
    sys.path.insert(0, _LAGERNVS_ROOT)


@dataclass
class LagerNVSCompressorCfg:
    name: Literal["lagernvs_compressor"]
    ckpt_path: str = "/data1/cympyc1785/lagernvs/checkpoints/lagernvs_general_512/model.pt"
    token_dim: int = 768                 # rec_tokens channel == denoiser cond_dim
    num_scene_tokens: int = 12432        # nominal (v_input*p); cross-attn handles actual count
    scene_token_projection: Literal["simple"] = "simple"   # deterministic features, no KL
    img_norm: Literal["zero_one", "neg_one_one"] = "zero_one"  # input view value range
    load_strict: bool = False


def _freeze(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()


class LagerNVSCompressor(Compressor[LagerNVSCompressorCfg]):

    def __init__(
        self,
        cfg: LagerNVSCompressorCfg,
        in_channels: int = 3,
        num_views: int = 8,
        temporal_downsample: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(cfg)
        from models.encoder_decoder import EncDec_VitB8

        model = EncDec_VitB8(pretrained_vggt=False)
        sd = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(sd["model"], strict=cfg.load_strict)
        # Keep only the reconstructor (VGGT + geo_feature_connector); drop the renderer.
        self.reconstructor = model.reconstructor
        _freeze(self.reconstructor)
        print(f"(LagerNVSCompressor) reconstructor loaded & frozen from {cfg.ckpt_path}")

    @property
    def num_scene_tokens(self) -> int:
        return self.cfg.num_scene_tokens

    @property
    def output_dim(self) -> int:
        return self.cfg.token_dim

    def forward(self, inputs: CompressorInputs):
        # Override the base `@torch.compile` forward: the frozen VGGT reconstructor
        # uses data-dependent F.interpolate (longer side -> 518) which fullgraph
        # compile cannot trace. Dispatch straight to `_forward`.
        return self._forward(inputs=inputs)

    def load_weights(self, path: Path | str, **kwargs):
        # Weights are loaded in __init__ from cfg.ckpt_path; nothing to resume here.
        return self

    def _build_cam_token(self, pose) -> Tensor:
        """Scale-only camera token (b, v, 11) — mirrors viser/lagernvs_infer.

        general_512 is effectively unposed (conditioning rays ignored); we only feed the
        normalized scene scale in slot 9.
        """
        ext = pose.extrinsics.float()                       # (b, v, 4, 4) c2w
        b, v = ext.shape[:2]
        first_inv = torch.linalg.inv(ext[:, 0:1])           # (b,1,4,4)
        ext = first_inv @ ext                               # ctx0-relative
        t = ext[..., :3, 3]                                 # (b, v, 3)
        scene_scale = 1.35 * t.norm(dim=-1).amax(dim=1, keepdim=True).clamp(min=1e-6)  # (b,1)
        t = t / scene_scale.unsqueeze(-1)
        camera_scale = t.norm(dim=-1).amax(dim=1)           # (b,)
        cam = torch.zeros(b, v, 11, device=ext.device)
        cam[:, :, 9] = camera_scale.unsqueeze(1)
        return cam

    def _forward(
        self,
        inputs: CompressorInputs,
        latent_input: bool = False,
        return_qk: bool = False,
    ) -> Float[Tensor, "batch num dim"]:
        imgs = inputs.view                                  # (b, v, 3, H, W) RAW RGB
        if self.cfg.img_norm == "neg_one_one":
            imgs = (imgs + 1.0) * 0.5                        # [-1,1] -> [0,1] for VGGT
        cam_token = self._build_cam_token(inputs.pose).to(imgs.dtype)
        with torch.no_grad():
            rec = self.reconstructor(imgs, cam_token)        # (b, v, p, 768)
        rec = einops.rearrange(rec, "b v p c -> b (v p) c")  # dense tokens
        # Match the (tokens, qk) tuple contract that callers unpack via `tokens, *_`.
        return rec, None
