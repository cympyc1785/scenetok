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
import torch.nn.functional as F
from torch import nn, Tensor
from jaxtyping import Float

from .compressor import Compressor
from ..types import CompressorInputs


class PerceiverResampler(nn.Module):
    """Resample a variable-length set of features to `num_latents` fixed tokens.

    Learnable latent queries cross-attend to the (frozen) dense LagerNVS rec_tokens,
    interleaved with latent self-attention and an MLP per layer (Perceiver-IO /
    Flamingo style). Only the latents self-attend — the ~12k rec_tokens are used as
    keys/values only (no O(N²) self-attention), so this is cheap despite the large
    input. Output: (b, num_latents, dim) fixed-length scene tokens.
    """

    def __init__(self, dim: int, num_latents: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim), nn.LayerNorm(dim),  # cross: q-norm, kv-norm
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                nn.LayerNorm(dim),  # self-attn norm
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                nn.LayerNorm(dim),  # ffn norm
                nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                              nn.Linear(int(dim * mlp_ratio), dim)),
            ]))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, feats: Tensor) -> Tensor:  # feats: (b, N, dim)
        b = feats.shape[0]
        q = self.latents.expand(b, -1, -1)
        for qn, kn, cross, sn, selfa, fn, ff in self.layers:
            kv = kn(feats)
            q = q + cross(qn(q), kv, kv, need_weights=False)[0]
            qs = sn(q)
            q = q + selfa(qs, qs, qs, need_weights=False)[0]
            q = q + ff(fn(q))
        return self.out_norm(q)

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
    num_scene_tokens: int = 12432        # dense: nominal (v_input*p). perceiver: fixed count.
    scene_token_projection: Literal["simple"] = "simple"   # deterministic features, no KL
    img_norm: Literal["zero_one", "neg_one_one"] = "zero_one"  # input view value range
    load_strict: bool = False
    # "none" = dense-direct (feed all rec_tokens to the denoiser cross-attn).
    # "perceiver" = SceneTok-style: resample dense rec_tokens to `num_scene_tokens`
    #   fixed-length scene tokens via a trainable Perceiver (frozen encoder stays frozen).
    token_reduction: Literal["none", "perceiver"] = "none"
    perceiver_depth: int = 6
    perceiver_heads: int = 12
    # Trainable LayerNorm on the output rec_tokens (before the denoiser cnd_proj).
    # Re-scales/shifts the frozen VGGT features toward the distribution the
    # pretrained scene cross-attn expects → faster adaptation. Needs
    # freeze.compressor=false so this norm (only) trains (reconstructor stays frozen).
    output_norm: bool = False
    # #6: keep VGGT (+ camera_mlp) frozen but UNFREEZE the thin geo_feature_connector
    # (Linear 2048->768) + geo_feature_norm adapter, so LagerNVS features can align to
    # the denoiser. Needs freeze.compressor=false. Slightly deviates from the paper's
    # "encoder fully frozen" recipe (only the connector adapter trains, VGGT stays frozen).
    unfreeze_geo_connector: bool = False


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
        if cfg.unfreeze_geo_connector:
            # Freeze VGGT + camera_mlp (encoder); keep geo_feature_connector +
            # geo_feature_norm adapter trainable (#6).
            _freeze(self.reconstructor.vggt)
            _freeze(self.reconstructor.camera_mlp)
            for m in (self.reconstructor.geo_feature_connector, self.reconstructor.geo_feature_norm):
                for p in m.parameters():
                    p.requires_grad_(True)
                m.train()
            print(f"(LagerNVSCompressor) VGGT frozen; geo_feature_connector+norm TRAINABLE (#6), from {cfg.ckpt_path}")
        else:
            _freeze(self.reconstructor)
            print(f"(LagerNVSCompressor) reconstructor loaded & frozen from {cfg.ckpt_path}")

        # Optional trainable Perceiver resampler: dense rec_tokens (768) ->
        # `num_scene_tokens` fixed-length scene tokens (SceneTok-style). Encoder
        # stays frozen; this + the denoiser are trained.
        self.perceiver = None
        if cfg.token_reduction == "perceiver":
            self.perceiver = PerceiverResampler(
                dim=cfg.token_dim,
                num_latents=cfg.num_scene_tokens,
                depth=cfg.perceiver_depth,
                num_heads=cfg.perceiver_heads,
            )
            print(f"(LagerNVSCompressor) Perceiver resampler: {cfg.num_scene_tokens} tokens "
                  f"x {cfg.token_dim}d, depth {cfg.perceiver_depth}")

        self.output_norm = nn.LayerNorm(cfg.token_dim) if cfg.output_norm else None
        if cfg.output_norm:
            print(f"(LagerNVSCompressor) trainable output LayerNorm({cfg.token_dim}) on rec_tokens")

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
        if self.cfg.unfreeze_geo_connector:
            # VGGT runs no_grad+detach internally (freeze_vggt); grad flows only
            # through the trainable geo_feature_connector + norm.
            rec = self.reconstructor(imgs, cam_token)        # (b, v, p, 768)
        else:
            with torch.no_grad():
                rec = self.reconstructor(imgs, cam_token)    # (b, v, p, 768) fully frozen
        rec = einops.rearrange(rec, "b v p c -> b (v p) c")  # dense tokens (b, v*p, 768)
        if self.perceiver is not None:
            rec = self.perceiver(rec)                        # (b, num_scene_tokens, 768) trainable
        if self.output_norm is not None:
            rec = self.output_norm(rec)                      # trainable re-scale toward denoiser cnd_proj input dist
        # Match the (tokens, qk) tuple contract that callers unpack via `tokens, *_`.
        return rec, None
