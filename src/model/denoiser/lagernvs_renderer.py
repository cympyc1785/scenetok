"""Feed-forward (deterministic) LagerNVS renderer as a SceneTok decoder.

Pairs a FROZEN SceneTok compressor's scene tokens (b, N, token_dim) with the
LagerNVS *original* feed-forward Renderer (target Plücker rays = query, scene
tokens = cross-attention KV, in_noisy_channels=0, use_adaln=False → single
deterministic forward, pixel-space RGB). No diffusion / timestep.

The renderer runs at `hidden_size` (768 for B/8), so a small trainable adapter
projects the SceneTok tokens (token_dim, e.g. 64) up to `hidden_size` — it plays
the geo_feature_connector role that VGGT features had in native LagerNVS.

Two init modes (via `renderer_ckpt`):
  • general_512 path  → warm-start the renderer from pretrained LagerNVS.
  • None              → scratch (decoder sized to the SceneTok token channel).

Imported (not copied) from the sibling lagernvs repo, mirroring lagernvs_compressor.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import einops
from torch import nn, Tensor

_LAGERNVS_ROOT = "/data1/cympyc1785/lagernvs"
if _LAGERNVS_ROOT not in sys.path:
    sys.path.insert(0, _LAGERNVS_ROOT)


@dataclass
class LagerNVSRendererCfg:
    name: Literal["lagernvs_renderer"]
    scene_token_dim: int = 64              # SceneTok compressor token_dim
    hidden_size: int = 768                 # renderer transformer width (B/8 = 768)
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 8
    attention_type: Literal["cross_attention", "bidirectional_cross_attention"] = "bidirectional_cross_attention"
    out_channels: int = 3
    # Pretrained LagerNVS checkpoint to warm-start the renderer (renderer.* keys).
    # None → scratch init (decoder learns from the SceneTok token channel up).
    renderer_ckpt: Optional[str] = "/data1/cympyc1785/lagernvs/checkpoints/lagernvs_general_512/model.pt"
    load_strict: bool = False


class LagerNVSRenderer(nn.Module):

    def __init__(
        self,
        cfg: LagerNVSRendererCfg,
        **kwargs,  # cond_dim/num_scene_tokens/num_views/... from get_denoiser (unused)
    ) -> None:
        super().__init__()
        from models.renderer import Renderer

        self.cfg = cfg
        # SceneTok token (token_dim) -> renderer hidden. Replaces geo_feature_connector.
        self.scene_adapter = nn.Sequential(
            nn.Linear(cfg.scene_token_dim, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size, bias=False),
        )
        self.renderer = Renderer(
            cfg.depth,
            cfg.hidden_size,
            cfg.patch_size,
            cfg.num_heads,
            attention_to_features_type=cfg.attention_type,
            in_noisy_channels=0,       # feed-forward: rays only (deterministic)
            use_adaln=False,           # no timestep
            out_channels=cfg.out_channels,
        )
        if cfg.renderer_ckpt:
            sd = torch.load(cfg.renderer_ckpt, map_location="cpu")
            sd = sd.get("model", sd)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            rsd = {k[len("renderer."):]: v for k, v in sd.items() if k.startswith("renderer.")}
            missing, unexpected = self.renderer.load_state_dict(rsd, strict=False)
            print(f"(LagerNVSRenderer) renderer warm-start from {cfg.renderer_ckpt}: "
                  f"loaded {len(rsd)-len(unexpected)}/{len(rsd)}, missing={len(missing)}")
        else:
            print("(LagerNVSRenderer) renderer from SCRATCH (no pretrained), "
                  f"scene_token_dim={cfg.scene_token_dim} -> hidden {cfg.hidden_size}")

    @property
    def name(self):
        return self.cfg.name

    def render(self, scene_tokens: Tensor, target_rays: Tensor) -> Tensor:
        """scene_tokens (b, N, token_dim) + target_rays (b, V, 6, H, W) -> RGB (b, V, 3, H, W)."""
        b, v_target = target_rays.shape[:2]
        rec = self.scene_adapter(scene_tokens)                       # (b, N, hidden)
        rec = einops.repeat(rec, "b n d -> (b v) n d", v=v_target)   # per target view
        rgb = self.renderer(rec, target_rays)                        # (b, V, 3, H, W) in [0,1]
        return rgb

    def load_weights(self, path, **kwargs):
        return self
