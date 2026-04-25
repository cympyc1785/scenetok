from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffsynth.models.wan_video_dit import SelfAttention, modulate


class ReCam3DAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = SelfAttention(dim, num_heads, eps)
        self.projector = nn.Linear(dim, dim)
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def _match_sequence_length(self, camera_embedding: torch.Tensor, seq_len: int) -> torch.Tensor:
        if camera_embedding.shape[1] == seq_len:
            return camera_embedding
        camera_embedding = rearrange(camera_embedding, "b n c -> b c n")
        camera_embedding = F.interpolate(camera_embedding, size=seq_len, mode="linear", align_corners=False)
        return rearrange(camera_embedding, "b c n -> b n c")

    def forward(self, x: torch.Tensor, camera_embedding: torch.Tensor | None, freqs: torch.Tensor) -> torch.Tensor:
        if camera_embedding is None:
            return torch.zeros_like(x)
        camera_embedding = camera_embedding.to(device=x.device, dtype=x.dtype)
        camera_embedding = self._match_sequence_length(camera_embedding, x.shape[1])
        return self.projector(self.attn(self.norm(x + camera_embedding), freqs))


def _recam_block_forward(self, x, context, t_mod, freqs):
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2),
            scale_msa.squeeze(2),
            gate_msa.squeeze(2),
            shift_mlp.squeeze(2),
            scale_mlp.squeeze(2),
            gate_mlp.squeeze(2),
        )

    input_x = modulate(self.norm1(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
    x = x + self.recam_3d_attn(x, getattr(self, "_recam_camera_embedding", None), freqs)
    x = x + self.cross_attn(self.norm3(x), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x


def _batch_compatible_head_forward(self, x, t_mod):
    if t_mod.ndim == 3:
        shift, scale = (
            self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
            + t_mod.unsqueeze(2)
        ).chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))

    if t_mod.ndim == 2:
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
            + t_mod.unsqueeze(1)
        ).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)

    shift, scale = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        + t_mod
    ).chunk(2, dim=1)
    return self.head(self.norm(x) * (1 + scale) + shift)


def install_recam_attention(dit: nn.Module, eps: float = 1e-6) -> nn.Module:
    if getattr(dit, "_recam_attention_installed", False):
        return dit

    ref_param = next(dit.parameters())
    dit.head._recam_original_forward = dit.head.forward
    dit.head.forward = MethodType(_batch_compatible_head_forward, dit.head)
    if hasattr(dit, "head_global"):
        dit.head_global._recam_original_forward = dit.head_global.forward
        dit.head_global.forward = MethodType(_batch_compatible_head_forward, dit.head_global)

    for block in dit.blocks:
        block.recam_3d_attn = ReCam3DAttention(dit.dim, block.num_heads, eps).to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
        block.recam_3d_attn.train(dit.training)
        block._recam_original_forward = block.forward
        block.forward = MethodType(_recam_block_forward, block)
        block._recam_camera_embedding = None

    dit._recam_attention_installed = True
    return dit


def project_recam_camera_embedding(dit: nn.Module, camera_embedding: torch.Tensor | None) -> torch.Tensor | None:
    if camera_embedding is None:
        return None

    ref_param = next(dit.parameters())
    camera_embedding = camera_embedding.to(device=ref_param.device, dtype=ref_param.dtype)
    text_in_dim = dit.text_embedding[0].in_features
    if camera_embedding.shape[-1] == text_in_dim:
        return dit.text_embedding(camera_embedding)
    if camera_embedding.shape[-1] == dit.dim:
        return camera_embedding
    raise ValueError(
        f"ReCam camera embedding dim {camera_embedding.shape[-1]} must match "
        f"Wan text input dim {text_in_dim} or DiT dim {dit.dim}."
    )


def set_recam_camera_embedding(dit: nn.Module, camera_embedding: torch.Tensor | None) -> None:
    camera_embedding = project_recam_camera_embedding(dit, camera_embedding)
    for block in dit.blocks:
        block._recam_camera_embedding = camera_embedding


def clear_recam_camera_embedding(dit: nn.Module) -> None:
    if not hasattr(dit, "blocks"):
        return
    for block in dit.blocks:
        if hasattr(block, "_recam_camera_embedding"):
            block._recam_camera_embedding = None
