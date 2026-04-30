from types import MethodType

import torch
import torch.nn as nn

from diffsynth.models.wan_video_dit import modulate


def _match_sequence_length(camera_embedding: torch.Tensor, seq_len: int) -> torch.Tensor:
    if camera_embedding.shape[1] == seq_len:
        return camera_embedding

    if seq_len % camera_embedding.shape[1] != 0:
        raise ValueError(
            "ReCam camera embedding sequence length must divide self-attention input length: "
            f"camera_embedding={camera_embedding.shape}, seq_len={seq_len}"
        )
    repeat_count = seq_len // camera_embedding.shape[1]
    return camera_embedding.repeat_interleave(repeat_count, dim=1)


def _recam_block_forward(self, x, context, t_mod, freqs, camera_embedding=None):
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
    if camera_embedding is not None:
        camera_embedding = camera_embedding.to(device=input_x.device, dtype=input_x.dtype)
        camera_embedding_extended = _match_sequence_length(camera_embedding, input_x.shape[1])
        input_x = input_x + camera_embedding_extended
    x = self.gate(x, gate_msa, self.recam_projector(self.self_attn(input_x, freqs)))
    x = x + self.cross_attn(self.norm3(x), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x


def install_recam_attention(dit: nn.Module, camera_dim: int, eps: float = 1e-6) -> nn.Module:
    if getattr(dit, "_recam_attention_installed", False):
        return dit

    ref_param = next(dit.parameters())
    dim = dit.blocks[0].self_attn.q.weight.shape[0]

    dit.recam_camera_encoder = nn.Linear(camera_dim, dim).to(
        device=ref_param.device,
        dtype=ref_param.dtype,
    )
    dit.recam_camera_encoder.train(dit.training)

    for block in dit.blocks:
        block.recam_projector = nn.Linear(dim, dim).to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
        block.recam_projector.train(dit.training)
        block._recam_original_forward = block.forward
        block.forward = MethodType(_recam_block_forward, block)

    dit._recam_attention_installed = True
    return dit
