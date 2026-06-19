import glob
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, inject_adapter_in_model

import traceback

from .denoiser import Denoiser
from ..camera import CameraCfg, get_camera
from ..types import DenoiserInputs


DIFFSYNTH_ROOT = Path(__file__).resolve().parents[1] / "DiffSynth-Studio"
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

from diffsynth.core import load_state_dict
from diffsynth.core.gradient.gradient_checkpoint import gradient_checkpoint_forward
from diffsynth.models.model_loader import ModelPool
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.models.wan_video_dit import (
    CrossAttention,
    GateModule,
    SelfAttention,
    WanModel,
    modulate,
    sinusoidal_embedding_1d,
)
from diffsynth.pipelines.wan_video import model_fn_wan_video
from diffsynth.models.wan_video_camera_controller import SimpleAdapter

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@dataclass
class WanTI2VLoRACfg:
    enabled: bool = False
    rank: int = 32
    alpha: int | None = None
    target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    checkpoint: str | Path | None = None


@dataclass
class WanTI2V5BCfg:
    name: Literal["wan_ti2v_5b"]
    camera: CameraCfg | None = None
    model_root: str | Path = Path("src/model/DiffSynth-Studio/Wan2.2/Wan2.2-TI2V-5B")
    dit_pattern: str = "diffusion_pytorch_model*.safetensors"
    text_encoder_path: str | Path | None = None
    tokenizer_path: str | Path | None = None
    seq_len: int = 512
    clean: str = "whitespace"
    gradient_checkpointing: bool = True
    condition_latents_input_type: Literal["none", "width", "channel", "temporal", "first_frame", "first_frame_random", "first_frame_depth", "first_frame_depth_soft"] = "none"
    camera_input_type: Literal["none", "recam_attention", "cross_attention", "new_cross_attention", "adaln", "wan_control", "channel_concat", "controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit"] | None = None
    enable_recam_attention: bool | None = None
    camera_context_spatial_pool: int = 1
    scene_input_type: Literal["none", "cross_attention", "new_cross_attention", "latent_concat", "controlnet"] = "cross_attention"
    scene_projection: Literal["linear", "mlp"] = "linear"
    ac3d_num_layers: int = 2
    # VACE식 interval injection (camera_input_type="controlnet" parallel 모드 전용):
    # 1이면 기존대로 main block 0..N-1(연속) 주입. k>1이면 ctrl block j를 main block
    # j*k (= 0,k,2k,…)에 주입해 네트워크 전체에 분산 (VACE vace_layers 패턴).
    # k>1이면 ctrl block 수는 자동으로 ceil(n_blocks/k)가 됨(ac3d_num_layers 무시).
    controlnet_inject_interval: int = 1
    # controlnet_lightningdit 전용: LightningDiT 분기의 hidden_size / num_heads / SwiGLU
    # 등 하이퍼파라미터. 기본값은 va-wan_dl3dv_256-480.ckpt의 LightningDiT denoiser와 일치.
    lightningdit_hidden_size: int = 1024
    lightningdit_num_heads: int = 16
    lightningdit_mlp_ratio: float = 4.0
    lightningdit_ckpt_path: str | Path | None = None  # warm-start ckpt
    # Joint dynamic + background recon training: when > 0, the wrapper computes
    # an additional rectified-flow loss between the LightningDiT ctrl branch's
    # raw prediction (pre-gate) and the velocity for `target["recon_video"]`
    # (typically `inpaint_result.mp4` — background only). Adds
    # `λ · MSE(ctrl_pred_raw, v_recon)` to the main loss.
    litdit_recon_loss_weight: float = 0.0
    # If True, the parallel `controlnet` mode's ctrl blocks are instantiated
    # with `use_text_cross_attn=False` (AC3D-paper-strict: text only in main
    # DiT, ctrl branch text-blind). Default False keeps the existing behaviour.
    controlnet_no_text_cross_attn: bool = False
    num_target_split: int = 1
    input_shape: int | list[int] = 16
    noise_seed: int | None = None
    ckpt_path: str | Path | None = None
    load_strict: bool = True
    lora: WanTI2VLoRACfg = field(default_factory=WanTI2VLoRACfg)


class BatchHead:
    @staticmethod
    def forward(head: nn.Module, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if t_mod.ndim != 2:
            return head(x, t_mod)

        modulation = head.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift, scale = (modulation + t_mod.unsqueeze(1)).chunk(2, dim=1)
        x = head.norm(x) * (1 + scale) + shift
        return head.head(x)

def _match_sequence_length(embedding: torch.Tensor, seq_len: int) -> torch.Tensor:
    if embedding.shape[1] == seq_len:
        return embedding
    if seq_len % embedding.shape[1] != 0:
        raise ValueError(
            "Embedding sequence length must divide self-attention input length: "
            f"embedding={embedding.shape}, seq_len={seq_len}"
        )
    repeat_count = seq_len // embedding.shape[1]
    return embedding.repeat_interleave(repeat_count, dim=1)


class NewDiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        camera_input_type: str | None = None,
        scene_input_type: str = "none",
        use_text_cross_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.camera_input_type = camera_input_type or "none"
        self.scene_input_type = scene_input_type
        self.use_text_cross_attn = use_text_cross_attn

        self.self_attn = SelfAttention(dim, num_heads, eps)
        if self.use_text_cross_attn:
            self.cross_attn = CrossAttention(
                dim, num_heads, eps, has_image_input=has_image_input)
        else:
            # controlnet_ac3d 분기: ac3d 블록은 scene cross-attn만 가짐
            # (text는 main DiT 분기에서만 처리). norm3도 같이 빠짐.
            self.cross_attn = None
        self.use_scene_cross_attn = self.scene_input_type == "new_cross_attention"
        if self.use_scene_cross_attn:
            self.scene_cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
            self.scene_cross_attn_proj = nn.Linear(dim, dim)
            self.norm4 = nn.LayerNorm(dim, eps=eps)
            nn.init.zeros_(self.scene_cross_attn_proj.weight)
            nn.init.zeros_(self.scene_cross_attn_proj.bias)
            # self.new_modulation = nn.Parameter(torch.randn(1, 9, dim) / dim**0.5)
        else:
            self.scene_cross_attn = None
            self.scene_cross_attn_proj = None
            self.norm4 = None
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        if self.camera_input_type == "recam_attention":
            self.recam_projector = nn.Linear(dim, dim)
        else:
            self.recam_projector = None
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        if self.use_text_cross_attn:
            self.norm3 = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm3 = None
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.gate = GateModule()

    @classmethod
    def from_dit_block(
        cls,
        block: nn.Module,
        has_image_input: bool,
        camera_input_type: str | None,
        scene_input_type: str,
        use_text_cross_attn: bool = True,
    ) -> "NewDiTBlock":
        new_block = cls(
            has_image_input=has_image_input,
            dim=block.dim,
            num_heads=block.num_heads,
            ffn_dim=block.ffn_dim,
            eps=block.norm1.eps,
            camera_input_type=camera_input_type,
            scene_input_type=scene_input_type,
            use_text_cross_attn=use_text_cross_attn,
        ).to(device=block.modulation.device, dtype=block.modulation.dtype)
        new_block.load_state_dict(block.state_dict(), strict=False)
        # if new_block.use_scene_cross_attn:
        #     with torch.no_grad():
        #         new_block.new_modulation[:, : block.modulation.shape[1]].copy_(block.modulation)
        return new_block

    def forward(self, x, context, scene_context, t_mod, scene_t_mod, freqs, camera_embedding=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        
        # if self.use_scene_cross_attn:
        #     # modulation = self.new_modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #         modulation[:, :6] + t_mod).chunk(6, dim=chunk_dim)
        #     shift_scene, scale_scene, gate_scene = (
        #         modulation[:, 6:] + scene_t_mod).chunk(3, dim=chunk_dim)
        # else:
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        
        if has_seq:
            # if self.use_scene_cross_attn:
            #     shift_msa, scale_msa, gate_msa, \
            #     shift_mlp, scale_mlp, gate_mlp, \
            #     shift_scene, scale_scene, gate_scene = (
            #         shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            #         shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            #         shift_scene.squeeze(2), scale_scene.squeeze(2), gate_scene.squeeze(2),
            #     )
            # else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        if self.camera_input_type == "recam_attention" and camera_embedding is not None:
            camera_embedding = camera_embedding.to(device=input_x.device, dtype=input_x.dtype)
            input_x = input_x + _match_sequence_length(camera_embedding, input_x.shape[1])

        self_attn = self.self_attn(input_x, freqs)
        if self.recam_projector is not None:
            self_attn = self.recam_projector(self_attn)

        x = self.gate(x, gate_msa, self_attn)
        if self.use_text_cross_attn:
            x = x + self.cross_attn(self.norm3(x), context)

        if self.use_scene_cross_attn:
            input_x = self.norm4(x)
            # input_x = modulate(self.norm4(x), shift_scene, scale_scene)
            scene_residual = self.scene_cross_attn(input_x, scene_context)
            scene_residual = self.scene_cross_attn_proj(scene_residual)
            # x = self.gate(x, gate_scene, scene_residual)
            x = x + scene_residual
        
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x
        
class WanTI2V5BDenoiser(Denoiser[WanTI2V5BCfg]):
    def __init__(
        self,
        cfg: WanTI2V5BCfg,
        cond_dim: int | None = 1,
        num_scene_tokens: int = 1,
        temporal_downsample: int = 1,
        using_wan: bool = False,
        **_: object,
    ) -> None:
        super().__init__(cfg)
        self.model_root = Path(cfg.model_root)
        self.model = self._load_model("wan_video_dit", self._resolve_dit_paths())
        self.supports_scene_tokens = True
        self.supports_condition_latents = True
        self.condition_latents_input_type = cfg.condition_latents_input_type
        self.supports_per_view_timestep = False
        self.uses_internal_text_encoder = True
        self.camera_input_type = getattr(cfg, "camera_input_type", None)
        self.scene_input_type = getattr(cfg, "scene_input_type", None)
        if self.camera_input_type is None and cfg.enable_recam_attention is not None:
            self.camera_input_type = "recam_attention" if cfg.enable_recam_attention else "none"
        # scene_input_type="controlnet"은 ControlNet(AC3D-style) 분기에 scene tokens을
        # (camera ray와 함께) 주입하는 모드 → 분기를 만드는 camera_input_type="controlnet"이 전제.
        if self.scene_input_type == "controlnet" and self.camera_input_type not in ("controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit"):
            raise ValueError(
                "scene_input_type='controlnet' requires camera_input_type='controlnet' "
                "(scene tokens are injected into the AC3D-style controlnet branch). "
                f"Got camera_input_type={self.camera_input_type!r}."
            )
        self._replace_dit_blocks_with_new_blocks(self.model)
        self.num_scene_tokens = num_scene_tokens
        self.cond_dim = 1 if cond_dim is None else cond_dim
        self.text_embed_dim = 4096
        scene_proj_mode = getattr(cfg, "scene_projection", "linear")
        if scene_proj_mode == "mlp":
            self.cnd_proj = nn.Sequential(
                nn.Linear(self.cond_dim, self.model.dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.model.dim, self.model.dim),
            )
        else:
            self.cnd_proj = nn.Linear(self.cond_dim, self.model.dim)
        self.null_tokens = nn.Parameter(torch.zeros(1, 1, self.model.dim))
        self.text_proj = None
        self.pose_embed = None
        self.negative_prompt = None
        if cfg.camera is not None:
            camera_cfg = deepcopy(cfg.camera)
            if self.camera_input_type in ("cross_attention", "new_cross_attention", "adaln"):
                # adaln collapses the camera embedding to a single (B, dim)
                # summary; running `time_embed` at the full target spatial
                # resolution (e.g. 240×416) blows up the Fourier+MLP
                # intermediates to >100 GB. Mirror the cross_attention path
                # and downsize the camera grid to a tiny pool ahead of time.
                pool_size = cfg.camera_context_spatial_pool
                if pool_size < 1:
                    raise ValueError(f"camera_context_spatial_pool must be >= 1, got {pool_size}.")
                camera_cfg.input_shape = [pool_size, pool_size]
            embedding_cfg = getattr(camera_cfg, "embedding", None)
            if (
                using_wan
                and temporal_downsample > 1
                and embedding_cfg is not None
                and getattr(embedding_cfg, "name", None) == "time_embed"
            ):
                embedding_cfg.in_channels *= temporal_downsample
            self.pose_embed = get_camera(
                camera_cfg,
                num_split=cfg.num_target_split,
                using_wan=using_wan,
                embed_dim=self.model.dim,
                temporal_downsample=temporal_downsample,
            )

        ref_param = next(self.model.parameters())
        # self.model.scene_embedding = nn.Sequential(
        #     nn.Linear(self.text_embed_dim, self.model.dim),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(self.model.dim, self.model.dim),
        # ).to(device=ref_param.device, dtype=ref_param.dtype)
        if self.scene_input_type == "new_cross_attention":
            self.model.scene_time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(self.model.dim, self.model.dim * 3))
        if self.camera_input_type in ("cross_attention", "new_cross_attention"):
            self.model.camera_embedding = nn.Sequential(
                nn.Linear(self.text_embed_dim, self.model.dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.model.dim, self.model.dim),
            ).to(device=ref_param.device, dtype=ref_param.dtype)
        if self.camera_input_type == "recam_attention":
            self.model.recam_camera_encoder = nn.Linear(12, self.model.dim).to(
                device=ref_param.device,
                dtype=ref_param.dtype,
            )
        if self.camera_input_type == "adaln" and getattr(self.model, "adaln_camera_proj", None) is None:
            # Zero-initialized gating projector applied to the pooled camera
            # summary before it is added to the timestep embedding. At
            # initialization the projector outputs all zeros, so
            # `t + adaln_camera_embedding == t` — the Wan TI2V prior is
            # preserved exactly. As training progresses the projector learns
            # to inject camera information into the per-batch adaLN
            # modulation. (Same idea as ControlNet zero-conv.)
            self.model.adaln_camera_proj = nn.Linear(
                self.model.dim, self.model.dim, bias=True
            ).to(device=ref_param.device, dtype=ref_param.dtype)
            nn.init.zeros_(self.model.adaln_camera_proj.weight)
            nn.init.zeros_(self.model.adaln_camera_proj.bias)
        if self.camera_input_type == "wan_control" and getattr(self.model, "control_adapter", None) is None:
            patch_size = tuple(getattr(self.model, "patch_size", (1, 2, 2)))
            spatial_patch = patch_size[1:] if len(patch_size) == 3 else patch_size
            self.model.control_adapter = SimpleAdapter(
                in_dim=6 * max(temporal_downsample, 1),
                out_dim=self.model.dim,
                kernel_size=tuple(spatial_patch),
                stride=tuple(spatial_patch),
                num_residual_blocks=1,
            ).to(device=ref_param.device, dtype=ref_param.dtype)
        if self.camera_input_type == "controlnet":
            # AC3D-style camera controlnet: small parallel branch that runs first
            # `ac3d_num_layers` Wan blocks on (latent + ray channel-concat) input,
            # then injects layer-wise zero-init residuals into the corresponding
            # first N main DiT blocks. Mirrors AC3D / CogVideoX-ControlNet.
            extra_in_ac = 6 * max(temporal_downsample, 1)
            orig_pe_ac = self.model.patch_embedding
            self.model.ac3d_patch_embedding = nn.Conv3d(
                orig_pe_ac.in_channels + extra_in_ac,
                orig_pe_ac.out_channels,
                kernel_size=orig_pe_ac.kernel_size,
                stride=orig_pe_ac.stride,
                padding=orig_pe_ac.padding,
                bias=orig_pe_ac.bias is not None,
            )
            with torch.no_grad():
                self.model.ac3d_patch_embedding.weight.zero_()
                self.model.ac3d_patch_embedding.weight[:, : orig_pe_ac.in_channels].copy_(orig_pe_ac.weight)
                if orig_pe_ac.bias is not None:
                    self.model.ac3d_patch_embedding.bias.copy_(orig_pe_ac.bias)
            self.model.ac3d_patch_embedding = self.model.ac3d_patch_embedding.to(
                device=orig_pe_ac.weight.device, dtype=orig_pe_ac.weight.dtype
            )
            # Clone first N main blocks. Camera inside-block path stays disabled
            # (camera enters via the channel-concat ray input instead). The scene
            # path is enabled only for scene_input_type="controlnet" so the
            # controlnet branch's residual carries scene guidance too (zero-init
            # scene_cross_attn_proj → initial contribution 0, Wan prior preserved).
            # Injection block ids. interval=1 → first N consecutive (기존 동작).
            # interval=k>1 → VACE식 [0,k,2k,…] 전체 분산, ctrl block 수 = len(ids).
            interval = max(int(getattr(cfg, "controlnet_inject_interval", 1)), 1)
            if interval > 1:
                inject_ids = list(range(0, len(self.model.blocks), interval))
            else:
                ac3d_n = min(max(int(cfg.ac3d_num_layers), 1), len(self.model.blocks))
                inject_ids = list(range(ac3d_n))
            self.model.ac3d_inject_ids = inject_ids
            ac3d_n = len(inject_ids)
            ac3d_has_image_input = bool(getattr(self.model, "has_image_input", False))
            ac3d_scene_input = "new_cross_attention" if self.scene_input_type == "controlnet" else "none"
            self.model.ac3d_blocks = nn.ModuleList([
                NewDiTBlock.from_dit_block(
                    self.model.blocks[bid],   # clone from the target main block (VACE: role-matched)
                    has_image_input=ac3d_has_image_input,
                    camera_input_type="none",
                    scene_input_type=ac3d_scene_input,
                    use_text_cross_attn=not bool(cfg.controlnet_no_text_cross_attn),
                )
                for bid in inject_ids
            ])
            # Per-layer zero-init projectors (ControlNet's signature pattern —
            # initial contribution = 0 → Wan prior preserved at t=0)
            self.model.ac3d_projectors = nn.ModuleList([
                nn.Linear(self.model.dim, self.model.dim)
                for _ in range(ac3d_n)
            ])
            for proj in self.model.ac3d_projectors:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            self.model.ac3d_projectors = self.model.ac3d_projectors.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
        if self.camera_input_type == "controlnet_feedback":
            # AC3D paper architecture (VDiT-CC): bidirectional controlnet.
            # Builds same parallel modules as `controlnet` PLUS per-layer
            # FC_down that lets the main DiT hidden state feed back into the
            # ac3d branch before each ac3d block. Both directions go through
            # zero-init linears so the model starts identical to vanilla Wan
            # and learns to use camera information gradually.
            extra_in_ac = 6 * max(temporal_downsample, 1)
            orig_pe_ac = self.model.patch_embedding
            self.model.ac3d_patch_embedding = nn.Conv3d(
                orig_pe_ac.in_channels + extra_in_ac,
                orig_pe_ac.out_channels,
                kernel_size=orig_pe_ac.kernel_size,
                stride=orig_pe_ac.stride,
                padding=orig_pe_ac.padding,
                bias=orig_pe_ac.bias is not None,
            )
            with torch.no_grad():
                self.model.ac3d_patch_embedding.weight.zero_()
                self.model.ac3d_patch_embedding.weight[:, : orig_pe_ac.in_channels].copy_(orig_pe_ac.weight)
                if orig_pe_ac.bias is not None:
                    self.model.ac3d_patch_embedding.bias.copy_(orig_pe_ac.bias)
            self.model.ac3d_patch_embedding = self.model.ac3d_patch_embedding.to(
                device=orig_pe_ac.weight.device, dtype=orig_pe_ac.weight.dtype
            )
            ac3d_n = max(int(cfg.ac3d_num_layers), 1)
            ac3d_n = min(ac3d_n, len(self.model.blocks))
            ac3d_has_image_input = bool(getattr(self.model, "has_image_input", False))
            ac3d_scene_input = "new_cross_attention" if self.scene_input_type == "controlnet" else "none"
            self.model.ac3d_blocks = nn.ModuleList([
                NewDiTBlock.from_dit_block(
                    self.model.blocks[i],
                    has_image_input=ac3d_has_image_input,
                    camera_input_type="none",
                    scene_input_type=ac3d_scene_input,
                )
                for i in range(ac3d_n)
            ])
            # ac3d → main residual projector (zero-init).
            self.model.ac3d_projectors = nn.ModuleList([
                nn.Linear(self.model.dim, self.model.dim)
                for _ in range(ac3d_n)
            ])
            for proj in self.model.ac3d_projectors:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            self.model.ac3d_projectors = self.model.ac3d_projectors.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
            # NEW (vs `controlnet`): per-layer FC_down for main → ac3d feedback.
            # Zero-init so the feedback contribution is 0 at start (ac3d only
            # sees ray/latent → ac3d_proj is also 0 → main is vanilla Wan).
            self.model.ac3d_fc_down = nn.ModuleList([
                nn.Linear(self.model.dim, self.model.dim)
                for _ in range(ac3d_n)
            ])
            for fc in self.model.ac3d_fc_down:
                nn.init.zeros_(fc.weight)
                nn.init.zeros_(fc.bias)
            self.model.ac3d_fc_down = self.model.ac3d_fc_down.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
        if self.camera_input_type == "controlnet_ac3d":
            # AC3D paper VDiT-CC architecture (paper-faithful variant).
            # 차이점 vs `controlnet`/`controlnet_feedback`:
            #   1. Noise/latent은 main DiT에만 들어감. ac3d 분기는 latent를 직접 안 봄.
            #   2. ctrl block i input = FC_main2ctrl(main_x_i) + cam_emb + ctrl_x_{i-1}
            #      → main의 매 layer 출력을 ctrl이 흡수.
            #   3. ctrl block output → zero-init Linear → main_x_i에 add → main block_{i+1} 입력.
            #   4. ctrl block 내부: self_attn → scene_cross_attn (zero-init proj) → FFN
            #      (text cross-attn 없음 — `use_text_cross_attn=False`).
            # Camera embedding은 ray map을 main patch grid에 맞게 patchify해서 per-token으로
            # 더해줌. ctrl block마다 같은 cam_emb를 더함 (per-layer 분리 X — 파라미터 절감).
            ac3d_n = max(int(cfg.ac3d_num_layers), 1)
            ac3d_n = min(ac3d_n, len(self.model.blocks))
            ac3d_has_image_input = bool(getattr(self.model, "has_image_input", False))
            ac3d_scene_input = "new_cross_attention" if self.scene_input_type == "controlnet" else "none"
            self.model.ac3d_blocks = nn.ModuleList([
                NewDiTBlock.from_dit_block(
                    self.model.blocks[i],
                    has_image_input=ac3d_has_image_input,
                    camera_input_type="none",
                    scene_input_type=ac3d_scene_input,
                    use_text_cross_attn=False,
                )
                for i in range(ac3d_n)
            ])
            # main → ctrl FC (forward path; NOT zero-init).
            self.model.ac3d_main2ctrl = nn.ModuleList([
                nn.Linear(self.model.dim, self.model.dim)
                for _ in range(ac3d_n)
            ])
            self.model.ac3d_main2ctrl = self.model.ac3d_main2ctrl.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
            # ctrl → main residual projector (zero-init, ControlNet signature).
            self.model.ac3d_projectors = nn.ModuleList([
                nn.Linear(self.model.dim, self.model.dim)
                for _ in range(ac3d_n)
            ])
            for proj in self.model.ac3d_projectors:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            self.model.ac3d_projectors = self.model.ac3d_projectors.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
            # Camera ray map → per-token embedding. Mirror main `patch_embedding`의 spatial
            # config (Conv3d, same kernel/stride/padding)이라 출력 grid가 main patch token
            # 시퀀스와 정확히 1:1 매칭. 6·td 입력 채널 (Plücker ray, origin 3 + dir 3) × Wan
            # temporal downsample factor.
            extra_in_ac = 6 * max(temporal_downsample, 1)
            orig_pe_ac = self.model.patch_embedding
            self.model.ac3d_camera_patch_embed = nn.Conv3d(
                extra_in_ac,
                orig_pe_ac.out_channels,
                kernel_size=orig_pe_ac.kernel_size,
                stride=orig_pe_ac.stride,
                padding=orig_pe_ac.padding,
                bias=orig_pe_ac.bias is not None,
            ).to(device=orig_pe_ac.weight.device, dtype=orig_pe_ac.weight.dtype)
        if self.camera_input_type == "controlnet_lightningdit":
            # SceneTok rectified flow decoder (LightningDiT)를 통째로 ctrl 분기로
            # instantiate. Main Wan DiT와 평행하게 독립 forward (no feedback) 후 final
            # prediction을 zero-init Conv3d gate 통해 main_pred에 잔차 add.
            #
            # Ctrl 분기 == SceneTok 학습된 LightningDiT 그대로:
            #   - x_embedder        : Conv2d(48 → 1024, k=1)           ← ckpt 그대로
            #   - pose_embed        : get_camera(cfg) (plücker → pos)  ← ckpt 그대로
            #   - t_embedder        : TimestepEmbedder(1024, pose-aware) ← ckpt 그대로
            #   - blocks × 24       : LightningDiTBlock (SwiGLU+RMSNorm+qk_norm, 3D RoPE) ← ckpt 그대로
            #   - cnd_proj          : Linear(64 → 1024)                ← ckpt 그대로 (raw cond_dim 입력)
            #   - final_layer + unpatchify → (B, V, C_lat, H, W) velocity prediction
            # 추가 학습 layer (zero-init):
            #   - litdit_output_gate: Conv3d(C_lat, C_lat, k=1)        ← ControlNet residual gate
            from .lightningdit import (
                LightningDiT,
                LightningDiTCfg,
                LightningDiTKwargsCfg,
            )

            litdit_kwargs = LightningDiTKwargsCfg(
                patch_size=1,
                in_channels=int(self.model.patch_embedding.in_channels),
                hidden_size=int(cfg.lightningdit_hidden_size),
                depth=24,
                num_heads=int(cfg.lightningdit_num_heads),
                mlp_ratio=float(cfg.lightningdit_mlp_ratio),
                class_dropout_prob=0.0,
                num_classes=1000,
                learn_sigma=False,
                use_qknorm=True,
                use_swiglu=True,
                use_rope=False,
                use_rope_3d=True,
                use_rmsnorm=True,
                wo_shift=False,
                frequency_embedding_size=256,
            )
            # Build a LightningDiT-side camera_cfg with the SAME in_channels
            # inflation main wrapper does — otherwise pose_embed.linear_1
            # expects 6 channels while Wan-style ray maps arrive as
            # 6 * temporal_downsample channels.
            ld_camera_cfg = deepcopy(cfg.camera)
            ld_emb_cfg = getattr(ld_camera_cfg, "embedding", None)
            if (
                temporal_downsample > 1
                and ld_emb_cfg is not None
                and getattr(ld_emb_cfg, "name", None) == "time_embed"
            ):
                ld_emb_cfg.in_channels *= temporal_downsample
            litdit_cfg = LightningDiTCfg(
                name="lightningdit",
                camera=ld_camera_cfg,
                kwargs=litdit_kwargs,
                single_dim_tokens=False,
                num_target_split=int(cfg.num_target_split),
                camera_conditioning="add",
                input_shape=list(cfg.input_shape) if not isinstance(cfg.input_shape, int) else [cfg.input_shape, cfg.input_shape],
                gradient_checkpointing=True,
                pretrained_from=None,
                ckpt_path=str(cfg.lightningdit_ckpt_path) if cfg.lightningdit_ckpt_path is not None else None,
                load_strict=False,
                causal_attention=False,
                text_cond_dim=None,
            )
            # cond_dim mismatch check (ckpt expects raw cond_dim=64 for cnd_proj).
            # If wrapper's cond_dim differs, LightningDiT.load_weights (strict=False
            # + shape-mismatch pop) will silently drop cnd_proj; init random.
            ld_cond_dim = 64
            num_target_views = int(getattr(cfg, "ac3d_num_layers", 0)) or 10  # placeholder; over-written by data shape at forward
            self.model.litdit_branch = LightningDiT(
                cfg=litdit_cfg,
                cond_dim=ld_cond_dim,
                num_scene_tokens=int(num_scene_tokens),
                num_views=int(num_target_views),
                temporal_downsample=int(temporal_downsample),
                using_wan=True,
                cfg_train=False,
            ).to(device=ref_param.device, dtype=ref_param.dtype)
            # Zero-init residual gate: ctrl pred → main pred sum.
            C_out = int(self.model.patch_embedding.in_channels)
            self.model.litdit_output_gate = nn.Conv3d(
                C_out, C_out, kernel_size=1, bias=True
            ).to(device=ref_param.device, dtype=ref_param.dtype)
            nn.init.zeros_(self.model.litdit_output_gate.weight)
            nn.init.zeros_(self.model.litdit_output_gate.bias)
            if cfg.lightningdit_ckpt_path is not None:
                print(
                    f"[controlnet_lightningdit] ctrl branch warm-started from "
                    f"{cfg.lightningdit_ckpt_path} (strict=False); output_gate zero-init."
                )
        if self.camera_input_type == "channel_concat":
            # Channel-concat ray map with latent BEFORE the patch_embedding conv.
            # `pose_embed(skip_embedding=True)` returns (B, V_lat, 6*T, H, W) when
            # `embedding.in_channels` was inflated by `temporal_downsample` in
            # `__init__` (matches wan_control's SimpleAdapter). So extra channels
            # at patch_embedding input = 6 * temporal_downsample.
            # Original 48 channels copied from pretrained; extra channels
            # zero-init so the model starts identical to the Wan TI2V prior,
            # then learns to use camera info.
            orig_pe = self.model.patch_embedding
            extra_in = 6 * max(temporal_downsample, 1)
            new_in = orig_pe.in_channels + extra_in
            new_pe = nn.Conv3d(
                new_in,
                orig_pe.out_channels,
                kernel_size=orig_pe.kernel_size,
                stride=orig_pe.stride,
                padding=orig_pe.padding,
                bias=orig_pe.bias is not None,
            )
            with torch.no_grad():
                new_pe.weight.zero_()
                new_pe.weight[:, : orig_pe.in_channels].copy_(orig_pe.weight)
                if orig_pe.bias is not None:
                    new_pe.bias.copy_(orig_pe.bias)
            self.model.patch_embedding = new_pe.to(
                device=orig_pe.weight.device, dtype=orig_pe.weight.dtype
            )
        self.text_encoder = self._load_model(
            "wan_video_text_encoder",
            self._resolve_text_encoder_path(),
        )
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_tokenizer = HuggingfaceTokenizer(
            name=str(self._resolve_tokenizer_path()),
            seq_len=cfg.seq_len,
            clean=cfg.clean,
        )

        if cfg.lora.enabled and cfg.scene_input_type != "new_cross_attention":
            self._enable_lora(cfg.lora)
        if cfg.ckpt_path is not None:
            self.load_weights(cfg.ckpt_path, strict=cfg.load_strict)
        self._set_trainable_parameters()
        self._log_trainable_modules()

    def _resolve_dit_paths(self) -> list[str]:
        paths = sorted(glob.glob(str(self.model_root / self.cfg.dit_pattern)))
        if not paths:
            raise FileNotFoundError(
                f"Could not find Wan TI2V weights under {self.model_root / self.cfg.dit_pattern}"
            )
        return paths

    def _resolve_text_encoder_path(self) -> str:
        if self.cfg.text_encoder_path is not None:
            return str(self.cfg.text_encoder_path)
        return str(self.model_root / "models_t5_umt5-xxl-enc-bf16.pth")

    def _resolve_tokenizer_path(self) -> Path:
        if self.cfg.tokenizer_path is not None:
            return Path(self.cfg.tokenizer_path)
        return self.model_root / "google" / "umt5-xxl"

    def _load_model(self, model_name: str, path: str | list[str]) -> nn.Module:
        pool = ModelPool()
        pool.auto_load_model(path)
        model = pool.fetch_model(model_name)
        if model is None:
            raise RuntimeError(f"Failed to load `{model_name}` from {path}.")
        return model

    def _replace_dit_blocks_with_new_blocks(self, model: WanModel) -> None:
        has_image_input = bool(getattr(model, "has_image_input", False))
        model.blocks = nn.ModuleList(
            NewDiTBlock.from_dit_block(
                block,
                has_image_input=has_image_input,
                camera_input_type=self.camera_input_type,
                scene_input_type=self.cfg.scene_input_type,
            )
            for block in model.blocks
        )

    def _map_lora_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                key = key.replace("lora_A.weight", "lora_A.default.weight")
                key = key.replace("lora_B.weight", "lora_B.default.weight")
            mapped[key] = value
        return mapped

    def _enable_lora(self, lora_cfg: WanTI2VLoRACfg) -> None:
        target_modules = [name.strip() for name in lora_cfg.target_modules.split(",") if name.strip()]
        lora_alpha = lora_cfg.alpha if lora_cfg.alpha is not None else lora_cfg.rank
        self.model = inject_adapter_in_model(
            LoraConfig(
                r=lora_cfg.rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
            ),
            self.model,
        )
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name or "recam_" in name

        if lora_cfg.checkpoint is not None:
            state_dict = load_state_dict(str(lora_cfg.checkpoint))
            state_dict = self._map_lora_state_dict(state_dict)
            self.model.load_state_dict(state_dict, strict=False)

    def _set_trainable_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_model_substrings = ("recam_projector", "recam_camera_encoder", "lora_")
        for name, param in self.model.named_parameters():
            trainable = any(key in name for key in trainable_model_substrings)
            trainable = trainable or (
                self.camera_input_type == "recam_attention"
                and ".self_attn." in name
            )
            trainable = trainable or (
                self.camera_input_type == "wan_control"
                and name.startswith("control_adapter.")
            )
            trainable = trainable or (
                self.camera_input_type == "channel_concat"
                and name.startswith("patch_embedding.")
            )
            trainable = trainable or (
                self.camera_input_type == "controlnet"
                and (
                    name.startswith("ac3d_patch_embedding.")
                    or name.startswith("ac3d_blocks.")
                    or name.startswith("ac3d_projectors.")
                )
            )
            trainable = trainable or (
                self.camera_input_type == "controlnet_feedback"
                and (
                    name.startswith("ac3d_patch_embedding.")
                    or name.startswith("ac3d_blocks.")
                    or name.startswith("ac3d_projectors.")
                    or name.startswith("ac3d_fc_down.")
                )
            )
            trainable = trainable or (
                self.camera_input_type == "controlnet_ac3d"
                and (
                    name.startswith("ac3d_camera_patch_embed.")
                    or name.startswith("ac3d_blocks.")
                    or name.startswith("ac3d_main2ctrl.")
                    or name.startswith("ac3d_projectors.")
                )
            )
            trainable = trainable or (
                self.camera_input_type == "controlnet_lightningdit"
                and (
                    name.startswith("litdit_branch.")
                    or name.startswith("litdit_output_gate.")
                )
            )
            trainable = trainable or (
                self.camera_input_type == "adaln"
                and name.startswith("adaln_camera_proj.")
            )
            if not self.cfg.lora.enabled:
                trainable = trainable or name.startswith("text_embedding.")
            # trainable = trainable or name.startswith("scene_embedding.")
            trainable = trainable or (
                self.camera_input_type in ("cross_attention", "new_cross_attention")
                and name.startswith("camera_embedding.")
            )
            trainable = trainable or (
                self.cfg.scene_input_type == "new_cross_attention"
                and (
                    ".scene_cross_attn." in name
                    or ".scene_cross_attn_proj." in name
                    or ".norm4." in name
                    or name.endswith(".new_modulation")
                )
            )
            param.requires_grad = trainable

        for param in self.cnd_proj.parameters():
            param.requires_grad = True
        self.null_tokens.requires_grad = True

        if self.pose_embed is not None:
            for param in self.pose_embed.parameters():
                param.requires_grad = True

    def _log_trainable_modules(self) -> None:
        grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"params": 0, "tensors": 0})
        total_params = 0
        total_tensors = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "lora_" in name:
                group_name = "model.lora"
            elif "recam_camera_encoder" in name:
                group_name = "model.recam_camera_encoder"
            elif "recam_projector" in name:
                group_name = "model.recam_projector"
            elif name.startswith("model.control_adapter."):
                group_name = "model.control_adapter"
            elif ".self_attn." in name:
                group_name = "model.self_attn"
            elif ".scene_cross_attn." in name:
                group_name = "model.scene_cross_attn"
            elif ".scene_cross_attn_proj." in name:
                group_name = "model.scene_cross_attn_proj"
            elif name.endswith(".modulation") or name.endswith(".new_modulation"):
                group_name = "model.modulation"
            # elif name.startswith("model.scene_embedding."):
            #     group_name = "model.scene_embedding"
            # elif name.startswith("model.camera_embedding."):
            #     group_name = "model.camera_embedding"
            elif name.startswith("pose_embed."):
                group_name = "pose_embed"
            elif name.startswith("cnd_proj."):
                group_name = "cnd_proj"
            elif name == "null_tokens":
                group_name = "null_tokens"
            elif name.startswith("model.text_embedding."):
                group_name = "model.text_embedding"
            else:
                parts = name.split(".")
                group_name = ".".join(parts[:2]) if len(parts) > 1 else name

            grouped[group_name]["params"] += param.numel()
            grouped[group_name]["tensors"] += 1
            total_params += param.numel()
            total_tensors += 1

        print(cyan("\n\n[WanTI2V] Trainable modules:"))
        for group_name in sorted(grouped):
            stats = grouped[group_name]
            print(f"  - {group_name}: {stats['params']:,} params across {stats['tensors']} tensors")
        print(cyan(f"[WanTI2V] Total trainable: {total_params:,} params across {total_tensors} tensors\n\n"))

    def encode_text_condition(self, text: str | list[str], device: torch.device) -> torch.Tensor | None:
        if text is None:
            text = ""
        ids, mask = self.text_tokenizer(text, return_mask=True)
        ids = ids.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            text_state = self.text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, v in enumerate(seq_lens):
                text_state[i, v:] = 0

            text_state = self.model.text_embedding(text_state)
        return text_state.to(dtype=next(self.model.parameters()).dtype)

    def load_weights(
        self,
        path: Path | str,
        **kwargs,
    ):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(key.startswith("denoiser.model.") for key in state_dict):
            state_dict = {
                key.replace("denoiser.model.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("denoiser.model.")
            }
        self.model.load_state_dict(state_dict, **kwargs)

    def _get_camera_embedding(
        self,
        inputs: DenoiserInputs,
        temporal_downsample: int,
        chunk_targets: bool=True,
    ) -> torch.Tensor | None:
        if self.camera_input_type == "none" or self.pose_embed is None or inputs.pose is None:
            return None

        if self.camera_input_type == "recam_attention":
            extrinsics = inputs.pose.extrinsics[..., :3, :4]
            if temporal_downsample > 1 and extrinsics.shape[1] != inputs.view.shape[1]:
                indices = torch.arange(inputs.view.shape[1], device=extrinsics.device)
                indices = torch.where(
                    indices == 0,
                    torch.zeros_like(indices),
                    1 + (indices - 1) * temporal_downsample,
                )
                if indices[-1] >= extrinsics.shape[1]:
                    raise ValueError(
                        "Not enough camera extrinsics for ReCam temporal sampling: "
                        f"extrinsics={extrinsics.shape}, view={inputs.view.shape}, "
                        f"temporal_downsample={temporal_downsample}, indices={indices}"
                    )
                extrinsics = extrinsics[:, indices]
            return extrinsics.flatten(-2)

        if self.camera_input_type in ("wan_control", "channel_concat", "controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit"):
            # Both modes consume raw Plücker rays (6 ch) without `time_embed`
            # blow-up. SimpleAdapter / channel-concat handle dim alignment themselves.
            # SimpleAdapter unpacks `(B, C, F, H, W)` (NCFHW), but
            # `RayCamera` returns NFCHW — permute frames and channels.
            rays = self.pose_embed(
                inputs.pose,
                temporal_downsample=temporal_downsample,
                chunk_targets=chunk_targets,
                skip_embedding=True,
            )
            return rays.permute(0, 2, 1, 3, 4).contiguous()  # (B, V, C, H, W) -> (B, C, V, H, W)

        pose_tokens = self.pose_embed(
            inputs.pose,
            temporal_downsample=temporal_downsample,
            chunk_targets=chunk_targets,
        )
        if pose_tokens.shape[1] != inputs.view.shape[1]:
            raise ValueError(
                "Shape mismatch",
                inputs.pose.extrinsics.shape,
                pose_tokens.shape,
                inputs.view.shape,
            )
        if self.camera_input_type == "adaln":
            return pose_tokens
        return rearrange(pose_tokens, "b v c h w -> b (v h w) c")

    def _get_camera_context(
        self,
        camera_embedding: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.camera_input_type == "none" or camera_embedding is None:
            return None
        if self.camera_input_type == "adaln":
            # `pose_embed` output is (B, V, dim, H, W). LightningDiT mixes the
            # full per-token pose grid into a per-token time embedding inside
            # `t_embedder(t, pemb=p)`; Wan TI2V's adaLN is per-batch scalar
            # (t_mod shape is (B, 6, dim)), so we collapse the (V, H, W) axes
            # to a (B, dim) summary that `simple_wan_video_fn` will add to
            # the timestep embedding before computing `t_mod`. The pooled
            # vector is gated through a zero-initialized linear so the
            # initial contribution is exactly 0 — Wan TI2V's pretrained
            # behaviour is preserved at the start of training, then camera
            # info ramps in as `adaln_camera_proj` learns.
            ref_param = next(self.model.parameters())
            camera_embedding = camera_embedding.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
            pooled = camera_embedding.mean(dim=(1, 3, 4))  # (b v c h w) -> (b c)
            return self.model.adaln_camera_proj(pooled)
        if self.camera_input_type in ("wan_control", "channel_concat", "controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit"):
            return camera_embedding
        ref_param = next(self.model.parameters())
        camera_embedding = camera_embedding.to(device=ref_param.device, dtype=ref_param.dtype)
        return camera_embedding
        # return self.model.camera_embedding(camera_embedding)

    def _condition_latents_concat_dim(self) -> int:
        if self.condition_latents_input_type == "channel":
            return 1
        if self.condition_latents_input_type == "temporal":
            return 2
        if self.condition_latents_input_type == "width":
            return -1
        raise ValueError(
            f"Unsupported condition_latents_input_type={self.condition_latents_input_type!r}. "
            "Expected 'none', 'width', 'channel', or 'temporal'."
        )

    def _concat_condition_latents(
        self,
        latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int | None]:
        if condition_latents is None or self.condition_latents_input_type in ("none", "first_frame", "first_frame_random", "first_frame_depth", "first_frame_depth_soft"):
            return latents, None, None

        condition_latents = rearrange(condition_latents, "b v c h w -> b c v h w")
        concat_dim = self._condition_latents_concat_dim()
        normalized_dim = concat_dim % latents.ndim
        condition_shape = condition_latents.shape
        latent_shape = latents.shape
        if condition_shape[:normalized_dim] + condition_shape[normalized_dim + 1:] != (
            latent_shape[:normalized_dim] + latent_shape[normalized_dim + 1:]
        ):
            raise ValueError(
                "condition_latents must match target latents except the "
                f"{self.condition_latents_input_type}-concat axis: "
                f"condition_latents={condition_shape}, latents={latent_shape}"
            )

        latents = torch.cat([condition_latents, latents], dim=concat_dim)
        expected_channels = getattr(getattr(self.model, "patch_embedding", None), "in_channels", None)
        if expected_channels is not None and latents.shape[1] != expected_channels:
            raise ValueError(
                "Condition latent concat produced an input channel count that does not match "
                "the Wan patch embedding. For channel concat, load or configure a model with "
                f"matching in_dim. latents={latents.shape}, patch_embedding.in_channels={expected_channels}"
            )
        return latents, condition_latents, concat_dim

    def _crop_condition_latents_prediction(
        self,
        pred: torch.Tensor,
        condition_latents: torch.Tensor | None,
        concat_dim: int | None,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        if condition_latents is None or concat_dim is None:
            return pred

        crop_size = condition_latents.shape[concat_dim]
        if concat_dim == 1:
            if pred.shape[1] == target_shape[1]:
                return pred
            if pred.shape[1] != target_shape[1] + crop_size:
                raise ValueError(
                    "Channel-concat prediction must either match target channels or include "
                    "condition channels before target channels: "
                    f"pred={pred.shape}, condition_latents={condition_latents.shape}, "
                    f"target_shape={target_shape}"
                )
            return pred[:, crop_size:]
        if concat_dim == 2:
            return pred[:, :, crop_size:]
        if concat_dim == -1 or concat_dim == pred.ndim - 1:
            return pred[..., crop_size:]
        raise ValueError(f"Unsupported condition latent concat dim for crop: {concat_dim}")

    def _forward(
        self,
        inputs: DenoiserInputs,
        **kwargs,
    ):
        temporal_downsample = kwargs.get("temporal_downsample", 1)
        chunk_targets = kwargs.get("chunk_targets", True)
        latents = rearrange(inputs.view, "b v c h w -> b c v h w")
        timestep = inputs.timestep
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])
        elif timestep.ndim > 1:
            timestep = timestep[:, 0]

        target_latent_shape = latents.shape
        latents, condition_latents, condition_latents_concat_dim = self._concat_condition_latents(
            latents,
            inputs.condition_latents,
        )

        context = inputs.text
        scene_context = inputs.state
        # if context is None and self.cfg.scene_input_type != "cross_attention":
            # context = self.null_tokens.expand(latents.shape[0], -1, -1).to(
            #     device=latents.device,
            #     dtype=latents.dtype,
            # )
        if context is None:
            context = self.encode_text_condition("", device=latents.device)

        camera_embedding = self._get_camera_embedding(inputs, temporal_downsample, chunk_targets)
        camera_context = self._get_camera_context(camera_embedding)

        if self.camera_input_type == "controlnet_lightningdit":
            # Main Wan DiT forward — camera off (camera 정보는 ctrl 분기에서만 사용)
            main_pred = simple_wan_video_fn(
                dit=self.model,
                latents=latents,
                timestep=timestep,
                context=context,
                scene_context=scene_context,
                scene_input_type=self.cfg.scene_input_type,
                camera_context=None,
                camera_input_type="none",
                fuse_vae_embedding_in_latents=False,
                use_gradient_checkpointing=self.cfg.gradient_checkpointing and self.training,
            )
            # Ctrl branch (SceneTok LightningDiT 그대로) — independent forward.
            # latents currently `(B, C, V, H, W)`; LightningDiT expects `(B, V, C, H, W)`.
            latents_v = rearrange(latents, "b c v h w -> b v c h w")
            raw_scene = getattr(inputs, "raw_state", None)
            # SceneTok convention: outer wrapper applies `cnd_proj` BEFORE passing
            # `state` into the denoiser. LightningDiT._forward itself never invokes
            # self.cnd_proj — its blocks' `attn2.to_k` expects `y` already in
            # hidden_size (1024) space. So project here using the branch's own
            # ckpt-warm-started `cnd_proj` (64 → 1024).
            if raw_scene is not None:
                projected_scene_ld = self.model.litdit_branch.cnd_proj(raw_scene)
            else:
                projected_scene_ld = None
            ctrl_inputs = DenoiserInputs(
                view=latents_v,
                pose=inputs.pose,
                timestep=inputs.timestep if inputs.timestep.ndim >= 1 else inputs.timestep.unsqueeze(0),
                state=projected_scene_ld,
                text=None,
                condition_latents=None,
            )
            ctrl_pred, _ = self.model.litdit_branch._forward(
                inputs=ctrl_inputs,
                temporal_downsample=temporal_downsample,
                chunk_targets=chunk_targets,
            )
            # ctrl_pred shape `(B, V, C, H, W)` → `(B, C, V, H, W)` to match main.
            ctrl_pred = rearrange(ctrl_pred, "b v c h w -> b c v h w")
            # Stash pre-gate raw ctrl prediction for joint recon loss.
            self._last_ctrl_pred_raw = ctrl_pred
            ctrl_pred = self.model.litdit_output_gate(ctrl_pred)
            pred = main_pred + ctrl_pred
        else:
            pred = simple_wan_video_fn(
                dit=self.model,
                latents=latents,
                timestep=timestep,
                context=context,
                scene_context=scene_context,
                scene_input_type=self.cfg.scene_input_type,
                camera_context=camera_context,
                camera_input_type=self.camera_input_type,
                fuse_vae_embedding_in_latents=False,
                use_gradient_checkpointing=self.cfg.gradient_checkpointing and self.training,
            )
        pred = self._crop_condition_latents_prediction(
            pred,
            condition_latents,
            condition_latents_concat_dim,
            target_latent_shape,
        )
        pred = rearrange(pred, "b c v h w -> b v c h w")
        return pred, None

def simple_wan_video_fn(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    scene_context: torch.Tensor = None,
    scene_input_type: Literal["none", "cross_attention", "new_cross_attention", "latent_concat", "controlnet"] = "cross_attention",
    camera_context: torch.Tensor = None,
    camera_input_type: Literal["none", "recam_attention", "cross_attention", "new_cross_attention", "adaln", "wan_control", "channel_concat", "controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit"] | None = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    if scene_input_type not in ("none", "cross_attention", "new_cross_attention", "latent_concat", "controlnet"):
        raise ValueError(
            f"Unsupported scene_input_type={scene_input_type!r}. "
            "Expected 'none', 'cross_attention', 'new_cross_attention', 'latent_concat', or 'controlnet'."
        )
    if camera_input_type is None:
        camera_input_type = "none"
    if camera_input_type not in ("none", "recam_attention", "cross_attention", "new_cross_attention", "adaln", "wan_control", "channel_concat", "controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit"):
        raise ValueError(
            f"Unsupported camera_input_type={camera_input_type!r}. "
            "Expected 'none', 'recam_attention', 'cross_attention', 'new_cross_attention', 'adaln', 'wan_control', 'channel_concat', 'controlnet', 'controlnet_feedback', 'controlnet_ac3d', or 'controlnet_lightningdit'."
        )

    adaln_camera_embedding = camera_context if camera_input_type == "adaln" else None
    recam_camera_embedding = camera_context if camera_input_type == "recam_attention" else None
    cross_attention_camera_context = camera_context if camera_input_type == "cross_attention" else None
    wan_control_camera_input = camera_context if camera_input_type == "wan_control" else None
    channel_concat_camera_input = camera_context if camera_input_type == "channel_concat" else None
    ac3d_camera_input = camera_context if camera_input_type in ("controlnet", "controlnet_feedback", "controlnet_ac3d", "controlnet_lightningdit") else None
    ac3d_feedback_mode = camera_input_type == "controlnet_feedback"
    ac3d_paper_mode = camera_input_type == "controlnet_ac3d"
    ac3d_litdit_mode = camera_input_type == "controlnet_lightningdit"
    cross_attention_scene_context = None
    ac3d_scene_context = None

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ) * timestep.reshape(()),
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if adaln_camera_embedding is not None:
            adaln_camera_embedding = adaln_camera_embedding.to(dtype=t.dtype, device=t.device)
            t = t + adaln_camera_embedding.unsqueeze(1)
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        if adaln_camera_embedding is not None:
            adaln_camera_embedding = adaln_camera_embedding.to(dtype=t.dtype, device=t.device)
            t = t + adaln_camera_embedding
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        scene_t_mod = None
        if scene_input_type == "new_cross_attention":
            scene_t_mod = dit.scene_time_projection(t).unflatten(1, (3, dit.dim))

    # embedding해서 넣으쇼
    # if context is not None:
    #     context = dit.text_embedding(context)
    if cross_attention_camera_context is not None:
        cross_attention_camera_context = cross_attention_camera_context.to(dtype=latents.dtype, device=latents.device)
        if context is None:
            context = cross_attention_camera_context
        else:
            if context.shape[0] != cross_attention_camera_context.shape[0]:
                raise ValueError(
                    "camera_context batch size must match text context batch size: "
                    f"camera_context={cross_attention_camera_context.shape[0]}, context={context.shape[0]}"
                )
            context = torch.cat([context, cross_attention_camera_context], dim=1)

    if recam_camera_embedding is not None:
        recam_camera_embedding = recam_camera_embedding.to(dtype=latents.dtype, device=latents.device)
        recam_dim = dit.blocks[0].self_attn.q.weight.shape[0]
        if recam_camera_embedding.shape[-1] != recam_dim:
            recam_camera_embedding = dit.recam_camera_encoder(recam_camera_embedding)

    scene_latent_tokens = None
    if scene_context is not None and scene_input_type != "none":
        scene_context = scene_context.to(dtype=latents.dtype, device=latents.device)
        scene_context = dit.embed_scene_context(scene_context)
        if scene_input_type == "new_cross_attention":
            cross_attention_scene_context = scene_context
        elif scene_input_type == "controlnet":
            # scene tokens은 main DiT가 아니라 AC3D 분기의 scene cross-attn으로 주입.
            ac3d_scene_context = scene_context
        elif scene_input_type == "cross_attention":
            if context is None:
                context = scene_context
            else:
                # Broadcast/repeat batch dims so concat works when CFG uncond
                # encodes a single "" prompt (batch=1) while scene_context has
                # the full val/train batch.
                if context.shape[0] != scene_context.shape[0]:
                    if context.shape[0] == 1:
                        context = context.expand(scene_context.shape[0], -1, -1)
                    elif scene_context.shape[0] == 1:
                        scene_context = scene_context.expand(context.shape[0], -1, -1)
                    elif scene_context.shape[0] % context.shape[0] == 0:
                        context = context.repeat_interleave(
                            scene_context.shape[0] // context.shape[0], dim=0
                        )
                    elif context.shape[0] % scene_context.shape[0] == 0:
                        scene_context = scene_context.repeat_interleave(
                            context.shape[0] // scene_context.shape[0], dim=0
                        )
                    else:
                        raise ValueError(
                            "text context and scene_context batch sizes must match or be broadcastable: "
                            f"text={context.shape[0]}, scene={scene_context.shape[0]}"
                        )
                context = torch.cat([context, scene_context], dim=1)
        else:
            scene_latent_tokens = scene_context
    if context is None and cross_attention_scene_context is None:
        raise ValueError("simple_wan_video_fn requires `context` or `scene_context` for cross-attention.")

    x = latents
    conditioning_batch = x.shape[0]
    if context is not None:
        conditioning_batch = max(conditioning_batch, context.shape[0])
    if cross_attention_scene_context is not None:
        conditioning_batch = max(conditioning_batch, cross_attention_scene_context.shape[0])

    if context is not None and context.shape[0] != conditioning_batch:
        if context.shape[0] != 1:
            raise ValueError(
                "context batch size must match latent batch size or be broadcastable: "
                f"context={context.shape[0]}, latents={x.shape[0]}, conditioning_batch={conditioning_batch}"
            )
        context = context.expand(conditioning_batch, -1, -1)
    if cross_attention_scene_context is not None and cross_attention_scene_context.shape[0] != conditioning_batch:
        if cross_attention_scene_context.shape[0] != 1:
            raise ValueError(
                "scene_context batch size must match latent batch size or be broadcastable: "
                f"scene_context={cross_attention_scene_context.shape[0]}, latents={x.shape[0]}, "
                f"conditioning_batch={conditioning_batch}"
            )
        cross_attention_scene_context = cross_attention_scene_context.expand(conditioning_batch, -1, -1)
    if x.shape[0] != conditioning_batch:
        if conditioning_batch % x.shape[0] != 0:
            raise ValueError(
                "latent batch size must divide conditioning batch size: "
                f"latents={x.shape[0]}, conditioning_batch={conditioning_batch}"
            )
        x = torch.cat([x] * (conditioning_batch // x.shape[0]), dim=0)

    if wan_control_camera_input is not None:
        wan_control_camera_input = wan_control_camera_input.to(
            dtype=latents.dtype, device=latents.device
        )
        if wan_control_camera_input.shape[0] != x.shape[0]:
            if wan_control_camera_input.shape[0] == 1:
                wan_control_camera_input = wan_control_camera_input.expand(
                    x.shape[0], -1, -1, -1, -1
                )
            elif x.shape[0] % wan_control_camera_input.shape[0] == 0:
                repeat = x.shape[0] // wan_control_camera_input.shape[0]
                wan_control_camera_input = wan_control_camera_input.repeat_interleave(
                    repeat, dim=0
                )
            else:
                raise ValueError(
                    "wan_control camera input batch must match or divide latent batch: "
                    f"camera={wan_control_camera_input.shape[0]}, latent={x.shape[0]}"
                )
        x = dit.patch_embedding(x)
        y_camera = dit.control_adapter(wan_control_camera_input)
        x = x + y_camera
    elif channel_concat_camera_input is not None:
        # Channel-concat ray map (6 ch) with latent before patch_embedding.
        # patch_embedding was extended to `in_channels + 6` in __init__, with
        # the extra 6 channels zero-init so initial behaviour matches Wan prior.
        channel_concat_camera_input = channel_concat_camera_input.to(
            dtype=latents.dtype, device=latents.device
        )
        if channel_concat_camera_input.shape[0] != x.shape[0]:
            if channel_concat_camera_input.shape[0] == 1:
                channel_concat_camera_input = channel_concat_camera_input.expand(
                    x.shape[0], -1, -1, -1, -1
                )
            elif x.shape[0] % channel_concat_camera_input.shape[0] == 0:
                repeat = x.shape[0] // channel_concat_camera_input.shape[0]
                channel_concat_camera_input = channel_concat_camera_input.repeat_interleave(
                    repeat, dim=0
                )
            else:
                raise ValueError(
                    "channel_concat camera input batch must match or divide latent batch: "
                    f"camera={channel_concat_camera_input.shape[0]}, latent={x.shape[0]}"
                )
        # Both `x` (latent) and ray map are NCFHW; spatial (H, W) and temporal
        # (F) dims must match. If they don't, raise — config has wrong shapes.
        if channel_concat_camera_input.shape[-3:] != x.shape[-3:]:
            raise ValueError(
                "channel_concat ray map shape must match latent (F, H, W). Got "
                f"ray={tuple(channel_concat_camera_input.shape)}, latent={tuple(x.shape)}. "
                "Set `model.denoiser.camera.input_shape=[H_latent, W_latent]` to match."
            )
        x = torch.cat([x, channel_concat_camera_input], dim=1)
        x = dit.patch_embedding(x)
    else:
        x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    scene_latent_token_count = 0
    if scene_latent_tokens is not None:
        if scene_latent_tokens.shape[0] != x.shape[0]:
            if x.shape[0] % scene_latent_tokens.shape[0] != 0:
                raise ValueError(
                    "scene_context batch size must match or divide the latent batch size: "
                    f"scene_context={scene_latent_tokens.shape[0]}, latents={x.shape[0]}"
                )
            repeat_count = x.shape[0] // scene_latent_tokens.shape[0]
            scene_latent_tokens = torch.cat([scene_latent_tokens] * repeat_count, dim=0)
        scene_latent_token_count = scene_latent_tokens.shape[1]
        x = torch.cat([x, scene_latent_tokens], dim=1)

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)
    # Preserve a copy of freqs without scene_latent_tokens for any side branch
    # (e.g. AC3D controlnet) that runs on the same (f, h, w) sequence.
    ac3d_freqs = freqs
    if scene_latent_token_count:
        scene_freqs = torch.ones(
            scene_latent_token_count,
            *freqs.shape[1:],
            dtype=freqs.dtype,
            device=freqs.device,
        )
        freqs = torch.cat([freqs, scene_freqs], dim=0)

    # AC3D ControlNet branch: 세 가지 모드 지원
    #   `controlnet`        — parallel: ac3d branch가 (latent⊕ray) end-to-end forward,
    #                         per-layer 잔차 미리 수집해서 main block 직후 add.
    #   `controlnet_feedback` — 인터리브: ac3d block_i 입력 = ac3d_x + FC_down(main_x),
    #                         그 output을 main block_i 직전에 잔차로 add.
    #   `controlnet_ac3d`   — AC3D 논문 VDiT-CC: noise는 main에만, ctrl block_i 입력 =
    #                         FC_main2ctrl(main_x_i) + cam_emb + ctrl_x_{i-1},
    #                         output → zero-init proj → main_x_i에 add → main block_{i+1} 입력.
    #                         ctrl block 내부엔 self_attn + scene_cross_attn + FFN만 (text 없음).
    ac3d_residuals = None
    ac3d_x = None
    ac3d_scene = None
    ac3d_cam_emb = None  # controlnet_ac3d 전용 per-token camera embedding
    if ac3d_camera_input is not None and getattr(dit, "ac3d_blocks", None) is not None:
        ac3d_camera_input = ac3d_camera_input.to(dtype=latents.dtype, device=latents.device)
        if ac3d_camera_input.shape[0] != latents.shape[0]:
            if ac3d_camera_input.shape[0] == 1:
                ac3d_camera_input = ac3d_camera_input.expand(latents.shape[0], -1, -1, -1, -1)
            elif latents.shape[0] % ac3d_camera_input.shape[0] == 0:
                ac3d_camera_input = ac3d_camera_input.repeat_interleave(
                    latents.shape[0] // ac3d_camera_input.shape[0], dim=0
                )
            else:
                raise ValueError(
                    "ac3d camera input batch must match or divide latent batch: "
                    f"camera={ac3d_camera_input.shape[0]}, latent={latents.shape[0]}"
                )
        if ac3d_camera_input.shape[-3:] != latents.shape[-3:]:
            raise ValueError(
                "ac3d ray map shape must match latent (F, H, W). "
                f"Got ray={tuple(ac3d_camera_input.shape)}, latent={tuple(latents.shape)}. "
                "Set `model.denoiser.camera.input_shape=[H_latent, W_latent]`."
            )

        if ac3d_paper_mode or ac3d_litdit_mode:
            # AC3D paper / lightningdit 변종: ray만 patchify해서 per-token camera
            # embedding 생성. latent은 ctrl 분기에 직접 들어가지 않음.
            cam_emb_3d = dit.ac3d_camera_patch_embed(ac3d_camera_input)
            if cam_emb_3d.shape[0] != x.shape[0]:
                if x.shape[0] % cam_emb_3d.shape[0] == 0:
                    cam_emb_3d = torch.cat([cam_emb_3d] * (x.shape[0] // cam_emb_3d.shape[0]), dim=0)
            ac3d_cam_emb = rearrange(cam_emb_3d, "b c f h w -> b (f h w) c").contiguous()
            ref_branch = ac3d_cam_emb
        else:
            # parallel / feedback 모드: (latent ⊕ ray) → ac3d_patch_embedding → ac3d_x
            ac3d_input = torch.cat([latents, ac3d_camera_input], dim=1)
            ac3d_x = dit.ac3d_patch_embedding(ac3d_input)
            # Replicate batch broadcast that vanilla path does to x (line ~913).
            if ac3d_x.shape[0] != x.shape[0]:
                if x.shape[0] % ac3d_x.shape[0] == 0:
                    ac3d_x = torch.cat([ac3d_x] * (x.shape[0] // ac3d_x.shape[0]), dim=0)
            ac3d_x = rearrange(ac3d_x, "b c f h w -> b (f h w) c").contiguous()
            ref_branch = ac3d_x

        # scene_input_type="controlnet": scene tokens → ac3d 블록의 scene cross-attn.
        # batch가 ref_branch와 안 맞으면 broadcast/repeat (CFG uncond 등).
        ac3d_scene = ac3d_scene_context
        if ac3d_scene is not None:
            ac3d_scene = ac3d_scene.to(dtype=ref_branch.dtype, device=ref_branch.device)
            if ac3d_scene.shape[0] != ref_branch.shape[0]:
                if ac3d_scene.shape[0] == 1:
                    ac3d_scene = ac3d_scene.expand(ref_branch.shape[0], -1, -1)
                elif ref_branch.shape[0] % ac3d_scene.shape[0] == 0:
                    ac3d_scene = ac3d_scene.repeat_interleave(
                        ref_branch.shape[0] // ac3d_scene.shape[0], dim=0
                    )
                else:
                    raise ValueError(
                        "ac3d scene_context batch must match or divide ac3d branch batch: "
                        f"scene={ac3d_scene.shape[0]}, ref_branch={ref_branch.shape[0]}"
                    )
        if not ac3d_feedback_mode and not ac3d_paper_mode and not ac3d_litdit_mode:
            # Parallel mode (AC3D公개 코드 / CogVideoX-ControlNet style):
            # ac3d branch runs end-to-end first, residuals collected, then
            # added to main blocks per-layer.
            ac3d_residuals = []
            for ac3d_block, ac3d_proj in zip(dit.ac3d_blocks, dit.ac3d_projectors):
                ac3d_x = ac3d_block(
                    ac3d_x, context, ac3d_scene, t_mod, None, ac3d_freqs, None
                )
                ac3d_residuals.append(ac3d_proj(ac3d_x))
        # feedback / ac3d_paper mode → ac3d block 실행은 main loop 안에서 처리

    ctrl_x = None  # controlnet_ac3d carry state
    for block_id, block in enumerate(dit.blocks):
        # Feedback mode: ac3d block i takes (ac3d_x + FC_down(main_x)) as input,
        # then its projected output is added to main_x BEFORE main block i.
        if (
            ac3d_feedback_mode
            and ac3d_x is not None
            and block_id < len(dit.ac3d_blocks)
        ):
            ac3d_n_patches = ac3d_x.shape[1]
            # Slice main patch tokens (exclude scene_latent_tokens appended at end)
            main_patches = x[:, :ac3d_n_patches]
            feedback = dit.ac3d_fc_down[block_id](main_patches)
            ac3d_x = dit.ac3d_blocks[block_id](
                ac3d_x + feedback,
                context,
                ac3d_scene,
                t_mod,
                None,
                ac3d_freqs,
                None,
            )
            res = dit.ac3d_projectors[block_id](ac3d_x)
            if x.shape[1] == ac3d_n_patches:
                x = x + res
            else:
                x = torch.cat([x[:, :ac3d_n_patches] + res, x[:, ac3d_n_patches:]], dim=1)

        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x, context, cross_attention_scene_context, t_mod, scene_t_mod, freqs, recam_camera_embedding
        )

        # Parallel mode: residual added AFTER main block. interval injection 시
        # ctrl residual j는 main block ac3d_inject_ids[j]에 더해짐 (VACE식 분산).
        _inject_ids = getattr(dit, "ac3d_inject_ids", None)
        _ridx = (_inject_ids.index(block_id) if (_inject_ids is not None and block_id in _inject_ids)
                 else (block_id if _inject_ids is None else None))
        if (
            not ac3d_feedback_mode
            and not ac3d_paper_mode
            and not ac3d_litdit_mode
            and ac3d_residuals is not None
            and _ridx is not None
            and _ridx < len(ac3d_residuals)
        ):
            res = ac3d_residuals[_ridx]
            # Main x may have scene_latent_tokens appended → only add residual
            # to the patch-token portion (first `n_patches` tokens).
            n_patches = res.shape[1]
            if x.shape[1] == n_patches:
                x = x + res
            else:
                x = torch.cat([x[:, :n_patches] + res, x[:, n_patches:]], dim=1)

        # AC3D paper mode: main block forward 직후 → ctrl block_i 실행 → 잔차 add.
        #   ctrl_input = FC_main2ctrl(main_x_i) + cam_emb + ctrl_x_{i-1}
        #   ctrl_x_i   = ctrl_block_i(ctrl_input, scene_tokens, t_mod, ac3d_freqs)
        #               (self_attn → scene_cross_attn → FFN; text cross-attn 없음)
        #   res        = ac3d_projectors[i](ctrl_x_i)   ← zero-init
        #   main_x     = main_x + res                    → main block_{i+1} 입력
        if (
            ac3d_paper_mode
            and ac3d_cam_emb is not None
            and block_id < len(dit.ac3d_blocks)
        ):
            n_patches_ac = ac3d_cam_emb.shape[1]
            main_for_ctrl = dit.ac3d_main2ctrl[block_id](x[:, :n_patches_ac])
            ctrl_input = main_for_ctrl + ac3d_cam_emb
            if ctrl_x is not None:
                ctrl_input = ctrl_input + ctrl_x
            ctrl_x = dit.ac3d_blocks[block_id](
                ctrl_input,
                None,  # text context — use_text_cross_attn=False라 무시됨
                ac3d_scene,
                t_mod,
                None,
                ac3d_freqs,
                None,
            )
            res = dit.ac3d_projectors[block_id](ctrl_x)
            if x.shape[1] == n_patches_ac:
                x = x + res
            else:
                x = torch.cat([x[:, :n_patches_ac] + res, x[:, n_patches_ac:]], dim=1)

        # controlnet_lightningdit mode: AC3D paper orchestration이지만 ctrl block이
        # LightningDiTBlock. 추가 차이:
        #   - dim adapter (`ac3d_main2ctrl`: wan_dim→ld_dim, `ac3d_projectors`: ld_dim→wan_dim).
        #   - Scene tokens은 `ac3d_cnd_proj`로 ld_dim에 투사 (한 번 계산 후 layer 간 재사용).
        #   - Timestep cond: LightningDiT의 `ac3d_t_embedder(timestep) → (B, 1, ld_dim)` 별도 계산.
        #   - LightningDiTBlock signature: (x, c, y, feat_rope, num_views, num_split).
        #     feat_rope=None (사전학습된 RoPE patch grid가 우리 Wan grid와 달라 skip).
        if (
            ac3d_litdit_mode
            and ac3d_cam_emb is not None
            and block_id < len(dit.ac3d_blocks)
        ):
            n_patches_ac = ac3d_cam_emb.shape[1]
            main_for_ctrl = dit.ac3d_main2ctrl[block_id](x[:, :n_patches_ac])
            ctrl_input = main_for_ctrl + ac3d_cam_emb
            if ctrl_x is not None:
                ctrl_input = ctrl_input + ctrl_x
            # LightningDiT가 요구하는 conditioning 벡터 c — timestep만 (class label 없음)
            if not hasattr(dit, "_ac3d_litdit_c_cache") or dit._ac3d_litdit_c_cache is None or dit._ac3d_litdit_c_cache.shape[0] != ctrl_input.shape[0]:
                # ac3d_t_embedder는 (B,) → (B, 1, ld_dim) 반환
                c_litdit = dit.ac3d_t_embedder(timestep.to(ctrl_input.device))
                dit._ac3d_litdit_c_cache = c_litdit
            else:
                c_litdit = dit._ac3d_litdit_c_cache
            # Scene tokens을 ctrl dim에 투사 — 한 번 계산해서 layer마다 재사용
            if not hasattr(dit, "_ac3d_litdit_y_cache") or dit._ac3d_litdit_y_cache is None:
                if ac3d_scene is not None:
                    y_litdit = dit.ac3d_cnd_proj(ac3d_scene)
                else:
                    y_litdit = None
                dit._ac3d_litdit_y_cache = y_litdit
            else:
                y_litdit = dit._ac3d_litdit_y_cache
            ctrl_x = dit.ac3d_blocks[block_id](
                ctrl_input,
                c_litdit,
                y_litdit,
                feat_rope=None,
                num_views=1,
                num_split=1,
            )
            res = dit.ac3d_projectors[block_id](ctrl_x)
            if x.shape[1] == n_patches_ac:
                x = x + res
            else:
                x = torch.cat([x[:, :n_patches_ac] + res, x[:, n_patches_ac:]], dim=1)

    # litdit cache 정리 (다음 forward 호출에 carry되지 않도록)
    if ac3d_litdit_mode:
        if hasattr(dit, "_ac3d_litdit_c_cache"):
            dit._ac3d_litdit_c_cache = None
        if hasattr(dit, "_ac3d_litdit_y_cache"):
            dit._ac3d_litdit_y_cache = None

    x = BatchHead.forward(dit.head, x, t)
    if scene_latent_token_count:
        x = x[:, :-scene_latent_token_count]
    x = dit.unpatchify(x, (f, h, w))
    return x
