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
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video import model_fn_wan_video

from .recam_wan import install_recam_attention, set_recam_camera_embedding, clear_recam_camera_embedding


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
    lock_first_frame: bool = True
    use_condition_latents: bool = True
    enable_recam_attention: bool = True
    num_target_split: int = 1
    input_shape: int | list[int] = 16
    ckpt_path: str | Path | None = None
    load_strict: bool = True
    lora: WanTI2VLoRACfg = field(default_factory=WanTI2VLoRACfg)


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
        self.supports_scene_tokens = True
        self.supports_condition_latents = True
        self.use_condition_latents = cfg.use_condition_latents
        self.lock_condition_frame = cfg.lock_first_frame
        self.supports_per_view_timestep = False
        self.uses_internal_text_encoder = True
        self.enable_recam_attention = cfg.enable_recam_attention
        self.num_scene_tokens = num_scene_tokens
        self.cond_dim = 1 if cond_dim is None else cond_dim
        self.text_embed_dim = 4096
        self.cnd_proj = nn.Linear(self.cond_dim, self.text_embed_dim)
        self.null_tokens = nn.Parameter(torch.zeros(1, 1, self.text_embed_dim))
        self.text_proj = None
        self.pose_embed = None
        if cfg.camera is not None:
            camera_cfg = deepcopy(cfg.camera)
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
                embed_dim=self.text_embed_dim,
                temporal_downsample=temporal_downsample,
            )

        self.model = self._load_model("wan_video_dit", self._resolve_dit_paths())
        self.model.scene_embedding = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.model.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.model.dim, self.model.dim),
        )
        if self.enable_recam_attention:
            install_recam_attention(self.model)
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

        if cfg.lora.enabled:
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

        trainable_model_substrings = ("recam_3d_attn", "lora_")
        for name, param in self.model.named_parameters():
            trainable = any(key in name for key in trainable_model_substrings)
            if not self.cfg.lora.enabled:
                trainable = trainable or name.startswith("text_embedding.")
            trainable = trainable or name.startswith("scene_embedding.")
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
            elif "recam_3d_attn" in name:
                group_name = "model.recam_3d_attn"
            elif name.startswith("model.scene_embedding."):
                group_name = "model.scene_embedding"
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

        print("[WanTI2V] Trainable modules:")
        for group_name in sorted(grouped):
            stats = grouped[group_name]
            print(f"  - {group_name}: {stats['params']:,} params across {stats['tensors']} tensors")
        print(f"[WanTI2V] Total trainable: {total_params:,} params across {total_tensors} tensors")

    def encode_text_condition(self, text: str | list[str], device: torch.device) -> torch.Tensor | None:
        if text is None:
            return None
        ids, mask = self.text_tokenizer(text, return_mask=True)
        ids = ids.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            text_state = self.text_encoder(ids, mask)
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

    def _forward(
        self,
        inputs: DenoiserInputs,
        **kwargs,
    ):
        temporal_downsample = kwargs.get("temporal_downsample", 1)
        latents = rearrange(inputs.view, "b v c h w -> b c v h w")
        timestep = inputs.timestep
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])
        elif timestep.ndim > 1:
            timestep = timestep[:, 0]

        if inputs.condition_latents is not None:
            condition_latents = rearrange(inputs.condition_latents, "b v c h w -> b c v h w")
            latents = latents.clone()
            latents[:, :, : condition_latents.shape[2]] = condition_latents

        context = inputs.text
        scene_context = inputs.state
        
        if self.enable_recam_attention:
            camera_embedding = None
            if self.pose_embed is not None and inputs.pose is not None:
                pose_tokens = self.pose_embed(inputs.pose, temporal_downsample=temporal_downsample)
                if pose_tokens.shape[1] != inputs.view.shape[1]:
                    raise ValueError(
                        "Shape mismatch",
                        inputs.pose.extrinsics.shape,
                        pose_tokens.shape,
                        inputs.view.shape,
                    )
                camera_embedding = rearrange(pose_tokens, "b v c h w -> b (v h w) c")
            
            set_recam_camera_embedding(self.model, camera_embedding)
        pred = simple_wan_video_fn(
            dit=self.model,
            latents=latents,
            timestep=timestep,
            context=context,
            scene_context=scene_context,
            fuse_vae_embedding_in_latents=False,
            use_gradient_checkpointing=self.cfg.gradient_checkpointing and self.training,
        )
        if self.enable_recam_attention:
            clear_recam_camera_embedding(self.model)
        pred = rearrange(pred, "b c v h w -> b v c h w")
        return pred, None

def simple_wan_video_fn(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    scene_context: torch.Tensor = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
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
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if context is not None:
        context = dit.text_embedding(context)
    if scene_context is not None:
        scene_context = scene_context.to(dtype=latents.dtype, device=latents.device)
        scene_context = dit.embed_scene_context(scene_context)
        if context is None:
            context = scene_context
        else:
            context = torch.cat([context, scene_context], dim=1)
    if context is None:
        raise ValueError("simple_wan_video_fn requires `context` or `extra_context`.")

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)

    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    for block_id, block in enumerate(dit.blocks):
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x, context, t_mod, freqs
        )

    x = dit.head(x, t.unsqueeze(1))
    x = dit.unpatchify(x, (f, h, w))
    return x
