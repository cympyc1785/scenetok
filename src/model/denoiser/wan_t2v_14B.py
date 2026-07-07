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

from .recam_wan import install_recam_attention

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@dataclass
class WanT2V14BLoRACfg:
    enabled: bool = False
    rank: int = 32
    alpha: int | None = None
    target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    checkpoint: str | Path | None = None


@dataclass
class WanT2V14BCfg:
    name: Literal["wan_t2v_14b"]
    camera: CameraCfg | None = None
    model_root: str | Path = Path("src/model/DiffSynth-Studio/Wan2.2/Wan2.2-T2V-A14B")
    high_noise_dit_pattern: str = "high_noise_model/diffusion_pytorch_model*.safetensors"
    low_noise_dit_pattern: str = "low_noise_model/diffusion_pytorch_model*.safetensors"
    dit_pattern: str = "diffusion_pytorch_model*.safetensors"
    switch_dit_boundary: float = 0.875
    text_encoder_path: str | Path | None = None
    tokenizer_path: str | Path | None = None
    seq_len: int = 512
    clean: str = "whitespace"
    gradient_checkpointing: bool = True
    condition_latents_input_type: Literal["none", "width", "channel", "temporal", "first_frame", "first_frame_random"] = "none"
    camera_input_type: Literal["none", "recam_attention", "cross_attention", "adaln"] | None = None
    enable_recam_attention: bool | None = None
    camera_context_spatial_pool: int = 1
    scene_input_type: Literal["none", "cross_attention", "latent_concat"] = "cross_attention"
    num_target_split: int = 1
    input_shape: int | list[int] = 16
    noise_seed: int | None = None
    ckpt_path: str | Path | None = None
    load_strict: bool = True
    lora: WanT2V14BLoRACfg = field(default_factory=WanT2V14BLoRACfg)


class BatchHead:
    @staticmethod
    def forward(head: nn.Module, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if t_mod.ndim != 2:
            return head(x, t_mod)

        modulation = head.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift, scale = (modulation + t_mod.unsqueeze(1)).chunk(2, dim=1)
        x = head.norm(x) * (1 + scale) + shift
        return head.head(x)


class WanT2V14BDenoiser(Denoiser[WanT2V14BCfg]):
    def __init__(
        self,
        cfg: WanT2V14BCfg,
        cond_dim: int | None = 1,
        num_scene_tokens: int = 1,
        temporal_downsample: int = 1,
        using_wan: bool = False,
        **_: object,
    ) -> None:
        super().__init__(cfg)
        self.model_root = Path(cfg.model_root)
        self.high_noise_model = self._load_model("wan_video_dit", self._resolve_dit_paths("high"))
        self.low_noise_model = self._load_model("wan_video_dit", self._resolve_dit_paths("low"))
        self.supports_scene_tokens = True
        self.supports_condition_latents = True
        self.condition_latents_input_type = cfg.condition_latents_input_type
        self.supports_per_view_timestep = False
        self.uses_internal_text_encoder = True
        self.camera_input_type = getattr(cfg, "camera_input_type", None)
        if self.camera_input_type is None and cfg.enable_recam_attention is not None:
            self.camera_input_type = "recam_attention" if cfg.enable_recam_attention else "none"
        self.num_scene_tokens = num_scene_tokens
        self.cond_dim = 1 if cond_dim is None else cond_dim
        self.text_embed_dim = 4096
        self.cnd_proj = nn.Linear(self.cond_dim, self.model.dim)
        self.null_tokens = nn.Parameter(torch.zeros(1, 1, self.model.dim))
        self.text_proj = None
        self.pose_embed = None
        self.negative_prompt = None
        if cfg.camera is not None:
            camera_cfg = deepcopy(cfg.camera)
            if self.camera_input_type == "cross_attention":
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
        if self.camera_input_type == "cross_attention":
            for model in self._iter_dit_models():
                model.camera_embedding = nn.Sequential(
                    nn.Linear(self.text_embed_dim, model.dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(model.dim, model.dim),
                ).to(device=ref_param.device, dtype=ref_param.dtype)
        if self.camera_input_type == "recam_attention":
            for model in self._iter_dit_models():
                install_recam_attention(model, camera_dim=12)
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

    @property
    def model(self) -> WanModel:
        return self.high_noise_model

    def _iter_dit_models(self) -> tuple[nn.Module, nn.Module]:
        return self.high_noise_model, self.low_noise_model

    def _resolve_dit_paths(self, noise_level: Literal["high", "low"] | None = None) -> list[str]:
        if noise_level == "high":
            pattern = self.cfg.high_noise_dit_pattern
        elif noise_level == "low":
            pattern = self.cfg.low_noise_dit_pattern
        else:
            pattern = self.cfg.dit_pattern
        paths = sorted(glob.glob(str(self.model_root / pattern)))
        if not paths:
            raise FileNotFoundError(
                f"Could not find Wan T2V 14B weights under {self.model_root / pattern}"
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

    def _enable_lora(self, lora_cfg: WanT2V14BLoRACfg) -> None:
        target_modules = [name.strip() for name in lora_cfg.target_modules.split(",") if name.strip()]
        lora_alpha = lora_cfg.alpha if lora_cfg.alpha is not None else lora_cfg.rank
        config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
        )
        self.high_noise_model = inject_adapter_in_model(config, self.high_noise_model)
        self.low_noise_model = inject_adapter_in_model(deepcopy(config), self.low_noise_model)
        for model in self._iter_dit_models():
            for name, param in model.named_parameters():
                param.requires_grad = "lora_" in name or "recam_" in name

        if lora_cfg.checkpoint is not None:
            state_dict = load_state_dict(str(lora_cfg.checkpoint))
            state_dict = self._map_lora_state_dict(state_dict)
            self.high_noise_model.load_state_dict(state_dict, strict=False)
            self.low_noise_model.load_state_dict(state_dict, strict=False)

    def _set_trainable_parameters(self) -> None:
        for model in self._iter_dit_models():
            for param in model.parameters():
                param.requires_grad = False

        trainable_model_substrings = ("recam_projector", "recam_camera_encoder", "lora_")
        for model in self._iter_dit_models():
            for name, param in model.named_parameters():
                trainable = any(key in name for key in trainable_model_substrings)
                trainable = trainable or (
                    self.camera_input_type == "recam_attention"
                    and ".self_attn." in name
                )
                if not self.cfg.lora.enabled:
                    trainable = trainable or name.startswith("text_embedding.")
                # trainable = trainable or name.startswith("scene_embedding.")
                trainable = trainable or (
                    self.camera_input_type == "cross_attention"
                    and name.startswith("camera_embedding.")
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
            elif ".self_attn." in name:
                group_name = "model.self_attn"
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

        print(cyan("\n\n[WanT2V14B] Trainable modules:"))
        for group_name in sorted(grouped):
            stats = grouped[group_name]
            print(f"  - {group_name}: {stats['params']:,} params across {stats['tensors']} tensors")
        print(cyan(f"[WanT2V14B] Total trainable: {total_params:,} params across {total_tensors} tensors\n\n"))

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
        return text_state.to(dtype=next(self.model.parameters()).dtype)

    def load_weights(
        self,
        path: Path | str,
        **kwargs,
    ):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        high_prefix = "denoiser.high_noise_model."
        low_prefix = "denoiser.low_noise_model."
        if any(key.startswith(high_prefix) or key.startswith(low_prefix) for key in state_dict):
            high_state_dict = {
                key.replace(high_prefix, "", 1): value
                for key, value in state_dict.items()
                if key.startswith(high_prefix)
            }
            low_state_dict = {
                key.replace(low_prefix, "", 1): value
                for key, value in state_dict.items()
                if key.startswith(low_prefix)
            }
            if high_state_dict:
                self.high_noise_model.load_state_dict(high_state_dict, **kwargs)
            if low_state_dict:
                self.low_noise_model.load_state_dict(low_state_dict, **kwargs)
            return
        if any(key.startswith("denoiser.model.") for key in state_dict):
            state_dict = {
                key.replace("denoiser.model.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("denoiser.model.")
            }
        self.high_noise_model.load_state_dict(state_dict, **kwargs)
        self.low_noise_model.load_state_dict(state_dict, **kwargs)

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
            return rearrange(camera_embedding, "b v c h w -> b v (h w) c")
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
        if condition_latents is None or self.condition_latents_input_type in ("none", "first_frame", "first_frame_random"):
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

    def _slice_batch_tensor(
        self,
        tensor: torch.Tensor | None,
        indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.shape[0] == batch_size:
            return tensor[indices]
        return tensor

    def _run_expert(
        self,
        model: WanModel,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None,
        scene_context: torch.Tensor | None,
        camera_context: torch.Tensor | None,
        condition_latents: torch.Tensor | None,
        condition_latents_concat_dim: int | None,
        target_latent_shape: torch.Size,
    ) -> torch.Tensor:
        pred = simple_wan_video_fn(
            dit=model,
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
        return self._crop_condition_latents_prediction(
            pred,
            condition_latents,
            condition_latents_concat_dim,
            target_latent_shape,
        )

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
        if context is None and self.cfg.scene_input_type != "cross_attention":
            context = self.null_tokens.expand(latents.shape[0], -1, -1).to(
                device=latents.device,
                dtype=latents.dtype,
            )

        camera_embedding = self._get_camera_embedding(inputs, temporal_downsample, chunk_targets)
        camera_context = self._get_camera_context(camera_embedding)

        boundary = self.cfg.switch_dit_boundary * 1000
        high_noise_mask = timestep >= boundary
        if bool(high_noise_mask.all()):
            pred = self._run_expert(
                self.high_noise_model,
                latents,
                timestep,
                context,
                scene_context,
                camera_context,
                condition_latents,
                condition_latents_concat_dim,
                target_latent_shape,
            )
        elif not bool(high_noise_mask.any()):
            pred = self._run_expert(
                self.low_noise_model,
                latents,
                timestep,
                context,
                scene_context,
                camera_context,
                condition_latents,
                condition_latents_concat_dim,
                target_latent_shape,
            )
        else:
            batch_size = latents.shape[0]
            if context is not None and context.shape[0] != batch_size:
                raise ValueError(
                    "WanT2V14BDenoiser cannot split mixed high/low timestep batches "
                    f"when context batch differs from latent batch: context={context.shape}, latents={latents.shape}"
                )
            pred = torch.empty(target_latent_shape, dtype=latents.dtype, device=latents.device)
            for model, mask in (
                (self.high_noise_model, high_noise_mask),
                (self.low_noise_model, ~high_noise_mask),
            ):
                indices = mask.nonzero(as_tuple=True)[0]
                subset_target_shape = torch.Size([indices.shape[0], *target_latent_shape[1:]])
                subset_pred = self._run_expert(
                    model,
                    latents[indices],
                    timestep[indices],
                    self._slice_batch_tensor(context, indices, batch_size),
                    self._slice_batch_tensor(scene_context, indices, batch_size),
                    self._slice_batch_tensor(camera_context, indices, batch_size),
                    self._slice_batch_tensor(condition_latents, indices, batch_size),
                    condition_latents_concat_dim,
                    subset_target_shape,
                )
                pred[indices] = subset_pred
        pred = rearrange(pred, "b c v h w -> b v c h w")
        return pred, None

def simple_wan_video_fn(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    scene_context: torch.Tensor = None,
    scene_input_type: Literal["none", "cross_attention", "latent_concat"] = "cross_attention",
    camera_context: torch.Tensor = None,
    camera_input_type: Literal["none", "recam_attention", "cross_attention", "adaln"] | None = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    if scene_input_type not in ("none", "cross_attention", "latent_concat"):
        raise ValueError(
            f"Unsupported scene_input_type={scene_input_type!r}. "
            "Expected 'none', 'cross_attention', or 'latent_concat'."
        )
    if camera_input_type is None:
        camera_input_type = "none"
    if camera_input_type not in ("none", "recam_attention", "cross_attention", "adaln"):
        raise ValueError(
            f"Unsupported camera_input_type={camera_input_type!r}. "
            "Expected 'none', 'recam_attention', 'cross_attention', or 'adaln'."
        )

    adaln_camera_embedding = camera_context if camera_input_type == "adaln" else None
    recam_camera_embedding = camera_context if camera_input_type == "recam_attention" else None
    cross_attention_camera_context = camera_context if camera_input_type == "cross_attention" else None

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

    if context is not None and context.shape[-1] != dit.dim:
        context = dit.text_embedding(context)
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
        # `embed_scene_context` was never defined on WanModel — scene tokens are
        # already projected to model.dim by the wrapper's `cnd_proj`. Identity
        # unless a model defines its own projection (mirrors wan_ti2v.py).
        if hasattr(dit, "embed_scene_context"):
            scene_context = dit.embed_scene_context(scene_context)
        if scene_input_type == "cross_attention":
            if context is None:
                context = scene_context
            else:
                context = torch.cat([context, scene_context], dim=1)
        else:
            scene_latent_tokens = scene_context
    if context is None:
        raise ValueError("simple_wan_video_fn requires `context` for cross-attention.")

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)

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
    if scene_latent_token_count:
        scene_freqs = torch.ones(
            scene_latent_token_count,
            *freqs.shape[1:],
            dtype=freqs.dtype,
            device=freqs.device,
        )
        freqs = torch.cat([freqs, scene_freqs], dim=0)

    for block_id, block in enumerate(dit.blocks):
        if recam_camera_embedding is not None:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs, recam_camera_embedding
            )
        else:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs
            )

    x = BatchHead.forward(dit.head, x, t)
    if scene_latent_token_count:
        x = x[:, :-scene_latent_token_count]
    x = dit.unpatchify(x, (f, h, w))
    return x
