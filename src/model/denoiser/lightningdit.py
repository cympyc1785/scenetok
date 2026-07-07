
import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from functools import reduce 
from dataclasses import dataclass
from jaxtyping import Float
from typing import Literal,Union, Tuple
from einops import rearrange

from .denoiser import Denoiser
from .layers.lightningdit import LitDiT
from ..types import DenoiserInputs
from ..camera import CameraCfg, get_camera
from ...misc.torch_utils import pop_state_dict_by_prefix

MODEL_PREFIX = "denoiser."

def zero_initialize(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.zeros_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)

@dataclass
class LightningDiTKwargsCfg:
    patch_size: int=1
    in_channels: int=32
    hidden_size: int=1152
    depth: int=28
    num_heads: int=16
    mlp_ratio: float=4.0
    class_dropout_prob: float=0.1
    num_classes: int=1000
    learn_sigma: bool=False
    use_qknorm: bool=False
    use_swiglu: bool=True
    use_rope: bool=True
    use_rope_3d: bool=True
    use_rmsnorm: bool=True
    wo_shift: bool=False
    frequency_embedding_size: int=256

@dataclass
class LightningDiTCfg:


    name: Literal["lightningdit"]
    
    camera: CameraCfg
    kwargs: LightningDiTKwargsCfg
    single_dim_tokens: bool=False
    num_target_split: int=1
    camera_conditioning: Literal["add", "adaLN"] = "adaLN"
    input_shape: Union[int, Tuple[int], list[int]]= 16
    # RoPE training grid for position interpolation at non-native resolution.
    # None → no interpolation (RoPE built at input_shape grid, original). Set to
    # the TRAINED input_shape (e.g. [8,8]) when running at a larger input_shape.
    rope_pt_input_shape: Union[int, Tuple[int], list[int], None] = None
    gradient_checkpointing: bool=False
    pretrained_from: str | Path | None = None
    ckpt_path: str | Path | None = None
    load_strict: bool=True
    causal_attention: bool=False
    text_cond_dim: int | None = None
    # (①) Channel-concat the raw Plücker target ray map (6*temporal_downsample ch,
    # generated at the LATENT grid = input_shape) onto the noisy latent before the
    # patch-embed conv — paper-faithful (tgt_embedder concats 6ch ray + noisy image)
    # and mirrors wan_ti2v's `channel_concat`. x_embedder input channels are inflated
    # by 6*td, EXTRA channels zero-init (orig channels kept) so at init the concat ray
    # contributes 0 → warm-start preserved. The existing lvsm-adaLN ray is kept too.
    ray_channel_concat: bool = False
    # (②) Per-view 2D RoPE on the scene cross-attn (attn2): scene-token keys are
    # rotated by their (context-view, h_c, w_c) grid position and the target queries
    # by their (target-frame, h_t, w_t) grid, restoring the spatial structure lost when
    # rec_tokens were flattened (V*P). Requires `scene_rope_grid=[V_ctx, h_c, w_c]`.
    # Replaces the 3D feat_rope on the attn2 query (attn1 keeps 3D). Default off.
    scene_2d_rope: bool = False
    scene_rope_grid: Union[Tuple[int], list[int], None] = None  # [V_ctx, h_c, w_c]
class LightningDiT(Denoiser[LightningDiTCfg]):
    def __init__(
        self, 
        cfg: LightningDiTCfg,
        cond_dim: int | None=16,
        num_scene_tokens: int=256,
        num_views: int=8,
        temporal_downsample: int=1,
        using_wan: bool=False,
        cfg_train: bool=False,
        **kwargs
    ) -> None:
        super().__init__(cfg)
        self.pretrained_from = cfg.pretrained_from
        self.num_scene_tokens = num_scene_tokens
        self.cond_dim = cond_dim
        inner_dim = cfg.kwargs.hidden_size
        num_split = cfg.num_target_split
        self.pose_embed = get_camera(cfg.camera, num_split=num_split, using_wan=using_wan, embed_dim=cfg.kwargs.hidden_size, temporal_downsample=temporal_downsample)
        print("(Denoiser) Using gradient checkpointing: ", cfg.gradient_checkpointing)

        self.model = LitDiT(
            input_size=cfg.input_shape,
            patch_size=cfg.kwargs.patch_size,
            in_channels=cfg.kwargs.in_channels,
            hidden_size=cfg.kwargs.hidden_size,
            depth=cfg.kwargs.depth,
            num_heads=cfg.kwargs.num_heads,
            mlp_ratio=cfg.kwargs.mlp_ratio,
            class_dropout_prob=cfg.kwargs.class_dropout_prob,
            num_classes=cfg.kwargs.num_classes,
            learn_sigma=cfg.kwargs.learn_sigma,
            use_qknorm=cfg.kwargs.use_qknorm,
            use_swiglu=cfg.kwargs.use_swiglu,
            use_rope=cfg.kwargs.use_rope,
            use_rope_3d=cfg.kwargs.use_rope_3d,
            use_rmsnorm=cfg.kwargs.use_rmsnorm,
            wo_shift=cfg.kwargs.wo_shift,
            use_checkpoint=cfg.gradient_checkpointing,
            frequency_embedding_size=cfg.kwargs.frequency_embedding_size,
            num_views=num_views,
            num_split=num_split,
            causal_attention=cfg.causal_attention,
            rope_pt_input_size=cfg.rope_pt_input_shape,
            scene_2d_rope=getattr(cfg, "scene_2d_rope", False),
            scene_rope_grid=getattr(cfg, "scene_rope_grid", None),
        )
        self.ray_channel_concat = getattr(cfg, "ray_channel_concat", False)
        self.temporal_downsample = temporal_downsample
        if self.ray_channel_concat:
            # Second camera (skip_embedding) that emits the raw Plücker ray at the
            # LATENT grid (input_shape) for channel-concat. embed_dim unused.
            import copy
            concat_cam_cfg = copy.deepcopy(cfg.camera)
            try:
                concat_cam_cfg.input_shape = cfg.input_shape
            except Exception:
                from omegaconf import OmegaConf
                OmegaConf.set_struct(concat_cam_cfg, False)
                concat_cam_cfg.input_shape = cfg.input_shape
            self.concat_pose = get_camera(
                concat_cam_cfg, num_split=num_split, using_wan=using_wan,
                embed_dim=cfg.kwargs.hidden_size, temporal_downsample=temporal_downsample,
            )
            print(f"(Denoiser) ① ray_channel_concat ON: +{6*max(temporal_downsample,1)} ray ch "
                  f"at latent grid {cfg.input_shape} (extra x_embedder ch zero-init)")

        if self.pretrained_from is not None:
            print("(Denoiser) Loading from pretrained: ", self.pretrained_from)
            weights = torch.load(self.pretrained_from, map_location=torch.device("cpu"))["model"]
            weights = pop_state_dict_by_prefix(weights, "module.")

            self.model.load_state_dict(weights, strict=False)

        self.cnd_proj = nn.Linear(cond_dim, inner_dim)
        self.text_proj = None
        if cfg.text_cond_dim is not None:
            self.text_proj = nn.Linear(cfg.text_cond_dim, inner_dim)

        if cfg_train:
            self.null_tokens = nn.Parameter(torch.zeros(1, 1, inner_dim))
            if self.text_proj is not None:
                self.null_text_tokens = nn.Parameter(torch.zeros(1, 1, inner_dim))

        if cfg.ckpt_path is not None:
            self.load_weights(cfg.ckpt_path, strict=cfg.load_strict)

        # (①) x_embedder surgery AFTER warm-start: expand input channels by the ray
        # channel count, copy the (warm-started) original channels, zero-init the extra
        # → concat ray contributes 0 at init (identical to the pre-concat model).
        if self.ray_channel_concat:
            extra = 6 * max(self.temporal_downsample, 1)
            orig = self.model.x_embedder.proj
            new = nn.Conv2d(
                orig.in_channels + extra, orig.out_channels,
                kernel_size=orig.kernel_size, stride=orig.stride,
                padding=orig.padding, bias=orig.bias is not None,
            )
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, : orig.in_channels].copy_(orig.weight)
                if orig.bias is not None:
                    new.bias.copy_(orig.bias)
            self.model.x_embedder.proj = new.to(orig.weight.device, orig.weight.dtype)
            print(f"(Denoiser) ① x_embedder in_channels {orig.in_channels} -> "
                  f"{orig.in_channels + extra} (extra zero-init, warm-start preserved)")

    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        print(f"(Denoiser) Loading weights (strict={self.cfg.load_strict}) from: ", path)
        weights = torch.load(path, map_location=torch.device("cpu"))
        weights = pop_state_dict_by_prefix(weights["state_dict"], MODEL_PREFIX)
        if not self.cfg.load_strict:
            for key in list(weights.keys()):
                try:
                    param_shape = reduce(getattr, [self, *key.split(".")]).shape
                    if param_shape != weights[key].shape:
                        del weights[key]
                except AttributeError:
                    continue
        self.load_state_dict(weights, **kwargs)

    def _forward(
        self,
        inputs: DenoiserInputs,
        temporal_downsample: int,
        chunk_targets: bool=True,
    ) -> Float[Tensor, "batch view channel height width"]:
        

        latents, pose, timestep, state = inputs.view, inputs.pose, inputs.timestep, inputs.state
        text = inputs.text
        # latents = latents.to(torch.half)
        pemb = self.pose_embed(
            pose,
            temporal_downsample=temporal_downsample,
            chunk_targets=chunk_targets,
        )

        if pemb.shape[1] != latents.shape[1]:
            raise ValueError("Shape mismatch", pose.extrinsics.shape, pemb.shape, latents.shape)

        if text is not None:
            if self.text_proj is not None:
                text = self.text_proj(text)
            elif state is not None and text.shape[-1] != state.shape[-1]:
                raise ValueError(
                    f"Text embedding dim {text.shape[-1]} does not match conditioning dim {state.shape[-1]}. "
                    "Set denoiser.text_cond_dim to enable text projection."
                )
            if state is None:
                state = text
            else:
                state = torch.cat([state, text], dim=1)

        # (①) Channel-concat the raw Plücker ray (latent-grid, 6*td ch) onto the latent.
        if self.ray_channel_concat:
            raw_ray = self.concat_pose(
                pose,
                temporal_downsample=temporal_downsample,
                chunk_targets=chunk_targets,
                skip_embedding=True,
            )  # (b, v, 6*td, h, w) at latent grid
            if raw_ray.shape[1] != latents.shape[1] or raw_ray.shape[-2:] != latents.shape[-2:]:
                raise ValueError("ray_channel_concat shape mismatch", raw_ray.shape, latents.shape)
            latents = torch.cat([latents, raw_ray.to(latents.dtype)], dim=2)

        pemb = rearrange(pemb, "b v c h w -> b v (h w) c")
        sample, qk_list = self.model(latents=latents, pose=pemb.bfloat16(), timestep=timestep, cond_state=state)
        return sample, qk_list
