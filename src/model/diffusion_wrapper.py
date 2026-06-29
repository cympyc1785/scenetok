
import torch
import einops
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from typing import Literal
from torch.nn import Parameter
from torch import optim, nn
from einops import repeat, rearrange
from lightning.pytorch import LightningModule
from typing import Any, Dict, Iterator, Optional
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from colorama import Fore

from .metrics import Metric
from .denoiser import get_denoiser
from .scheduler import get_scheduler
from .compressor import get_compressor
from .autoencoder import get_autoencoder
from .diffusion import get_images, get_latents, sample
from .types import CameraInputs, CompressorInputs, DenoiserInputs
from .sampler import SamplerListCfg, Sampler, get_sampler, SamplerCfg
from .config import ModelCfg, OptimizerCfg, TestCfg, TrainCfg, FreezeCfg, ValCfg
from ..dataset import DatasetCfg
from ..misc.step_tracker import StepTracker
from ..misc.wandb_tools import log_tensor_as_video
from ..misc.image_io import prep_image, save_image_video
from ..misc.torch_utils import freeze, convert_to_buffer
from ..misc.batch_utils import repeat, sequence_concatenate, preprocess_batch, repeat_batch, sequence_reverse
from ..misc.mask_utils import generate_random_context_mask, generate_random_context_mask_tail_decay, generate_biased_boolean_mask, random_mask_biased
from ..visualization.layout import  hcat
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.spline import interpolate_extrinsics_batched

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

def get_target_latent_shape(dataset_cfg: DatasetCfg, model_cfg: ModelCfg) -> list[int] | None:
    target_shape = getattr(dataset_cfg, "target_shape", None)
    if target_shape is None:
        return None

    target_cfg = getattr(model_cfg.autoencoders, "target", None)
    target_name = None if target_cfg is None else target_cfg.name
    if target_name is None:
        spatial_compression_ratio = 1
    elif target_name in {"wan", "wan_single"}:
        spatial_compression_ratio = 8 if target_cfg.kwargs.latent_channels == 16 else 16
    elif target_name == "va":
        spatial_compression_ratio = 16
    elif target_name == "video_dc":
        spatial_compression_ratio = target_cfg.kwargs.spatial_compression_ratio
    else:
        print(
            cyan(
                f"Skipping automatic denoiser input_shape for unsupported "
                f"target autoencoder: {target_name}"
            )
        )
        return None

    h, w = target_shape
    if h % spatial_compression_ratio != 0 or w % spatial_compression_ratio != 0:
        raise ValueError(
            "target_shape must be divisible by the target autoencoder spatial "
            f"compression ratio: target_shape={target_shape}, "
            f"spatial_compression_ratio={spatial_compression_ratio}"
        )
    return [h // spatial_compression_ratio, w // spatial_compression_ratio]

def _get_ae_spatial_compression(ae_cfg) -> int | None:
    """Spatial downsample factor for a configured autoencoder. None = unknown."""
    if ae_cfg is None:
        return None
    name = ae_cfg.name
    if name in {"wan", "wan_single"}:
        return 8 if ae_cfg.kwargs.latent_channels == 16 else 16
    if name == "va":
        return 16
    if name == "video_dc":
        return ae_cfg.kwargs.spatial_compression_ratio
    return None


def get_context_latent_shape(dataset_cfg: DatasetCfg, model_cfg: ModelCfg) -> list[int] | None:
    """Mirror of get_target_latent_shape for the context autoencoder."""
    context_shape = getattr(dataset_cfg, "context_shape", None)
    if context_shape is None:
        return None
    ratio = _get_ae_spatial_compression(getattr(model_cfg.autoencoders, "context", None))
    if ratio is None:
        return None
    h, w = context_shape
    if h % ratio != 0 or w % ratio != 0:
        raise ValueError(
            f"context_shape must be divisible by the context autoencoder spatial "
            f"compression ratio: context_shape={context_shape}, ratio={ratio}"
        )
    return [h // ratio, w // ratio]


def _override_if_diff(parent, attr: str, new_value, label: str) -> None:
    """Set parent.<attr> = new_value and print a cyan diff log if it differed."""
    old_value = getattr(parent, attr)
    old_list = list(old_value) if isinstance(old_value, (list, tuple)) else old_value
    new_list = list(new_value) if isinstance(new_value, (list, tuple)) else new_value
    if old_list == new_list:
        return
    print(cyan(f"Setting {label}: {old_value} -> {new_value}"))
    setattr(parent, attr, new_value)


# Camera input modes that consume raw rays at the *latent* grid (i.e. rays are
# channel-concatted into the main `patch_embedding` or its ac3d-branch clone).
# For these, `camera.input_shape == latent_shape` (= target_shape // vae_down).
#
# Other modes — `wan_control` (SimpleAdapter does its own 16× downsample),
# `recam_attention` (uses extrinsics only, ignores rays), `cross_attention` /
# `new_cross_attention` / `adaln` (overridden to `[pool_size, pool_size]` at
# model init in wan_ti2v.py) — follow the pixel/2 convention by default.
_LATENT_DOMAIN_CAMERA_MODES = {
    "channel_concat",
    "controlnet",
    "controlnet_feedback",
    "controlnet_ac3d",
    "controlnet_lightningdit",
    "ac3d",
}


def _derive_camera_shapes(
    denoiser_cfg,
    *,
    pixel_shape: list[int],
    latent_shape: list[int],
    label_prefix: str,
) -> None:
    """Derive `camera.input_shape` and `camera.embedding.patch_size` so that the
    camera tokens line up with the latent grid.

    Two conventions, picked by `camera_input_type`:

    - **pixel-domain** (default for `recam_attention` / `cross_attention` /
      `new_cross_attention`): ray map at half pixel resolution, LVSM patch
      embed so `input_shape // patch_size == latent_shape`. With `vae_down=16`
      this gives `patch_size = 8`.

    - **latent-domain** (`channel_concat`, `controlnet`, `controlnet_feedback`,
      `ac3d`, `adaln`, `wan_control`): raw rays already at latent grid, so
      `camera.input_shape == latent_shape`. `patch_size` (if any) is left to
      the yaml — these modes use ad-hoc adapters rather than LVSM.

    Note: `recam_attention` is later re-overridden at runtime inside
    `T2VWrapper`/`wan_ti2v.py` (`camera_cfg.input_shape = [pool_size, pool_size]`),
    so the value we set here for that mode is just a placeholder.
    """
    camera_cfg = getattr(denoiser_cfg, "camera", None)
    if camera_cfg is None:
        return

    cam_mode = getattr(denoiser_cfg, "camera_input_type", None)
    if cam_mode in _LATENT_DOMAIN_CAMERA_MODES:
        if latent_shape[0] == 0:
            return
        _override_if_diff(
            camera_cfg,
            "input_shape",
            list(latent_shape),
            f"{label_prefix}.camera.input_shape (latent-domain, mode={cam_mode})",
        )
        return  # patch_size is mode-specific; leave to yaml.

    # Pixel-domain (or unknown/none — treat as pixel-domain).
    cam_input = [pixel_shape[0] // 2, pixel_shape[1] // 2]
    if cam_input[0] == 0 or cam_input[1] == 0:
        return
    _override_if_diff(camera_cfg, "input_shape", cam_input, f"{label_prefix}.camera.input_shape")

    emb_cfg = getattr(camera_cfg, "embedding", None)
    if emb_cfg is None or not hasattr(emb_cfg, "patch_size"):
        return
    if latent_shape[0] == 0:
        return
    if cam_input[0] % latent_shape[0] != 0 or cam_input[1] % latent_shape[1] != 0:
        raise ValueError(
            f"{label_prefix}.camera: derived input_shape {cam_input} not divisible by "
            f"latent grid {latent_shape}; can't derive patch_size."
        )
    derived_patch = cam_input[0] // latent_shape[0]
    _override_if_diff(
        emb_cfg, "patch_size", derived_patch, f"{label_prefix}.camera.embedding.patch_size"
    )


def derive_shape_dependent_fields(dataset_cfg: DatasetCfg, model_cfg: ModelCfg) -> None:
    """Auto-derive every `*.input_shape` (and the corresponding camera
    `embedding.patch_size`) from `dataset.context_shape` / `dataset.target_shape`
    and each autoencoder's spatial-compression ratio.

    Backward compatible: a field is overridden only when it differs from the
    derived value, and the change is logged in cyan. This catches yaml staleness
    when shape settings get edited but downstream fields don't.

    Target side (denoiser):
      - `denoiser.input_shape`           = target_shape // target_ae_ratio
      - `denoiser.camera.input_shape`    = target_shape // 2
      - `denoiser.camera.embedding.patch_size` such that camera token grid ==
        latent grid (typically 8 for vae_ratio=16).

    Context side (compressor):
      - `compressor.input_shape`         = context_shape // context_ae_ratio
      - `compressor.camera.input_shape`  = context_shape // 2
      - `compressor.camera.embedding.patch_size` such that camera token grid
        equals `compressor.input_shape // compressor.kwargs.patch_size`
        (so compressor camera tokens align with compressor video tokens).
    """
    # ── Target / denoiser side ──────────────────────────────────────────
    target_latent = get_target_latent_shape(dataset_cfg, model_cfg)
    if target_latent is not None:
        _override_if_diff(model_cfg.denoiser, "input_shape", target_latent, "denoiser.input_shape")
        _derive_camera_shapes(
            model_cfg.denoiser,
            pixel_shape=list(dataset_cfg.target_shape),
            latent_shape=target_latent,
            label_prefix="denoiser",
        )

    # ── Context / compressor side ───────────────────────────────────────
    compressor_cfg = getattr(model_cfg, "compressor", None)
    if compressor_cfg is None:
        return
    context_latent = get_context_latent_shape(dataset_cfg, model_cfg)
    if context_latent is None:
        return
    _override_if_diff(compressor_cfg, "input_shape", context_latent, "compressor.input_shape")

    # compressor.kwargs.patch_size patches the latent further; the camera token
    # grid must equal latent_grid // compressor_patch_size.
    cmp_patch = getattr(getattr(compressor_cfg, "kwargs", None), "patch_size", 1)
    cmp_token_grid = [context_latent[0] // cmp_patch, context_latent[1] // cmp_patch]
    cam_cfg = getattr(compressor_cfg, "camera", None)
    if cam_cfg is not None:
        cam_input = [dataset_cfg.context_shape[0] // 2, dataset_cfg.context_shape[1] // 2]
        if cam_input[0] > 0 and cam_input[1] > 0:
            _override_if_diff(cam_cfg, "input_shape", cam_input, "compressor.camera.input_shape")
            emb_cfg = getattr(cam_cfg, "embedding", None)
            if emb_cfg is not None and hasattr(emb_cfg, "patch_size") and cmp_token_grid[0] > 0:
                if cam_input[0] % cmp_token_grid[0] != 0 or cam_input[1] % cmp_token_grid[1] != 0:
                    raise ValueError(
                        f"compressor.camera: input_shape {cam_input} not divisible by "
                        f"compressor camera token grid {cmp_token_grid}."
                    )
                derived_patch = cam_input[0] // cmp_token_grid[0]
                _override_if_diff(
                    emb_cfg, "patch_size", derived_patch, "compressor.camera.embedding.patch_size"
                )


def set_denoiser_input_shape_from_target(dataset_cfg: DatasetCfg, model_cfg: ModelCfg) -> None:
    """Back-compat alias. The actual derivation now also covers camera and
    compressor shapes — see `derive_shape_dependent_fields`."""
    derive_shape_dependent_fields(dataset_cfg, model_cfg)

class DiffusionWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model_cfg: ModelCfg
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    freeze_cfg: FreezeCfg
    step_tracker: StepTracker | None
    output_dir: Path | None = None

    def __init__(
        self,
        model_cfg: ModelCfg,
        dataset_cfg: DatasetCfg,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        val_cfg: ValCfg,
        sampler_cfg: SamplerListCfg | SamplerCfg,
        freeze_cfg: FreezeCfg,
        batch_size: int,
        step_tracker: StepTracker | None,
        output_dir: Path | None = None,
        val_check_interval: int=5000,
        mode: Literal["train", "val", "test", "predict_train", "predict_test"]="train",
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.freeze_cfg = freeze_cfg
        self.step_tracker = step_tracker
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.val_check_interval = val_check_interval
        self.mode = mode
        set_denoiser_input_shape_from_target(self.dataset_cfg, self.model_cfg)
        self.text_tokenizer = None
        self.text_encoder = None
        self.text_condition_proj = None
        torch._dynamo.config.optimize_ddp = model_cfg.optimize_ddp
        # if bool(int(os.getenv("DEBUG", 0))):
        #     print("Anomaly detection is enabled during DEBUG")
        #     torch.autograd.set_detect_anomaly(False)

        # self.automatic_optimization = False

        print("(Main Model) Optimize DDP: ", model_cfg.optimize_ddp)
        print("(Main Model) Using Plucker Coordinates: ", model_cfg.use_plucker)
        print("(Main Model) Using Compressor: ", True if model_cfg.compressor is not None else False)
        print("(Main Model) Using SRT Ray Encoding: ", self.model_cfg.srt_ray_encoding)
        print("(Main Model) Using Standard Ray Encoding: ", model_cfg.use_ray_encoding)
        print("(Main Model) Using EMA: ", self.model_cfg.ema)
        print("(Main Model) Using Memory Efficient Attention: ", self.model_cfg.enable_xformers_memory_efficient_attention)
        print("(Main Model) Using Scheduler from: ", self.model_cfg.scheduler.pretrained_from)
       
        num_target_split = self.model_cfg.denoiser.num_target_split
        
        print("(Main Model) Number of Target Splits: ", num_target_split)
        print(f"(Sampler) Timestep Shift: {self.model_cfg.scheduler.kwargs.timestep_shift}")
        print(f"(Sampler) Clean Targets: {sampler_cfg.clean_targets}")

        self.sampler = get_sampler(sampler_cfg)
        self.override_applied = False
        if model_cfg.autoencoders is not None:
            _dict = {}
            if model_cfg.autoencoders.context is not None:
                _dict["context"] = get_autoencoder(model_cfg.autoencoders.context)
            if model_cfg.autoencoders.target is not None:
                _dict["target"] = get_autoencoder(model_cfg.autoencoders.target)

            self.autoencoder = nn.ModuleDict(_dict)

        
        if model_cfg.compressor is not None:
            if getattr(model_cfg.autoencoders, "context") is not None:
                in_channels = model_cfg.autoencoders.context.kwargs.latent_channels
            else:
                in_channels = 3
            self.compressor = get_compressor(
                model_cfg.compressor, 
                in_channels=in_channels, 
                num_views=self.dataset_cfg.view_sampler.num_context_views, 
                temporal_downsample=1
            )
            cond_dim = self.compressor.output_dim
            num_scene_tokens = self.compressor.num_scene_tokens
        else:
            cond_dim = 64   
            num_scene_tokens = 256
        
        temporal_downsample = 1
        if getattr(model_cfg.autoencoders, "target") is not None:
            if getattr(self.model_cfg.autoencoders, "target").name == "video_dc":
                temporal_downsample = 4
            
            if not model_cfg.force_incorrect and getattr(self.model_cfg.autoencoders, "target").name == "wan":
                temporal_downsample = 4
        
        self.denoiser = get_denoiser(
            model_cfg.denoiser, 
            cond_dim=cond_dim, 
            num_scene_tokens=num_scene_tokens, 
            num_views=self.dataset_cfg.view_sampler.num_target_views, 
            temporal_downsample=temporal_downsample, 
            using_wan=True if "wan" in getattr(self.model_cfg.autoencoders, "target").name else False)
        self._init_text_encoder()
        self.scheduler = get_scheduler(model_cfg.scheduler)

        if self.freeze_cfg.denoiser:
            print("(Main Model) Freezing Denoiser")
            freeze(self.denoiser)
        if self.freeze_cfg.compressor:
            print("(Main Model) Freezing Compressor")
            freeze(self.compressor)
        if self.freeze_cfg.autoencoder and self.model_cfg.autoencoders is not None:
            print("(Main Model) Freezing Autoencoder")
            freeze(self.autoencoder)

        if self.model_cfg.autoencoders is not None:
            print("(Main Model) Converting to buffer Autoencoder")
            convert_to_buffer(self.autoencoder, persistent=False)
        
        self.metric = Metric().eval()
        freeze(self.metric)
        convert_to_buffer(self.metric, persistent=False)


        if self.model_cfg.ema:
            self.ema = AveragedModel(
                self.denoiser, 
                multi_avg_fn=get_ema_multi_avg_fn(0.995)
            )
                
        self.validation_type = None
        self.frozen_compressor = self.freeze_cfg.compressor
        self.unfrozen_compressor = False
        self.frozen_scene_query = False
        self.test_step_outputs = []
        self.validation_loader_names = {0: "standard", 1: "unseen"}
        if getattr(self.dataset_cfg, "name", None) == "multi":
            # multi val builds one loader per (val_key × sub-dataset), in order
            # standard×subs then unseen×subs (see data_module.val_dataloader). Map
            # ALL subs of a key to ONE panel name so DL3DV + DynamicVerse log and
            # metric together under "standard"/"unseen" (e.g. 4 DL3DV + 4 DV = 8).
            n_subs = len(self.dataset_cfg.datasets)
            self.validation_loader_names = {
                ki * n_subs + j: name
                for ki, name in enumerate(["standard", "unseen"])
                for j in range(n_subs)
            }
        self.predicted = {name: [] for name in self.validation_loader_names.values()}
        self.generated = {name: [] for name in self.validation_loader_names.values()}
        # Video/context logging accumulates up to `val_vis_num` samples across
        # validation batches, then flushes once in `on_validation_end`. This
        # decouples how many videos get logged from `data_loader.val.batch_size`
        # — a reduced batch_size (e.g. 1, to save VAE-decode memory) still logs
        # the full set instead of just the first batch.
        self.val_vis_num = 8
        self.val_vis_buffer = {
            name: {"sampled": [], "target": [], "context": [], "scene": []}
            for name in self.validation_loader_names.values()
        }
        # Loss NaN/Inf occurrence counter (process-lifetime cumulative).
        # Used by training_step to drop into pdb after 3 occurrences.
        self.nan_loss_count = 0



    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_cfg.ema:
            self.ema.update_parameters(self.denoiser)   

    def setup(self, stage: str) -> None:
        # Scale base learning rates to effective batch size
        if stage == "fit":
            # assumes one fixed batch_size for all train dataloaders!
            effective_batch_size = self.trainer.accumulate_grad_batches \
                * self.trainer.num_devices \
                * self.trainer.num_nodes \
                * self.batch_size

            self.lr = effective_batch_size * self.optimizer_cfg.lr \
                if self.optimizer_cfg.scale_lr else self.optimizer_cfg.lr
        return super().setup(stage)



    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        print(f"[load_state_dict] ckpt={len(state_dict)} keys; "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing[:8]: {missing[:8]}")
        if unexpected:
            print(f"  unexpected[:8]: {unexpected[:8]}")
        return result

    
    def set_timesteps(self, num: Optional[int]=None, name: str="validation"):
        """
            Args:
                num (Optional[int]): Override the number of inference steps. Default to None.
        """
        num_inference_timesteps = self.model_cfg.scheduler.num_inference_steps if num is None else num
        print(f"Setting Max Timesteps for {name} to: ", num_inference_timesteps)
        self.scheduler.set_timesteps(num_inference_timesteps)    

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps(name="validation")
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps(name="testing")
    
    def get_conditioning_mask(self, shape, device, dtype):
        if self.model_cfg.compressor is not None:
            
            context_mask = generate_random_context_mask(shape=shape, device=device).to(dtype)
        else:
            context_mask = generate_random_context_mask_tail_decay(shape=shape, device=device).to(dtype)
        
        return context_mask
    
    def get_noise_level(self, shape: tuple=(1,), dtype: torch.dtype = torch.float, mu: float=0.0, sigma: float=1.0):
        
        if self.model_cfg.scheduler.name == "rectified_flow":
            if self.model_cfg.scheduler.kwargs.weighting == "uniform":
                timesteps = torch.rand(
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ) 
            elif self.model_cfg.scheduler.kwargs.weighting == "logit_normal":
                timesteps = torch.normal(mu, sigma,
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ).sigmoid_()
            elif self.model_cfg.scheduler.kwargs.weighting == "shifted":
                timesteps = torch.rand(
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ) 
                timesteps = self.scheduler.shift_timestep(timestep=timesteps, shift=self.model_cfg.scheduler.kwargs.timestep_shift)

            else:
                raise NotImplementedError(f"{self.model_cfg.scheduler.kwargs.weighting} weighting is not implemented!")
            
            
        else:
            timesteps = torch.randint(
                0, 
                self.model_cfg.scheduler.num_train_timesteps, 
                size=shape, 
                device=self.device,
                dtype=torch.long
            ) 
        return timesteps
    
    def rescale_timesteps(self, timestep):
        
        if self.model_cfg.scheduler.name == "rectified_flow":
            t = (timestep * self.model_cfg.scheduler.num_train_timesteps - 1).clip(min=0)
        else:
            t = timestep
        return t

    
    def preprocess_scene_tokens(self, scene_tokens, shape, device, token_mask=None):
        
        if scene_tokens is None:
            if self.model_cfg.compressor is None:
                scene_tokens = torch.zeros(shape, device=device)
                scene_tokens = self.denoiser.cnd_proj(scene_tokens)
            else:
                if self.model_cfg.no_null_expand:
                    scene_tokens = self.denoiser.null_tokens.expand(shape[0], -1, -1)
                else:
                    scene_tokens = self.denoiser.null_tokens.expand(*shape[:2], -1)
        else:
            scene_tokens = self.denoiser.cnd_proj(scene_tokens)
        if token_mask is not None:
            scene_tokens[~token_mask] = self.denoiser.null_tokens[0].to(scene_tokens.dtype)
        return scene_tokens

    def _init_text_encoder(self):
        if self.model_cfg.text_encoder is None:
            return
        if self.model_cfg.text_encoder.name != "umt5":
            raise ValueError(f"Unsupported text encoder: {self.model_cfg.text_encoder.name}")

        try:
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError as err:
            raise ImportError(
                "transformers is required for umT5 conditioning. "
                "Install it with `pip install transformers sentencepiece`."
            ) from err

        text_cfg = self.model_cfg.text_encoder
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_cfg.pretrained_model_name_or_path
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            text_cfg.pretrained_model_name_or_path
        )

        if not text_cfg.trainable:
            freeze(self.text_encoder)
            self.text_encoder.eval()

        if getattr(self.denoiser, "text_proj", None) is None:
            hidden_size = self.text_encoder.config.d_model
            target_dim = self.denoiser.cnd_proj.out_features
            if hidden_size != target_dim:
                self.text_condition_proj = nn.Linear(hidden_size, target_dim)

    def get_text_condition(self, batch, device=None):
        if "text_embedding" in batch and batch["text_embedding"] is not None:
            text = batch["text_embedding"]
            if self.text_condition_proj is not None:
                text = self.text_condition_proj(text.to(self.device))
            if device is None:
                return text
            return text.to(device)

        text = batch.get("text")
        if text is None or self.text_encoder is None or self.text_tokenizer is None:
            return None
        if isinstance(text, str):
            text = [text]

        text_inputs = self.text_tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length=self.model_cfg.text_encoder.max_length,
            return_tensors="pt",
        )
        target_device = self.device if device is None else device
        text_inputs = {k: v.to(target_device) for k, v in text_inputs.items()}

        if self.model_cfg.text_encoder.trainable:
            text_outputs = self.text_encoder(**text_inputs)
        else:
            with torch.no_grad():
                text_outputs = self.text_encoder(**text_inputs)

        text_state = text_outputs.last_hidden_state
        if self.text_condition_proj is not None:
            text_state = self.text_condition_proj(text_state)
        return text_state
    
    def process_gt(self, latents, noise, timestep):
        
        if self.model_cfg.scheduler.kwargs.prediction_type == "epsilon":
            target = noise
        
        elif self.model_cfg.scheduler.kwargs.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timestep)
        
        elif self.model_cfg.scheduler.kwargs.prediction_type == "sample":
            target = latents
        
        elif self.model_cfg.scheduler.kwargs.prediction_type == "flow":

            target = self.scheduler.get_flow(latents, noise)
        else:
            raise NotImplementedError()
        
        return target



    @torch.no_grad()
    def generate_batch_with_scene(self, batch, sampler: Sampler, repeat_factor: int=1):
        
        context_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["context"], 
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )

        target = repeat_batch(batch["target"], repeat_factor)
        device = context_latents.device
        dtype = context_latents.dtype
        text_state = self.get_text_condition(batch, device=device)
        if text_state is not None:
            text_state = einops.repeat(text_state, "b n d -> (b r) n d", r=repeat_factor)
        b_c, v_c, *_ = context_latents.shape
        b, v_t, *_ = target["extrinsics"].shape

        if getattr(self.model_cfg.autoencoders, "target") is not None:
            if self.model_cfg.autoencoders.target.name in ["kl"]:
                c = self.model_cfg.autoencoders.target.kwargs.latent_channels // 2
            else:
                c = self.model_cfg.autoencoders.target.kwargs.latent_channels
        else:
            c = 3

        temporal_downsample = 1
        num = v_t
        if getattr(self.model_cfg.autoencoders, "target") is not None:
            if getattr(self.model_cfg.autoencoders, "target").name == "video_dc":
                temporal_downsample = 4
                num = (v_t // temporal_downsample)
                target["extrinsics"] = target["extrinsics"][:, :num*temporal_downsample]
                target["intrinsics"] = target["intrinsics"][:, :num*temporal_downsample]
            
            elif getattr(self.model_cfg.autoencoders, "target").name == "wan":
                temporal_downsample = 4
                if getattr(self.dataset_cfg.view_sampler, "chunk_targets", True):
                    num = (v_t // 17) * 5
                    num_pose_views = (num // 5) * 17
                else:
                    num = 1 + (v_t - 1) // temporal_downsample
                    num_pose_views = 1 + (num - 1) * temporal_downsample

                target["extrinsics"] = target["extrinsics"][:, :num_pose_views]
                target["intrinsics"] = target["intrinsics"][:, :num_pose_views]

        input_shape = self.model_cfg.denoiser.input_shape
        if isinstance(input_shape, int):
            h = w = input_shape
        else:
            h, w = input_shape
        x_t = torch.randn((b, num, c, h, w), device=device, dtype=dtype)
        x_t *= self.scheduler.init_noise_sigma 
        
        context_camera = CameraInputs(
            intrinsics=batch["context"]["intrinsics"],
            extrinsics=batch["context"]["extrinsics"]
        )


        target_pose = CameraInputs(
            intrinsics=target["intrinsics"],
            extrinsics=target["extrinsics"]
        )

        context_mask = None

        if not self.model_cfg.mask_context:
            context_inputs = CompressorInputs(
                view=context_latents,
                pose=context_camera,
                mask=None
            )
        else:
            context_inputs = CompressorInputs(
                view=context_latents[:, ~context_mask[0]],
                pose=context_camera[:, ~context_mask[0]],
                mask=None
            )

        tokens, qks = self.compressor._forward(inputs=context_inputs)
        

        if self.model_cfg.compressor.scene_token_projection == "kl":
            scene_tokens = tokens.sample()
        else:
            scene_tokens = tokens

        scene_tokens = repeat(scene_tokens, "b ... -> (b n) ...", n=repeat_factor)
        
        sampler.set_scheduling_matrix(
            horizon=num,
            steps=self.model_cfg.scheduler.num_inference_steps, 
            concurrency=self.dataset_cfg.view_sampler.num_target_views, 
            device=device,
            dtype=dtype,
            cond_mask_indices=None
        )
        if self.model_cfg.scheduler.kwargs.weighting == "shifted":
            print(f"(Sampler) Shifting scheduling matrix by {self.model_cfg.scheduler.kwargs.timestep_shift}")
            sampler.shift_scheduling_matrix(self.model_cfg.scheduler.kwargs.timestep_shift)
        sampler.log_vis(self.logger, step=self.step_tracker, name=f"({sampler.cfg.name})")
        print("Shape of latents: ", x_t.shape)
        return *sample(
            model=self.denoiser,
            x_t=x_t, 
            target_pose=target_pose,
            cond_state=scene_tokens,
            text_state=text_state,
            sampler=sampler,
            scheduler=self.scheduler,
            autoencoder=self.autoencoder,
            temporal_downsample=temporal_downsample,
            cfg_scale=self.model_cfg.cfg_scale,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_index_gap=self.dataset_cfg.view_sampler.chunk_index_gap,
            offset=self.dataset_cfg.view_sampler.offset,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        ), scene_tokens

    @staticmethod
    def _diagnose_batch_for_blacklist(batch, target_latents) -> list[dict]:
        """Inspect a NaN-loss batch and return per-scene anomaly entries.

        Checks per-sample (i in 0..B-1):
          - extrinsics/intrinsics: non-finite, or |translation| > 50
          - raw input ("latent" key — actually pre-VAE RGB for DL3DV): non-finite,
            or |val| > 5 (post-augmentation RGB should stay in roughly [-2, 2])
          - encoded `target_latents`: non-finite, or |val| > 200
        """
        scenes_field = batch.get("scene", [])
        scenes = list(scenes_field) if isinstance(scenes_field, (list, tuple)) else [scenes_field]
        B = len(scenes)
        if B == 0:
            return []
        issues: list[dict] = []

        def _per_sample_flatten(t):
            return t.reshape(t.shape[0], -1) if t.ndim > 1 else t.reshape(t.shape[0], 1)

        for view_key in ("context", "target"):
            view = batch.get(view_key)
            if not isinstance(view, dict):
                continue
            ext = view.get("extrinsics")
            if ext is not None and torch.is_tensor(ext) and ext.shape[0] == B:
                t = ext[..., :3, 3]  # (B, V, 3)
                finite = torch.isfinite(ext).reshape(B, -1).all(dim=1)
                max_t = t.abs().reshape(B, -1).amax(dim=1)
                for i in range(B):
                    if not bool(finite[i]):
                        issues.append({"scene": scenes[i], "reason": f"{view_key}_extrinsics_nonfinite", "detail": ""})
                    elif float(max_t[i]) > 50.0:
                        issues.append({"scene": scenes[i], "reason": f"{view_key}_extrinsics_large", "detail": f"max|t|={float(max_t[i]):.3g}"})
            intr = view.get("intrinsics")
            if intr is not None and torch.is_tensor(intr) and intr.shape[0] == B:
                finite = torch.isfinite(intr).reshape(B, -1).all(dim=1)
                for i in range(B):
                    if not bool(finite[i]):
                        issues.append({"scene": scenes[i], "reason": f"{view_key}_intrinsics_nonfinite", "detail": ""})
            lat = view.get("latent")
            if lat is not None and torch.is_tensor(lat) and lat.shape[0] == B:
                finite = torch.isfinite(lat).reshape(B, -1).all(dim=1)
                max_v = lat.abs().reshape(B, -1).amax(dim=1)
                for i in range(B):
                    if not bool(finite[i]):
                        issues.append({"scene": scenes[i], "reason": f"{view_key}_input_nonfinite", "detail": ""})
                    elif float(max_v[i]) > 5.0:
                        issues.append({"scene": scenes[i], "reason": f"{view_key}_input_huge", "detail": f"max|x|={float(max_v[i]):.3g}"})

        if target_latents is not None and torch.is_tensor(target_latents) and target_latents.shape[0] == B:
            tl = target_latents
            finite = torch.isfinite(tl).reshape(B, -1).all(dim=1)
            max_v = tl.abs().reshape(B, -1).amax(dim=1)
            for i in range(B):
                if not bool(finite[i]):
                    issues.append({"scene": scenes[i], "reason": "encoded_target_latents_nonfinite", "detail": ""})
                elif float(max_v[i]) > 200.0:
                    issues.append({"scene": scenes[i], "reason": "encoded_target_latents_huge", "detail": f"max|z|={float(max_v[i]):.3g}"})

        # Dedup by (scene, reason) keeping first detail.
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for iss in issues:
            key = (iss["scene"], iss["reason"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(iss)
        return deduped

    def training_step(self, batch, batch_idx):
        if batch is None:  # safe_collate returned None (entire batch was None-filtered)
            return None
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log(f"step_tracker/step", self.step_tracker.get_step())

        # convert all camera poses for context and target to relative w.r.t a random context camera
        # during test time, you can select any context index to be the origin
        # during training of scenegen, make sure the origin is present and that the first conditioning
        # image starts at the origin 
        batch = preprocess_batch(batch)

        # Get latents if input is rgb otherwise scale latents by the scaling_factor
        # In case of VA-VAE, scale and shift by the predefined latent statistics
        target_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["target"], 
            view_type="target",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        
        device = target_latents.device
        dtype = target_latents.dtype
        b, v_t, c, h, w = target_latents.shape

        # CFG if enabled
        conditional_tokens = True
        if self.train_cfg.cfg_train and not self.freeze_cfg.denoiser:
            # Randomly choose to train conditionally or unconditionally
            conditional_tokens = np.random.choice([True, False], 1, p=[0.90, 0.10])
        
        token_mask = None
        if conditional_tokens and self.model_cfg.compressor is not None:
            # Get latents if input is rgb otherwise scale latents by the scaling_factor
            # In case of VA-VAE, scale and shift by the predefined latent statistics
            context_latents = get_latents(
                autoencoder=self.autoencoder,
                inputs=batch["context"], 
                view_type="context",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
                scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
                chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
            )
            
            b, v_c, *_ = context_latents.shape
            # Experimental: masking context views until min context views
            if self.model_cfg.mask_context and np.random.choice([True, False], p=[0.4, 0.6]):
                context_mask = generate_biased_boolean_mask((b, v_c), self.dataset_cfg.view_sampler.min_context_views).to(context_latents.device)
            else:
                context_mask = None

            context_inputs = CompressorInputs(
                view=context_latents,
                pose=CameraInputs(
                    intrinsics=batch["context"]["intrinsics"],
                    extrinsics=batch["context"]["extrinsics"]
                ),
                mask=context_mask
            )
            if self.frozen_compressor:
                with torch.no_grad():
                    tokens, *_ = self.compressor(inputs=context_inputs)
            else:
                tokens, *_ = self.compressor(inputs=context_inputs)
            
            if self.model_cfg.compressor.scene_token_projection == "kl":
                scene_tokens = tokens.sample()
            else:
                scene_tokens = tokens


            # Experimental: Add noise to tokens
            if self.model_cfg.noisy_scene_tokens and np.random.choice([True, False], p=[self.model_cfg.noise_prob, 1-self.model_cfg.noise_prob]):
                scene_noise = torch.randn_like(scene_tokens, device=device)  
                timestep_scene = self.get_noise_level((b, self.compressor.num_scene_tokens), dtype=dtype, mu=self.model_cfg.mu, sigma=self.model_cfg.sigma)
                scene_tokens = self.scheduler.add_noise(scene_tokens, scene_noise, timestep_scene)

            # Experimental: Masking tokens
            if self.model_cfg.mask_tokens:
                token_mask, ratios, num_false = random_mask_biased(B=scene_tokens.shape[0], N=scene_tokens.shape[1], M=0.6, device="cpu")


        else:
            # for unconditional sampling (rendering)
            scene_tokens = None
        
        
        # Define noise-level per tokens
        if self.model_cfg.scheduler.sampling_type == "random_uniform":
            timestep_shape = (b, )
        elif self.model_cfg.scheduler.sampling_type == "random_chunked_uniform":
            timestep_shape = (b, self.dataset_cfg.view_sampler.num_target_split)
        elif self.model_cfg.scheduler.sampling_type == "random_independent":
            timestep_shape = (b, v_t)
        else:
            raise NotImplementedError(f"Sampling type in scheduler is not correctly specified and instead got {self.model_cfg.scheduler.sampling_type}")
        
        # Allow targets with same noise-levels
        if np.random.choice([True, False], p=[0.2, 0.8]) and self.model_cfg.enforce_uniform_noise:
            timestep_shape = (b, )
        
        # Get noise-level
        timestep = self.get_noise_level(timestep_shape, dtype=dtype)
        # Repeat timestep in case of (chunked) uniform noise-levels
        if timestep.ndim == 1:
            timestep = repeat(timestep, "b -> b v", v=v_t)

        elif self.model_cfg.scheduler.sampling_type == "random_chunked_uniform":
            timestep = repeat(timestep, "b n -> b (n v)", v=self.dataset_cfg.view_sampler.num_target_views // self.dataset_cfg.view_sampler.num_target_split)
        
        # Experimental: Force zero noise-levels for conditioning 
        if self.model_cfg.force_clean:
            target_cond_mask = self.get_conditioning_mask((b, v_t), device=device, dtype=dtype) 
            timestep = timestep * target_cond_mask
        
        # Sample Noise
        noise = torch.randn_like(target_latents, device=device)  

        # Add noise to targets
        noisy_latents = self.scheduler.add_noise(target_latents, noise, timestep)

        # If masking tokens, then define mask either before or after up-projection layer
        # Otherwise no masking, and simply up-project the tokens for rendering
        # Parameters of this up-projection is part of the denoiser
        scene_tokens = self.preprocess_scene_tokens(
            scene_tokens=scene_tokens, 
            shape=(b, self.denoiser.num_scene_tokens, self.denoiser.cond_dim), 
            device=device, 
            token_mask=token_mask
        )
        
        t = self.rescale_timesteps(timestep=timestep)   
            
        # Denoise
        denoiser_input = DenoiserInputs(
            view=noisy_latents, 
            pose=CameraInputs(
                intrinsics=batch["target"]["intrinsics"],
                extrinsics=batch["target"]["extrinsics"]
            ), 
            timestep=t, 
            state=scene_tokens,
            text=self.get_text_condition(batch, device=device)

        )
        
        # flow prediction for targets
        pred, _ = self.denoiser(
            inputs=denoiser_input, 
            temporal_downsample=self.dataset_cfg.view_sampler.temporal_downsample,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        
        # Get ground truth flow for targets
        gt = self.process_gt(target_latents, noise, timestep)

        # Loss
        loss = F.mse_loss(pred, gt, reduction='none')

        # Only apply on noisy targets
        if self.model_cfg.force_clean:
            loss = einops.reduce(loss, "b v c h w -> b v", "mean")
            loss = loss * target_cond_mask
            loss = loss.sum(-1) / target_cond_mask.sum(-1)
        else:
            loss = einops.reduce(loss, "b v c h w -> b", "mean")

        # Apply KL divergence weighting and scheduling
        kl_weight = 0.0
        if self.model_cfg.compressor.scene_token_projection == "kl" and conditional_tokens and self.model_cfg.compressor is not None and not self.frozen_compressor:
            kl_raw = tokens.kl()
            kl = kl_raw / (tokens.mean.shape[1] * tokens.mean.shape[2])
            if self.global_step <= self.model_cfg.compressor.kl_schedule[0]:
                kl_weight = self.model_cfg.compressor.kl_weights[0]
            elif self.global_step <= self.model_cfg.compressor.kl_schedule[1] and self.global_step > self.model_cfg.compressor.kl_schedule[0]:
                t = (self.global_step - self.model_cfg.compressor.kl_schedule[0]) / (self.model_cfg.compressor.kl_schedule[1] - self.model_cfg.compressor.kl_schedule[0])
                kl_weight = (1 - t) * self.model_cfg.compressor.kl_weights[0] + t*self.model_cfg.compressor.kl_weights[1]
            else:
                kl_weight = self.model_cfg.compressor.kl_weights[1]
            loss = loss + kl_weight * kl
            self.log("loss/kl", kl.mean())
            self.log("loss/kl_raw", kl_raw.mean())
        loss = loss.mean()

        # Loss NaN/Inf tracking. On every NaN we (a) diagnose the batch for
        # data-side anomalies (huge extrinsics translations, non-finite inputs,
        # exploded latents) and (b) append offending scenes to a per-dataset
        # blacklist CSV so subsequent runs filter them out via
        # `load_blacklist(...)` in dataset_dl3dv. Auto-continue is fine — the
        # gradient guard around line 1144 zeros the bad step's grads.
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.nan_loss_count += 1
            step = self.step_tracker.get_step() if self.step_tracker is not None else -1
            scenes_field = batch.get("scene", [])
            scenes = list(scenes_field) if isinstance(scenes_field, (list, tuple)) else [scenes_field]
            loss_str = f"{loss.item():.3g}" if torch.isfinite(loss).all() else "nan"
            print(
                f"[NaN loss #{self.nan_loss_count}] step={step} "
                f"loss={loss_str} batch_idx={batch_idx} scenes={scenes}"
            )
            try:
                issues = self._diagnose_batch_for_blacklist(batch, target_latents)
                if issues:
                    for iss in issues:
                        iss["step"] = str(step)
                        iss["loss"] = loss_str
                    if getattr(self.dataset_cfg, "name", None) == "dl3dv":
                        from src.dataset.dataset_dl3dv import (
                            append_blacklist,
                            _resolve_blacklist_path,
                        )
                        bl_path = _resolve_blacklist_path(
                            self.dataset_cfg, Path(self.dataset_cfg.root) / "train"
                        )
                        n_added = append_blacklist(bl_path, issues)
                        print(
                            f"[NaN loss] blacklist += {n_added} (of {len(issues)} flagged) → {bl_path}"
                        )
                    else:
                        print(
                            f"[NaN loss] {len(issues)} anomalies flagged but dataset is "
                            f"{getattr(self.dataset_cfg, 'name', None)!r} (blacklist unsupported); just logging"
                        )
                    for iss in issues:
                        print(f"  - {iss['scene']}: {iss['reason']} ({iss['detail']})")
                else:
                    print(f"[NaN loss] no data anomaly detected — likely model-side/transient, not blacklisting")
            except Exception as e:
                print(f"[NaN loss] blacklist diagnostic failed: {e!r}")

        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        if self.global_rank == 0:
            print(
                f"Train step {self.step_tracker.get_step()}; "
                # f"scene = {batch['scene']}; "
                # f"context = {batch['context']['index'].tolist()}; "
                # f"target = {batch['target']['index'][:, [0, 16, -1]].tolist()}; "
                f"loss = {loss.item():.4f} lr = {current_lr}"
            )
        self.log("loss/diffusion", loss) 

        return loss
    
    # @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx: Optional[int]=None):
        if batch is None:
            return None

        # val_step = global step // val_check_interval → clean validation counter
        # (0,1,2,...), exactly as the ORIGINAL SceneTok release code does.
        # ⚠️ MEMO: keep this `(step+1)//self.val_check_interval` form. Do NOT
        # change it to the raw step (`val_step = step`) — that was a wrong detour.
        step = self.step_tracker.get_step()
        val_step = (step + 1) // self.val_check_interval
        loader_name = self.validation_loader_names.get(dataloader_idx or 0, f"val_{dataloader_idx}")
        self.predicted.setdefault(loader_name, [])
        self.generated.setdefault(loader_name, [])

        print(
            f"Name = {loader_name}; "
            f"Validation Step {val_step}; "
            f"Step {batch_idx}; "
            f"Scene = {batch['scene']}; "
            f"Context = {batch['context']['index'].tolist()}; "
            f"Target = {batch['target']['index'].tolist()}; "
            f"Rank = {self.global_rank}; "
        )

        # In case if latent inputs e.g., during training with precomputed latents
        context_views = get_images(
            autoencoder=self.autoencoder, 
            inputs=batch["context"], 
            view_type="context", 
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        target_views = get_images(
            autoencoder=self.autoencoder, 
            inputs=batch["target"], 
            view_type="target", 
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        b, v_c, *_ = context_views.shape

        # Relative pose w.r.t the middle context index
        batch = preprocess_batch(batch, index=v_c//2)

        # Sample target views
        sampled_views, _, _ = self.generate_batch_with_scene(batch, self.sampler)
        b, v_t, c, h, w = sampled_views.shape
        target_views = target_views[:, :v_t]

        # Diagnostic + finite-mask guard. NaN/Inf in sampled_views poisons FVD's
        # linalg.sqrtm via NaN activations → covariance → Schur decomposition hang.
        # Training-side already skips NaN gradients (line 1144); val side previously
        # had no sanitize, so this closes the loop.
        nan_count = torch.isnan(sampled_views).sum().item()
        inf_count = torch.isinf(sampled_views).sum().item()
        if nan_count or inf_count:
            print(
                f"[Sample guard] {loader_name} batch {batch_idx}: "
                f"nan={nan_count} inf={inf_count} "
                f"min={sampled_views.float().min().item():.3f} "
                f"max={sampled_views.float().max().item():.3f}"
            )
        finite_mask = torch.isfinite(sampled_views).reshape(b, -1).all(dim=1)
        if not finite_mask.all():
            sampled_views = sampled_views[finite_mask]
            target_views = target_views[finite_mask]
        if sampled_views.shape[0] == 0:
            print(f"[Sample guard] {loader_name} batch {batch_idx}: entire batch dropped (all NaN/Inf)")
            return None

        self.generated[loader_name].append(sampled_views.detach().cpu())
        self.predicted[loader_name].append(target_views.detach().cpu())
        # Only do remaining on Rank: 0 in case of multi-gpu/node
        if self.global_rank != 0:
            return None

        batch_scene = []
        for j in range(b):
            scene = batch["scene"][j]
            batch_scene.append(scene)

        # Accumulate up to `val_vis_num` samples for video/context logging
        # (flushed once in `on_validation_end`). Replaces the old
        # `batch_idx == 0` immediate log so a reduced val batch_size still
        # produces the full set of logged videos.
        buf = self.val_vis_buffer.setdefault(
            loader_name, {"sampled": [], "target": [], "context": [], "scene": []}
        )
        remaining = self.val_vis_num - len(buf["scene"])
        if remaining > 0:
            take = min(remaining, sampled_views.shape[0])
            buf["sampled"].append(sampled_views[:take].detach().cpu())
            buf["target"].append(target_views[:take].detach().cpu())
            buf["context"].append(context_views[:take].detach().cpu())
            buf["scene"].extend(batch_scene[:take])

        # Do a spline interpolation using context poses as "knot" and sample a video
        if self.val_cfg.video and batch_idx == 0:
            print(cyan("Generating Context Interpolation Video..."))
            b, v_c, c, h, w = batch["context"]["latent"].shape
            t = self.val_cfg.video_length
            start = batch["context"]["extrinsics"][:, 0]
            target_extrinsics = interpolate_extrinsics_batched(batch["context"]["extrinsics"], self.val_cfg.video_length - v_c + 1)
            indices = torch.linspace(0, t, steps=t, device=start.device)
            new_target = {
                "extrinsics": target_extrinsics.to(start.device),
                "intrinsics": batch["target"]["intrinsics"][:, 0:1].expand(-1, t, -1, -1).clone(),
                "latent": torch.zeros((b, t, c, h, w), device=start.device).to(start.dtype),
                "index": indices[None].expand(b, -1)
            }
            batch["target"] = new_target  
            try:    
                sampled_views, _, _ = self.generate_batch_with_scene(batch, self.sampler)
                log_tensor_as_video(self.logger, sampled_views, f"{loader_name}/Context Interpolation ({self.sampler.cfg.name})", fps=24, step=val_step, caption=batch_scene)
            except:
                pass
        torch.cuda.empty_cache()

        return None

    def on_validation_epoch_end(self):
        
        step = self.step_tracker.get_step()
        
        assert self.predicted.keys() == self.generated.keys()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.empty_cache()
        return None

    def on_validation_end(self):
        # if step == 0:
        #     return

        # val_step = global step // val_check_interval → clean validation counter
        # (0,1,2,...), exactly as the ORIGINAL SceneTok release code does.
        # ⚠️ MEMO: keep this `(step+1)//self.val_check_interval` form. Do NOT
        # change it to the raw step (`val_step = step`) — that was a wrong detour.
        step = self.step_tracker.get_step()
        val_step = (step + 1) // self.val_check_interval

        for loader_name in self.predicted.keys():
            if len(self.predicted[loader_name]) == 0 or len(self.generated[loader_name]) == 0:
                continue
            sampled_views = torch.concat(self.generated[loader_name]).to(self.device, non_blocking=True)
            target_views = torch.concat(self.predicted[loader_name]).to(self.device, non_blocking=True)
            print(loader_name, sampled_views.shape)
            print(loader_name, target_views.shape)
            # metrics = self.metric(sampled_views.flatten(0, 1), target_views.flatten(0, 1), psnr=True, ssim=True, lpips=True)
            # self.log("lpips", metrics["lpips"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log("ssim", metrics["ssim"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log("psnr", metrics["psnr"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # for key, value in metrics.items():
            #     self.logger.log_metrics({f"{loader_name}/full_sequence/{key}": value}, val_step)
            
            # --- sequence-level metrics ---
            gathered_sampled = self.all_gather(sampled_views)
            gathered_target = self.all_gather(target_views)
            num_views = gathered_sampled.shape[-4]
            chunk_size = min(16, num_views)
            if getattr(self.model_cfg.autoencoders, "target") is not None:
                if getattr(self.model_cfg.autoencoders, "target").name == "wan":
                    if getattr(self.dataset_cfg.view_sampler, "chunk_targets", True):
                        chunk_size = min(17, num_views)
                    else:
                        chunk_size = num_views
            if num_views % chunk_size != 0:
                chunk_size = num_views
            
            if gathered_sampled.shape[1] != gathered_target.shape[1]:
                print(pred.shape, gt.shape)
                breakpoint()
                raise RuntimeError("prediction shape mismatch", pred.shape, gt.shape)

            # flatten across world size
            gathered_sampled = rearrange(gathered_sampled, "... (k v) c h w -> (... k) v c h w", v=chunk_size)
            gathered_target = rearrange(gathered_target, "... (k v) c h w -> (... k) v c h w", v=chunk_size)

            # flatten views for metric
            gathered_sampled = rearrange(gathered_sampled, "... c h w -> (...) c h w")
            gathered_target = rearrange(gathered_target, "... c h w -> (...) c h w")

            if self.global_rank == 0:
                # DEBUG (remove after LPIPS-saturation diagnosis): show LPIPS internal dtypes
                _lpips_param_dtypes = {p.dtype for p in self.metric.lpips.parameters()}
                _lpips_buf_dtypes = {b.dtype for b in self.metric.lpips.buffers()}
                print(
                    f"[LPIPS dtype] params={_lpips_param_dtypes} "
                    f"buffers={_lpips_buf_dtypes} "
                    f"pred={gathered_sampled.dtype} gt={gathered_target.dtype}"
                )
                full_sequence_metrics = self.metric(
                    gathered_sampled,
                    gathered_target,
                    psnr=True,
                    ssim=True,
                    lpips=True,
                )

                for key, value in full_sequence_metrics.items():
                    if value is None:
                        continue
                    self.logger.log_metrics({f"{loader_name}/full_sequence/{key}": value}, val_step)
                    if loader_name == "standard":
                        self.logger.log_metrics({f"full_sequence/{key}": value}, val_step)

                general_metrics = self.metric(
                    gathered_sampled.cpu(),
                    gathered_target.cpu(),
                    num_views=chunk_size,
                    fvd=True,
                    fid=True,
                )
                
                self.metric.reset_fid()
                self.metric.reset_fvd()

                for key, value in general_metrics.items():
                    if value is not None:
                        self.logger.log_metrics({f"{loader_name}/{key}": value}, val_step)
                        if loader_name == "standard":
                            self.logger.log_metrics({f"{key}": value}, val_step)

        # Flush accumulated video/context logs (up to `val_vis_num` samples,
        # collected across val batches). Rank 0 only — matches the original
        # logging guard.
        if self.global_rank == 0:
            idx0_name = self.validation_loader_names.get(0)
            for loader_name, buf in self.val_vis_buffer.items():
                if len(buf["scene"]) == 0:
                    continue
                n = self.val_vis_num
                sampled = torch.cat(buf["sampled"])[:n].to(self.device, non_blocking=True)
                target = torch.cat(buf["target"])[:n].to(self.device, non_blocking=True)
                scenes = buf["scene"][:n]
                # Context view-count can differ across sub-datasets sharing one
                # panel (multi: DL3DV 16 ctx vs DynamicVerse 12) → keep per-sample,
                # don't cat into one tensor. (Single-dataset behaviour unchanged.)
                context_list = [c for chunk in buf["context"] for c in chunk][:n]
                log_tensor_as_video(self.logger, sampled, f"{loader_name}/Sampled Video", fps=8, step=val_step, caption=scenes)
                log_tensor_as_video(self.logger, target, f"{loader_name}/Original Video", fps=8, step=val_step, caption=scenes)
                vis_list = []
                for j, ctx in enumerate(context_list):
                    ctx = ctx.to(self.device, non_blocking=True)
                    context_vis = add_label(hcat(*[ctx[i, ...] for i in range(ctx.shape[0])]), "Context Views")
                    vis = add_label(context_vis, scenes[j])
                    vis_list.append(prep_image(vis))
                self.logger.log_image(f"{loader_name}/Context ({self.sampler.cfg.name})", vis_list, step=val_step, caption=scenes)
                if loader_name == idx0_name:
                    log_tensor_as_video(self.logger, sampled, "Sampled Video", fps=8, step=val_step, caption=scenes)
                    log_tensor_as_video(self.logger, target, "Original Video", fps=8, step=val_step, caption=scenes)
                    self.logger.log_image(f"Context ({self.sampler.cfg.name})", vis_list, step=val_step, caption=scenes)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        for loader_name in self.predicted.keys():
            self.predicted[loader_name].clear()
            self.generated[loader_name].clear()
        for buf in self.val_vis_buffer.values():
            for v in buf.values():
                v.clear()

        print("Setting Max Timesteps for training to: ", self.model_cfg.scheduler.num_train_timesteps)
        self.scheduler.set_timesteps(self.model_cfg.scheduler.num_train_timesteps)
        return None

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None

        step = self.step_tracker.get_step()

        print(
            f"Current epoch {step}; "
            f"Step {batch_idx}; "
            f"Scene = {batch['scene']}; "
            f"Context = {batch['context']['index'].tolist()}; "
            f"Target = {batch['target']['index'].tolist()}; "
        )

        b, v_t, *_ = batch["target"]["extrinsics"].shape
        b, v_c, *_ = batch["context"]["extrinsics"].shape

        print(f"Number of context views: {v_c}")
        print(f"Number of target views: {v_t}\n")

        target_views=get_images(
            autoencoder=self.autoencoder,
            inputs=batch["target"],
            view_type="target",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        context_views=get_images(
            autoencoder=self.autoencoder,
            inputs=batch["context"],
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )

        # Relative camera w.r.t middle context camera (can be any other context camera)
        batch = preprocess_batch(batch, index=v_c//2)
        sampled_views, uncertainty_maps, _ = self.generate_batch_with_scene(
            batch, 
            self.sampler, 
            repeat_factor=1
        )

        # for j in tqdm(range(b), desc="Saving Uncertainty Maps: "):
        #     save_image_video(
        #         images=uncertainty_maps[j], 
        #         indices=torch.arange(0, uncertainty_maps[j].shape[0]), 
        #         output_dir=self.output_dir / "uncertainty" / batch["scene"][j],
        #         name=self.sampler.cfg.name, save_img=True, save_video=True, fps=self.dataset_cfg.fps
        #     )

        for j in tqdm(range(b), desc="Saving Sampled Views: "):
            save_image_video(
                images=sampled_views[j].float(), 
                indices=torch.arange(0, sampled_views[j].shape[0]), 
                output_dir=self.output_dir / "predicted" / batch["scene"][j],
                name=self.sampler.cfg.name, save_img=True, save_video=True, fps=self.dataset_cfg.fps
            )
        for j in tqdm(range(b), desc="Saving Original Views: "):
            save_image_video(
                images=target_views[j].float(), 
                indices=torch.arange(0, target_views[j].shape[0]), 
                output_dir=self.output_dir / "gt" / batch["scene"][j],
                name="original", save_img=True, save_video=True, fps=self.dataset_cfg.fps
            )

            save_image_video(
                images=context_views[j].float(), 
                indices=batch["context"]["index"][j], 
                output_dir=self.output_dir / "context" / batch["scene"][j],
                name="context", save_img=True, save_video=True, fps=self.dataset_cfg.fps
            )
        return None

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        if step >= self.model_cfg.compressor.freeze_after and self.model_cfg.compressor.freeze_after != -1 and not self.frozen_compressor:
            print(f"[INFO] Freezing Compressor after {step} steps!")
            freeze(self.compressor)
            self.frozen_compressor = True


        if type(self.optimizer_cfg.scheduler) != list:

            warmup_iters = self.optimizer_cfg.scheduler.kwargs.get("total_iters", 0)
            if step < warmup_iters and not self.override_applied:
                self.print(f"[INFO] Warmup not done yet! Current step {step} < {warmup_iters}. Overriding will happen afterwards!")
                self.override_applied = True


            if step >= warmup_iters and not self.override_applied:
                for group in self.trainer.optimizers[0].param_groups:
                    ckpt_lr = group["lr"]
                    if ckpt_lr != self.lr:
                        group["lr"] = self.lr
                        self.print(f"[INFO] Warmup done at step {step}. Overriding LR from {ckpt_lr} to {self.lr}")
                        self.override_applied = True
        else:
            if self.optimizer_cfg.override_lr is not None and not self.override_applied:


                for group in self.trainer.optimizers[0].param_groups:
                    ckpt_lr = group["lr"]
                    if ckpt_lr != self.optimizer_cfg.override_lr:
                        group["lr"] = self.optimizer_cfg.override_lr
                        self.print(f"[INFO] Overriding LR from {ckpt_lr} to {self.optimizer_cfg.override_lr}")
                        self.override_applied = True

    # NOTE: Check for nans in gradients otherwise skip the optimizer step by zeroing out grad
    def on_after_backward(self):
        if self.global_step == 0 and self.global_rank == 0:
            print("\n[DEBUG] Checking for parameters with grad=None after backward:")
            for name, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    print("  UNUSED PARAM (no grad):", name)
            print("[DEBUG] End unused-param scan\n")
        
        # DEBUG: Compute and log gradient norms
        grad_norms = []
        total_norm_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm_sq += param_norm ** 2
        
        if grad_norms:
            total_norm = total_norm_sq ** 0.5
            grad_norms_tensor = torch.tensor(grad_norms)
            avg_norm = grad_norms_tensor.mean().item()
            std_norm = grad_norms_tensor.std().item() if len(grad_norms) > 1 else 0.0
            
            self.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
            self.log("grad/avg_norm", avg_norm, on_step=True, on_epoch=False)
            self.log("grad/std_norm", std_norm, on_step=True, on_epoch=False)
        
        # scan all grads for NaN or Inf
        for name, p in self.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any() or (self.train_cfg.grad_norm_skip and avg_norm > self.train_cfg.grad_norm_skip_threshold):
                    print(f"Skipping Nans! in {name}")
                    self.log("nan_grad_skipped", 1.0, prog_bar=True)
                    # zero out everything—this makes the upcoming optimizer.step() a no-op
                    for q in self.parameters():
                        if q.grad is not None: 
                            q.grad.detach().zero_()
                        # q.grad = None
                    break  # done
    @staticmethod
    def get_optimizer(
        optimizer_cfg: OptimizerCfg,
        params: Iterator[Parameter] | list[Dict[str, Any]],
        lr: float
    ) -> optim.Optimizer:
        return getattr(optim, optimizer_cfg.name)(
            params,
            lr=lr,
            **(optimizer_cfg.kwargs if optimizer_cfg.kwargs is not None else {})       
        )
    
    
    @staticmethod
    def get_lr_scheduler(
        opt: optim.Optimizer, 
        optim_cfg: OptimizerCfg
    ) -> optim.lr_scheduler.LRScheduler:
        lr_scheduler_cfg = optim_cfg.scheduler

        if type(lr_scheduler_cfg) == list:
            return optim.lr_scheduler.SequentialLR(
                optimizer=opt,
                schedulers=[
                    getattr(optim.lr_scheduler, cfg.name)(
                        opt,
                        **(cfg.kwargs if cfg.kwargs is not None else {})     
                    ) for cfg in lr_scheduler_cfg
                ],
                milestones=optim_cfg.milestones
            )
        else:
            return getattr(optim.lr_scheduler, lr_scheduler_cfg.name)(
                opt,
                **(lr_scheduler_cfg.kwargs if lr_scheduler_cfg.kwargs is not None else {})     
            )

    def configure_optimizers(self):
        param_list = [{"params": self.denoiser.parameters()}]
        if self.model_cfg.compressor is not None:
            param_list.append({"params": self.compressor.parameters()})
        optimizer = self.get_optimizer(self.optimizer_cfg, param_list, self.lr)
        if self.optimizer_cfg.scheduler is not None:
            if type(self.optimizer_cfg.scheduler) == list:
                frequency = self.optimizer_cfg.scheduler[0].frequency
                interval = self.optimizer_cfg.scheduler[0].interval
            else:
                frequency = self.optimizer_cfg.scheduler.frequency
                interval = self.optimizer_cfg.scheduler.interval
            lr_scheduler_config = {
                "scheduler": self.get_lr_scheduler(optimizer, self.optimizer_cfg),
                "frequency": frequency,
                "interval": interval
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
