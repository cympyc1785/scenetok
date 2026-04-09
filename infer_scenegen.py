

import os
import sys

import torch
import numpy as np

from tqdm import tqdm
from jaxtyping import Float, Bool
from torch import Tensor
from einops import rearrange
from typing import Optional
from hydra import initialize, compose


# Custom imports
from src.dataset import get_dataset, DatasetDL3DVCfg, DatasetRE10kCfg
from src.config import load_typed_config, ModelCfg
from src.model.types import CompressorInputs, DenoiserInputs, SceneGeneratorInputs, CameraInputs
from src.misc.batch_utils import preprocess_batch, batch_expand, batch_cast
from src.misc.image_io import play_tensor_video
from src.misc.torch_utils import freeze
from src.model.autoencoder import get_autoencoder, AutoencodersCfg
from src.model.compressor import MVAECompressorCfg, MVAECompressor
from src.model.denoiser import LightningDiTCfg, LightningDiT, Denoiser
from src.model.scheduler import RectifiedFlowMatchingScheduler, RectifiedFlowMatchingSchedulerCfg
from src.model.sampler.full_sequence import FullSequenceSampler, FullSequenceSamplerCfg
from src.model.diffusion import get_images, get_latents, latent_to_original_index, original_to_latent_index, last_stage_decode

from src.model.scene_generator import SceneGenerator, SceneGeneratorCfg

# ===== Setup

with initialize(version_base=None, config_path="./config"):
    cfg = compose(config_name="main.yaml", overrides=[
        
        #----------------------------------------------
        # RE10K - SceneGenerator
        #----------------------------------------------
        "dataset=re10k",
        "dataset.root=<RE10K-DATA_ROOT>",
        "dataset/view_sampler=evaluation_video",
        "dataset.view_sampler.max_cond_number=3",

        # Scene Generator with VAVAE + VideoDC

        # # Shift: 1
        # "+experiment=scenegen_shift1_re10k",
        # "model.denoiser.ckpt_path=./checkpoints/va-videodc_re10k_scene.ckpt",
        # "model.compressor.ckpt_path=./checkpoints/va-videodc_re10k_scene.ckpt",
        # "model.scene_generator.ckpt_path=./checkpoints/scenegen_shift1_re10k.ckpt",

        # # Shift: 4
        "+experiment=scenegen_shift4_re10k",
        "model.denoiser.ckpt_path=./checkpoints/va-videodc_re10k_scene.ckpt",
        "model.compressor.ckpt_path=./checkpoints/va-videodc_re10k_scene.ckpt",
        "model.scene_generator.ckpt_path=./checkpoints/scenegen_shift4_re10k.ckpt",

        # Shift: 12
        # "+experiment=scenegen_shift12_re10k",
        # "model.denoiser.ckpt_path=./checkpoints/va-videodc_re10k_scene.ckpt",
        # "model.compressor.ckpt_path=./checkpoints/va-videodc_re10k_scene.ckpt",
        # "model.scene_generator.ckpt_path=./checkpoints/scenegen_shift12_re10k.ckpt",


        "model.scene_generator.load_strict=false",
        "dataset.view_sampler.num_target_views=8",
        "dataset.view_sampler.temporal_downsample=4",
        "dataset.view_sampler.num_context_views=12",
        "dataset.view_sampler.index_path=./assets/evaluation_index/re10k_c1_192.json",
        "dataset.precomputed_latents.context=false", 
        "dataset.precomputed_latents.target=false",

    ])

    model_cfg = load_typed_config(cfg.model, ModelCfg)
    compressor_cfg = load_typed_config(cfg.model.compressor, MVAECompressorCfg)    
    denoiser_cfg = load_typed_config(cfg.model.denoiser, LightningDiTCfg)
    autoencoders_cfg = load_typed_config(cfg.model.autoencoders, AutoencodersCfg)
    scheduler_cfg = load_typed_config(cfg.model.scheduler, RectifiedFlowMatchingSchedulerCfg)
    scene_scheduler_cfg = load_typed_config(cfg.model.scene_scheduler, RectifiedFlowMatchingSchedulerCfg)
    sampler_cfg = load_typed_config(cfg.sampler, FullSequenceSamplerCfg)
    scenegen_cfg = load_typed_config(cfg.model.scene_generator, SceneGeneratorCfg)

    # Select the correct dataset
    dataset_cfg = load_typed_config(cfg.dataset, DatasetRE10kCfg)

# Only Video DCAE has temporal compression of factor 4
if getattr(autoencoders_cfg, "target").name in ["video_dc", "wan"]:
    temporal_downsample = 4
else:
    temporal_downsample = 1

num_scene_tokens = compressor_cfg.num_scene_tokens
token_dim = compressor_cfg.token_dim

# Scene generator configs
scene_cfg_scale = model_cfg.scene_cfg_scale if hasattr(model_cfg, 'scene_cfg_scale') else 3.0

# Make sure to load from root dir since the current directory is /notebook
autoencoders_cfg.context.pretrained_from = "./" + autoencoders_cfg.context.pretrained_from
autoencoders_cfg.target.pretrained_from = "./" + autoencoders_cfg.target.pretrained_from


# ===== Load Models

scheduler = RectifiedFlowMatchingScheduler(
    **scheduler_cfg.kwargs.__dict__
) 

scene_scheduler = RectifiedFlowMatchingScheduler(
    **scene_scheduler_cfg.kwargs.__dict__
) 
sampler = FullSequenceSampler(cfg=sampler_cfg)

# Load the multi-view compressor
compressor = MVAECompressor(
    cfg=compressor_cfg,
    in_channels=autoencoders_cfg.context.kwargs.latent_channels,
    num_views=dataset_cfg.view_sampler.num_context_views,
    temporal_downsample=1  # We never use temporal compression in the compressor

).to("cuda").to(torch.bfloat16)

# Load the main renderer / denoiser
denoiser = LightningDiT(
    cfg=denoiser_cfg,
    cond_dim=token_dim,
    num_scene_tokens=num_scene_tokens,
    num_views=dataset_cfg.view_sampler.num_target_views,
    temporal_downsample=temporal_downsample if not model_cfg.force_incorrect else 1,
    using_wan=True if "wan" in getattr(autoencoders_cfg, "target").name else False

).to("cuda").to(torch.bfloat16)

# Load scene generator
scenegen = SceneGenerator(
    cfg=scenegen_cfg,
    cond_dim=token_dim,
    num_scene_tokens=num_scene_tokens,
    temporal_downsample=1

).to("cuda").to(torch.bfloat16)

# Load separate autoencoder for context and target
autoencoders = {
    "context": get_autoencoder(autoencoders_cfg.context).to("cuda").to(torch.bfloat16),
    "target": get_autoencoder(autoencoders_cfg.target).to("cuda").to(torch.bfloat16)
}

dataset = get_dataset(dataset_cfg, stage="test", step_tracker=None)

freeze(denoiser)
freeze(compressor)
freeze(scenegen)
freeze(autoencoders["context"])
freeze(autoencoders["target"])


# ===== Sampler

def step(
    model: Denoiser, 
    x_t: Float[Tensor, "batch view channel height width"], 
    ts: Float[Tensor, "batch view"], 
    target_pose: CameraInputs, 
    cond_state: Optional[Float[Tensor, "batch num _"]]=None, 
    temporal_downsample: int=1,
    cfg_scale: float=3.0,
):
    
    b, v_t, *_ = x_t.shape 
    x_t_inputs = scheduler.scale_model_input(x_t, ts)
    
    t = (ts * scheduler.num_train_timesteps - 1).clip(min=0)
    
    inputs = x_t_inputs.clone()

    # Conditional Forward Pass
    cond_state = model.cnd_proj(cond_state)

    denoiser_input = DenoiserInputs(
        view=inputs, 
        pose=target_pose, 
        timestep=t, 
        state=cond_state
    )

    with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
        pred_conditional, qk_list = model._forward(inputs=denoiser_input, temporal_downsample=temporal_downsample)

        if cfg_scale > 1.0:
            cond_state_uc = model.null_tokens.expand(b, model.num_scene_tokens, -1)            
            denoiser_input.state = cond_state_uc
            pred_unconditional, _ = model._forward(inputs=denoiser_input, temporal_downsample=temporal_downsample)
            pred_out = pred_unconditional + cfg_scale * (pred_conditional - pred_unconditional)
        else:
            pred_out = pred_conditional

    sch_out = scheduler.step(pred_out, ts, x_t).prev_sample
    return sch_out, qk_list, pred_conditional

def step_scene(
    model: SceneGenerator, 
    x_t: Float[Tensor, "batch num dim"], 
    ts: Float[Tensor, "batch"], 
    anchor_pose: CameraInputs,
    cond_latents: Float[Tensor, "batch view channels height width"],
    cond_pose: CameraInputs,
    cond_mask: Bool[Tensor, "batch cond_view"],
    guidance_scale: float=3.0
):
    
    b, v_t, *_ = x_t.shape 
    x_t_inputs = scheduler.scale_model_input(x_t, ts)
    
    t = (ts * scheduler.num_train_timesteps - 1).clip(min=0)
    
    inputs = x_t_inputs.clone()

    generator_input = SceneGeneratorInputs(
        view=cond_latents, 
        pose=cond_pose, 
        anchor_pose=anchor_pose,
        timestep=t, 
        state=inputs
    )
    with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
        pred_conditional, qk_list = model._forward(inputs=generator_input, cond_mask=cond_mask)
    
    if guidance_scale > 1.0:
        uncond_mask = torch.zeros_like(cond_mask, dtype=torch.bool, device=cond_mask.device)
        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            pred_unconditional, _ = model._forward(inputs=generator_input, cond_mask=uncond_mask)
        pred_out = pred_unconditional + guidance_scale * (pred_conditional - pred_unconditional)
    else:
        pred_out = pred_conditional

    sch_out = scheduler.step(pred_out, ts, x_t).prev_sample
    return sch_out, qk_list, pred_conditional

@torch.no_grad()
def sample_scene(
    model: SceneGenerator,
    x_t: Tensor,
    cond_latents: Float[Tensor, "batch view channels height width"],
    anchor_pose: CameraInputs,
    cond_pose: CameraInputs,
    sampler: FullSequenceSampler,
    num_cond: int=1,
    guidance_scale: float=3.0
):

    b = anchor_pose.extrinsics.shape[0]
    device = anchor_pose.extrinsics.device

    if x_t is None:
        x_t = torch.randn((b, num_scene_tokens, compressor.output_dim), device=device)

    cond_mask = torch.zeros((b, dataset_cfg.view_sampler.max_cond_number), device=device, dtype=torch.bool)
    cond_mask[:, :num_cond] = True
    pbar = tqdm(range(sampler.global_steps), desc=f"Sampling Scene with {num_cond} conditioning: ")
    for m in pbar:
        ts, denoise_mask = sampler(m)
        ts_next, _ = sampler(m+1)
        
        ts = repeat(ts, "n -> b n", b=b).to(device)
        ts_next = repeat(ts_next, "n -> b n", b=b).to(device)

        scheduler.set_scheduling_matrix(ts_next)
        # Denoise within the sliding window
        x_t, _, _ = step_scene(
            model=model, 
            x_t=x_t, 
            ts=ts, 
            anchor_pose=anchor_pose, 
            cond_latents=cond_latents,
            cond_pose=cond_pose,
            cond_mask=cond_mask,
            guidance_scale=guidance_scale
        )
        scheduler.unset_scheduling_matrix()

    return x_t

@torch.no_grad()
def sample(
    model: Denoiser,
    x_t: Float[Tensor, "batch view channel height width"], 
    target_pose: CameraInputs, 
    cond_state: torch.Tensor, 
    sampler: FullSequenceSampler,
    temporal_downsample: int=1,
    cfg_scale: int=3.0,
):
    device = x_t.device
    b, v_t, c, h, w = x_t.shape

    pbar = tqdm(range(sampler.global_steps), desc=f"Sampling ({sampler.cfg.name}): ")
    for m in pbar:
        ts, denoise_mask = sampler(m)
        ts_next, _ = sampler(m+1)

        ts = repeat(ts, "v -> b v", b=b).to(x_t.device)
        ts_next = repeat(ts_next, "v -> b v", b=b).to(x_t.device)
        scheduler.set_scheduling_matrix(ts_next[:, denoise_mask])
        new_denoise_mask = repeat(denoise_mask, "n -> (n t)", t=temporal_downsample)
        num_done = sampler.current_frame(m)+dataset_cfg.view_sampler.num_target_views - sampler_cfg.clean_targets

        if getattr(autoencoders_cfg, "target") is not None:
            if getattr(autoencoders_cfg, "target").name == "wan":
                num_pose = (v_t // 5) * 17 
                views_done = (num_done // 5) * 17
                view_step=17
                idx = torch.nonzero(denoise_mask).squeeze(1)
                start = latent_to_original_index(idx[0], temporal_downsample, dataset_cfg.view_sampler.chunk_index_gap, dataset_cfg.view_sampler.offset)
                end = latent_to_original_index(idx[-1]+1, temporal_downsample, dataset_cfg.view_sampler.chunk_index_gap, dataset_cfg.view_sampler.offset)
                range_idx = torch.arange(start, end)
                new_denoise_mask = torch.zeros((num_pose,), device=denoise_mask.device, dtype=torch.bool)
                new_denoise_mask[range_idx] = True
            if getattr(autoencoders_cfg, "target").name == "video_dc":
                view_step=16
                num_pose = v_t * 4 
                views_done = num_done * 4 
        pbar.set_postfix({
            "Latents:": f" {num_done}/{sampler.total_frames}",
            "Views:": f" {views_done}/{num_pose}"
        })
        
        # Denoise within the sliding window
        x_t[:, denoise_mask], qk_list, pred_conditional = step(
            model=model, 
            x_t=x_t[:, denoise_mask], 
            ts=ts[:, denoise_mask], 
            target_pose=target_pose[:, new_denoise_mask], 
            cond_state=cond_state, 
            temporal_downsample=temporal_downsample,
            cfg_scale=cfg_scale
        )
        scheduler.unset_scheduling_matrix()

    uncertainty_map = pred_conditional.norm(dim=2) 
    print("Decoding to images...")
    if autoencoders_cfg.target.name == "video_dc":
        decoded = last_stage_decode(
                autoencoder=autoencoders,
                latents=x_t, 
                view_type="target",
                autoencoder_name=autoencoders_cfg.target.name,
                scaling_factor=autoencoders_cfg.target.kwargs.scaling_factor,
            )
    elif autoencoders_cfg.target.name == "wan":
        decoded = []
        for x in torch.split(x_t, split_size_or_sections=5, dim=1):
            decoded.append(last_stage_decode(
                autoencoder=autoencoders,
                latents=x, 
                view_type="target",
                autoencoder_name=autoencoders_cfg.target.name,
                scaling_factor=autoencoders_cfg.target.kwargs.scaling_factor,
            ))
        decoded = torch.concat(decoded, dim=1)
    return decoded, uncertainty_map


# ===== Load data

# Specify a scene to visualize
# RE10K (scenes available in re10k_c1_192.json index)
dataset.overfit_to_scene = ["c24b14dbd06b6f12"]
idx = 0

batch = dataset[idx]
batch["context"] = batch_expand(batch["context"])    
batch["target"] = batch_expand(batch["target"]) 

# Index defines the index of the context camera to be used as reference for computing relative poses
batch = preprocess_batch(batch, index=0) 

batch["context"] = batch_cast(batch["context"], torch.bfloat16)    
batch["target"] = batch_cast(batch["target"], torch.bfloat16)
batch["context"] = batch_cast(batch["context"], torch.device("cuda"))    
batch["target"] = batch_cast(batch["target"], torch.device("cuda"))

# Create conditioning input (c1 = 1 conditioning view from first target frame)
batch["cond"] = {
    "extrinsics": batch["target"]["extrinsics"].clone()[:, [0, -1, -1]],
    "intrinsics": batch["target"]["intrinsics"].clone()[:, [0, -1, -1]],
    "latent": batch["target"]["latent"].clone()[:, [0, -1, -1]],  
    "index": batch["target"]["index"].clone()[:, [0, -1, -1]]
}

# ===== Conditions

# Select context views (uniformly sampled from target views)
ctx_idx = torch.linspace(0, batch["target"]["extrinsics"].shape[1]-1, dataset_cfg.view_sampler.num_context_views, device=batch["target"]["extrinsics"].device).long()
batch["context"] = {
    "extrinsics": batch["target"]["extrinsics"].clone()[:,ctx_idx],
    "intrinsics": batch["target"]["intrinsics"].clone()[:,ctx_idx],
    "latent": batch["target"]["latent"].clone()[:,ctx_idx], 
    "index": batch["target"]["index"].clone()[:,ctx_idx],
}
print(f"Context indices: {ctx_idx}")

# Compute context latents
context_latents = get_latents(
    autoencoder=autoencoders,
    inputs=batch["context"],  
    view_type="context",
    precomputed_latents=dataset_cfg.precomputed_latents,
    autoencoder_name=autoencoders_cfg.context.name,
    scaling_factor=autoencoders_cfg.context.kwargs.scaling_factor,
)

# Compute conditioning latents for scene generator
cond_latents = get_latents(
    autoencoder=autoencoders,
    inputs=batch["cond"],  
    view_type="context",
    precomputed_latents=dataset_cfg.precomputed_latents,
    autoencoder_name=autoencoders_cfg.context.name,
    scaling_factor=autoencoders_cfg.context.kwargs.scaling_factor,
)

# ===== Inference

# Predict scene tokens from context views
# NOTE: use ._forward to avoid torch.compile during debug
# Forward pass only possible in fp16 or bf16 because of flash attention
# Outputs are scene tokens of shape (B, N, C) where N is number of scene tokens
# and C is token dimension

context_camera = CameraInputs(
    intrinsics=batch["context"]["intrinsics"].bfloat16(),
    extrinsics=batch["context"]["extrinsics"].bfloat16(),
)

context_inputs = CompressorInputs(
    view=context_latents,
    pose=context_camera,
    mask=None
)

with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
    tokens, qks = compressor._forward(inputs=context_inputs)

if compressor_cfg.scene_token_projection == "kl":
    tokens = tokens.sample().to(torch.bfloat16)
else:
    tokens = tokens.to(torch.bfloat16)

print(f"Scene tokens shape: {tokens.shape}")

# ===== Helper

# Define token generation wrapper function to handle conditioning and sampling
def get_scene_tokens(model, batch, shift=None, x_t=None, repeat_factor=1, num_cond: int=1, steps=150, guidance_scale=3.0):
    """Generate scene tokens using the scene generator model."""
    
    scene_sampler = FullSequenceSampler(FullSequenceSamplerCfg(name="full_sequence"))

    # Get conditioning latents
    cond_latents_local = get_latents(
        autoencoder=autoencoders,
        inputs=batch["cond"],
        view_type="context",
        precomputed_latents=dataset_cfg.precomputed_latents,
        autoencoder_name=autoencoders_cfg.context.name,
        scaling_factor=autoencoders_cfg.context.kwargs.scaling_factor,
    )
    
    device = cond_latents_local.device
    dtype = cond_latents_local.dtype
    b, v_t, *_ = batch["target"]["extrinsics"].shape

    anchor_pose = CameraInputs(
        intrinsics=batch["context"]["intrinsics"].repeat(repeat_factor, 1, 1, 1),
        extrinsics=batch["context"]["extrinsics"].repeat(repeat_factor, 1, 1, 1)
    )
    cond_pose = CameraInputs(
        intrinsics=batch["cond"]["intrinsics"].repeat(repeat_factor, 1, 1, 1),
        extrinsics=batch["cond"]["extrinsics"].repeat(repeat_factor, 1, 1, 1)
    )

    scene_sampler.set_scheduling_matrix(
        horizon=num_scene_tokens,
        steps=steps, 
        concurrency=num_scene_tokens, 
        device=device,
        dtype=dtype,
        cond_mask_indices=None
    )
    scene_sampler.shift_scheduling_matrix(shift)
    
    scene_tokens_gen = sample_scene(
        model=model, 
        x_t=x_t,
        cond_latents=cond_latents_local.repeat(repeat_factor, 1, 1, 1, 1), 
        anchor_pose=anchor_pose, 
        cond_pose=cond_pose, 
        num_cond=num_cond,
        sampler=scene_sampler,
        guidance_scale=guidance_scale
    )

    scene_tokens_gen = scene_tokens_gen / scenegen_cfg.scale_factor

    return scene_tokens_gen


# Define rendering wrapper function to handing token and pose conditioning 
from matplotlib import pyplot as plt

def render(model, tokens, batch, shift=None, x_t=None, repeat_factor=1, clean_targets=0):
    """Render images from scene tokens."""
    
    repeated_tokens = tokens.repeat(repeat_factor, 1, 1)
    batch_size = repeated_tokens.shape[0]
    
    target_pose = CameraInputs(
        intrinsics=batch["target"]["intrinsics"].repeat(repeat_factor, 1, 1, 1),
        extrinsics=batch["target"]["extrinsics"].repeat(repeat_factor, 1, 1, 1),
    )

    v_t = target_pose.extrinsics.shape[1]
    temporal_downsample_local = 1
    
    if getattr(autoencoders_cfg, "target") is not None:
        if getattr(autoencoders_cfg, "target").name == "video_dc":
            temporal_downsample_local = 4
            num = (v_t // temporal_downsample_local)
            target_pose.extrinsics = target_pose.extrinsics[:, :num*temporal_downsample_local]
            target_pose.intrinsics = target_pose.intrinsics[:, :num*temporal_downsample_local]
        
        elif getattr(autoencoders_cfg, "target").name == "wan":
            temporal_downsample_local = 4
            num = (v_t // 17) * 5
            target_pose.extrinsics = target_pose.extrinsics[:, :(num//5)*17]
            target_pose.intrinsics = target_pose.intrinsics[:, :(num//5)*17]

    if x_t is None:
        if autoencoders_cfg.target.name in ["kl"]:
            c = autoencoders_cfg.target.kwargs.latent_channels // 2
        else:
            c = autoencoders_cfg.target.kwargs.latent_channels

        h, w = denoiser_cfg.input_shape
        x_t = torch.randn((batch_size, num, c, h, w)).to(tokens.dtype).to(tokens.device)  
    
    sampler.set_scheduling_matrix(
        horizon=num,
        steps=scheduler_cfg.num_inference_steps, 
        concurrency=dataset_cfg.view_sampler.num_target_views, 
        device=tokens.device,
        dtype=tokens.dtype,
        cond_mask_indices=None,
        clean_targets=clean_targets
    )

    sampler.shift_scheduling_matrix(shift=shift if shift is not None else scheduler_cfg.kwargs.timestep_shift)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(sampler.get_visualization())
    plt.title("Sampling Schedule")
    plt.show()

    target_rendering, _ = sample(
        model=model,
        x_t=x_t.clone(), 
        target_pose=target_pose, 
        cond_state=repeated_tokens, 
        temporal_downsample=temporal_downsample_local,
        sampler=sampler,
        cfg_scale=1.0
    )

    target_rendering = target_rendering.clamp(0, 1)
    target_rendering = target_rendering.cpu().float()
    uncertainty_map = None
    
    if repeat_factor > 1:
        uncertainty_map = target_rendering.std(dim=0, keepdim=True).mean(dim=2, keepdim=True)
        u_min, _ = uncertainty_map.reshape(1, -1).min(-1)
        u_max, _ = uncertainty_map.reshape(1, -1).max(-1)
        u_min = u_min[:, None, None, None, None]
        u_max = u_max[:, None, None, None, None]
        normalized_uncertainty = (uncertainty_map - u_min) / (u_max - u_min + 1e-8)

    return target_rendering, uncertainty_map


# Initialize scene tokens from random noise
repeat_factor = 1
x_t = torch.randn((repeat_factor, num_scene_tokens, compressor.output_dim), device=tokens.device)
print(f"Initial noise shape: {x_t.shape}")


# Generate scene tokens using the scene generator
tokens_gen = get_scene_tokens(
    scenegen, 
    batch, 
    shift=scene_scheduler_cfg.kwargs.timestep_shift if scene_scheduler_cfg.kwargs.timestep_shift is not None else 1,  # Use the shift from config
    x_t=x_t, 
    repeat_factor=repeat_factor, 
    num_cond=1,  # Number of conditioning views to use
    steps=150, 
    guidance_scale=3.0 
)

# ===== Render

# Render the generated scene tokens (using variance-corrected tokens)
target_rendering, uncertainty_map = render(
    denoiser, 
    tokens_gen.bfloat16(),  # Use corrected tokens
    batch, 
    shift=1,  # Use the shift from config
    repeat_factor=repeat_factor,
    clean_targets=sampler_cfg.clean_targets
)
print(f"Rendered video shape: {target_rendering.shape}")

# Visualize the rendered video
play_tensor_video(target_rendering[0].float().cpu(), fps=24)



# Render using predicted tokens (from context views)
target_rendering_enc, _ = render(
    denoiser, 
    tokens,  # tokens from compressor, not generated
    batch, 
    shift=1, 
    repeat_factor=1,
    clean_targets=sampler_cfg.clean_targets
)
print(f"Rendered video from compressor tokens shape: {target_rendering_enc.shape}")

# Play video from compressor tokens (encoding from context views)
play_tensor_video(target_rendering_enc[0].float().cpu(), fps=30)