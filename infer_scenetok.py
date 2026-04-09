

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
from src.model.types import CompressorInputs,  DenoiserInputs
from src.misc.batch_utils import preprocess_batch, batch_expand, batch_cast
from src.misc.image_io import play_tensor_video
from src.misc.torch_utils import freeze
from src.model.autoencoder import get_autoencoder, AutoencodersCfg
from src.model.compressor import MVAECompressorCfg, MVAECompressor
from src.model.denoiser import LightningDiTCfg, LightningDiT, Denoiser
from src.model.scheduler import RectifiedFlowMatchingScheduler, RectifiedFlowMatchingSchedulerCfg
from src.model.sampler.full_sequence import FullSequenceSampler, FullSequenceSamplerCfg
from src.model.diffusion import get_images, get_latents, latent_to_original_index, original_to_latent_index, last_stage_decode

from src.model.types import CameraInputs

# ===== Config

with initialize(version_base=None, config_path="./config"):
    cfg = compose(config_name="main.yaml", overrides=[
        
        
        #----------------------------------------------
        # RE10K
        #----------------------------------------------
        "dataset=re10k",
        "dataset.root=<RE10K-DATA_ROOT>",
        
        # VAVAE + VideoDC
        "+experiment=scenetok_va-vdc_lognorm_re10k_scratch",
        "model.denoiser.ckpt_path=./checkpoints/va-videodc_re10k.ckpt",
        "model.compressor.ckpt_path=./checkpoints/va-videodc_re10k.ckpt",
        "dataset/view_sampler=evaluation_video",
        "dataset.view_sampler.num_target_views=8",
        "dataset.view_sampler.temporal_downsample=4",
        "dataset.view_sampler.num_context_views=16",
        "dataset.view_sampler.index_path=./assets/evaluation_index/re10k_c12_128.json"

        #----------------------------------------------
        # DL3DV
        #----------------------------------------------

        # "dataset=dl3dv",
        # "dataset.root=<DL3DV-DATA_ROOT>",
        
        # # VAVAE + VideoDC
        # "+experiment=scenetok_va-vdc_shift8_dl3dv_finetuned",
        # "model.denoiser.ckpt_path=./checkpoints/va-videodc_dl3dv.ckpt",
        # "model.compressor.ckpt_path=./checkpoints/va-videodc_dl3dv.ckpt",
        # "dataset/view_sampler=evaluation_video",
        # "dataset.view_sampler.num_target_views=8",
        # "dataset.view_sampler.temporal_downsample=4",
        # "dataset.view_sampler.num_context_views=16",
        # "dataset.view_sampler.index_path=./assets/evaluation_index/dl3dv_c16_64.json"

        # # VAVAE + WAN
        # "+experiment=scenetok_va-wan_shift4_dl3dv_finetuned",
        # "model.denoiser.ckpt_path=./checkpoints/va-wan_dl3dv.ckpt",
        # "model.compressor.ckpt_path=./checkpoints/va-wan_dl3dv.ckpt",
        # "dataset/view_sampler=evaluation_video_wan",
        # "dataset.view_sampler.num_target_views=10",
        # "dataset.view_sampler.temporal_downsample=4",
        # "dataset.view_sampler.num_context_views=16",
        # "dataset.view_sampler.index_path=./assets/evaluation_index/dl3dv_c16_64.json"

    ])


    model_cfg = load_typed_config(cfg.model, ModelCfg)
    compressor_cfg = load_typed_config(cfg.model.compressor, MVAECompressorCfg)    
    denoiser_cfg = load_typed_config(cfg.model.denoiser, LightningDiTCfg)
    autoencoders_cfg = load_typed_config(cfg.model.autoencoders, AutoencodersCfg)
    scheduler_cfg = load_typed_config(cfg.model.scheduler, RectifiedFlowMatchingSchedulerCfg)
    sampler_cfg = load_typed_config(cfg.sampler, FullSequenceSamplerCfg)

    # Select the correct dataset type
    dataset_cfg = load_typed_config(cfg.dataset, DatasetDL3DVCfg)
    # dataset_cfg = load_typed_config(cfg.dataset, DatasetRE10kCfg)
    
# autoencoders_cfg.target.kwargs.tile_overlap_factor = 0.0

# Only Video DCAE has temporal compression of factor 4
if getattr(autoencoders_cfg, "target").name in ["video_dc", "wan"]:
    temporal_downsample = 4

else:
    temporal_downsample = 1

num_scene_tokens = compressor_cfg.num_scene_tokens
token_dim = compressor_cfg.token_dim


# ===== Load Models


scheduler = RectifiedFlowMatchingScheduler(
    **scheduler_cfg.kwargs.__dict__
) 

sampler = FullSequenceSampler(cfg=sampler_cfg)
# Load the multi-view compressor
compressor = MVAECompressor(
    cfg=compressor_cfg,
    in_channels=autoencoders_cfg.context.kwargs.latent_channels,
    num_views=dataset_cfg.view_sampler.num_context_views,
    temporal_downsample=1 # We never use temporal compression in the compressor

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

# Load separate autoencoder for context and target
autoencoders = {
    "context": get_autoencoder(autoencoders_cfg.context).to("cuda").to(torch.bfloat16),
    "target": get_autoencoder(autoencoders_cfg.target).to("cuda").to(torch.bfloat16)
}

dataset = get_dataset(dataset_cfg, stage="test", step_tracker=None)

freeze(denoiser)
freeze(compressor)
freeze(autoencoders["context"])
freeze(autoencoders["target"])


# ===== Sampling Functions

# Defining it separately but does the same as that defined in src/model/diffusion.py

from einops import repeat

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
            cond_state_uc = denoiser.null_tokens.expand(b, denoiser.num_scene_tokens, -1)            
            denoiser_input.state = cond_state_uc
            pred_unconditional, _ = model._forward(inputs=denoiser_input, temporal_downsample=temporal_downsample)
            pred_out = pred_unconditional + cfg_scale * (pred_conditional - pred_unconditional)

        else:
            pred_out = pred_conditional


    sch_out = scheduler.step(pred_out, ts, x_t).prev_sample


    return sch_out, qk_list, pred_conditional

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

# Renderers

from matplotlib import pyplot as  plt


def render(model, tokens, batch, shift=None, x_t=None, repeat_factor=1):

    repeated_tokens = tokens.repeat(repeat_factor, 1, 1)

    batch_size = repeated_tokens.shape[0]
    target_pose = CameraInputs(
        intrinsics=batch["target"]["intrinsics"].repeat(repeat_factor, 1, 1, 1),
        extrinsics=batch["target"]["extrinsics"].repeat(repeat_factor, 1, 1, 1),
    )


    v_t = target_pose.extrinsics.shape[1]

    temporal_downsample = 1
    if getattr(autoencoders_cfg, "target") is not None:
        if getattr(autoencoders_cfg, "target").name == "video_dc":
            temporal_downsample = 4
            num = (v_t // temporal_downsample)
            target_pose.extrinsics = target_pose.extrinsics[:, :num*temporal_downsample]
            target_pose.intrinsics = target_pose.intrinsics[:, :num*temporal_downsample]
        
        elif getattr(autoencoders_cfg, "target").name == "wan":
            temporal_downsample = 4
            num = (v_t // 17) * 5

            target_pose.extrinsics = target_pose.extrinsics[:, :(num//5)*17]
            target_pose.intrinsics = target_pose.intrinsics[:, :(num//5)*17]
    # print(len(target_pose))
    if x_t is None:
        if autoencoders_cfg.target.name in ["kl"]:
            c = autoencoders_cfg.target.kwargs.latent_channels // 2
        else:
            c = autoencoders_cfg.target.kwargs.latent_channels

        # print("Total target views: ", num)
        h, w = denoiser_cfg.input_shape
        x_t = torch.randn((batch_size, num, c, h, w)).to(tokens.dtype).to(tokens.device)  
    
    
    sampler.set_scheduling_matrix(
        horizon=num,
        steps=scheduler_cfg.num_inference_steps, 
        concurrency=dataset_cfg.view_sampler.num_target_views, 
        device=tokens.device,
        dtype=tokens.dtype,
        cond_mask_indices=None,
        # cond_mask_indices=[0, 1, 2, 3],
        clean_targets=sampler_cfg.clean_targets   # Number of clean input target to give to the denoiser. Note that total window size is 8
    )

    sampler.shift_scheduling_matrix(shift=shift if shift is not None else scheduler_cfg.kwargs.timestep_shift)
    sampler.get_visualization()

    target_rendering, _ = sample(
        model=model,
        x_t=x_t.clone(), 
        target_pose=target_pose, 
        cond_state=repeated_tokens, 
        temporal_downsample=temporal_downsample,
        sampler=sampler,
        cfg_scale=1.0
    )

    plt.imshow(sampler.get_visualization())

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

# ===== Load Data

# Specifiy specific scene with the scene id
# Or manually set the index

# RE10K
dataset.overfit_to_scene = ["8f06df2fca9350ba"]
idx = 0 # Index is now relative to the above ovefit list

# DL3DV
# idx = dataset.chunks.index("165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557")


batch = dataset[idx]
batch["context"] = batch_expand(batch["context"])    
batch["target"] = batch_expand(batch["target"]) 

# Index defines the index of the context camera to be used as reference for computing relative poses
batch = preprocess_batch(batch, index=0) 

batch["context"] = batch_cast(batch["context"], torch.bfloat16)    
batch["target"] = batch_cast(batch["target"], torch.bfloat16)
batch["context"] = batch_cast(batch["context"], torch.device("cuda"))    
batch["target"] = batch_cast(batch["target"], torch.device("cuda"))

context_latents = get_latents(
    autoencoder=autoencoders,
    inputs=batch["context"],  
    view_type="context",
    precomputed_latents=dataset_cfg.precomputed_latents,
    autoencoder_name=autoencoders_cfg.context.name,
    scaling_factor=autoencoders_cfg.context.kwargs.scaling_factor,
    
)

# ===== Compute Scene Tokens

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


# ===== Render NVS

# Single chunk sampling is default to 25 steps
target_rendering, uncertainty_map = render(
    denoiser, 
    tokens, 
    batch, 
    shift=scheduler_cfg.kwargs.timestep_shift if scheduler_cfg.kwargs.timestep_shift is not None else 1, 
    repeat_factor=1 # Set to > 1 to sample multiple times for visualizing uncertainty
)

# ===== Visualize

play_tensor_video(target_rendering[0].float().cpu())
