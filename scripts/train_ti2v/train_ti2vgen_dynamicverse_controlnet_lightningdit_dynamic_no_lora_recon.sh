#!/usr/bin/env bash
# Joint dynamic + background reconstruction training for the new
# `controlnet_lightningdit` (SceneTok rectified flow decoder as ControlNet
# branch + zero-init output gate).
#
#   - Main Wan DiT  → predicts `video_input.mp4` velocity (dynamic foreground).
#   - LightningDiT ctrl branch (pre-gate raw output)
#                   → supervised toward `inpaint_result.mp4` velocity
#                     (background only — preserves SceneTok recon objective).
#   - Total loss = main_diffusion + λ * recon_loss   (λ = litdit_recon_loss_weight).
# Same dataset/condition routing as dynamic_no_lora; lora off. GPU 2.

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_controlnet_lightningdit_scene_camera_2_no_lora_recon"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=controlnet
camera_input_type=controlnet_lightningdit
condition_latents_input_type=none

# ── LightningDiT ctrl branch warm-start ──────────────────────────────────
lightningdit_ckpt_path=checkpoints/va-wan_dl3dv_256-480.ckpt
litdit_recon_loss_weight=1.0

# ── LoRA on main Wan DiT — off (AC3D-strict: main frozen) ────────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,controlnet_lightningdit,scene+camera,video_input,category,warmstart,joint_recon]'

export WANDB_API_KEY=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
export DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=2 exec -a litdit_recon_no_lora_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.recon_target_video_name=inpaint_result.mp4 \
  dataset.prompt_style=category_first \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  +model.denoiser.lightningdit_ckpt_path=${lightningdit_ckpt_path} \
  +model.denoiser.litdit_recon_loss_weight=${litdit_recon_loss_weight} \
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
