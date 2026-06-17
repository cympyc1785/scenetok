#!/usr/bin/env bash
# Variant of train_ti2vgen_dynamicverse_controlnet_dynamic_no_lora_no_text_ctrl.sh:
# parallel controlnet + dynamic + LoRA off + ctrl text-CA off + **ac3d_num_layers=6**
# (default 2 → 6, AC3D paper 비율). GPU 3.

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_no_text_ctrl_layer_6"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=controlnet
camera_input_type=controlnet
condition_latents_input_type=none

# ── Ctrl branch depth (default 2 → 6) ─────────────────────────────────────
ac3d_num_layers=6

# ── LoRA on main Wan DiT (off) ─────────────────────────────────────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,controlnet,scene+camera,video_input,category,no_text_ctrl,layer_6]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=3 exec -a no_text_ctrl_layer_6_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.prompt_style=category_first \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  +model.denoiser.controlnet_no_text_cross_attn=true \
  model.denoiser.ac3d_num_layers=${ac3d_num_layers} \
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
