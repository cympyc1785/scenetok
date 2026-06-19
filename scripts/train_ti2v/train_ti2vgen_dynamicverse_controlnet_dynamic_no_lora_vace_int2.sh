#!/usr/bin/env bash
# VACE식 interval injection 변종: ctrl 잔차를 main block 0..N-1(연속)이 아니라
# interval=2로 [0,2,4,…,28] 전체에 분산 주입 (Wan 30블록 → 15 injection points).
# ctrl block 수는 interval로 자동 결정(ac3d_num_layers 무시). camera_input_type=controlnet
# (parallel) 모드에서만 동작. GPU 0.

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_vace_int2"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=controlnet
camera_input_type=controlnet
condition_latents_input_type=none

# ── VACE-style interval injection ─────────────────────────────────────────
controlnet_inject_interval=2

# ── LoRA on main Wan DiT (off) ─────────────────────────────────────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,controlnet,scene+camera,video_input,category,vace_interval2]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} exec -a vace_int2_scenetok_lets_go python -m src.main +experiment=${config} \
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
  +model.denoiser.controlnet_inject_interval=${controlnet_inject_interval} \
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
