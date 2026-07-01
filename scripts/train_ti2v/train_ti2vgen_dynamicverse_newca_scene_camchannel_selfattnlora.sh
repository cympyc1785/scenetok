#!/usr/bin/env bash
# Condition B of the camera-blindness probe: identical to
# train_ti2vgen_dynamicverse_newca_scene_camchannel_dynamic_no_lora.sh
# (channel_concat camera + new_cross_attention scene) but ALSO adds LoRA on the
# SELF-ATTENTION QKV/O only, to test whether opening self-attn lets camera reach
# the foreground (vs gate-only Condition A = the _no_lora run).
#
# Differences from Condition A (all else — data/lr/scheduler/seed — identical):
#   * lora.enabled=true, target_modules='self_attn.q,self_attn.k,self_attn.v,self_attn.o'
#     → LoRA on self-attn ONLY (NOT text-CA, NOT scene-CA, NOT FFN).
#   * +lora.allow_with_new_ca=true → bypass the default new_cross_attention LoRA-skip
#     (additive default-off flag; other runs unaffected).
# trainable = self-attn LoRA + patch_embedding(camera gate) + scene_cross_attn + pose_embed + cnd_proj.
# base self-attn / FFN / text-CA / modulation stay frozen.
#
# WANDB_API_KEY(VCAI_Vid) env 필요. GPU 는 CUDA_VISIBLE_DEVICES 로 지정(0~3).

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_selfattnlora"

scene_input_type=new_cross_attention
camera_input_type=channel_concat
condition_latents_input_type=none

lora_enabled=true
lora_rank=32
lora_alpha=32
lora_target_modules='self_attn.q,self_attn.k,self_attn.v,self_attn.o'
resume_lora_ckpt=null

wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,new_cross_attn,channel_concat,selfattn_lora,probe_conditionB]'

: "${WANDB_API_KEY:?set WANDB_API_KEY (VCAI_Vid) in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a camchlora_scenetok_lets_go python -m src.main +experiment=${config} \
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
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  +model.denoiser.lora.allow_with_new_ca=true \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
