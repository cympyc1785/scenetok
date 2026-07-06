#!/usr/bin/env bash
# camchannel_selfattnlora(Condition B) 세팅을 DynamicVerse의 EffectErase 배경
# 입력에 적용한 변형. train_ti2vgen_dynamicverse_newca_scene_camchannel_selfattnlora.sh
# 와 동일(scene=new_cross_attention, camera=channel_concat, self-attn LoRA QKVO)하되,
# context 배경 영상만 inpaint_result.mp4 → inpaint_result_effecterase.mp4 로 교체
# (dynamic foreground를 EffectErase로 지운 더 깨끗한 배경 → scene token 출처).
# target 은 그대로 video_input.mp4(dynamic). effecterase 미생성 scene 은 dataset이
# 자동 drop(2080→1980). max_steps=50001 → 50k+val 후 자동 종료.
#
# WANDB_API_KEY(VCAI_Vid) env 필요. GPU 는 CUDA_VISIBLE_DEVICES 로 지정(0~3).

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
compressor_ckpt=checkpoints/va-wan_dl3dv_256-480_unscaled_intrins.ckpt
exp_name="va-wan-ti2v_dynamicverse_effecterase_newca_scene_camchannel_selfattnlora_unscaledcomp"

scene_input_type=new_cross_attention
camera_input_type=channel_concat
condition_latents_input_type=none

lora_enabled=true
lora_rank=32
lora_alpha=32
lora_target_modules='self_attn.q,self_attn.k,self_attn.v,self_attn.o'
resume_lora_ckpt=null

wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,new_cross_attn,channel_concat,selfattn_lora,effecterase,conditionB]'

: "${WANDB_API_KEY:?set WANDB_API_KEY (VCAI_Vid) in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} exec -a camchlora_eff_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.video_name=inpaint_result_effecterase.mp4 \
  dataset.target_video_name=video_input.mp4 \
  dataset.prompt_style=category_first \
  trainer.max_steps=50001 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  model.compressor.ckpt_path=${compressor_ckpt} \
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
