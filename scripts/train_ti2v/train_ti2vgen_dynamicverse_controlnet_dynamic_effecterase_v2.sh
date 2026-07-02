#!/usr/bin/env bash
# Variant of train_ti2vgen_dynamicverse_controlnet_dynamic.sh
# (= va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora) with:
#   * context input video: inpaint_result.mp4 → inpaint_result_effecterase.mp4
#     (EffectErase-removed background). target stays video_input.mp4.
#   * max_steps = 50000.
#   * run_name += _effecterase.
# Scenes lacking inpaint_result_effecterase.mp4 are auto-dropped by the dataset
# (_collect_scenes existence check), so the ~140 un-EffectErase'd scenes are skipped.
#
# Set WANDB_API_KEY in env (VCAI_Vid key) before running:
#   export WANDB_API_KEY=<vcai_vid_key>

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
# compressor: scaled-trained va-wan_dl3dv_256x448 → unscaled-trained ckpt로 교체.
# dynamicverse는 normalized intrinsic(scale off)을 먹이는데 기존 scaled compressor엔
# OOD였음 → unscaled compressor는 in-distribution. exp_name += _unscaledcomp (새 wandb run).
compressor_ckpt=checkpoints/va-wan_dl3dv_256-480_unscaled_intrins.ckpt
exp_name="va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora_effecterase_v2_unscaledcomp"

scene_input_type=controlnet
camera_input_type=controlnet
condition_latents_input_type=none

lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,controlnet,scene+camera,video_input,category,effecterase]'

# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 VCAI_Vid 키 export.
: "${WANDB_API_KEY:?set WANDB_API_KEY (VCAI_Vid) in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} exec -a effecterase_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.video_name=inpaint_result_effecterase.mp4 \
  dataset.target_video_name=video_input.mp4 \
  dataset.prompt_style=category_first \
  trainer.max_steps=50000 \
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
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
