#!/usr/bin/env bash
# DL3DV + DynamicVerse MultiDataset controlnet training (scene+camera, no LoRA),
# variant of train_ti2vgen_multi_controlnet_dynamic.sh with ONLY the DynamicVerse
# background input swapped to the EffectErase-inpainted video:
#   * DynamicVerse sub (datasets[1]) video_name: inpaint_result.mp4
#     → inpaint_result_effecterase.mp4 (배경=scene token 출처가 더 깨끗한 inpaint).
#     target stays video_input.mp4 (dynamic). DL3DV sub unchanged (recon, text="").
#   * multi.yaml(사용 중 config)은 수정하지 않고 CLI list-index override 사용.
#   * effecterase 미생성 DynamicVerse scene은 dataset existence-check가 자동 drop.
#
# ⚠️ TEMPORARY: compressor.ckpt_path = va-wan_dl3dv_256x448.ckpt (scaled). Swap to
#    the unscaled retrain once done (multi.yaml과 동일 caveat).
#
# Set WANDB_API_KEY (VCAI_Vid) in env before running:  export WANDB_API_KEY=<key>
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (0~3 범위).

config=custom/scenetok_va-wan-ti2v_multi
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_multi_dynamic_controlnet_scene_camera_no_lora_effecterase"

scene_input_type=controlnet
camera_input_type=controlnet
condition_latents_input_type=none

: "${WANDB_API_KEY:?set WANDB_API_KEY (VCAI_Vid) in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} exec -a multieff_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.datasets.1.video_name=inpaint_result_effecterase.mp4 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  model.denoiser.lora.enabled=false \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
