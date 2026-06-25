#!/usr/bin/env bash
# DL3DV + DynamicVerse MultiDataset controlnet training (scene+camera, no LoRA).
# Same model recipe as `..._dynamicverse_dynamic_controlnet_scene_camera_2_no_lora`
# but dataset=multi (weighted 0.5/0.5 mix; DL3DV bounded+unscaled+text="", DV
# caption_window). Frozen compressor.
# ⚠️ TEMPORARY: compressor.ckpt_path = va-wan_dl3dv_256x448.ckpt (scaled). Swap to
#    the unscaled retrain (scenetok_va-wan_dl3dv_256-480_unscaled_intrin) once done
#    — else DL3DV unscaled input ↔ scaled compressor mismatch.
#
# Set WANDB_API_KEY in env before running:  export WANDB_API_KEY=<key>

config=custom/scenetok_va-wan-ti2v_multi
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_multi_dynamic_controlnet_scene_camera_no_lora"

scene_input_type=controlnet
camera_input_type=controlnet
condition_latents_input_type=none

: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a multi_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
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
