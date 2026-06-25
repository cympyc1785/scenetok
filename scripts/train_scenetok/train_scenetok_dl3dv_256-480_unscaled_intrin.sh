#!/usr/bin/env bash
# SceneTok va-wan DL3DV 256-480 finetune with UNSCALED context intrinsics.
# Base = `_fixed_intrin` shell, but override `dataset.scale_context_focal_by_256=false`
# so the compressor learns the SAME intrinsic convention as DynamicVerse
# (normalized K, ~64° FOV) instead of the va-wan ×15/×8.4375 context-focal hack.
# Goal: a frozen-able compressor whose context-K convention matches dynamicverse,
# a prerequisite for combining DL3DV + DynamicVerse training. GPU 3.

config=custom/scenetok_va-wan_shift8_dl3dv_finetuned_wide
num_workers=4
gpus=1
num_nodes=1
exp_name="scenetok_va-wan_dl3dv_256-480_unscaled_intrin"

# Set WANDB_API_KEY in your environment before running (avoid committing secrets):
#   export WANDB_API_KEY=<your key>
: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.smallset=true \
  dataset.scale_context_focal_by_256=false \
  dataset.context_shape=[256,448] \
  dataset.target_shape=[480,832] \
  model.compressor.input_shape=[16,28] \
  model.compressor.camera.input_shape=[128,224] \
  model.denoiser.input_shape=[30,52] \
  model.denoiser.camera.input_shape=[240,416] \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true
