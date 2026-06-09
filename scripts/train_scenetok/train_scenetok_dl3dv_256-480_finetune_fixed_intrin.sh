#!/usr/bin/env bash
# Re-run of `scenetok_va-wan_dl3dv_256-480_finetune_small` with `derive_shape_dependent_fields`
# in effect (commit a546443). exp_name suffix `_fixed_intrin` to keep wandb run and
# checkpoint dir separate from the original. Same yaml, same overrides — derivation
# produces 0 changes for this shape combo so behaviour identical; the run serves as
# a clean post-fix baseline.

config=custom/scenetok_va-wan_shift8_dl3dv_finetuned_wide
num_workers=4
gpus=1
num_nodes=1
exp_name="scenetok_va-wan_dl3dv_256-480_finetune_small_fixed_intrin"

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=3 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.smallset=true \
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
