#!/usr/bin/env bash
# Structured latent coarse_kl_scale sweep (smallset, num_workers=0).
# Usage: bash train_scenetok_dl3dv_256-480_structured_klsweep.sh <coarse_kl_scale> <gpu> [num_workers]
KLC=${1:-10}
GPU=${2:-4}
NW=${3:-0}   # train DataLoader workers (default 0: fork+CUDA deadlock 회피, FIX.log 2026-06-17)

config=custom/scenetok_va-wan_shift4_dl3dv_finetuned_wide_structured
exp_name="scenetok_va-wan_dl3dv_256-480_structured_klc${KLC}_small"

export WANDB_API_KEY=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
export DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=${GPU} python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${NW} \
  data_loader.val.standard.num_workers=0 \
  data_loader.val.unseen.num_workers=0 \
  mode=train \
  dataset.smallset=true \
  model.compressor.coarse_kl_scale=${KLC} \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
