#!/usr/bin/env bash
# Structured-latent ablation runner (smallset, train num_workers=4).
# Usage: bash train_scenetok_dl3dv_256-480_structured_abl.sh <tag> <gpu> [extra hydra overrides...]
#   #1 structured vs flat:
#     ... structured 4
#     ... flat 5 model.compressor.structured_latent=false
#   #2 mask on/off:  ... nomask 6 model.compressor.group_mask_enabled=false
#   #3 recep on/off: ... norecep 7 model.compressor.coarse_receptive_field=false
TAG=$1; GPU=$2; shift 2 || true

config=custom/scenetok_va-wan_shift4_dl3dv_finetuned_wide_structured
exp_name="scenetok_va-wan_dl3dv_256-480_abl_${TAG}_small"

export WANDB_API_KEY=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
export DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=${GPU} python -m src.main +experiment=${config} \
  data_loader.train.num_workers=4 \
  data_loader.val.standard.num_workers=0 \
  data_loader.val.unseen.num_workers=0 \
  mode=train \
  dataset.smallset=true \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.num_sanity_val_steps=1 \
  "$@" \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
