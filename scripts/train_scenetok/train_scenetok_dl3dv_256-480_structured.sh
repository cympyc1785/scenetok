#!/usr/bin/env bash
# Structured (coarse/fine) scene latent — 256x448 context -> 480x832 target.
# config: custom/scenetok_va-wan_shift4_dl3dv_finetuned_wide_structured
#   compressor.structured_latent=true (coarse 128 / fine 896, level-emb, group masking,
#   coarse KL bottleneck), denoiser outputs 480x832.
# num_workers=0: 이 노드의 train DataLoader fork+CUDA deadlock 회피 (FIX.log 2026-06-17).

config=custom/scenetok_va-wan_shift4_dl3dv_finetuned_wide_structured
num_workers=0
gpus=1
num_nodes=1
exp_name="scenetok_va-wan_dl3dv_256-480_structured_small"

export WANDB_API_KEY=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
export DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=3 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  data_loader.val.standard.num_workers=0 \
  data_loader.val.unseen.num_workers=0 \
  mode=train \
  dataset.smallset=true \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
