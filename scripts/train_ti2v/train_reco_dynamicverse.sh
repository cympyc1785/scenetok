#!/usr/bin/env bash
# ReCo(Wan2.1 VACE 1.3B) + LightningDiT ctrl branch, dynamicverse.
# main 16ch ReCo latent: 좌=inpaint_result(recon) / 우=video_input(dynamic) width-doubled.
# ldt branch: 48ch Wan2.2 background(inpaint_result) latent을 same-t denoise → ReCo VACE source.
# loss는 ReCo 출력만 (recon + dynamic). LightningDiT trainable (joint).

config=custom/scenetok_reco_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-reco_dynamicverse_ldt_ctrl"

wandb_activated=true
wandb_tags='[dynamicverse,reco,wan2.1-vace,lightningdit-ctrl,width-doubled]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} exec -a reco_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.recon_target_video_name=inpaint_result.mp4 \
  dataset.prompt_style=category_first \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
