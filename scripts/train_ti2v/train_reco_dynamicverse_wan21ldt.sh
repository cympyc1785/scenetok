#!/usr/bin/env bash
# ReCo(Wan2.1 VACE 1.3B) + LightningDiT ctrl branch — Wan2.1 16ch ldt 변형.
# 개선점: ldt를 Wan2.1 16ch(va-wan2.1_dl3dv_256x448_50k)로 교체 → ldt 출력과 ReCo ref
# slot이 동일 latent space. ldt2reco_proj=identity로 pretrained ldt를 그대로 ref에 주입
# (기존 48→16 zero-init OOD 브릿지 제거). compressor도 동일 ckpt에서 로드.
# main 16ch ReCo latent: 좌=inpaint_result(recon) / 우=video_input(dynamic) width-doubled.

config=custom/scenetok_reco_dynamicverse_wan21ldt
num_workers=4
gpus=1
num_nodes=1
exp_name="va-reco_dynamicverse_ldt_ctrl_wan21ldt"

wandb_activated=true
wandb_tags='[dynamicverse,reco,wan2.1-vace,lightningdit-ctrl,wan21ldt,identity-proj]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a reco_wan21ldt python -m src.main +experiment=${config} \
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
