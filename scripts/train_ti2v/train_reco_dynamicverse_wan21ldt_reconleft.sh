#!/usr/bin/env bash
# ReCo + LightningDiT (Wan2.1 16ch) — **recon_left 입력 수정판**.
# ldt 입력을 teacher-forced GT 배경(bg_clean) 대신 ReCo latent의 recon-left 절반으로 교체
# → test-time에 GT 불필요, train/inference 일치. va-reco_dynamicverse_ldt_ctrl_wan21ldt의
# step5000 ckpt를 seed(hardlink last.ckpt)로 이어서 학습.
# (ablation 짝: ..._reconleft_ldtloss = ldt 출력에 clean-recon loss 추가)

config=custom/scenetok_reco_dynamicverse_wan21ldt
num_workers=4
gpus=1
num_nodes=1
exp_name="va-reco_dynamicverse_ldt_ctrl_wan21ldt_reconleft"

wandb_activated=true
wandb_tags='[dynamicverse,reco,wan2.1-vace,lightningdit-ctrl,wan21ldt,recon_left,no-teacher-force]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a reco_reconleft python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.recon_target_video_name=inpaint_result.mp4 \
  dataset.prompt_style=category_first \
  ++model.denoiser.ldt_input_type=recon_left \
  ++model.denoiser.ldt_x0_ref=true \
  ++model.denoiser.ldt_loss_weight=0.0 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
