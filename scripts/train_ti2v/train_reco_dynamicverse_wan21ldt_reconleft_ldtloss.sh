#!/usr/bin/env bash
# ReCo + LightningDiT (Wan2.1 16ch) — recon_left + **ldt output loss ablation**.
# _reconleft와 동일하되 ldt 출력(ref_latent)을 clean recon latent(x0)로 직접 supervise
# (ldt_loss_weight=1.0). 같은 step5000 ckpt를 seed로 이어서 학습 → ldt loss 유무 ablation.

config=custom/scenetok_reco_dynamicverse_wan21ldt
num_workers=4
gpus=1
num_nodes=1
exp_name="va-reco_dynamicverse_ldt_ctrl_wan21ldt_reconleft_ldtloss"

wandb_activated=true
wandb_tags='[dynamicverse,reco,wan2.1-vace,lightningdit-ctrl,wan21ldt,recon_left,ldt_loss,ablation]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} exec -a reco_reconleft_ldtloss python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.recon_target_video_name=inpaint_result.mp4 \
  dataset.prompt_style=category_first \
  ++model.denoiser.ldt_input_type=recon_left \
  ++model.denoiser.ldt_loss_weight=1.0 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
