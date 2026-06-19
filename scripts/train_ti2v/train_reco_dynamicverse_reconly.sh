#!/usr/bin/env bash
# recon-우선 phase: ReCo(DiT/VACE LoRA) FREEZE + LightningDiT ctrl branch + ldt2reco_proj만 학습,
# loss는 recon(좌)만 (dynamic_loss_weight=0). va-reco_dynamicverse_ldt_ctrl(joint, step~15k)에서
# weight만 load(checkpointing.load, resume=false → optimizer/step fresh). dynamic은 이후 phase.

config=custom/scenetok_reco_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-reco_dynamicverse_ldt_ctrl_reconly"
init_ckpt="my_checkpoints/va-reco_dynamicverse_ldt_ctrl/last.ckpt"   # recoX joint run → LDT/LoRA/proj weight 승계

wandb_activated=true
wandb_tags='[dynamicverse,reco,wan2.1-vace,lightningdit-ctrl,recon-only,freeze-reco]'

export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} exec -a reco_reconly_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.recon_target_video_name=inpaint_result.mp4 \
  dataset.prompt_style=category_first \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  +model.denoiser.freeze_reco=true \
  +model.denoiser.dynamic_loss_weight=0.0 \
  checkpointing.load=${init_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
