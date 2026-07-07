#!/usr/bin/env bash
# SceneTok + Wan TI2V 5B on the CineScene Scene-Decoupled dataset — ReCamMaster
# camera variant (vs channel_concat). scene→new_cross_attention 동일, camera만
# recam_attention (extrinsic 3x4(=12) → recam_camera_encoder(12→dim) → per-frame
# embedding을 self-attn 입력에 addition + recam_projector 출력 보정, self_attn
# unfreeze). ControlNet 없음, no LoRA. dynamicverse newca_scene_camrecam 셸과
# 동일 recipe, dataset만 scene_decoupled (decoupled: context=wohuman 배경,
# target=whuman 인물, text=caption_action_only — dataset이 네이티브 처리).
#
# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 키 export.
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (0~3 범위).

config=custom/scenetok_va-wan-ti2v_scene_decoupled
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_scene_decoupled_newca_scene_camrecam_no_lora"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=new_cross_attention
camera_input_type=recam_attention
condition_latents_input_type=none

# ── LoRA (off; recam self_attn unfreeze + 새 scene CA 가 어댑터) ─────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[scene_decoupled,wan-ti2v,new_cross_attn,recam_attention,scene+camera,decoupled,no_lora]'

: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
