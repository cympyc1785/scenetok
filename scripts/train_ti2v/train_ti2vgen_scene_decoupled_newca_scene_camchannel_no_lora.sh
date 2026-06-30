#!/usr/bin/env bash
# SceneTok + Wan TI2V 5B on the CineScene Scene-Decoupled Video Dataset
# (DECOUPLED: context=wohuman background → scene tokens, target=whuman subject,
# text=foreground action). Mirrors the dynamicverse newca+channel_concat recipe:
#   * scene token → new_cross_attention (블록별 zero-init scene_cross_attn KV)
#   * camera      → channel_concat      (Plücker ray map → patch_embedding 확장)
#   * ControlNet 없음. base Wan DiT freeze (no LoRA).
# 데이터 분기/소스(wohuman→whuman, action caption)는 scene_decoupled dataset이
# 네이티브로 처리 → dynamicverse 셸의 target_video_name/prompt_style override 불필요.
#
# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 키 export.
#   export WANDB_API_KEY=<key>
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (0~3 범위).

config=custom/scenetok_va-wan-ti2v_scene_decoupled
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_scene_decoupled_newca_scene_camchannel_no_lora"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=new_cross_attention
camera_input_type=channel_concat
condition_latents_input_type=none

# ── LoRA on main Wan DiT (off) ──────────────────────────────────────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[scene_decoupled,wan-ti2v,new_cross_attn,channel_concat,scene+camera,decoupled,no_lora]'

: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m src.main +experiment=${config} \
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
