#!/usr/bin/env bash
# Variant of train_ti2vgen_dynamicverse_controlnet_dynamic.sh (= baseline
# `va-wan-ti2v_dynamicverse_dynamic_controlnet_scene_camera_2_no_lora`):
#   * ControlNet 제거. scene token + camera embedding 을 각 Wan DiT 블록에 추가된
#     새 cross-attention 레이어(zero-init proj)의 k,v 로 **concat**해서 주입.
#       - scene_input_type=new_cross_attention  → 블록별 scene_cross_attn 생성
#       - camera_input_type=new_cross_attention → per-frame 1토큰(pool)으로
#         scene token 뒤에 sequence concat → [scene ⊕ camera] 가 하나의 k,v
#   * 학습 범위: base Wan DiT freeze, 새 cross-attn + pose_embed + cnd_proj 만
#     학습 (zero-init 이라 init 에서 Wan prior 보존). LoRA off.
#   * dynamic 세팅 동일: target = video_input.mp4, text = category_first.
# GPU 1.

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_newca_scene_camera_no_lora"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=new_cross_attention
camera_input_type=new_cross_attention
condition_latents_input_type=none

# ── LoRA on main Wan DiT (off; 새 cross-attn 자체가 어댑터) ──────────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,new_cross_attn,scene+camera,video_input,category,no_lora]'

# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 export 할 것.
: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a newca_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.target_video_name=video_input.mp4 \
  dataset.prompt_style=category_first \
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
