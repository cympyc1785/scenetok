#!/usr/bin/env bash
# Variant of train_ti2vgen_dynamicverse_newca_dynamic_no_lora.sh with the CAMERA
# injected ReCamMaster-style into the main DiT (vs newca KV / vs channel_concat):
#   * scene token → new_cross_attention  (블록별 zero-init scene_cross_attn KV; 동일)
#   * camera      → recam_attention      (extrinsic 3x4(=12) → recam_camera_encoder
#                   (12→dim) → per-frame embedding을 self-attn 입력에 addition +
#                   recam_projector로 self-attn 출력 보정. self_attn 자체를 unfreeze
#                   해 attention이 cross-view fusion 을 학습 — ReCamMaster 방식.)
#   * ControlNet 없음. dynamic 세팅 동일 (target=video_input.mp4, category_first).
#
# 학습 범위 (no LoRA): base FFN/text-CA/modulation freeze. trainable =
#   - 모든 블록 self_attn (recam unfreeze) + recam_projector + recam_camera_encoder
#   - 블록별 scene_cross_attn / scene_cross_attn_proj / norm4 (scene new CA)
#   - pose_embed (가드용 존재; recam은 extrinsics 직접 사용), cnd_proj, null_tokens
# scene proj zero-init 이라 scene 경로는 init 에서 Wan prior 보존.
#
# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 VCAI_Vid 키 export.
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (0~3 범위).

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_newca_scene_camrecam_no_lora"

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
wandb_tags='[dynamicverse,wan-ti2v,new_cross_attn,recam_attention,scene+camera,video_input,category,no_lora]'

: "${WANDB_API_KEY:?set WANDB_API_KEY (VCAI_Vid) in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a newcamrecam_scenetok_lets_go python -m src.main +experiment=${config} \
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
