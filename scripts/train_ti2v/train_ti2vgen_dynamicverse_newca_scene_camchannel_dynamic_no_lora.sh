#!/usr/bin/env bash
# Variant of train_ti2vgen_dynamicverse_newca_dynamic_no_lora.sh
# (= va-wan-ti2v_dynamicverse_dynamic_newca_scene_camera_no_lora) with the
# CAMERA injection moved off the new cross-attention KV and onto the main DiT:
#   * scene token  → new_cross_attention  (블록별 zero-init scene_cross_attn KV; 동일)
#   * camera       → channel_concat       (Plücker ray map(6*td ch)을 latent 채널에
#                    concat → patch_embedding 확장(+24ch, extra zero-init).
#                    per-pixel spatial 정렬 + self-attn 스트림 직접 주입)
#   * ControlNet 없음. dynamic 세팅 동일 (target=video_input.mp4, category_first).
#
# 학습 범위 (no LoRA): base Wan DiT freeze. trainable =
#   - 블록별 scene_cross_attn / scene_cross_attn_proj / norm4 (scene new CA)
#   - patch_embedding (channel_concat 확장 conv)
#   - pose_embed (ray → latent-grid Plücker), cnd_proj (scene 64→dim), null_tokens
# zero-init (scene proj + patch_embed extra ch) 이라 init 에서 Wan prior 보존.
#
# dynamicverse config 의 camera 는 이미 input_shape=[30,52](latent grid) +
# time_embed in_channels=6 → channel_concat 형식과 정합 (config 수정 불필요).
#
# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 VCAI_Vid 키 export.
#   export WANDB_API_KEY=<vcai_vid_key>
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (0~3 범위).

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_newca_scene_camchannel_no_lora"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=new_cross_attention
camera_input_type=channel_concat
condition_latents_input_type=none

# ── LoRA on main Wan DiT (off; 새 cross-attn + channel-concat 가 어댑터) ──────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,new_cross_attn,channel_concat,scene+camera,video_input,category,no_lora]'

: "${WANDB_API_KEY:?set WANDB_API_KEY (VCAI_Vid) in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} exec -a newcamch_scenetok_lets_go python -m src.main +experiment=${config} \
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
