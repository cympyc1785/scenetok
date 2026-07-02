#!/usr/bin/env bash
# SceneTok + Wan TI2V 5B on the CineScene Scene-Decoupled Video Dataset
# (DECOUPLED: context=wohuman background → scene tokens, target=whuman subject,
# text=foreground action). ControlNet 라우팅 버전:
#   * scene token → controlnet (AC3D-style parallel branch)
#   * camera      → controlnet (same branch, Plücker ray)
#   * ac3d_num_layers=2, base Wan DiT freeze (no LoRA).
# 데이터 분기/소스(wohuman→whuman, action caption)는 scene_decoupled dataset이
# 네이티브로 처리 → dynamicverse 셸의 target_video_name/prompt_style override 불필요.
# ckpt 이어받지 않음 — base Wan DiT(model_root)에서 fresh 시작 (denoiser.ckpt_path=null).
#
# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 키 export.
#   export WANDB_API_KEY=<key>
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (0~3 범위).

config=custom/scenetok_va-wan-ti2v_scene_decoupled
num_workers=4
gpus=1
num_nodes=1
# compressor를 unscaled-trained ckpt로 교체 (scene_decoupled normalized intrinsic in-distribution). exp_name += _unscaledcomp.
compressor_ckpt=checkpoints/va-wan_dl3dv_256-480_unscaled_intrins.ckpt
exp_name="va-wan-ti2v_scene_decoupled_controlnet_scene_camera_no_lora_unscaledcomp"

# ── Condition routing ─────────────────────────────────────────────────────
scene_input_type=controlnet
camera_input_type=controlnet
condition_latents_input_type=none

# ── Ctrl branch depth ──────────────────────────────────────────────────────
ac3d_num_layers=2

# ── LoRA on main Wan DiT (off) ──────────────────────────────────────────────
lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

# ── wandb ────────────────────────────────────────────────────────────────
wandb_activated=true
wandb_tags='[scene_decoupled,wan-ti2v,controlnet,scene+camera,decoupled,no_lora]'

: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1
# 이 노드의 /tmp 는 noexec 로 마운트되어 있어 torch-inductor/triton 이 거기에
# 컴파일한 .so 를 dlopen(exec-mmap) 하지 못함 → "failed to map segment from
# shared object". eager 로 돌려 triton 컴파일 자체를 회피 (기존 학습들도 inductor
# cuda_utils.so 를 map 하지 않아 eager-equivalent → 속도 손해 없음).
export TORCHDYNAMO_DISABLE=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} exec -a scdec_ctrl_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  model.compressor.ckpt_path=${compressor_ckpt} \
  model.denoiser.ac3d_num_layers=${ac3d_num_layers} \
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
