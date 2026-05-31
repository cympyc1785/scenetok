#!/usr/bin/env bash
# Standalone fast inference wrapper. No Hydra/Lightning.
# 영상만 저장 (save_img=false 내부 고정).

ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok

# exp_name="va-wan-t2v_recon_aggressive_train_480_scene_ca_recam_small"
# exp_name="va-wan-ti2v_recon_aggressive_train_256-480_finetuned_scene_ca_cam_channel_small"
exp_name="va-wan-ti2v_recon_aggressive_train_256-480_finetuned_scene_new_ca_recam_small"
scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

prompt="A teddy bear is running across the road."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

export DEBUG=1
CUDA_VISIBLE_DEVICES=1 exec -a infer_scenetok python "${ROOT_DIR}/scripts/fast_infer_t2v.py" \
  --exp_name "${exp_name}" \
  --scene_id "${scene_id}" \
  --stage_override train \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --noise_seed 0 \
  --target_shape 480,832 \
  --context_shape 256,448 \
  --num_context_views 16 \
  --num_target_views 10
