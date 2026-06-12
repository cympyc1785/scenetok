#!/usr/bin/env bash
# Run the dynamicverse-recon-controlnet ckpt on a DL3DV scene via
# `scripts/fast_infer_t2v_swap_dataset.py` (Hydra-compose with dataset=dl3dv).

ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok

exp_name="va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera"
scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

prompt="An indoor garden features a bonsai tree on a display stand amidst a variety of lush plants, with a pathway and a large duct visible."
ood_prompt="A large shopping mall with a curved facade, multiple stories of parking visible, situated at a street intersection with traffic lights and crosswalks."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

export DEBUG=1
CUDA_VISIBLE_DEVICES=1 exec -a infer_scenetok_dl3dv python "${ROOT_DIR}/scripts/fast_infer_t2v_swap_dataset.py" \
  --exp_name "${exp_name}" \
  --experiment custom/scenetok_va-wan-ti2v_dynamicverse \
  --dataset dl3dv \
  --dataset_root ./DATA/DL3DV/DL3DV-960 \
  --scene_id "${scene_id}" \
  --prompt "${prompt}" \
  --ood_prompt "${ood_prompt}" \
  --negative_prompt "${negative_prompt}" \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --noise_seed 0 \
  --context_shape 256,448 \
  --target_shape 480,832 \
  --num_context_views 12 \
  --num_target_views 10 \
  --load_prompts \
  --scene_input_type controlnet \
  --camera_input_type controlnet \
  --controlnet_ablation
