#!/usr/bin/env bash
# Fast standalone inference for the dynamicverse-recon-controlnet ckpt.
# - exp_name: `va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera`
# - 1-scene custom eval index (assets/evaluation_index/dynamicverse_infer_1scene.json,
#   key=0004 → MVS-Synth/0004, has `prompt_scene` so dataset doesn't drop it).

ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok

exp_name="va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera"
evaluation_index_path="${ROOT_DIR}/assets/evaluation_index/dynamicverse_infer_1scene.json"

# 데이터셋이 자체적으로 scene별 `prompt_scene`을 prompts.json에서 불러오지만,
# fast_infer_t2v.py는 단일 글로벌 prompt를 cfg.test.prompt로 강제 override함.
# 학습 때 모델이 본 prompt 형태와 비슷한 한 줄을 넣음.
prompt="A scene with smooth camera motion through an urban environment, capturing buildings, streets, and surrounding objects."
ood_prompt="An outdoor fitness station stands on cracked pavement beside a white bench, with a garden, fence, and apartment buildings in the background."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

export DEBUG=1
CUDA_VISIBLE_DEVICES=1 exec -a infer_scenetok python "${ROOT_DIR}/scripts/fast_infer_t2v.py" \
  --exp_name "${exp_name}" \
  --evaluation_index_path "${evaluation_index_path}" \
  --stage_override train \
  --prompt "${prompt}" \
  --ood_prompt "${ood_prompt}" \
  --negative_prompt "${negative_prompt}" \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --noise_seed 0 \
  --repeat_factor 1 \
  --controlnet_ablation \
  --val_seen true \
  --target_shape 480,832 \
  --context_shape 256,448 \
  --num_context_views 12 \
  --num_target_views 10
