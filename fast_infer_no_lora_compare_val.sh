#!/usr/bin/env bash
# Inference on dynamicverse val (DAVIS blackswan) scene for both no_lora recon ckpts.
# GPU 2 (train2와 공유 — train2 ~57 GB, 추론 ~50 GB, 178 GB GPU 헤드룸 OK).

set -e
ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok
export DEBUG=1
export CUDA_VISIBLE_DEVICES=2

ckpts=(
  "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_no_lora"
  "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_scenenorm_no_lora"
)

user_prompt="A scene with smooth camera motion through an urban environment, capturing buildings, streets, and surrounding objects."
ood_prompt="An outdoor fitness station stands on cracked pavement beside a white bench, with a garden, fence, and apartment buildings in the background."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

for exp in "${ckpts[@]}"; do
  echo ""
  echo "════════════════════════════════════════════════════════════════════════"
  echo "  CKPT: $exp  (val, DAVIS blackswan)"
  echo "════════════════════════════════════════════════════════════════════════"
  python "${ROOT_DIR}/scripts/fast_infer_t2v.py" \
    --exp_name "${exp}" \
    --evaluation_index_path "${ROOT_DIR}/assets/evaluation_index/dynamicverse_infer_val_blackswan.json" \
    --stage_override train \
    --prompt "${user_prompt}" \
    --ood_prompt "${ood_prompt}" \
    --negative_prompt "${negative_prompt}" \
    --cfg_scale 5.0 \
    --num_inference_steps 50 \
    --noise_seed 0 \
    --repeat_factor 1 \
    --controlnet_ablation \
    --val_seen false \
    --target_shape 480,832 \
    --context_shape 256,448 \
    --num_context_views 12 \
    --num_target_views 10 \
    --output_dir "${ROOT_DIR}/results/fast_infer_${exp}_val"
done

echo ""
echo "Done. Results under results/fast_infer_<exp>_val/blackswan/*"
