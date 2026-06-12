#!/usr/bin/env bash
# Standard inference recipe for a given exp checkpoint.
#
#   Scenes (6 total):
#     train (dynamicverse standard): 0004, sav_014713_manual
#     test  (dynamicverse unseen):   blackswan, bike-packing
#     OOD   (DL3DV):                 1K/a4c20f668ce179db..., 1K/aeea987ca3dea5b7...
#
#   Combos per scene (8):
#     text ∈ {empty, user, ood_text, dataset}  ×  cfg ∈ {1.0, 5.0}
#
#   Output: results/fast_infer_<exp>{,_val,_swap-dl3dv}/<scene>/cfg*_text-*/full_sequence.mp4
#
# Usage:
#   bash scripts/run_standard_inference.sh <exp_name> [GPU_ID]
#     e.g. bash scripts/run_standard_inference.sh va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_no_lora 0

set -e
exp="${1:?usage: bash run_standard_inference.sh <exp_name> [GPU_ID]}"
GPU="${2:-0}"

ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok
export DEBUG=1
export CUDA_VISIBLE_DEVICES="${GPU}"

train_index="${ROOT_DIR}/assets/evaluation_index/dynamicverse_infer_train_2scene.json"
val_index="${ROOT_DIR}/assets/evaluation_index/dynamicverse_infer_val_2scene.json"

# OOD DL3DV scenes (kept fixed across runs for comparability)
dl3dv_scene_1="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"
dl3dv_user_prompt_1="An indoor garden features a bonsai tree on a display stand amidst a variety of lush plants, with a pathway and a large duct visible."
dl3dv_ood_prompt_1="A large shopping mall with a curved facade, multiple stories of parking visible, situated at a street intersection with traffic lights and crosswalks."

dl3dv_scene_2="1K/aeea987ca3dea5b7b55abeef9d9807629a091fbc0b78db65041ab4c71d9298b4"
dl3dv_user_prompt_2="A paved path runs beside a body of water, with exercise equipment and trees on the left."
dl3dv_ood_prompt_2="A large shopping mall with a curved facade, multiple stories of parking visible, situated at a street intersection with traffic lights and crosswalks."

# Generic prompts for the dynamicverse side
user_prompt="A scene with smooth camera motion through an urban environment, capturing buildings, streets, and surrounding objects."
ood_prompt_dyn="An outdoor fitness station stands on cracked pavement beside a white bench, with a garden, fence, and apartment buildings in the background."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

echo "════════════════════════════════════════════════════════════════════════"
echo "  CKPT: ${exp}    GPU: ${GPU}"
echo "  6 scenes × 8 combos = 48 mp4"
echo "════════════════════════════════════════════════════════════════════════"

# ── 1. dynamicverse train (2 scenes) ─────────────────────────────────────
echo ""
echo "──── [1/4] dynamicverse train (0004, sav_014713_manual) ────"
python "${ROOT_DIR}/scripts/fast_infer_t2v.py" \
  --exp_name "${exp}" \
  --evaluation_index_path "${train_index}" \
  --stage_override train \
  --prompt "${user_prompt}" \
  --ood_prompt "${ood_prompt_dyn}" \
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

# ── 2. dynamicverse val (2 scenes) ───────────────────────────────────────
echo ""
echo "──── [2/4] dynamicverse val (blackswan, bike-packing) ────"
python "${ROOT_DIR}/scripts/fast_infer_t2v.py" \
  --exp_name "${exp}" \
  --evaluation_index_path "${val_index}" \
  --stage_override train \
  --prompt "${user_prompt}" \
  --ood_prompt "${ood_prompt_dyn}" \
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

# ── 3. DL3DV OOD scene 1 ──────────────────────────────────────────────────
echo ""
echo "──── [3/4] DL3DV a4c20f668ce179db... (indoor garden) ────"
python "${ROOT_DIR}/scripts/fast_infer_t2v_swap_dataset.py" \
  --exp_name "${exp}" \
  --experiment custom/scenetok_va-wan-ti2v_dynamicverse \
  --dataset dl3dv \
  --dataset_root ./DATA/DL3DV/DL3DV-960 \
  --scene_id "${dl3dv_scene_1}" \
  --prompt "${dl3dv_user_prompt_1}" \
  --ood_prompt "${dl3dv_ood_prompt_1}" \
  --negative_prompt "${negative_prompt}" \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --noise_seed 0 \
  --controlnet_ablation \
  --scene_input_type controlnet \
  --camera_input_type controlnet \
  --load_prompts \
  --target_shape 480,832 \
  --context_shape 256,448 \
  --num_context_views 12 \
  --num_target_views 10

# ── 4. DL3DV OOD scene 2 ──────────────────────────────────────────────────
echo ""
echo "──── [4/4] DL3DV aeea987ca3dea5b7... (paved path by water) ────"
python "${ROOT_DIR}/scripts/fast_infer_t2v_swap_dataset.py" \
  --exp_name "${exp}" \
  --experiment custom/scenetok_va-wan-ti2v_dynamicverse \
  --dataset dl3dv \
  --dataset_root ./DATA/DL3DV/DL3DV-960 \
  --scene_id "${dl3dv_scene_2}" \
  --prompt "${dl3dv_user_prompt_2}" \
  --ood_prompt "${dl3dv_ood_prompt_2}" \
  --negative_prompt "${negative_prompt}" \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --noise_seed 0 \
  --controlnet_ablation \
  --scene_input_type controlnet \
  --camera_input_type controlnet \
  --load_prompts \
  --target_shape 480,832 \
  --context_shape 256,448 \
  --num_context_views 12 \
  --num_target_views 10

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Done. Results under:"
echo "    results/fast_infer_${exp}/{0004,sav_014713_manual}/cfg*_text-*/"
echo "    results/fast_infer_${exp}_val/{blackswan,bike-packing}/cfg*_text-*/"
echo "    results/fast_infer_${exp}_swap-dl3dv/{a4c20f6...,aeea987...}/cfg*_text-*/"
echo "════════════════════════════════════════════════════════════════════════"
