#!/usr/bin/env bash
# Sequential inference for 2 no_lora recon ckpts × 2 datasets (dynamicverse + dl3dv).
# 각 조합마다 8 combos (cfg ∈ {1.0, 5.0} × text ∈ {empty, user, ood_text, dataset})
# + ControlNet ablation. GPU 1 단독 사용.

set -e
ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok
export DEBUG=1
export CUDA_VISIBLE_DEVICES=0   # train4와 GPU 공유 (사용 가능 헤드룸 ~125 GB)

ckpts=(
  "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_no_lora"
  "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_scenenorm_no_lora"
)

# Prompts — recon 학습 셋업이라 user prompt는 generic urban, dataset prompt는 scene 자체
# prompts.json/category에서 자동.
user_prompt="A scene with smooth camera motion through an urban environment, capturing buildings, streets, and surrounding objects."
ood_prompt_dyn="An outdoor fitness station stands on cracked pavement beside a white bench, with a garden, fence, and apartment buildings in the background."
ood_prompt_dl3dv="A large shopping mall with a curved facade, multiple stories of parking visible, situated at a street intersection with traffic lights and crosswalks."
user_prompt_dl3dv="An indoor garden features a bonsai tree on a display stand amidst a variety of lush plants, with a pathway and a large duct visible."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

for exp in "${ckpts[@]}"; do
  echo ""
  echo "════════════════════════════════════════════════════════════════════════"
  echo "  CKPT: $exp"
  echo "════════════════════════════════════════════════════════════════════════"

  echo ""
  echo "──── [1/2] dynamicverse 0004 (MVS-Synth) ────"
  python "${ROOT_DIR}/scripts/fast_infer_t2v.py" \
    --exp_name "${exp}" \
    --evaluation_index_path "${ROOT_DIR}/assets/evaluation_index/dynamicverse_infer_1scene.json" \
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

  echo ""
  echo "──── [2/2] DL3DV a4c2... (1K) ────"
  python "${ROOT_DIR}/scripts/fast_infer_t2v_swap_dataset.py" \
    --exp_name "${exp}" \
    --experiment custom/scenetok_va-wan-ti2v_dynamicverse \
    --dataset dl3dv \
    --dataset_root ./DATA/DL3DV/DL3DV-960 \
    --scene_id "1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97" \
    --prompt "${user_prompt_dl3dv}" \
    --ood_prompt "${ood_prompt_dl3dv}" \
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
done

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Done. Results under results/fast_infer_<exp>/* and *_swap-dl3dv/*"
echo "════════════════════════════════════════════════════════════════════════"
