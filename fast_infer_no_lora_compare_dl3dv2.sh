#!/usr/bin/env bash
# Different DL3DV scene (aeea... 1K — paved path beside water + exercise equipment).
set -e
ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok
export DEBUG=1
export CUDA_VISIBLE_DEVICES=3   # train1과 GPU 공유

ckpts=(
  "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_no_lora"
  "va-wan-ti2v_dynamicverse_recon_controlnet_scene_camera_2_scenenorm_no_lora"
)
scene_id="1K/aeea987ca3dea5b7b55abeef9d9807629a091fbc0b78db65041ab4c71d9298b4"
user_prompt="A paved path runs beside a body of water, with exercise equipment and trees on the left."
ood_prompt="A large shopping mall with a curved facade, multiple stories of parking visible, situated at a street intersection with traffic lights and crosswalks."
negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

for exp in "${ckpts[@]}"; do
  echo ""
  echo "════════════════════════════════════════════════════════════════════════"
  echo "  CKPT: $exp on DL3DV ${scene_id}"
  echo "════════════════════════════════════════════════════════════════════════"
  python "${ROOT_DIR}/scripts/fast_infer_t2v_swap_dataset.py" \
    --exp_name "${exp}" \
    --experiment custom/scenetok_va-wan-ti2v_dynamicverse \
    --dataset dl3dv \
    --dataset_root ./DATA/DL3DV/DL3DV-960 \
    --scene_id "${scene_id}" \
    --prompt "${user_prompt}" \
    --ood_prompt "${ood_prompt}" \
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
echo "Done."
