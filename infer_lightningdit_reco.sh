#!/usr/bin/env bash
# Two-stage inference: lightningDiT (SceneTok) → ReCo (Wan2.1-VACE + LoRA).
# Inference only. Both stages frozen.
#
# Runs as two sequential python processes — Stage 1 writes a coarse mp4,
# Stage 2 reads it. Two procs are required because the main repo's
# `src/model/DiffSynth-Studio/diffsynth` and ReCo's vendored
# `src/model/ReCo/DiffSynth-Studio/diffsynth` share the package name
# `diffsynth` and conflict if loaded in the same Python process.
#
# This script runs four configurations sequentially. Output dirs include a
# resolution-tag suffix so they don't collide:
#   1) va-wan 256 model + bilinear upscale to 480x832 → ..._256_finetune_large_upscale480
#   2) va-wan 256 model, no upscale (ReCo at 256x448)  → ..._256_finetune_large_native256
#   3) va-wan 480 model, no upscale (ReCo at 480x832)  → ..._480_finetune_large_native480
#   4) published va-vdc DL3DV ckpt @ 256x256 (Hydra-compose) → ..._vavdc_native256

set -e

ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok

scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

# prompt="A teddy bear is running across the road."
prompt="Add a flying robot."
# prompt="Add a kitten walking on the floor."

export DEBUG=1
export CUDA_VISIBLE_DEVICES=0

run_pipeline() {
  local exp="$1"
  local shape="$2"          # e.g. "256,448" or "480,832"
  local upscale_flag="$3"   # "--upscale" or ""
  local tag="$4"            # output-dir suffix (e.g. "upscale480")
  # Optional args for published-ckpt mode:
  local experiment="${5:-}"    # `config/experiment/<name>.yaml`
  local ckpt="${6:-}"          # absolute ckpt path
  local out_dir="${ROOT_DIR}/results/lightningdit_reco_${exp}_${tag}"

  local stage1_exp_args
  if [[ -n "${experiment}" ]]; then
    stage1_exp_args=(--scenetok_experiment "${experiment}" --scenetok_ckpt "${ckpt}")
  else
    stage1_exp_args=(--scenetok_exp "${exp}")
  fi

  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  [${tag}] exp=${exp}  shape=${shape}  upscale='${upscale_flag}'"
  echo "  ${experiment:+experiment=${experiment}  ckpt=${ckpt}}"
  echo "  out_dir=${out_dir}"
  echo "════════════════════════════════════════════════════════════════════"

  # ── Stage 1 ────────────────────────────────────────────────────────────
  python "${ROOT_DIR}/scripts/stage1_lightningdit_coarse.py" \
    "${stage1_exp_args[@]}" \
    --scene_id "${scene_id}" \
    --stage_override train \
    --context_shape "${shape}" \
    --target_shape "${shape}" \
    --num_context_views 16 \
    --num_target_views 10 \
    --scenetok_inference_steps 25 \
    --scenetok_cfg_scale 1.0 \
    --scenetok_seed 0 \
    --out_dir "${out_dir}" \
    --fps 15

  local scene_name
  scene_name=$(cat "${out_dir}"/*/stage1/scene.txt | head -1 | tr -d '[:space:]')
  local coarse_mp4="${out_dir}/${scene_name}/stage1/coarse.mp4"
  echo "[chain] coarse mp4: ${coarse_mp4}"

  # ── Stage 2 ────────────────────────────────────────────────────────────
  # --height/--width matter only when --upscale is on; otherwise ReCo runs
  # at coarse mp4's native size.
  python "${ROOT_DIR}/scripts/stage2_reco_edit.py" \
    --coarse_mp4 "${coarse_mp4}" \
    --prompt "${prompt}" \
    --out_dir "${out_dir}" \
    --scene_name "${scene_name}" \
    --height 480 \
    --width 832 \
    ${upscale_flag} \
    --reco_num_inference_steps 50 \
    --reco_seed 1 \
    --reco_num_frames 37 \
    --fps 15

  echo "[done ${tag}] outputs under ${out_dir}/${scene_name}"
}

# Case 1: 256 model + bilinear upscale to 480x832 before ReCo.
run_pipeline "scenetok_va-wan_dl3dv_256_finetune_large" "256,448" "--upscale" "upscale480_2"

# # Case 2: 256 model, no upscale (ReCo runs at native 256x448).
# run_pipeline "scenetok_va-wan_dl3dv_256_finetune_large" "256,448" "" "native256"

# # Case 3: 480 model, no upscale (ReCo runs at native 480x832).
# run_pipeline "scenetok_va-wan_dl3dv_480_finetune_large" "480,832" "" "native480"

# # Case 4: published va-videodc DL3DV ckpt @ 256x256 (no exp dir → Hydra-compose).
# run_pipeline \
#   "scenetok_va-vdc_shift8_dl3dv_finetuned" \
#   "256,256" \
#   "" \
#   "vavdc_native256" \
#   "scenetok_va-vdc_shift8_dl3dv_finetuned" \
#   "${ROOT_DIR}/checkpoints/va-videodc_dl3dv.ckpt"

echo
echo "[all done]"
