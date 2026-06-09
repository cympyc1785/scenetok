#!/usr/bin/env bash
# Stage-1-only SceneTok inference at the paper's 256x256 setting with
# Monte-Carlo variance maps. Same scene as `infer_lightningdit_reco.sh` —
# but no ReCo stage 2.
#
# Two published ckpts, each in its own output dir:
#   1) va-wan_dl3dv.ckpt   (VA-VAE + Wan 2.2 VAE pair)
#   2) va-videodc_dl3dv.ckpt (VA-VAE + VideoDCAE pair)
#
# Each run saves:
#   coarse.mp4         — mean of N samples (paper's mean panel)
#   variance.mp4       — viridis-colored per-pixel std
#   variance_gray.mp4  — grayscale std (same min-max normalization)
#   variance.pt        — raw std tensor (F, H, W)

set -e

ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok

dl3dv_scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"
re10k_scene_id="93b36a54151e085e"   # 279-frame scene from /train/ — unbounded
                                    # sampler with chunk_index_gap=4 + num_target_views=10
                                    # needs ≥ ~140 frames after context; this is comfortably
                                    # long. (stage1 pins val_seen=True → loads /train/)
shape="256,256"
N=8

export DEBUG=1
export CUDA_VISIBLE_DEVICES=0

run_variance() {
  local experiment="$1"   # config/experiment/<name>.yaml (no .yaml)
  local ckpt="$2"         # absolute ckpt path
  local dataset="$3"      # dl3dv | re10k
  local scene_id="$4"     # scene_id for that dataset
  local tag="$5"          # output-dir suffix
  local td_override="${6:-}"  # optional: --view_sampler_td <N>; empty = leave default
  local out_dir="${ROOT_DIR}/results/scenetok_variance_${tag}"

  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  [${tag}] experiment=${experiment}  ckpt=${ckpt}"
  echo "  dataset=${dataset}  scene_id=${scene_id}  shape=${shape}  N=${N}"
  echo "  out_dir=${out_dir}"
  echo "════════════════════════════════════════════════════════════════════"

  local td_args=()
  if [[ -n "${td_override}" ]]; then
    td_args=(--view_sampler_td "${td_override}")
  fi

  python "${ROOT_DIR}/scripts/stage1_lightningdit_coarse.py" \
    --scenetok_experiment "${experiment}" \
    --scenetok_ckpt "${ckpt}" \
    --dataset "${dataset}" \
    --scene_id "${scene_id}" \
    --stage_override train \
    --context_shape "${shape}" \
    --target_shape "${shape}" \
    --num_context_views 16 \
    --num_target_views 10 \
    --scenetok_inference_steps 25 \
    --scenetok_cfg_scale 1.0 \
    --scenetok_seed 0 \
    --repeat_factor "${N}" \
    "${td_args[@]}" \
    --out_dir "${out_dir}" \
    --fps 15
}

# Case 1: DL3DV — VA-VAE + Wan 2.2 VAE
run_variance \
  "scenetok_va-wan_shift4_dl3dv_finetuned" \
  "${ROOT_DIR}/checkpoints/va-wan_dl3dv.ckpt" \
  "dl3dv" "${dl3dv_scene_id}" "va-wan_dl3dv_256"

# Case 2: DL3DV — VA-VAE + VideoDCAE
run_variance \
  "scenetok_va-vdc_shift8_dl3dv_finetuned" \
  "${ROOT_DIR}/checkpoints/va-videodc_dl3dv.ckpt" \
  "dl3dv" "${dl3dv_scene_id}" "va-videodc_dl3dv_256"

# Case 3: RE10K — VA-VAE + VideoDCAE
# `--view_sampler_td 1`: raw-mode RE10K — unbounded sampler's `td=4` multiplier
# would OOB the raw indices since num_latents==num_views in this path.
run_variance \
  "scenetok_va-vdc_re10k_scratch" \
  "${ROOT_DIR}/checkpoints/va-videodc_re10k.ckpt" \
  "re10k" "${re10k_scene_id}" "va-videodc_re10k_256" "1"

echo
echo "[all done]"
