#!/usr/bin/env bash
# Run EffectErase "remove" on given DynamicVerse subdatasets, saving
# inpaint_result_effecterase.mp4 (832x480, 49f) in each scene folder. Existing
# outputs are skipped by infer_dataset.py. Model reloads once per subdataset.
#
# Usage:  GPU=0 bash scripts/run_effecterase_dynamicverse.sh MOSE MVS-Synth SAV
set -uo pipefail
EE=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/EffectErase
BIN=/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/effecterase/bin
DV="$EE/WorldTraj/dynamicverse"
cd "$EE"
for sd in "$@"; do
  echo "===== [$(date '+%m-%d %H:%M:%S')] subdataset=$sd GPU=${GPU:-0} ====="
  CUDA_VISIBLE_DEVICES=${GPU:-0} "$BIN/python" infer_dataset.py \
    --dataset_root "$DV/$sd" \
    --text_encoder_path Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth \
    --dit_path Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors \
    --image_encoder_path Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --pretrained_lora_path EffectErase.ckpt
done
echo "===== ALL DONE: $* ====="
