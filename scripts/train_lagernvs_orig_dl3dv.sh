#!/usr/bin/env bash
# LagerNVS ORIGINAL code (submodules/lagernvs/train.py) from scratch on DL3DV
# 256x256 smallset. Train = 1K batch (985 scenes), eval = 11K batch (498 scenes).
# batch_size 8 (기존 SceneTok 계열 런과 맞춤), pretrained_vggt true, orig lr 4e-4.
# Logs/ckpts -> my_checkpoints/<exp>/ (TensorBoard, not wandb — original code).
set -e

GPU=${1:-2}
EXP=${2:-lagernvs_orig_dl3dv_1k}
PORT=${3:-29540}

cd "$(dirname "$0")/../submodules/lagernvs"
export CUDA_VISIBLE_DEVICES=$GPU
export LAGERNVS_DATA_ROOT=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/submodules/lagernvs/data_dl3dv_smallset
export TORCHDYNAMO_DISABLE=1
PY=/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/lagernvs/bin

exec $PY/torchrun --nproc_per_node=1 --master_port=$PORT train.py \
  -c config/train_dl3dv_smallset.yaml -e "$EXP"
