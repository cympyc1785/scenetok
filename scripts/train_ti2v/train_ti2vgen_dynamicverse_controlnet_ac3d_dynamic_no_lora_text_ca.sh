#!/usr/bin/env bash
# === GPU 6 ===
# train_ti2vgen_dynamicverse_controlnet_ac3d_dynamic_no_lora.sh 와 동일하되,
# AC3D ctrl block에 text cross-attn을 다시 추가 (ac3d_use_text_cross_attn=true) —
# 즉 ctrl 분기에도 text가 들어가게 함. run_name 에 _text_ca 접미사.
# 이 머신 전용 override: dataset.root, val batch_size=4 x max_batches=8 (=32 scenes, 8 videos logged) + expandable alloc, OOM 회피.

config=custom/scenetok_va-wan-ti2v_dynamicverse
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_dynamicverse_dynamic_controlnet_ac3d_scene_camera_2_no_lora_text_ca"

data_root=/data1/cympyc1785/data/dynamicverse

scene_input_type=controlnet
camera_input_type=controlnet_ac3d
condition_latents_input_type=none

lora_enabled=false
lora_rank=32
lora_alpha=32
lora_target_modules='q,k,v,o,ffn.0,ffn.2'
resume_lora_ckpt=null

wandb_activated=true
wandb_tags='[dynamicverse,wan-ti2v,controlnet_ac3d,scene+camera,video_input,category,text_ca]'

export WANDB_API_KEY=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
export DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=6 exec -a ac3d_text_ca_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.root=${data_root} \
  dataset.target_video_name=video_input.mp4 \
  dataset.prompt_style=category_first \
  data_loader.val.standard.batch_size=4 \
  data_loader.val.standard.max_batches=8 \
  data_loader.val.unseen.batch_size=4 \
  data_loader.val.unseen.max_batches=8 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  model.denoiser.scene_input_type=${scene_input_type} \
  model.denoiser.camera_input_type=${camera_input_type} \
  model.denoiser.condition_latents_input_type=${condition_latents_input_type} \
  +model.denoiser.ac3d_use_text_cross_attn=true \
  model.denoiser.lora.enabled=${lora_enabled} \
  model.denoiser.lora.rank=${lora_rank} \
  model.denoiser.lora.alpha=${lora_alpha} \
  model.denoiser.lora.target_modules="'${lora_target_modules}'" \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=${wandb_activated} \
  +wandb.tags=${wandb_tags} \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
