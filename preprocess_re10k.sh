config=custom/scenetok_wan-wan_lognorm_re10k_scratch
exp_name="preprocess"
resume_lora_ckpt=null

num_workers=8

export DEBUG=1
CUDA_VISIBLE_DEVICES=0 exec -a dynamic_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=preprocess_data \
  wandb.activated=false \
  hydra.run.dir=${exp_name}
