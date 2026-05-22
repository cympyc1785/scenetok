config=custom/scenetok_wan-wan_lognorm_re10k_scratch
num_workers=8
gpus=1
num_nodes=1
exp_name="scenetok_wan-wan_re10k_256_scratch"


export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=1 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
