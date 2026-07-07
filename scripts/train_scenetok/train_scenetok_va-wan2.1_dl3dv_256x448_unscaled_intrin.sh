config=custom/scenetok_va-wan2.1_dl3dv_256x448_unscaled_intrin
exp_name="scenetok_va-wan2.1_dl3dv_256x448_unscaled_intrin"
: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python -m src.main +experiment=${config} \
  data_loader.train.num_workers=8 mode=train trainer.devices=1 trainer.num_nodes=1 \
  trainer.num_sanity_val_steps=1 wandb.activated=true \
  hydra.run.dir=exp/${exp_name} checkpointing.dirpath=my_checkpoints/${exp_name}
