config=custom/scenetok_va-wan_shift4_dl3dv_finetuned_wide
num_workers=8
gpus=1
num_nodes=1
exp_name="scenetok_va-wan_dl3dv_480_finetune_large"


export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=3 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.context_shape=[480,832] \
  dataset.target_shape=[480,832] \
  model.denoiser.input_shape=[30,52] \
  model.denoiser.camera.input_shape=[240,416] \
  model.compressor.input_shape=[30,52] \
  model.compressor.camera.input_shape=[240,416] \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  trainer.max_steps=20000 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
