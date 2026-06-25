config=custom/scenetok_va-wan_shift8_dl3dv_finetuned_wide
num_workers=4
gpus=1
num_nodes=1
exp_name="aika_scenetok_va-wan_dl3dv_256-480_finetune_small"


export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=1 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.root=./DATA/DL3DV/DL3DV-960/DL3DV-10K \
  dataset.smallset=true \
  dataset.context_shape=[256,448] \
  dataset.target_shape=[480,832] \
  model.compressor.input_shape=[16,28] \
  model.compressor.camera.input_shape=[128,224] \
  model.denoiser.input_shape=[30,52] \
  model.denoiser.camera.input_shape=[240,416] \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  # optimizer.lr=1.0e-4 \
