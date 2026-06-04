config=custom/scenetok_wan-wan_shift4_dl3dv_scratch
num_workers=8
gpus=1
num_nodes=1
exp_name="scenetok_wan-wan_dl3dv_256_scratch_large"


export WANDB_API_KEY=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
export DEBUG=1

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  dataset.smallset=false \
  dataset.context_shape=[256,448] \
  dataset.target_shape=[256,448] \
  model.denoiser.input_shape=[16,28] \
  model.denoiser.camera.input_shape=[128,224] \
  model.compressor.input_shape=[16,28] \
  model.compressor.camera.input_shape=[128,224] \
  model.compressor.kl_weights=[1e-10,1e-10] \
  data_loader.train.batch_size=4 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
  # dataset.view_sampler.chunk_targets=true \
  # model.denoiser.num_target_split=2 \
  

# # RealEstate10
# scenetok_va-vdc_lognorm_re10k_scratch

# # DL3DV (VA-VAE + VideoDCAE)
# scenetok_va-vdc_shift4_dl3dv_finetuned
# scenetok_va-vdc_shift8_dl3dv_finetuned

# # DL3DV (VA-VAE + Wan)
# scenetok_va-wan_shift4_dl3dv_finetuned
# scenetok_va-wan_shift8_dl3dv_finetuned
