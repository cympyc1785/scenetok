config="scenetok_va-vdc_shift8_dl3dv_finetuned"
dataset=""
ckpt="./checkpoints/va-videodc_dl3dv.ckpt"
data_root="./DATA"
output_dir="./results"
view_sampler=""
index_path="./assets/evaluation_index/dl3dv_c16_32.json"


python -m src.main +experiment=${config} mode=test hydra.job.name=test \
  dataset=${dataset} \ 
  wandb.activated=false \ 
  trainer.limit_test_batches=1 \
  data_loader.test.batch_size=1 \
  model.compressor.ckpt_path=${ckpt} \
  model.compressor.load_strict=false \
  model.denoiser.ckpt_path=${ckpt} \
  model.denoiser.load_strict=false \
  dataset.root=${data_root} \
  hydra.run.dir=${output_dir} \
  dataset/view_sampler=${view_sampler} \
  dataset.view_sampler.index_path=${index_path}


# # RealEstate10
# scenetok_va-vdc_lognorm_re10k_scratch # va-videodc_re10k.ckpt

# # DL3DV (VA-VAE + VideoDCAE)
# scenetok_va-vdc_shift8_dl3dv_finetuned # va-videodc_dl3dv.ckpt

# # DL3DV (VA-VAE + Wan)
# scenetok_va-wan_shift4_dl3dv_finetuned # va-wan_dl3dv.ckpt