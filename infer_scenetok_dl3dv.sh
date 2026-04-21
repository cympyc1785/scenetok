io_mode="c16_16_extra"

config=scenetok_va-vdc_shift8_dl3dv_finetuned
dataset="dl3dv"
ckpt="./checkpoints/va-videodc_dl3dv.ckpt"
data_root="./DATA"
output_dir=./results/tok_${io_mode}
view_sampler="evaluation_video"
index_path=./assets/evaluation_index/dl3dv_${io_mode}_custom.json

exec -a scenetok_lets_go python -m src.main +experiment=${config} mode=test hydra.job.name=test \
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