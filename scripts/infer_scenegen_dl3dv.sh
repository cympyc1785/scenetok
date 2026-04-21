config=scenegen_shift12_re10k
scenetok_ckpt=checkpoints/va-videodc_re10k_scene.ckpt
scenegen_ckpt=checkpoints/scenegen_shift12_re10k.ckpt
data_root="./DATA/dl3dv_subset"
output_dir="./results/gen_dl3dv_c16_160_extra"
index_path="./assets/evaluation_index/dl3dv_c16_160_extra.json"

exec -a scenegen_lets_go python -m src.main_scene +experiment=${config} mode=test hydra.job.name=test \
  dataset=dl3dv \
  wandb.activated=false \
  trainer.limit_test_batches=1 \
  data_loader.test.batch_size=1 \
  model.compressor.ckpt_path=${scenetok_ckpt} \
  model.compressor.load_strict=false \
  model.denoiser.ckpt_path=${scenetok_ckpt} \
  model.denoiser.load_strict=false \
  model.scene_generator.ckpt_path=${scenegen_ckpt} \
  model.scene_generator.load_strict=false \
  dataset.root=${data_root} \
  hydra.run.dir=${output_dir} \
  dataset/view_sampler=evaluation_video \
  dataset.view_sampler.index_path=${index_path}

# # RealEstate10K
# scenegen_shift1_re10k   # scenegen_shift1_re10k.ckpt
# scenegen_shift4_re10k   # scenegen_shift4_re10k.ckpt
# scenegen_shift12_re10k  # scenegen_shift12_re10k.ckpt

# # All SceneGen configs require:
# # - scenetok_ckpt=checkpoints/va-videodc_re10k_scene.ckpt
# # - scenegen_ckpt=checkpoints/scenegen_shift{1,4,12}_re10k.ckpt