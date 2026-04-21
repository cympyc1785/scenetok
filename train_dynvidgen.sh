config=scenetok_va-wan_shift8_davis_finetuned_dynvid
num_workers=8
gpus=1
num_nodes=1
data_root=./WorldTraj/dynamicverse/DAVIS
meta_file=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/WorldTraj/dynamicverse/meta.csv
ckpt="./checkpoints/va-wan_dl3dv.ckpt"
exp_name="exp_va-wan"

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=1 exec -a dynamic_scenetok_lets_go_4 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  dataset.root=${data_root} \
  dataset.meta_file=${meta_file} \
  dataset.overfit_train_all_meta=true \
  dataset.overfit_val_from_meta_last=true \
  model.denoiser.ckpt_path=${ckpt} \
  freeze.denoiser=false \
  freeze.compressor=true \
  freeze.autoencoder=true \
  wandb.activated=true \
  hydra.run.dir=${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}

# # RealEstate10
# scenetok_va-vdc_lognorm_re10k_scratch

# # DL3DV (VA-VAE + VideoDCAE)
# scenetok_va-vdc_shift4_dl3dv_finetuned
# scenetok_va-vdc_shift8_dl3dv_finetuned

# # DL3DV (VA-VAE + Wan)
# scenetok_va-wan_shift4_dl3dv_finetuned
# scenetok_va-wan_shift8_dl3dv_finetuned
