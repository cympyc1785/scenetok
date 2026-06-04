config=custom/scenetok_va-wan-ti2v_dl3dv
num_workers=8
gpus=1
num_nodes=1
exp_name="exp_va-wan_t2v_recon_train_only_scene_lora"
resume_lora_ckpt=null

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=2 exec -a dynamic_scenetok_lets_go python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  freeze.denoiser=false \
  freeze.compressor=true \
  freeze.autoencoder=true \
  model.text_encoder=null \
  model.denoiser.lora.enabled=true \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
