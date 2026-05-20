ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok

config=custom/scenetok_va-wan-ti2v_dl3dv
num_workers=4
gpus=1
num_nodes=1
exp_name="va-wan-ti2v_recon_aggressive_train_480_scene_new_ca_recam_small"
resume_lora_ckpt=null

checkpoint_path="${ROOT_DIR}/my_checkpoints/${exp_name}/last.ckpt"

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=1 exec -a dynamic_scenetok_lets_go python -m src.main +experiment=${config} \
  mode=val \
  dataset.smallset=true \
  dataset.context_shape=[256,256] \
  dataset.target_shape=[480,832] \
  dataset.do_scale_and_pad=false \
  data_loader.val.standard.num_workers=${num_workers} \
  data_loader.val.standard.batch_size=1 \
  data_loader.val.standard.max_batches=32 \
  data_loader.val.unseen.num_workers=${num_workers} \
  data_loader.val.unseen.batch_size=1 \
  data_loader.val.unseen.max_batches=32 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=32 \
  freeze.denoiser=true \
  freeze.compressor=true \
  freeze.autoencoder=true \
  model.text_encoder=null \
  model.denoiser.scene_input_type=cross_attention \
  model.denoiser.condition_latents_input_type=none \
  model.denoiser.camera_input_type=recam_attention \
  model.denoiser.lora.enabled=true \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  checkpointing.load="${checkpoint_path}" \
  checkpointing.save=false \
  wandb.activated=false \
  hydra.run.dir=results/val_${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
