config=custom/scenetok_va-wan-ti2v_dl3dv
num_workers=0
gpus=1
num_nodes=1
exp_name="exp_va-wan_t2v_recon_train_only_scene_lora_overfit"
resume_lora_ckpt=null
train_scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=2 exec -a dynamic_scenetok_lets_go python -m src.main mode=test +experiment=${config} \
  data_loader.test.num_workers=${num_workers} \
  data_loader.test.batch_size=1 \
  dataset.stage_override=train \
  dataset.scene_id=${train_scene_id} \
  dataset.overfit=true \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.limit_test_batches=1 \
  freeze.denoiser=false \
  freeze.compressor=true \
  freeze.autoencoder=true \
  model.text_encoder=null \
  model.denoiser.lora.enabled=true \
  model.denoiser.lora.checkpoint=${resume_lora_ckpt} \
  wandb.activated=false \
  hydra.run.dir=exp/${exp_name}_infer_train_one \
  checkpointing.dirpath=my_checkpoints/${exp_name}
