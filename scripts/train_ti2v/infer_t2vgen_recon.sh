ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/

config=custom/scenetok_va-wan-ti2v_dl3dv
num_workers=0
gpus=1
num_nodes=1
exp_name="exp_va-wan_t2v_recon_train_latent_concat"
train_scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

checkpoint_path="${ROOT_DIR}/my_checkpoints/${exp_name}/last.ckpt"
hydra_config_dir="${ROOT_DIR}/exp/${exp_name}/.hydra"

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=1 exec -a dynamic_scenetok_lets_go python -m src.main +experiment=${config}\
  mode=test \
  data_loader.test.num_workers=${num_workers} \
  data_loader.test.batch_size=1 \
  +dataset.stage_override=train \
  +dataset.scene_id=${train_scene_id} \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.limit_test_batches=1 \
  model.text_encoder=null \
  model.denoiser.scene_input_type=none \
  model.denoiser.condition_latents_input_type=none \
  model.denoiser.camera_input_type=none \
  model.denoiser.camera_context_spatial_pool=16 \
  model.denoiser.lora.enabled=true \
  wandb.activated=false \
  hydra.run.dir=exp/infer \
  test.output_dir=results/infer_${exp_name}
  # checkpointing.load="${checkpoint_path}" \
  # --config-path "${hydra_config_dir}" \
  # --config-name config \
