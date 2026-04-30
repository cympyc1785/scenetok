ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/

config=custom/scenetok_va-wan-ti2v_dl3dv
num_workers=0
exp_name="exp_va-wan_t2v_recon_train_latent_concat"
train_scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

checkpoint_path="${ROOT_DIR}/my_checkpoints/${exp_name}/last.ckpt"
hydra_config_dir="${ROOT_DIR}/exp/${exp_name}/.hydra"

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=0 exec -a dynamic_scenetok_lets_go python -m src.main +experiment=${config}\
  mode=test \
  data_loader.test.num_workers=${num_workers} \
  data_loader.test.batch_size=1 \
  +dataset.stage_override=train \
  +dataset.scene_id=${train_scene_id} \
  test.prompt="rabbit is jumping on the grass" \
  test.negative_prompt="'色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'" \
  trainer.limit_test_batches=1 \
  model.text_encoder=null \
  model.denoiser.scene_input_type=none \
  model.denoiser.condition_latents_input_type=none \
  model.denoiser.camera_input_type=none \
  model.denoiser.lora.enabled=false \
  wandb.activated=false \
  hydra.run.dir=exp/infer \
  test.output_dir=results/infer \
  model.scheduler.num_inference_steps=50 \
  model.scheduler.kwargs.timestep_shift=8 \
  model.cfg_scale=5.0 \
  model.denoiser.noise_seed=0 \
  dataset.view_sampler.num_target_views=10 \
  model.denoiser.input_shape=[30,52]
  # checkpointing.load="${checkpoint_path}" \
  # --config-path "${hydra_config_dir}" \
  # --config-name config \
