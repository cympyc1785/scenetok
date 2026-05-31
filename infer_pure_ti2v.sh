ROOT_DIR=/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/

config=custom/scenetok_va-wan-ti2v_dl3dv
num_workers=0
gpus=1
num_nodes=1
exp_name="pure_ti2v"
# 학습 풀(train) 안에서 단일 scene 1개를 골라 inference. 다른 scene 보고싶으면 교체.
train_scene_id="1K/a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97"

# Text prompt — 학습은 text_encoder=null로 돌았지만 Wan TI2V DiT 내부의 text
# cross-attention과 pretrained text encoder는 그대로 살아있어서 추론 시 활용 가능.
# prompt="A static indoor scene with consistent lighting and texture detail"
prompt="A teddy bear is running across the road."
negative_prompt="'色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'"

export WANDB_API_KEY=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
export DEBUG=1
CUDA_VISIBLE_DEVICES=0 exec -a dynamic_scenetok_lets_go python -m src.main +experiment=${config} \
  mode=test \
  data_loader.test.num_workers=${num_workers} \
  data_loader.test.batch_size=1 \
  +dataset.stage_override=train \
  +dataset.scene_id=${train_scene_id} \
  dataset.smallset=true \
  +dataset.val_seen=true \
  +dataset.evaluation_index_path=${ROOT_DIR}/assets/evaluation_index/dl3dv_c16_37_standard.json \
  dataset.context_shape=[256,256] \
  dataset.target_shape=[480,832] \
  dataset.scale_context_focal_by_256=true \
  dataset.do_scale_and_pad=false \
  dataset.target_latent_type=wan \
  dataset.view_sampler.num_context_views=16 \
  dataset.view_sampler.num_target_views=10 \
  dataset.view_sampler.min_distance_between_context_views=0 \
  dataset.view_sampler.offset=1 \
  dataset.view_sampler.chunk_targets=false \
  test.prompt="${prompt}" \
  test.negative_prompt="${negative_prompt}" \
  model.compressor=null \
  model.denoiser.input_shape=[30,52] \
  model.denoiser.camera.input_shape=[240,416] \
  model.scheduler.num_inference_steps=50 \
  model.scheduler.kwargs.timestep_shift=8 \
  model.cfg_scale=5.0 \
  model.denoiser.noise_seed=0 \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.limit_test_batches=1 \
  freeze.denoiser=true \
  freeze.compressor=false \
  freeze.autoencoder=true \
  model.denoiser.scene_input_type=none \
  model.denoiser.condition_latents_input_type=none \
  model.denoiser.camera_input_type=none \
  model.denoiser.camera_context_spatial_pool=16 \
  model.denoiser.lora.enabled=false \
  wandb.activated=false \
  hydra.run.dir=results/infer_${exp_name} \
  test.output_dir=results/infer_${exp_name}
