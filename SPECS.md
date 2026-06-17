## Commands

이 레포는 두 개의 평행한 학습 라인이 동시에 존재한다. 작업 대상이 어느 쪽인지 먼저 확인할 것. **`f65f87d "Organize scripts" (2026-06-04)` 이후 학습/추론 셸들은 `scripts/train_scenetok/`와 `scripts/train_ti2v/` 하위로 이동됨.** root에 남아 있는 것은 `train_scenetok.sh` (RE10K va-vdc scratch baseline), `train_scenegen.sh`, `train_dynvidgen{,_exp_1}.sh`, plus eval/infer/convert/preprocess 셸들 + `fast_infer_t2v.sh`, `infer_pure_ti2v.sh`, `infer_lightningdit_reco.sh`.

**A. SceneTok (scene autoencoder) 학습** — `scripts/train_scenetok/`
- `train_scenetok_re10k_256.sh` → `scenetok_va-wan_re10k_256_scratch` (RE10K va-wan 256 scratch)
- `train_scenetok_re10k_vavdc.sh` → `scenetok_va-vdc_re10k_256_scratch` (RE10K va-vdc 256 scratch)
- `train_scenetok_dl3dv_256.sh` → `scenetok_va-wan_dl3dv_256_scratch_aligned` (DL3DV va-wan 256x448 scratch, large)
- `train_scenetok_dl3dv_256_finetune.sh` → `scenetok_va-wan_dl3dv_256_finetune_large` (DL3DV va-wan 256x448 finetune from `va-wan_dl3dv.ckpt`)
- `train_scenetok_dl3dv_480_finetune.sh` → `scenetok_va-wan_dl3dv_480_finetune_large` (DL3DV va-wan 480x832 finetune from `va-wan_dl3dv.ckpt`)
- `train_scenetok_dl3dv_256-480_finetune.sh` / `_aggressive.sh` → ctx 256x448 / tgt 480x832 비대칭 finetune (small)
- `train_scenetok_dl3dv_wan_256.sh` / `train_scenetok_dl3dv_wan_480.sh` — DL3DV wan-wan scratch (large)
- `train_scenetok_chunked.sh` — DL3DV wan-wan 480 chunked-target ablation
- `bash train_scenetok.sh` (root) — RE10K va-vdc scratch baseline (`scenetok_va-vdc_re10k_scratch`)
- `bash train_dynvidgen.sh` / `train_dynvidgen_exp_1.sh` (root) — DAVIS dynvid finetune; data root `./WorldTraj/dynamicverse/DAVIS`

**B. TI2V/T2V denoiser (scene-token-conditioned video gen)** — `scripts/train_ti2v/`
- Recon-overfit base: `train_ti2vgen_recon_overfit.sh` (256-480) / `_256_finetuned.sh` / `_480.sh` / `_480_finetuned.sh` / `_interp.sh`
- Recon-overfit ablations: `_256_finetuned_ac3d.sh` (scene_new_ca + ac3d camera), `_256_finetuned_ac3d_full_controlnet.sh` (scene controlnet + camera controlnet_feedback), `_256_finetuned_adaln_cam.sh`, `_256_finetuned_cam_concat_scene_ca_lora.sh`, `_256_finetuned_depth.sh` (first_frame_depth), `_256_finetuned_new_cam.sh` (wan_control camera), `_256_finetune_all.sh` / `_not_aggressive.sh` (encoder도 trainable)
- Full TI2V/T2V (non-overfit): `train_t2vgen.sh` (DAVIS, `scenetok_va-wan-ti2v_davis`), `train_t2vgen_recon.sh` (DL3DV, `scenetok_va-wan-ti2v_dl3dv`)
- 14B variant: `train_t2vgen_14B_recon_overfit.sh`
- Inference: `infer_t2vgen_recon.sh` / `infer_t2vgen_text.sh` (recon-style / text-cond), `infer_ti2vgen.sh` / `_recon_480.sh`
- Eval: `eval_ti2vgen_recon_480.sh` (학습 train 후 같은 split val)
- Smoke / load test: `load_ti2v_test.sh`
- Validation: `val_ti2vgen_recon_overfit.sh`

**B-2. Fast standalone TI2V inference (root)**
- `bash fast_infer_t2v.sh` (`scripts/fast_infer_t2v.py`) — Hydra/Lightning을 우회해 `exp/<exp_name>/.hydra/config.yaml` + `last.ckpt`를 직접 로드, 한 scene에 `{cfg, 1.0}×{prompt, ""}` 4 combo를 mp4로 sampling
- `bash infer_pure_ti2v.sh` — pure TI2V (scene token 없음) baseline 비교용

**C. lightningDiT → ReCo 2-stage 파이프라인 (root, 추론 전용)**
- `bash infer_lightningdit_reco.sh` — Stage 1 SceneTok lightningDiT coarse render → Stage 2 ReCo (Wan2.1-VACE-1.3B + LoRA) edit. 두 stage가 서로 다른 `diffsynth` 패키지(메인 repo vs ReCo 벤더링)를 써서 같은 Python 프로세스에 임포트하면 충돌 → sequential subprocess 분리.
- 셸 안에 4 케이스 (256+upscale480 / 256 native / 480 native / published va-vdc 256x256) `run_pipeline` 함수로 묶임.

**기타 (root)**
- Eval: `bash eval_scenetok_{re10k,dl3dv}.sh`, inference: `bash infer_scenetok_{re10k,dl3dv,davis}.sh`
- Latent precompute: `bash convert_{dl3dv,re10k}_{vavae,videodc}.sh`
- Dataset cache precompute (`mode=preprocess_data`): `bash preprocess_{dl3dv,re10k}.sh`
- Custom eval: `python scripts/eval_compare_256upscale_vs_480.py` — DL3DV standard/unseen 140 scene 두 ckpt 비교 (per-scene CSV + summary.json)

All shells call `python -m src.main +experiment=<config> ...` (Hydra). `src.main` is the standard entry; `src.main_scene` is used only for SceneGen.

Required env vars in the shells:
- `WANDB_API_KEY` — already inlined in each script
- `DEBUG=1` — disables `torch.compile` for all modules (set in every current shell). Drop it only when actually benchmarking compiled training.
- `CUDA_VISIBLE_DEVICES` — pinned per-shell (often `1`, `2`, or `3` depending on which run a script is for). Override at the shell level before invoking, do not delete from the script.
- `OMP_NUM_THREADS` (+ MKL/OpenBLAS) — `src/main.py`가 import 이전에 `setdefault("8")`. 72-core 머신에서 train job 여러 개 동시 실행 시 intra-op pool이 코어를 독점하는 것 방지. shell에서 export하면 override 가능. `src.main_scene`엔 미적용.

### Modes
The `mode=` Hydra arg routes the trainer:
- `train` — `trainer.fit`
- `val` — `trainer.validate`
- `test` — `trainer.test`
- `predict_test` / `predict_train` — `trainer.predict` over the respective loader
- `preprocess_data` — runs `preprocess_dataset_cache` over train/val/test and exits before building the trainer (see `src/main.py:101`)

## Architecture

### Entry points & model routing
`src/main.py` builds the Lightning trainer and chooses one of two wrappers based on `cfg.model.denoiser.name`:
- `wan_ti2v_5b` or `wan_t2v_14b` → `src/model/t2v_wrapper.py::T2VWrapper`
- everything else → `src/model/diffusion_wrapper.py::DiffusionWrapper`

`T2VWrapper` subclasses `DiffusionWrapper` and overrides text-encoder init, scene-token preprocessing, and condition-latents (first-frame conditioning, width-concat for Wan TI2V). When adding new denoiser families, decide which wrapper they extend rather than forking a third.

`src.main_scene` is the SceneGen entry (compressor + scene generator) and is independent of the T2V work in this branch — don't conflate the two.

### Models (src/model/)
- `autoencoder/` — pluggable VAEs. Current target encoders: `va` (VA-VAE), `videodc` (VideoDCAE), `wan` (Wan 2.2 VAE), plus context encoders.
- `compressor/` — multi-view perceiver that produces scene tokens from a context-view set. SceneTok checkpoint is loaded here.
- `denoiser/` — diffusion backbones. `wan_ti2v.py` and `wan_t2v_14B.py` wrap DiffSynth-Studio's Wan DiT and add LoRA + scene/camera/condition-latent injection points. `lightningdit.py` is the legacy SceneTok denoiser.
- `camera/` — Plücker / ray / Wan-Plücker camera encoders. Selected via `model.denoiser.camera`.
- `scene_generator/`, `scheduler/`, `sampler/` — used by SceneGen and inference paths.
- `DiffSynth-Studio/`, `ac3d/`, `GEN3C/` — vendored upstream repos (Wan DiT/VAE, AC3D camera-control reference, GEN3C). Treat as read-only references; our code imports from them, don't refactor them.

### Configs (Hydra, config/)
- `config/main.yaml` — defaults composition (dataset, autoencoders, scheduler, denoiser, …)
- `config/experiment/*.yaml` — published SceneTok / SceneGen experiments. **Do not edit these.**
- `config/experiment/custom/*.yaml` — in-house experiments for this project. Two groups:
  - **SceneTok-side (active):** `scenetok_va-vdc_lognorm_re10k_scratch.yaml`, `scenetok_wan-wan_lognorm_re10k_scratch.yaml`, `scenetok_wan-wan_shift4_dl3dv_scratch.yaml`, `scenetok_va-vdc_shift{4,8}_dl3dv_finetuned_fixed.yaml`, `scenetok_va-wan_shift8_dl3dv_scratch{,_wide}.yaml`, plus DAVIS finetune variants (`*_davis_finetuned_dynvid.yaml`). These train the autoencoder (compressor + decoder denoiser).
  - **TI2V denoiser-side:** `scenetok_va-wan-ti2v_dl3dv.yaml` (+ `_interp`), `scenetok_va-wan-t2v-14B_dl3dv.yaml`, `scenetok_va-wan-ti2v_davis.yaml` — override `denoiser=wan_ti2v_5b` / `wan_t2v_14b`.

Key knobs exposed on the denoiser side (driven from the shell scripts):
- `model.denoiser.scene_input_type` ∈ `{none, cross_attention, new_cross_attention, latent_concat}` — how scene tokens enter the DiT
- `model.denoiser.scene_projection` ∈ `{linear, mlp}` — scene-token projector (`cnd_proj`); `mlp` = `Linear→GELU(tanh)→Linear`
- `model.denoiser.camera_input_type` ∈ `{none, recam_attention, cross_attention, new_cross_attention, adaln, wan_control, channel_concat, controlnet, controlnet_feedback, controlnet_ac3d}` — camera conditioning path. `channel_concat` concats raw Plücker rays into `patch_embedding` (zero-init extra channels); `controlnet` is the AC3D-style parallel ControlNet (separate `ac3d_patch_embedding(latent⊕ray)` + first `ac3d_num_layers` block copies + zero-init projectors residual-injected after each main block); `controlnet_feedback` adds zero-init `ac3d_fc_down` for main→ctrl feedback per layer (interleaved); `controlnet_ac3d` is the AC3D paper VDiT-CC variant — noise stays in main only, ctrl block input = `FC_main2ctrl(main_x_i) + ac3d_cam_emb + ctrl_x_{i-1}`, ctrl blocks have NO text cross-attn (only self_attn + scene_cross_attn + FFN), zero-init `ac3d_projectors` gate the residual back to main
- `model.denoiser.condition_latents_input_type` ∈ `{none, width, channel, temporal, first_frame, first_frame_random, first_frame_depth, first_frame_depth_soft}` — how condition latents/frames are fused
- `model.denoiser.lora.{enabled,rank,alpha,target_modules,checkpoint}` — PEFT LoRA over Wan DiT projection / FFN layers
- `freeze.{denoiser,compressor,autoencoder}` — what gets gradients

### Output / checkpoint convention
Each shell sets:
- `hydra.run.dir=exp/${exp_name}` — wandb logs & Hydra run dir
- `checkpointing.dirpath=my_checkpoints/${exp_name}` — Lightning checkpoints (`last.ckpt`, plus top-k)
- For val: `hydra.run.dir=results/val_${exp_name}` and `checkpointing.load=...` pulls the train run's `last.ckpt`

Pretrained weights live under `checkpoints/` (downloaded SceneTok/VAE/SceneGen weights from the README) — that directory is for *upstream* checkpoints, while `my_checkpoints/` is for our own training runs.

### Data
- `src/dataset/__init__.py` registers `re10k`, `dl3dv`, `latent`, `davis`.
- `dl3dv` and `re10k` 둘 다 `meta.csv` 기반 — 첫 init 시 `build_{dl3dv,re10k}_meta`가 split별로 모든 chunk를 스캔해서 `DATA/{ds}/.../meta.csv (chunk, key, num_images)`를 작성하고, 이후엔 csv만 읽어 `num_images >= cfg.min_frames` scene만 사용. rebuild 필요 시 csv 삭제.
- View samplers (`src/dataset/view_sampler/`): training은 `bounded` (dl3dv) 또는 `unbounded` (re10k 등). `unbounded`는 `offset != 0`이면 Wan 4N+1 chunk 레이아웃을 따른다 (chunk당 raw `4N-3` 프레임). `chunk_targets`는 sampler 자체 동작은 그대로지만 downstream wrapper가 raw→latent 변환 경로를 선택하는 플래그.
- Eval은 `assets/evaluation_index/{ds}_c{NC}_{NT}_{standard,unseen}.json` 사전 샘플 파일을 사용. `(chunk_targets, val_seen)` 조합으로 자동 선택되며 (`load_evaluation_index`), `chunk_targets=true`면 `_34`, false면 `_37` (raw frame count). `standard`는 train pool, `unseen`은 test pool. config에서 `evaluation_index_path`를 명시하면 그 값이 우선.
- `dataset.smallset=true` is the overfit/dev subset; 일부 SceneTok 학습은 `smallset=false`로 full split 사용.
- **Scene blacklist** (dl3dv & re10k): `{cfg.root}/{train|test}/blacklist.csv` (`scene, reason, step, loss, detail`)에 적힌 scene은 meta.csv 재빌드 없이 `__init__` 필터에서 즉시 제외됨. helper는 `dataset_dl3dv.py`의 `load_blacklist`/`append_blacklist`/`_resolve_blacklist_path` (re10k가 재사용). DL3DV는 학습 중 NaN loss 발생 시 `diffusion_wrapper._diagnose_batch_for_blacklist`가 데이터 anomaly를 잡아 자동 append (auto-mode 친화적); re10k는 자동 append 분기가 없어 수동 편집 워크플로우. `scripts/blacklist_non_consecutive_frames.py`로 비연속 프레임 scene을 일괄 등록.

### Known gotchas
- **Wan 2.2 VAE + 1024 scene_tokens (wan-wan scratch):** published va-vdc는 512 tokens + `kl_weights=1e-10`로 안정했지만, 1024 tokens 환경에선 동일 anchor가 부족해 step ~2k–20k 사이에 cascading gradient NaN으로 발산한다. 대응: compressor `init_large=false, init_small=true` (scene_tokens 초기 std 5.0 → 0.02) + `kl_weights` 단계적 상향 (1e-10 → 1e-7 → 1e-6 → 1e-5). 새 wan-wan 학습 시작 전에 `scenetok_wan-wan_lognorm_re10k_scratch.yaml`의 현재 값을 그대로 따라가는 게 안전. 진단 스크립트: `python scripts/check_wan_vae_outliers.py`.
- **FVD `sqrtm` hang:** `submodules/fvd/frechet_video_distance.py`의 `calculate_fvd_from_activations`는 rank-deficient I3D feature cov에서 scipy `sqrtm`이 무한 스핀할 수 있음. 현재는 원본대로 두고 NaN/Inf fallback만 활용 (사용자 요청으로 1e-6 regularization은 revert됨). validation hang 보이면 sample 수 늘리거나 ε 패치 부활 검토.