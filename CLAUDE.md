# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dynamic video generation with scene tokens. Built on top of SceneTok (CVPR '26): a scene autoencoder that compresses multi-view scenes into 1D scene tokens. The current line of work plugs those scene tokens into a Wan-based TI2V/T2V denoiser to produce camera- and scene-aligned dynamic video.

연구 가설: SceneTok의 flexible한 scene tokens을 T2V 모델과 결합하면 scene과 camera에 align된 dynamic video를 생성할 수 있다.

## Commands

- 학습: `bash train_ti2vgen_recon_overfit.sh` (single-view-conditioned recon overfit)
- 학습 (interp variant): `bash train_ti2vgen_recon_overfit_interp.sh`
- 검증: `bash val_ti2vgen_recon_overfit.sh`
- Smoke / load test: `bash load_ti2v_test.sh`

All shells call `python -m src.main +experiment=<config> ...` (Hydra). `src.main` is the standard entry; `src.main_scene` is used only for SceneGen.

Required env vars in the shells:
- `WANDB_API_KEY` — already inlined in each script
- `DEBUG=1` — disables `torch.compile` for all modules (set in every current shell). Drop it only when actually benchmarking compiled training.
- `CUDA_VISIBLE_DEVICES` — pinned (most shells use device `1`). Override at the shell level before invoking, do not delete from the script.

### Modes
The `mode=` Hydra arg routes the trainer:
- `train` — `trainer.fit`
- `val` — `trainer.validate`
- `test` — `trainer.test`
- `predict_test` / `predict_train` — `trainer.predict` over the respective loader
- `preprocess_data` — runs `preprocess_dataset_cache` over train/val/test and exits before building the trainer (see `src/main.py:41`)

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

### Configs (Hydra, config/)
- `config/main.yaml` — defaults composition (dataset, autoencoders, scheduler, denoiser, …)
- `config/experiment/*.yaml` — published SceneTok / SceneGen experiments. **Do not edit these.**
- `config/experiment/custom/*.yaml` — in-house experiments for this project. The active one is `scenetok_va-wan-ti2v_dl3dv.yaml` (and `_interp` variant); both override `denoiser=wan_ti2v_5b` and target the Wan VAE.

Key knobs exposed on the denoiser side (driven from the shell scripts):
- `model.denoiser.scene_input_type` ∈ `{none, cross_attention, new_cross_attention, latent_concat}` — how scene tokens enter the DiT
- `model.denoiser.camera_input_type` ∈ `{none, recam_attention, cross_attention, new_cross_attention, adaln, wan_control}` — camera conditioning path
- `model.denoiser.condition_latents_input_type` ∈ `{none, width, channel, temporal, first_frame, first_frame_random}` — how condition latents/frames are fused
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
- `src/dataset/dataset_dl3dv.py` is the primary loader; `build_dl3dv_meta` is invoked from `src/main.py`.
- View samplers in `config/dataset/view_sampler/` — `bounded` is used for training; `evaluation_video_wan` / `evaluation_video` for eval depending on target autoencoder.
- `dataset.smallset=true` is the overfit/dev subset used in every current train script.

## Don't

- 실험 결과를 임의로 요약 금지 (wandb 원본 수치 그대로).
- 기존 (published) config 수정 금지. 새 실험은 새 config 파일 — drop it under `config/experiment/custom/`.
- `src/main_scene.py` 경로(SceneGen)는 이 브랜치의 TI2V 작업과 무관하니 함께 건드리지 말 것.
- `checkpoints/` 폴더에 학습 산출물 저장 금지. 우리 결과물은 `my_checkpoints/`.

## Coding conventions

- 기존의 핵심 작동 구조가 있다면 option에 따라 분기를 쳐서 기존의 방식도 똑같이 돌아가게끔 유지. 새 옵션을 추가할 때는 위의 `*_input_type` Literal에 새 값을 추가하고 기본값(`none` / 기존 동작)을 유지할 것.
- New experiment = new YAML in `config/experiment/custom/`. New denoiser variant = new file in `src/model/denoiser/` + register via `get_denoiser`.
- Wrapper routing은 `src/main.py:174` 의 `denoiser.name` 분기를 통해 결정됨 — 새로운 denoiser 패밀리를 도입하면 이 분기를 함께 갱신.


## Changelog

이 프로젝트는 [Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

코드를 변경할 때마다 `CHANGELOG.md`의 `[Unreleased]` 섹션에 항목을 추가할 것.

## 작업 흐름

코드 변경 작업이 끝나면 **반드시** 다음을 수행한다:
1. 변경 내용을 `CHANGELOG.md`의 `[Unreleased]`에 기록
2. 사용자에게 어느 카테고리에 추가했는지 알려줄 것

이 단계를 건너뛰지 말 것. 사소한 변경이라도 사용자에게 영향이 있으면 기록한다.