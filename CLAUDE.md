# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dynamic video generation with scene tokens. Built on top of SceneTok (CVPR '26): a scene autoencoder that compresses multi-view scenes into 1D scene tokens. The current line of work plugs those scene tokens into a Wan-based TI2V/T2V denoiser to produce camera- and scene-aligned dynamic video.

연구 가설: SceneTok의 flexible한 scene tokens을 T2V 모델과 결합하면 scene과 camera에 align된 dynamic video를 생성할 수 있다.

## Commands

이 레포는 두 개의 평행한 학습 라인이 동시에 존재한다. 작업 대상이 어느 쪽인지 먼저 확인할 것.

**A. SceneTok (scene autoencoder) 학습 — 현재 active focus**
- `bash train_scenetok.sh` — RE10K va-vdc scratch (`scenetok_va-vdc_re10k_scratch`)
- `bash train_scenetok_re10k_256.sh` — RE10K wan-wan 256 scratch
- `bash train_scenetok_re10k_vavdc.sh` — RE10K va-vdc 256 scratch
- `bash train_scenetok_dl3dv_256.sh` — DL3DV va-wan 256 scratch (large)
- `bash train_scenetok_dl3dv_wan_256.sh` — DL3DV wan-wan 256 scratch (large)
- `bash train_scenetok_dl3dv_wan_480.sh` — DL3DV wan-wan 480 scratch (large)
- `bash train_scenetok_chunked.sh` — DL3DV wan-wan 480 chunked-target ablation
- `bash train_dynvidgen.sh` / `train_dynvidgen_exp_1.sh` — DAVIS dynvid finetune (`scenetok_va-wan_shift8_davis_finetuned_dynvid`); data root `./WorldTraj/dynamicverse/DAVIS`

**B. TI2V/T2V denoiser (scene-token-conditioned video gen)**
- Single-view recon overfit: `bash train_ti2vgen_recon_overfit.sh` / `_interp.sh`; validate with `val_ti2vgen_recon_overfit.sh`
- Full TI2V/T2V (non-overfit): `bash train_t2vgen.sh` (DAVIS, `scenetok_va-wan-ti2v_davis`), `bash train_t2vgen_recon.sh` (DL3DV, `scenetok_va-wan-ti2v_dl3dv`)
- 14B variant: `bash train_t2vgen_14B_recon_overfit.sh`
- Inference: `bash infer_t2vgen_recon.sh` (recon-style), `bash infer_t2vgen_text.sh` (text-conditioned)
- Fast standalone inference: `bash fast_infer_t2v.sh` (`scripts/fast_infer_t2v.py`) — Hydra/Lightning을 우회해 `exp/<exp_name>/.hydra/config.yaml` + `last.ckpt`를 직접 로드, 한 scene에 `{cfg, 1.0}×{prompt, ""}` 4 combo를 mp4로 sampling
- Smoke / load test: `bash load_ti2v_test.sh`

**기타**
- Eval: `bash eval_scenetok_{re10k,dl3dv}.sh`, inference: `bash infer_scenetok_{re10k,dl3dv,davis}.sh`
- Latent precompute: `bash convert_{dl3dv,re10k}_{vavae,videodc}.sh`
- Dataset cache precompute (`mode=preprocess_data`): `bash preprocess_{dl3dv,re10k}.sh`

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
- `model.denoiser.camera_input_type` ∈ `{none, recam_attention, cross_attention, new_cross_attention, adaln, wan_control, channel_concat, ac3d}` — camera conditioning path. `channel_concat` concats raw Plücker rays into `patch_embedding` (zero-init extra channels); `ac3d` is the AC3D / CogVideoX-ControlNet pattern (separate `ac3d_patch_embedding` + first `ac3d_num_layers` block copies + zero-init projectors residual-injected into the main DiT)
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

## Git 커밋 워크플로우

역할 분담: **Claude = 현재 브랜치에 커밋까지, 사용자 = push.** Claude는 브랜치를 새로 만들지 않고 push도 하지 않는다. (이 프로젝트는 "요청받을 때만 커밋" 기본 동작과 "default 브랜치면 먼저 분기" 동작을 의도적으로 override 한다 — 코드 변경 작업이면 분기 없이 현재 브랜치에 자동 커밋.)

### Claude가 자동으로 하는 것
1. 코드 변경 작업이 끝나면 **현재 브랜치에 그대로 커밋**한다 (브랜치 분기 금지).
2. 코드 변경 + `CHANGELOG.md [Unreleased]` 갱신을 **하나의 커밋**으로 묶는다. 커밋 메시지는 `<type>: <한 줄 요약>` (`feat` 새 기능 / `fix` 버그 / `exp` 실험·ablation / `chore` 리팩터·문서·잡일; 예: `feat: ac3d camera_input_type 추가`) + 필요 시 본문에 why.
3. **스테이징은 변경한 파일만 명시적으로 `git add <경로>`.** `git add -A` / `git add .` **금지** — vendored 디렉토리(`src/model/{DiffSynth-Studio,ac3d,GEN3C}`, `WorldTraj`; DiffSynth-Studio는 ~417GB)가 인덱스에 끌려들어가는 사고를 gitignore 상태와 무관하게 차단.
4. 커밋 후 **커밋 요약을 사용자에게 보고하고 멈춘다.**

### Claude가 하지 않는 것 (전부 사용자가 수동)
- `git push`, `git merge`, `git pull`, 브랜치 생성/삭제.

### 커밋하지 않는 경우
- 읽기 전용 / 분석 / 질문 답변 등 코드 변경이 없는 작업.
- 사용자가 명시적으로 "커밋하지 마"라고 지시한 경우.