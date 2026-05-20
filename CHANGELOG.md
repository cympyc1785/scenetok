# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- `train_scenetok_256.sh`: 256x448 입력/출력 학습에 맞춰 denoiser/compressor latent grid를 `[16,28]`, camera input grid를 `[128,224]`로 override.
- `src/dataset/dataset_dl3dv.py`: context/target preprocessing 경로를 통일하고 출력 image shape가 요청한 `context_shape`/`target_shape`와 다르면 즉시 에러를 내도록 검증 추가.
- `train_scenetok.sh` (`scenetok_wan-wan_dl3dv_480_scratch_large`, 480×832): `data_loader.train.batch_size=2`로 낮추고 `trainer.accumulate_grad_batches=2`를 추가해 effective batch는 유지하면서 첫 training step의 backward activation OOM(B200, RoPE `rotate_half` 경로에서 발현되던 244 MiB 추가 할당 실패)을 회피.
