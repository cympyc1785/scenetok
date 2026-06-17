# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dynamic video generation with scene tokens. Built on top of SceneTok (CVPR '26): a scene autoencoder that compresses multi-view scenes into 1D scene tokens. The current line of work plugs those scene tokens into a Wan-based TI2V/T2V denoiser to produce camera- and scene-aligned dynamic video.

연구 가설: SceneTok의 flexible한 scene tokens을 T2V 모델과 결합하면 scene과 camera에 align된 dynamic video를 생성할 수 있다.

## Don't

- 실험 결과를 임의로 요약 금지 (wandb 원본 수치 그대로).
- 기존 (published) config 수정 금지. 새 실험은 새 config 파일 — drop it under `config/experiment/custom/`.
- `src/main_scene.py` 경로(SceneGen)는 이 브랜치의 TI2V 작업과 무관하니 함께 건드리지 말 것.
- `checkpoints/` 폴더에 학습 산출물 저장 금지. 우리 결과물은 `my_checkpoints/`.
- GPU 4~7 사용 금지. `CUDA_VISIBLE_DEVICES`는 항상 0~3 범위에서만 지정할 것.

## Coding conventions

- 기존의 핵심 작동 구조가 있다면 option에 따라 분기를 쳐서 기존의 방식도 똑같이 돌아가게끔 유지. 새 옵션을 추가할 때는 위의 `*_input_type` Literal에 새 값을 추가하고 기본값(`none` / 기존 동작)을 유지할 것.
- New experiment = new YAML in `config/experiment/custom/`. New denoiser variant = new file in `src/model/denoiser/` + register via `get_denoiser`.
- Wrapper routing은 `src/main.py` 의 `denoiser.name` 분기를 통해 결정됨 — 새로운 denoiser 패밀리를 도입하면 이 분기를 함께 갱신.

## Specifics

- 코드 구조 및 기타 프로젝트 convention들은 `SPECS.md`를 이용한다.

## Changelog

이 프로젝트는 [Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

코드를 변경할 때마다 `CHANGELOG.md`의 `[Unreleased]` 섹션에 항목을 추가할 것.

## 작업 흐름

코드 변경 작업이 끝나면 **반드시** 다음을 수행한다:
1. 변경 내용을 `CHANGELOG.md`의 `[Unreleased]`에 기록
2. 사용자에게 어느 카테고리에 추가했는지 알려줄 것

이 단계를 건너뛰지 말 것. 사소한 변경이라도 사용자에게 영향이 있으면 기록한다.

학습 진행 시 다음 과정을 따른다:
1. 학습 설정이 이전과 어떻게 달라졌는지 사용자의 지시에 맞게 변경되었는지 이상이 생길만한 부분은 있는지 확인하여 이상이 없을 시 wandb run name과 함께 `EXPERIMENTS.log`에 기록한다.
2. 먼저 smoke test를 진행하여 학습 코드가 정상적으로 돌아가는지 확인 후 이상이 있다면 고친 후 `FIX.log`에 기록 후 사용자에게 보고한다.
3. 사용자의 명시적 지시가 없으면 GPU 0~3 중 가장 여유가 있는 GPU만을 사용한다.
4. screen train1~4에서 돌리고 어떤 screen에 어떤 학습이 돌아가고 있는지 기억한다.
5. 학습을 돌려놓고 모니터링을 걸어 주기적으로 확인하여 정상적으로 학습이 돌아가고 있는지 확인하고 이상이 있다면 고친 후 `FIX.log`에 기록 후 사용자에게 보고한다.

추론 진행 시 다음 과정을 따른다:
1. 모델을 학습했을 때의 config와 일치하는지 먼저 확인하고 다른 설정값이 사용자의 명시적 지시에 의한 것이 아니라면 학습했을 때의 설정을 따르고 사용자에게 고지해준다.
2. 사용자의 명시적 지시가 없으면 GPU 0~3 중 가장 여유가 있는 GPU만을 사용한다.
3. screen infer1~4에서 돌리고 어떤 screen에서 어떤 추론이 돌아가고 있는지 기억한다.
4. 결과물은 results에 저장하고 추론 당시의 config값들과 input을 같이 결과 폴더 안에 정리하여 저장한다.

추론 결과물을 후처리 시 다음 과정을 따른다:
1. `scripts`의 기존의 후처리 코드에 사용자가 지시한 역할을 하는 코드가 있는지 찾고 있다면 최소한의 수정으로 고쳐서 사용한다.
2. 해당 역할의 코드가 없다면 새로 코드를 scripts에 작성하여 실행한다.

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