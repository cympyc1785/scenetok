# Extrapolation 비교 프로토콜 (va-wan_dl3dv vs g1020 vs lagernvs_dl3dv_2-6_v_256)

## 목표
세 모델이 **context 밖(extrapolation) target view**를 얼마나 잘 합성하는지 정량 비교.
핵심 산출물 = **"extrapolation 거리(Δ) → 화질 metric" 곡선**. Δ가 커질 때 누가 천천히 무너지는가.

## 확정 결정 (사용자)
- **Context**: native + matched **둘 다** 실행.
- **거리 축**: **frame-index Δ** (context window 밖 프레임 거리). 주축 only.
- **범위**: **소수 scene 정성 우선** → 파일럿(3~5 scene)으로 파이프라인·곡선 검증 후 전체 확장.

## 대상 모델 & native 설정 (confound)
| 모델 | 종류 | 학습 regime | native ctx | focal(Bug2) | camera 정규화 |
|---|---|---|---|---|---|
| va-wan_dl3dv | diffusion (LightningDiT, 25 step) | interpolation | 16 | on | re-center만 |
| g1020 (mvB1_ctx6_extrap) | diffusion (동일 decoder) | extrapolation [10,20] | 6 | on | re-center만 |
| lagernvs_dl3dv_2-6_v_256 | feed-forward (1-pass, VGGT+renderer) | extrapolation (expansion 1.0) | 2–6 (→6) | px K | 1.35×max(cond‖t‖) |

## 공통 held-out scene pool (확정)
- **DL3DV 11K** (lagernvs `dl3dv_6v_11k.json` 498 scene). 498/498 우리 DATA/DL3DV/DL3DV-960/train/11K 에 존재(transforms.json + ~300+ 프레임).
- 11K는 va-wan/g1020(1K smallset 학습)에게 held-out, lagernvs eval pool과 동일 → **세 모델 공통 held-out**.
- 파일럿: 11K에서 프레임 수 넉넉한(N≥120) 3~5 scene 선정.

## 공정성 통제
1. **동일 scene / 동일 target 카메라 / 동일 GT** — target은 실제 held-out 프레임(GT 존재 → PSNR/SSIM/LPIPS/FVD).
2. **동일 해상도** 256×256.
3. **각 모델은 자기 native camera convention으로 입력** (scenetok=re-center+focal-on K / lagernvs=1.35 정규화+px K). 남의 규약 강제 금지(부당 OOD 방지).
4. viser 편집 move는 GT 없어 정량 제외(정성 grid 보조만).

## Context 구성 (native + matched)
scene당 중앙 **context window [a,b]** 고정(Δ 비교 가능하도록 세 모델 공통):
- **native**: va-wan = [a,b]에 16 frame spread, g1020/lagernvs = [a,b]에 6 frame spread.
- **matched**: 세 모델 모두 동일 6 frame (window·프레임 동일).
- Δ = target 프레임의 window edge(a 또는 b)로부터 부호 있는 거리.

## Target / 거리 bin
- forward `b+Δ`, backward `a−Δ`, **Δ ∈ {5,10,20,30,40,60}** (GT 있는 데까지).
- interpolation baseline: window 내부 프레임 몇 개(Δ≤0).
- 각 target에 Δ metadata 저장.

## Metric (Δ-bin 별)
- per-frame full-ref: **PSNR↑ / SSIM↑ / LPIPS↓** (vs GT).
- per-clip 분포: **FVD↓** (연속 target).
- diffusion 2종: stochastic → **seed 3개 평균+std**. lagernvs: deterministic.
- 리포트: ① 표(model × Δbin × metric) ② **곡선(metric vs Δ, 3 line)** — 기울기/AUC = extrapolation robustness ③ 정성 grid.

## 구현 설계 — 통합 harness (cross-pipeline frame 정렬 회피)
두 codebase의 frame ordering/subsampling이 달라 eval-index를 각자 돌리면 "동일 물리 프레임" 보장이 깨진다.
→ **우리 DL3DV 로더에서 scene을 한 번 로드**해 context RGB + target 카메라 + GT를 확보하고, 세 모델 모두 **같은 batch**로 구동:
- va-wan / g1020: `generate_batch_with_scene` (diffusion), 우리 batch.
- lagernvs: context RGB + target c2w를 export → lagernvs conda env subprocess(원본 VGGT+renderer, 내부에서 1.35 정규화). 원본은 RGB 입력이라 scene token 경유 아님.
- GT: 같은 batch의 target 프레임.

## 구현 파일 (예정)
1. `scripts/eval_extrap/build_extrap_index.py` — 11K에서 파일럿 scene 선정 + window/Δ target 산출 + 우리-format eval index(native 16 / native 6 / matched 6) + Δ metadata.
2. `scripts/eval_extrap/run_scenetok_extrap.py` — va-wan/g1020 per-target 예측·metric (우리 batch 기반, seed 3).
3. `scripts/eval_extrap/run_lagernvs_extrap.py` — 동일 context RGB + target c2w를 lagernvs subprocess로 렌더·metric.
4. `scripts/eval_extrap/aggregate_extrap.py` — Δ-bin 집계 → 표 + 곡선 plot + 정성 grid.

## 문서화할 confound
ctx수(16/6/2-6), 정규화(1.35 유무), focal, diffusion vs feed-forward, 학습 regime(interp vs extrap).
native 프로토콜은 이들이 model과 교란 → matched-context(모두 6)로 부분 분리.

## 파일럿 → 전체 확장
파일럿(3~5 scene)에서 곡선·정성 확인 후 498 scene(또는 subset)으로 확장.
