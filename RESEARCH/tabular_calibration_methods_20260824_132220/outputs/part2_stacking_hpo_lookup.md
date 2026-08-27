# 2부: 스태킹 안전성 / HPO 실효성·과적합 위험 / lookup 세분화 표본희소성

조사일: 2026-08-24

## 1. 스태킹 안전성 (calibration-sensitive 이진분류)

**같은 KFold OOF를 보정과 메타러너 학습에 재사용해도 되는가?** 최소 기준(OOF 재사용, in-sample 금지)은
scikit-learn `StackingClassifier` 공식 문서가 명시: 베이스모델이 전체 데이터로 학습한 예측값을 메타러너에
그대로 넣으면 "매우 높은 과적합 위험"이 있다고 경고 — 우리 파이프라인은 `kfold_oof()`로 이 최소 기준은
이미 충족.
— scikit-learn `StackingClassifier` 공식 문서 (Grade B, 공식 문서)

다만 **2차 위험**이 있다: 같은 KFold split을 보정과 메타러너 둘 다에 재사용하면, CLAUDE.md에 이미 기록된
`asof_*` 시간누적 피처의 KFold 순환성(이 프로젝트가 issue #4/v7에서 다룬 문제)이 메타러너에도 그대로
전이된다. 2개 확률 컬럼만 쓰는 로지스틱회귀 메타러너는 과적합 위험이 낮지만(저용량 모델), GBM
메타러너는 위험이 커짐.
— Stacking Ensemble Practical Guide, mcpanalytics.ai (Grade D, 미확인 저자)

**미확정**: "2-GBM 블렌드 -> 스태킹, Brier 채점" 정확히 이 케이스의 Kaggle/Dacon 사례 연구는 이번 조사에서
찾지 못함 — 위 권고는 일반 스태킹 원칙에서의 추론이지 직접 매칭된 선례가 아님.

**결론**: 2개 확률 컬럼만 쓰는 로지스틱회귀 메타러너는 안전하고 실질적으로 "블렌드의 일반화판"(절편도
학습 가능). GBM 메타러너를 쓰려면 보정용 OOF와 별도의 독립 KFold split을 메타러너 학습에 써야 함(nested
CV 원칙).

## 2. Optuna/HPO 실효성과 과적합 위험

**기대 효과**: Mario Filho(Kaggle Grandmaster 자칭), Forecastegy 블로그(2023)는 "피처 엔지니어링
라운드마다 재튜닝은 이득이 작고 과적합 위험만 커진다"고 경고하면서도, 애초에 한번도 안 튜닝했다면
이득의 여지가 크다고 시사 — 우리 프로젝트는 v1 이후 9라운드 넘게 피처가 바뀌었는데 파라미터는 그대로라
이 경고의 "재튜닝 무익" 조건에 해당하지 않고 오히려 "여지가 큰" 쪽에 가까움.
— Filho, "How To Use Optuna to Tune LightGBM Hyperparameters," Forecastegy, 2023-04-07 (Grade D, 블로그)

**미확정**: "HPO 단독으로 얻는 정량적 이득"을 뒷받침하는 엄밀한 벤치마크는 이번 조사에서 못 찾음 — 커뮤니티
통설 수준.

**과적합 위험은 확립된 문헌으로 뒷받침됨** — 이게 핵심 발견:
- Schneider, Bischl & Feurer, "Overtuning in Hyperparameter Optimization," arXiv:2506.19540 (2025-06) —
  HPO 자체가 검증셋에 과적합될 수 있음을 공식화. 작은 검증셋 + 큰 탐색예산 + 노이즈 있는 성능추정
  + 고차원 탐색공간에서 특히 심함. (Grade D, 프리프린트)
- Cawley & Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance
  Evaluation," JMLR (2010) — 같은 CV 루프를 모델선택과 성능추정 둘 다에 쓰면 낙관 편향이 생기고 탐색이
  강할수록 커짐. 표준 해법은 **nested CV**(내부루프=튜닝, 외부루프=보고용 점수) — 이게 바로 우리
  블렌드가중치 grid search가 원래 결여했던 보호장치. (Grade B, 동료심사)

**우리 프로젝트에 대한 시사점**: 이미 이 정확한 유형의 문제로 두 번 데었다(v4-v6 expanding-window OOF
역전, 시즌워크로드 로컬+14.61/실측-11.03). Optuna를 2024 홀드아웃(또는 OOF)만으로 검증하면 같은 위험
프로필을 갖는다 — 로컬 신호만으로 믿지 말고 v7 때처럼 실측 재제출로 확인해야 함.

## 3. Lookup 세분화(count-state × pitch_type_group) 표본희소성

문헌에 고정된 "셀당 최소 N" 기준은 없고, 대신 **연속적 축소(shrinkage)**가 표준 해법.

- Micci-Barreca(2001) 방식(scikit-learn `TargetEncoder` 문서가 인용) — 셀 크기에 비례해 카테고리 평균을
  전역평균 쪽으로 축소. `smooth="auto"`가 경험적 베이즈 분산추정으로 자동 조정. (Grade B, 공식 문서 인용)
- Efron & Morris(1975)의 James-Stein/경험적 베이즈 축소 원조 사례가 바로 **야구 타율**(18명 선수, 첫
  45타석으로 시즌 나머지 예측) — 야구 데이터는 이 기법의 "고전적 테스트베드"로 여러 소스에서 언급됨.
  (Grade C, 2차 요약)
- **직접 관련**: Ludwig/Brill/Wyner(2025) xCTRL 프리프린트가 카운트별 위치분포를 카운트-무관 분포로
  축소하는 정확히 같은 패턴(세분화 키 + 상위 집계로의 축소)을 씀 — 다만 이번 조사에서 정확한 축소
  공식/가중치는 확인 못함(도구 중단). (Grade D, 프리프린트, 메커니즘 세부는 미확인)

**결론**: count-state×pitch_type_group을 "추가/기각"의 이분법으로 보지 말고, 기존 `shrink()` k=30
패턴처럼 **세분화된 셀 평균을 count-state 단독 평균(상위 집계) 쪽으로 셀 표본수 비례 축소**하는 방식을
쓰면 표본희소성 문제를 구조적으로 우회할 수 있음(`shrunk_rate = (n_cell*cell_rate + k*parent_rate)/(n_cell+k)`).

## 4. 선형 블렌드 대안

- **로짓(logit) 공간에서의 가중평균**이 "Kaggle Ensembling Guide"(Triskelion/MLWave)에서 확률공간 평균보다
  낫다고 보고된 사례 있음 — 스태킹보다 훨씬 작은 변경으로 시도 가능. (Grade D, 커뮤니티 가이드)
- 기하평균/순위평균은 문서화돼 있지만 **Brier score 맥락에서는 주의 필요**: 기하평균은 그 자체로 보정된
  확률이 아니고, 순위평균은 확률 스케일 자체를 깨뜨림(AUC류 지표에만 적합) — 둘 다 쓰려면 사후에
  isotonic 재보정이 필요해 "무료 업그레이드"가 아님.
- Bayesian Model Averaging은 이번 조사에서 tabular 대회 실전 사례를 못 찾음(미확정).

## 조사 방법 메모
general-purpose 리서치 에이전트 1개에 위임(WebSearch/WebFetch), 세션 한도로 1회 중단 후 재개. 전체 인용은
본문 각주 참고, 별도 claim_ledger 파일은 이번 라운드에서는 생략(경량 진행).
