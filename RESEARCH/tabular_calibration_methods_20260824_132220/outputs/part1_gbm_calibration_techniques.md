# 1부: GBM 태뷸러 이진분류 + Brier-score 보정 기법

조사일: 2026-08-24

## 1. Pseudo-labeling
Santander Kaggle 1위 솔루션(2019, "Wizardry")과 NVIDIA 기술블로그("Kaggle Grandmasters
Playbook", Onodera/Viel/Titericz/Deotte, 2025-09-18)가 핵심 규율 제시: (a) 단일모델이 아닌
**앙상블 예측**으로 pseudo-label 생성, (b) hard label 대신 **soft label(확률)**을 정규화 용도로,
(c) K-fold 안에서는 **fold별로 별도 pseudo-label 세트**를 만들어 검증폴드가 자기자신으로 학습된
모델의 라벨을 보지 않게 해야 함.

**함정(잘 뒷받침됨)**: confirmation bias(Arazo et al. arXiv:1908.02983), 그리고 결정적으로
**보정 왜곡** — Wei et al. "Do not trust what you trust" (arXiv:2403.15567, 2024): SSL/pseudo-label
방식은 정확도가 올라가도 **보정은 체계적으로 나빠짐**(confidence 기반 필터링이 과신 예측을
선호하기 때문) — Brier score로 채점하는 우리 대회엔 직접적 위험. 쓴다면 재학습 후 보정기를
반드시 재적합해야 함.

**적용 시 주의**: 대회 규칙(test.csv 행 간 집계·타겟인코딩 금지)이 개별 행 단위 pseudo-labeling
자체를 명시적으로 막진 않지만, test.csv의 피처 분포를 재학습에 쓰는 셈이라 규칙 해석이 필요 —
순수 ML 판단이 아니라 대회 규칙 재검토 필요 항목으로 플래그.

## 2. 고급 target encoding
- **Micci-Barreca류 스무딩**: Max Halford 블로그(2018) — `μ=(n·x̄+m·w)/(n+m)`, `m≈300` 권장,
  KFold target encoding보다 **단일 스무딩 파라미터 방식을 선호** — 소표본 카테고리를 그대로
  평균내면 과신 추정이 나와 Brier score를 직접 악화시킨다고 지적.
- **CatBoost Ordered TS + Ordered Boosting**: Prokhorenkova et al. NeurIPS 2018(arXiv:1706.09516) —
  이미 우리가 쓰는 CatBoost에 내장. 다만 KFold로 만든 target encoding을 CatBoost에 "미리 계산해서"
  넣으면 CatBoost 자체의 ordered-TS 로직과 중복/충돌할 수 있어 확인 필요.
- **베이지안 target encoding**: Slakey/Salas/Schamroth류(arXiv:2006.01317) — 사후분포 샘플링이
  암묵적 정규화 효과. "선호되는 방법"이라는 주장은 종합적 추론 수준(직접 벤치마크 아님, 미확정).

**우리 프로젝트 제약**: 이 스킴들 전부 test.csv 자기집계 금지 규칙 때문에 pitcher_id/batter_id
등에 test.csv 기준으로는 적용 불가 — train.csv/trackman_history 소스로만 가능(기존 asof_*
cold-start 스무딩과 같은 패턴).

## 3. Adversarial Validation — 이번 조사에서 가장 실전 적용 가치가 큰 항목

Zak Jost 블로그(2020): 표준 워크플로 — (1) train-vs-test 분류기 학습, (2) **피처 중요도**로
어떤 피처가 분포 분리를 유발하는지 확인, (3) 그 피처를 제거/재설계, (4) 재학습해 AUC가 0.5로
내려가는지 확인. "진단만 해주지 고치는 법을 알려주진 않는다"는 본인 명시 한계.

**우리 프로젝트에 대한 직접적 시사점**: 이 기법이 오늘 우리가 못 푼 미스터리(시즌워크로드 피처
로컬 +14.61 vs 실측 -11.03)를 **직접 진단하는 다음 단계**가 될 수 있다 — 2019~2023을 "train",
2024를 "test"로 놓고 어드버서리얼 분류기를 돌려 시즌워크로드 관련 피처들이 유난히 분리력이
높은지(=연도별로 불안정한 아티팩트인지) 확인 가능. (Zak Jost, blog.zakjost.com, 2020-03-31;
Anil Ozturk, Medium/Kaggle Blog — reweighting까지 확장한 "exploiter" 버전도 소개)

## 4. 확률보정 기법 비교 — 두 번째로 중요한 발견

- **Isotonic**: scikit-learn 공식 문서 — "표본이 ~1000개 넘으면 sigmoid만큼 좋거나 더 좋음."
  우리는 KFold OOF가 수십만 행이라 순수 표본수 관점에서는 과적합 위험이 낮음.
- **Platt scaling**: Niculescu-Mizil & Caruana(ICML 2005, arXiv:1207.1403) — **부스팅 트리
  전용 확률보정의 원조 근거**: 부스팅은 확률을 극단(0/1)으로 밀어붙이는 특유의 왜곡이 있어서
  보정이 필요하다는 원 논거.
- **Beta calibration**: Kull/Filho/Flach(AISTATS 2017) — Platt을 일반화한 베타분포 기반
  3파라미터 모델. 이미 잘 보정된 분류기에는 항등함수에 가깝게 수렴(isotonic처럼 노이즈에
  과적합하는 실패 없이 우아하게 축소). **다만 GBM에 직접 테스트한 원논문 근거는 못 찾음(간극)**.
- **Venn-Abers predictors — 가장 신선하고 직접적인 발견**: Manokhin & Grønhaug,
  "Classifier Calibration at Scale" (arXiv:2601.19944, 2026-01-19) — CatBoost/XGBoost/LightGBM
  포함 21개 분류기를 TabArena-v0.1 태뷸러 벤치마크에서 isotonic/Platt/beta/Venn-Abers/Pearsonify로
  비교, **log-loss/Brier score 기준**. 핵심 발견: **Venn-Abers가 평균 log-loss 감소폭 최대, beta가
  근소한 2위, Platt이 가장 약함**. 더 중요하게: **"isotonic/Platt 같은 흔한 보정법이 강한 현대
  태뷸러 모델(=우리 LGB+CB급)에서는 오히려 proper scoring을 체계적으로 악화시킬 수 있다"**는
  경고 — 즉 지금 쓰는 isotonic이 "이미 강한 모델"에는 최선이 아닐 수 있음을 시사하는 최신(2026-01)
  근거. Venn-Abers는 클래스0/클래스1 가정 각각에 대해 isotonic을 2번 적합해 결합하는 방식이라
  단일 isotonic적합의 과적합 실패모드를 구조적으로 완화. **주의**: 논문 원문은 세션 한도로
  전체 확인 못함(초록/2차 요약 기반, 수치는 검증 필요).
- **Temperature scaling**: 원래 NN용, GBM 이진분류에는 Platt의 부분집합 수준이라 별도 이점
  확인 못함(미확정).

**결론**: 표본수가 커서 isotonic의 "소표본 과적합" 걱정은 낮지만, 2026-01 벤치마크는 "강한
모델엔 isotonic/Platt이 최선이 아닐 수 있다"는 다른 각도의 위험을 제기 — 기존 2024 홀드아웃
하네스 그대로 Venn-Abers/beta calibration을 isotonic 대신 끼워넣는 ablation은 낮은 비용으로
시도해볼 가치가 있음(`venn-abers` PyPI 패키지 존재).

## 5. 기타
- **스태킹이 보정을 오히려 날카롭게(overconfident) 만들 위험**: "Ensembling Tabular Foundation
  Models" (arXiv:2605.18696) — 선형 스태킹은 베이스모델보다 더 날카로운(overconfident) 확률을
  뱉어 보정을 악화시킬 수 있음. 대안으로 "Temperature-Scaled Blending"(모델별 온도파라미터를
  검증셋에서 개별 적합 후 블렌드) 제안 — 이건 기존 calibrate_then_blend와 비슷하지만 isotonic
  대신 온도 스칼라를 쓰는 버전.
- **CatBoost의 prediction-shift 방지가 우리 블렌드가중치 OOF에도 일관되게 적용됐는지 재확인
  권장** — 보정 OOF는 TK_LOOKUP_VAL/FULL 시즌분리를 지키지만, 블렌드가중치 격자탐색용 OOF도
  같은 리크 방지 규율을 지켰는지 별도 확인할 가치.

## 조사 방법 메모
general-purpose 리서치 에이전트 1개 위임, 세션 한도로 1회 중단 후 재개. 원문 확보 실패 소스는
인용에서 제외(SSL/403 에러 2건).
