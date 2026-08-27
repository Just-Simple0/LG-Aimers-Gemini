# 요약: 태뷸러 GBM + Brier 보정 신규 기법 리서치

조사일: 2026-08-24 | 세션: `tabular_calibration_methods_20260824_132220`
전체 3부: `part1_gbm_calibration_techniques.md`, `part2_stacking_hpo_lookup.md`,
`part3_nested_cv_regression_prevention.md`

## 가장 중요한 3가지 발견

1. **Venn-Abers 보정이 isotonic보다 나을 수 있다는 2026-01 최신 벤치마크** — CatBoost/XGBoost/
   LightGBM 포함 21개 분류기 비교(arXiv:2601.19944)에서 "isotonic/Platt 같은 흔한 보정법이
   **강한 현대 태뷸러 모델에서는 오히려 성능을 악화시킬 수 있다**"고 경고. 기존 하네스 그대로
   isotonic 대신 Venn-Abers(`venn-abers` PyPI 패키지)로 교체하는 ablation은 구현 비용이 낮고
   시도할 가치가 큼.

2. **Adversarial validation이 오늘 못 푼 미스터리(시즌워크로드 로컬+14.61/실측-11.03)를
   진단할 다음 단계일 수 있다** — 2019~2023 vs 2024로 분류기를 학습해 시즌워크로드 관련 피처가
   유난히 분리력이 높은지 확인하면, 우리가 "가설1(리크)/가설2(콜드스타트 분포차)"로 남겨뒀던
   질문에 직접 답을 줄 수 있음.

3. **우리가 스스로 재발견한 "OOF-only 선택, 홀드아웃 평가전용" 절차는 표준 명칭은 없지만
   nested cross-validation의 "outer fold 1개" 특수사례로 정확히 자리매김된다** — 이론적으로
   정당하지만, Dwork et al.의 "reusable holdout" 이론에 따르면 이 홀드아웃은 v1~v9 내내 반복
   확인되며 이미 유효성이 상당히 소모됐을 수 있음. 업그레이드하려면 롤링-오리진(다중 시즌)
   outer fold가 정석.

## 구조적 후보 3가지에 대한 조사 결론

| 후보 | 결론 |
|---|---|
| 스태킹 | 2개 확률컬럼만 쓰는 **로지스틱회귀 메타러너**는 안전(블렌드의 일반화판). GBM 메타러너는 보정용 OOF와 별도의 독립 KFold가 필요(nested CV 원칙) — 다만 정확히 이 케이스(2-GBM 블렌드→스태킹)의 실전 사례연구는 못 찾음. |
| HPO(Optuna) | v1 이후 한 번도 재탐색 안 했으니 여지는 있어 보이지만(추론), HPO 자체도 검증셋에 과적합될 수 있다는 게 공식화돼 있음(arXiv:2506.19540) — 로컬 신호만으로 믿지 말고 v7 때처럼 실측 재제출로 확인 필요. |
| lookup 세분화(count×pitch_type) | 이분법(추가/기각)이 아니라 **셀 표본수 비례로 상위집계(count-state 단독) 쪽으로 축소**하는 게 표준 해법 — 야구 데이터 자체가 이 축소기법(Efron-Morris)의 고전적 사례. |

## 안 써본 것 중 새로 발견한 것 (우선순위 순)

1. Venn-Abers / beta calibration ablation (isotonic 대체 시도, 낮은 비용)
2. Adversarial validation (시즌워크로드 미스터리 진단 + 향후 모든 신규 피처의 "연도별 안정성"
   사전 체크 도구로 상시 활용 가능)
3. 로지스틱회귀 메타러너 스태킹 (블렌드의 자연스러운 일반화)
4. 롤링-오리진 다중 outer fold (가중치선택 검증 자체를 더 견고하게)
5. HPO는 여지는 있으나 리스크 관리(로컬만 믿지 말 것) 필요 — 우선순위는 위 4개보다 낮음
   (이미 구조가 아니라 파라미터 문제라 상한이 상대적으로 작을 가능성)
6. lookup 세분화는 셀 축소(shrinkage) 설계 없이는 추천 안 함 — 오늘 기각된 reverse_rate/
   middle_trend와 유사한 표본희소성 위험

## 확인 필요(미확정으로 표시된 주장)
Venn-Abers 논문 원문 수치, 베이지안 target encoding의 "선호 방법" 주장, BMA의 실전 태뷸러 사례,
temperature scaling의 GBM 이진분류 이점 — 전부 2차 요약/초록 기반이거나 직접 벤치마크 미확인.
본문 각 파트에 상세 플래그.
