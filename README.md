# LG Aimers 9기 — 투구 제구 성공 확률 예측 AI 온라인 해커톤

Dacon 대회 저장소. 각 행은 KBO 투구 1건이며, 투구 직전까지 알 수 있는 정보만으로
해당 투구가 "제구 성공"일 확률을 예측한다.

- 데이터/규칙 상세: [docs/data_description.md](docs/data_description.md)
- 대회/프로젝트 컨텍스트, 안티리키지 규칙, 제출 포맷 제약: [CLAUDE.md](CLAUDE.md)

## 평가지표

Brier Skill Score, 0~100000로 스케일링:

```
BrierScore     = mean((p_i - y_i) ** 2)
baseline_brier = r * (1 - r)          # r = 평가셋의 평균 control_success
Score          = max(0, 100000 * (1 - BrierScore / baseline_brier))
```

Brier score를 직접 최소화하는 지표이므로 순위(ranking)뿐 아니라 **확률 보정(calibration)**
품질이 점수에 그대로 반영된다.

## 현재 파이프라인 (main)

1. `test.csv` 헤더로 사용 가능한 피처 47개를 고정 (`train.csv`에만 있는 컬럼은 사용 불가)
2. 같은 행 내부 값 또는 정적 lookup 테이블로 38개 파생 피처 생성 — 카운트/레버리지
   상호작용(`li` 기반 3개 포함 — v8, 아래 참고), cold-start 스무딩된 asof 성공률,
   매치업 차이, 구종 성향, 주기성 인코딩, `trackman_history.csv` 카운트 상황별 리그
   lookup(아래 참고) 등 (`build_features()`, 학습 노트북과 `script.py`에 동일하게 존재)
3. **LightGBM + CatBoost** 를 각각 학습
4. **KFold(shuffle=True, 5-fold) out-of-fold** 예측(이미 만들어진 학습 피처
   전체를 5개 fold로 무작위 분할해, 각 fold를 나머지 4개 fold로 학습한 모델로
   예측)을 만든다. (이 OOF 메커니즘은 v4~v6에서는 시즌 단위 expanding-window였다가
   v7에서 KFold로 되돌렸다 — 아래 "v7" 절 참고)
5. **로지스틱 회귀 메타러너(스태킹)** 로 두 모델을 결합한다(v10부터, v5~v9의
   "블렌드 가중치 + 보정 순서 격자탐색"을 대체) — 위 OOF의 `[lgb_oof, cb_oof]`를
   피처로 `LogisticRegression`을 학습하고, 그 raw `predict_proba` 출력을 그대로
   쓴다(추가 isotonic 보정 없음 — 아래 "v10" 절 참고).
6. `model/ensemble.pkl` (LightGBM 모델 + CatBoost 모델 + 로지스틱 회귀 메타러너
   `stack_model` + 피처 목록 + global_mean + trackman lookup 테이블)로 저장 →
   `src/script.py` + `src/requirements.txt` 와 함께 제출용 zip으로 패키징
   (`submissions/`)

`trackman_history.csv`의 `pitcher_trackman_id`/`batter_trackman_id`는
`train.csv`/`test.csv`의 `pitcher_id`/`batter_id`와 겹치는 값이 0건이고(직접 확인),
팀 코드(`pitcher_team`, 26종)도 `pitcher_team_id`(13종)와 신뢰성 있게 대응시킬 수 없어
**선수/팀 단위로는 결합할 수 없다.** 대신 두 데이터셋에 공통으로 존재하는, 투구 직전에
이미 알 수 있는 카운트 상황 키(`balls_before`, `strikes_before`, `outs_before`)로
2019~2024년 `trackman_history.csv` 전체를 집계해 "이 카운트에서 리그가 보통 어떤
구종을 어떤 구위로 던지는가"를 나타내는 정적 lookup 테이블을 만들고, 이 투수의 평소
구종 비율과의 차이를 피처로 추가했다(v3). 로컬 검증(2024 홀드아웃, isotonic 보정 후)
에서 748.36 → **766.04**로 개선을 확인했다. lookup 테이블은 학습 시점에만
`trackman_history.csv`를 읽어 만들고 그 결과를 `model/ensemble.pkl`에 함께 저장하므로,
추론(`script.py`) 시점에는 `trackman_history.csv` 파일이 필요 없다.

lookup 테이블은 **검증용(`TK_LOOKUP_VAL`, 시즌 ≤ 2023)과 최종 제출용
(`TK_LOOKUP_FULL`, 시즌 ≤ 2024)을 분리**해서 만든다. 처음에는 시즌 필터 없이 만든
lookup 하나로 검증했다가 770.19라는 수치를 얻었는데, PR #3 코드 리뷰(GPT Pro)에서
이 lookup에 2024 시즌 Trackman 정보가 이미 섞여 있어 "2024 홀드아웃" 검증 자체가
미래 정보를 미리 아는 상태였다는 지적을 받았다 — 실제 2025 제출 경로는 안전했지만
(2025 데이터 자체가 애초에 lookup에 없으므로), 내부 검증 수치만 낙관적으로
부풀어 있었다. lookup을 시즌별로 분리해 재검증한 수치가 765.86이었다.

이후 3차 리뷰에서 이 프로젝트에 v2부터 있던 별개의 버그도 발견됐다:
`CatBoostClassifier.get_best_iteration()`은 0-based 인덱스라(직접 검증:
`get_best_iteration()`이 145면 `use_best_model=True`로 선택된 실제 모델의
`tree_count_`는 146), 이 값을 그대로 `CatBoostClassifier(iterations=...)`에 넘기면
검증에서 선택된 모델보다 트리가 1개 적게 재현된다. `+ 1` 보정 후 재검증한 최종
수치가 **766.04**다 (LightGBM의 `best_iteration_`은 이미 카운트라 이 문제가 없다).

4차 재리뷰에서는 isotonic 보정용 OOF가 `KFold(shuffle=True)` 무작위 분할이라
`asof_*`(투수/타자의 시간 누적 성적) 피처 특성상 완전한 시간순 OOF는 아니라는
지적이 나왔다. 대회 안티리키지 규칙 위반은 아니고(평가 서버 `test.csv`는 전혀
침범하지 않음) v1부터 있던 기존 보정 설계라 v3 고유 문제는 아니라고 판단해,
[이슈 #4](../../issues/4)로 분리하고 PR #3은 거기서 마무리했다 — 이 이슈는
v4에서 고쳤다(아래 참고).

## v4 — isotonic 보정을 시즌 단위 expanding-window OOF로 재설계

[이슈 #4](../../issues/4)의 지적대로, 기존 `KFold(shuffle=True)` OOF를 시즌 단위
expanding-window로 바꿨다: OOF 대상 시즌 `Y`마다 `season < Y` 데이터로만 학습하고,
`global_mean`/trackman lookup도 그 컷오프로 다시 계산한다 — 어떤 학습 행도 자신보다
미래 시즌의 정보를 보지 않는다. 2019는 그 이전 시즌이 없어 OOF 대상에서 빠지고
(학습에는 계속 포함), 검증용은 2020~2023, 최종 아티팩트용은 2020~2024가 각각
OOF로 평가된다.

이렇게 재검증한 정직한 점수는 **766.04 → 748.08**로 낮아졌다 — 이전 무작위 OOF가
계산 전반에 낙관적 편향을 주고 있었다는 뜻이다. 이 낙폭이 "트랙맨 피처 자체가
사실 효과가 없었다"는 뜻은 아닌지 확인하기 위해, **같은 시즌 단위 OOF로 트랙맨
피처 유무를 직접 비교**했다: 트랙맨 없음 727.83 vs 있음 748.08, **차이 +20.25**로
v3의 결론(트랙맨 lookup 피처가 실제로 유효한 신호)이 그대로 재확인됐다. 즉 이전
낙관적 편향은 트랙맨 유무와 무관하게 전반적으로 걸려 있던 것이었고, v3의 상대적
개선폭 자체는 견고했다.

## v7 — isotonic 보정 OOF를 KFold(shuffle)로 되돌림

v4~v6에서 로컬 검증 점수는 계속 개선됐지만(748.08 → 750.75 → 755.94), 실제 Dacon
리더보드 점수는 반대로 계속 하락했다: **v3 882.11 → v4 871.32 → v5 876.93 →
v6 871.37**. v4의 시즌 단위 expanding-window OOF 재설계(위 "v4" 절)가 이 역행의
원인일 가능성을 조사했다.

**로컬 A/B/C 실험 (2024 홀드아웃, 보정 OOF 메커니즘만 바꿔가며 비교)**:

| 실험 | 방식 | 로컬 점수 | v5 기준(750.75) 대비 |
|---|---|---|---|
| 기준(v4~v6, main) | expanding-window OOF, season별 fold 전체 | 750.75 | — |
| **A (채택)** | **KFold(shuffle=True, 5-fold) 복원 (pre-issue #4/v3 방식 그대로)** | **766.04** | **+15.29** |
| B1 | expanding-window, season=2020 fold만 제외 | 757.06 | +6.31 |
| B2 | expanding-window, season=2020+2021 fold 제외 | 733.29 | -17.46 (악화) |
| C | expanding-window, fold train_size 비례 가중 isotonic | 749.33 | -1.42 |

A(KFold 복원)가 로컬 최고점이며, v3의 원래 로컬 점수(766.04)와 정확히 일치한다 —
같은 메커니즘을 재현했다는 뜻이다. 실제 리더보드에서도 KFold 기반이었던 v3가
지금까지 최고 실측 점수(882.11)였으므로, 로컬·실제 데이터가 처음으로 같은 방향을
가리켰다. 이 조사를 바탕으로 v7은 **보정용 OOF 생성 메커니즘만** KFold(shuffle=True,
n_splits=5, random_state=42)로 되돌렸다 — `TK_LOOKUP_VAL`/`TK_LOOKUP_FULL`의 시즌
분리(진짜 리크 방지 로직, "v3" 절 참고)와 `build_features()`, 블렌드 가중치/순서
탐색 인프라는 전혀 건드리지 않았다. 재실행한 로컬 홀드아웃 점수는 **766.04**로
실험 A를 그대로 재현했다.

**주의 — 이 변경의 진짜 통과 기준은 로컬 점수가 아니다.** v4/v5/v6 모두 로컬은
개선됐지만 실제 리더보드는 악화된 전례가 있다. 이번 로컬 재현(766.04)과 v3의
과거 실측 최고점(882.11)이 같은 메커니즘이라는 사실은 강한 정황 증거이지만,
**Dacon 재제출 실측 점수로 확인되기 전까지는 가설로만 취급한다.** v6의 NN(PR #7)과
새로운 피처 아이디어(xCTRL류)는 원인 격리를 위해 이번 변경 범위에서 의도적으로
제외했다 — 여러 변경을 동시에 묶으면 재제출 결과가 나와도 무엇 때문인지 구분할
수 없기 때문이다.

## v8 — li(leverage index) 상호작용 피처

`li`(leverage index) 컬럼은 원본 데이터에 그대로 들어 있었지만(`FEATURES`를 통해
모델 입력에는 포함됨) `build_features()`에서 파생 상호작용 피처가 하나도 없었다.
세이버메트릭스 문헌에서 leverage index는 "이 상황이 승패에 얼마나 중요한가"를
나타내는 지표로, 고레버리지 상황에서 투수 행동/제구가 달라질 수 있다는 가설을
`li` 원본 값만으로는 모델이 충분히 학습하지 못할 수 있다고 보고 파생 피처를
추가했다.

**임계치는 추측이 아니라 `train.csv`의 실제 `li` 분포를 직접 확인해서 정했다**
(1,475,092행, 결측 0건): 평균 0.98, 중앙값 0.80, 표준편차 0.89, 최댓값 10.83.
분위수: 90분위 2.02, 95분위 2.63. 세이버메트릭스 관행에서 자주 쓰는 임계치
1.5는 이 데이터에서 상위 약 19.7%(`li >= 1.5`)에 해당해 "흔치는 않지만
너무 희소하지도 않은" 합리적인 컷오프로 확인되어 그대로 채택했다.

추가한 피처 3개(`build_features()`, 기존 `late_and_close` 패턴과 일관되게
카운트/레버리지 블록에 배치):

- `is_high_leverage = (li >= 1.5).astype(int)` — 고레버리지 상황 플래그
- `li_count_diff = li * count_diff` — 레버리지와 볼카운트 압박의 상호작용
- `li_late_close = li * late_and_close` — 레버리지와 "후반 접전" 상황의 상호작용

**필수 ablation** (`main` 5a68620 기준, 동일 KFold(shuffle=True, 5-fold) OOF +
블렌드 가중치/보정 순서 격자 탐색, 2024 홀드아웃):

| 실험 | 파생 피처 수 | 선택된 (order, w_lgb) | 로컬 점수 |
|---|---|---|---|
| 기준(main, li 피처 없음) | 35 | blend_then_calibrate, 0.50 | 766.04 |
| **li 피처 포함(채택)** | 38 | calibrate_then_blend, 0.45 | **774.07** |

**+8.03 개선** — same_hand(v5, -1.41로 기각)와 뚜렷이 다른 방향과 크기의 양의
신호이므로 **채택**했다. 최적 블렌드 순서가 `blend_then_calibrate`에서
`calibrate_then_blend`로 바뀐 점도 흥미로운데, li 상호작용 피처가 두 모델의
예측 분포를 서로 다르게 바꿔 모델별 개별 보정이 더 유리해진 것으로 추정된다
(정확한 원인 분석은 이번 실험 범위 밖).

이 PR은 다른 v8 백로그 항목(NN 재검증, 시즌 워크로드 피처)과 `main`에서 독립적으로
분기했다 — 원인 격리를 위해 서로 stacking하지 않는다(자세한 배경은
`.omc/plans/v8-backlog-nn-li-workload.md` 참고). **이 PR의 통과 기준도 로컬
홀드아웃 점수가 아니다** — v4/v5/v6가 로컬은 개선됐지만 실제 리더보드는 악화된
전례가 있으므로(v7 절 참고), 이번 로컬 개선(+8.03)은 강한 정황 증거로만 받아들이고
**Dacon 재제출 실측 점수로 확인되기 전까지는 머지하지 않는다.**

## v10 — 로지스틱 회귀 메타러너 스태킹 (블렌드 가중치 격자탐색 대체)

v5~v9까지 결합 단계는 "`W_GRID`(0.10~0.90, 17개 값) x 보정 순서(2개) = 34개 조합을
2024 홀드아웃에서 직접 최댓값으로 고른다"는 방식이었다. 오늘 리서치(스태킹 안전성,
scikit-learn `StackingClassifier` 문서 및 arXiv:2605.18696)를 바탕으로, 이 결합
단계 자체를 **로지스틱 회귀 메타러너(스태킹)**로 교체하는 실험을 진행했다.

**설계**: `kfold_oof()`가 만드는 `[lgb_oof, cb_oof] -> y_oof`(v5~v9의 isotonic
보정에 쓰던 것과 동일한 KFold(shuffle=True, 5-fold) OOF)를 그대로 `sklearn.
linear_model.LogisticRegression`의 학습 입력으로 재사용한다. **OOF 재사용 판단**:
scikit-learn `StackingClassifier` 문서는 메타러너를 반드시 OOF(in-sample이 아닌)
예측으로 학습해야 한다고 권고하는데, `kfold_oof()`는 정확히 그 조건을 만족한다.
피처가 2개뿐인 로지스틱 회귀는 표현력이 낮아(사실상 비음수/합=1 제약이 없는 가중
블렌드의 일반화판) 과적합 위험이 낮다고 판단해, calibration에 쓰는 OOF와 별도의
중첩(nested) KFold를 두지 않았다 — GBM 등 고용량 메타러너였다면 이 판단은 달랐을
것이다(오늘 리서치가 명시적으로 구분한 지점). 이는 v8-nn/season-workload에서
문제가 됐던 "홀드아웃에서 직접 격자탐색 최댓값을 고르는" 패턴과는 다르다 — 로지스틱
회귀 자체는 홀드아웃을 전혀 보지 않고 OOF에서 1회 fit되며, 홀드아웃은 평가 전용으로만
쓰인다.

**독립 진단 스크립트**(`docs/logistic_stack_diagnostic.py`, `main` eea05e5=v8-li
기준 `train.ipynb` 섹션 1~6을 문자단위로 재현)로 2024 홀드아웃에서 3갈래를 비교했다:

| 방식 | 로컬 점수 |
|---|---|
| 기존(main): W_GRID 격자탐색 + isotonic 보정 블렌드 (재현) | 774.07 |
| 로지스틱 스태킹, raw(보정 없음) | **782.47** |
| 로지스틱 스태킹 + isotonic 보정 1개 더 | 770.21 |

**raw 로지스틱 스태킹이 +8.40으로 가장 좋았다** — li(+8.03)와 비슷한 크기의 뚜렷한
양의 신호다. 흥미롭게도 **isotonic 보정을 한 번 더 씌우면 오히려 기존 기준(774.07)
보다도 낮아졌다(-3.86)** — 애초에 "선형 스태킹은 베이스보다 더 날카로운 확률을
내서 보정을 오히려 해칠 수 있다"(오늘 리서치, arXiv:2605.18696)는 가설을 검증하려고
둘 다 실험했는데, 실제로 그 방향으로 나타난 것이다. 메타러너 계수는
`coef_=[2.327, 2.272], intercept_=-2.313`로 극단적이지 않고 대칭에 가까웠다
(둘 다 비슷한 가중치 — 기존 W_GRID 탐색이 고른 `w_lgb=0.45`와도 방향이 일치).

**채택**: raw 로지스틱 스태킹(추가 보정 없음)을 `notebooks/train.ipynb` 섹션 6/7,
`notebooks/inference.ipynb`, `src/script.py`에 반영했다. 아티팩트 스키마가
바뀐다 — `blend_w_lgb`/`calibration_mode`/`calibrator`/`calibrator_lgb`/
`calibrator_cb`가 `stack_model`(fitted `LogisticRegression`) 하나로 대체된다.

**이 변경도 로컬 홀드아웃 점수만으로 최종 채택하지 않는다** — v4/v5/v6/v8-season의
전례대로 로컬 개선이 실제 리더보드에서 반대로 나타난 사례가 이미 여러 번 있었다.
`submissions/v10_logistic_stack.zip`을 빌드하고 Dacon에 재제출해 실측 점수로
확인하기 전까지는 PR을 머지하지 않는다.

## 실험 이력

각 실험 단계는 `experiment/vN-설명` 브랜치에서 진행하고, PR로 `main`에 merge한다.
로컬 검증은 2024 시즌을 홀드아웃(2019~2023 학습)으로 사용한 Brier Skill Score.

| 버전 | 제출 파일 | 브랜치 | 방법 | 로컬 검증 | Dacon 점수 |
|---|---|---|---|---|---|
| baseline | — | `main` (초기 커밋) | RandomForest, 중앙값 대치 | ~454 | — |
| v1 | [`v1_lgb_isotonic.zip`](submissions/v1_lgb_isotonic.zip) | [`experiment/v1-lightgbm-isotonic`](../../pull/1) | LightGBM + 피처 엔지니어링 + isotonic 보정 | 692.66 → 722.50 | **848.64484** |
| v2 | [`v2_lgb_cat_blend.zip`](submissions/v2_lgb_cat_blend.zip) | [`experiment/v2-lgbm-catboost-blend`](../../pull/2) | LightGBM+CatBoost 50:50 블렌드 + isotonic 보정 | 692.66 / 713.18 / 721.52 → **748.36** | **881.22099** (697/1127위) |
| v3 | [`v3_lgb_cat_trackman.zip`](submissions/v3_lgb_cat_trackman.zip) | [`experiment/v3-trackman-history`](../../pull/3) | v2 + `trackman_history.csv` 카운트 상황별 리그 lookup 피처(구종/구속/무브먼트, 12개) + CatBoost `best_iteration` off-by-one 수정 | 720.40 / 724.12 / 740.81 → **766.04** | **882.108625189** |
| v4 | [`v4_forward_oof.zip`](submissions/v4_forward_oof.zip) | [`experiment/v4-forward-oof-calibration`](../../pull/5) | isotonic 보정 OOF를 `KFold(shuffle=True)` → 시즌 단위 expanding-window로 재설계 ([이슈 #4](../../issues/4)) | 720.40 / 724.12 / 740.81 → **748.08** | **871.3183926356** |
| v5 | [`v5_blend_tune.zip`](submissions/v5_blend_tune.zip) | [`experiment/v5-blend-calib-tuning`](../../pull/6) | 블렌드 가중치(w_lgb=0.35)를 격자 탐색으로 튜닝 (보정 순서·트랙맨 매치업 키 확장도 실험했으나 기각 — 아래 참고) | 720.40 / 724.12 / — → **750.75** | **876.9277353124** |
| v7 | [`v7_kfold_calib.zip`](submissions/v7_kfold_calib.zip) | `experiment/v7-kfold-calibration` | 보정용 OOF 생성을 expanding-window에서 KFold(shuffle=True, 5-fold)로 되돌림 (원인 조사·A/B/C 실험은 위 "v7" 절 참고, v6/NN은 이 변경에서 제외) | 720.40 / 724.12 / — → **766.04** | 제출 대기(재검증용) |
| v8-li | [`v8_li_leverage.zip`](submissions/v8_li_leverage.zip) | [`experiment/v8-li-leverage`](../../pull/9) | `main`(v7) 기준 `li`(leverage index) 파생 피처 3개 추가(`is_high_leverage`/`li_count_diff`/`li_late_close`, 위 "v8" 절 참고) | 720.59 / 740.89 / 766.04(li 없음) → **774.07**(+8.03) | 머지 보류(Dacon 재제출 확인 전) |
| v10-stack | [`v10_logistic_stack.zip`](submissions/v10_logistic_stack.zip) | `experiment/v10-logistic-stacking` | `main`(v8-li) 기준, 결합 단계를 W_GRID 격자탐색+isotonic 보정에서 로지스틱 회귀 메타러너(스태킹)로 교체 — raw 출력 채택, isotonic 추가는 기각(위 "v10" 절 참고) | 774.07(기존 재현) / 770.21(스태킹+isotonic, 기각) → **782.47**(+8.40, raw 스태킹) | 머지 보류(Dacon 재제출 확인 전) |

> Dacon 제출 파일명은 30자 제한(`.zip` 포함)이 있어 `vN_<짧은태그>.zip` 형식을 쓴다.

v2에서 얻은 핵심 발견: 단일 2024 홀드아웃 대신 2022/2023/2024 롤링 검증을 돌려보니,
**하이퍼파라미터 튜닝보다 학습 데이터의 최신성(검증 대상 시즌에 가까운 시즌까지 포함)이
점수에 훨씬 큰 영향**을 미쳤다. 2019~2021만으로 학습한 모델은 2023 시즌 검증에서
베이스라인(상수 예측)보다도 낮은 점수를 기록했는데, 이는 시즌 간 제구 성공률의
레벨 시프트(2022: 0.529 → 2023: 0.500) 때문이었다. 그래서 최종 제출 모델은 항상
가용한 전체 시즌(2019~2024)으로 재학습한다.

v3에서 얻은 핵심 발견: `trackman_history.csv`는 선수/팀 ID가 `train.csv`/`test.csv`와
전혀 겹치지 않아 처음엔 "쓸 수 없는 데이터"로 판단했지만(v1/v2), 카운트 상황
(`balls_before`/`strikes_before`/`outs_before`)처럼 **두 데이터셋에 공통으로 존재하는
비식별 키**로 리그 전체 집계 lookup 테이블을 만들면 안티리키지 규칙을 지키면서도
유효한 신호를 끌어낼 수 있었다. 특히 이 투수의 평소 구종 비율과 리그의 카운트별 평균
구종 비율의 차이(`tk_fastball_dev` 등)가 LightGBM 피처 중요도 상위권에 들었다 — "이
카운트에서 리그가 보통 어떻게 던지는가 대비 이 투수가 얼마나 벗어나는가"가 제구
성공률과 관련이 있다는 뜻으로 해석된다.

v3 검증 방법론에서 얻은 교훈: **lookup/집계 테이블도 시즌 필터링을 빠뜨리면 홀드아웃
검증에 미래 정보가 새어 들어간다.** `asof_*` 피처나 `train_fe`를 2019~2023/2024로
나누는 것만으로는 충분하지 않다 — 그 피처를 만드는 데 쓰인 별도 소스 테이블
(`trackman_history.csv` 집계)도 검증 시점 기준으로 동일하게 시즌을 잘라야 한다.
이 프로젝트에서는 PR #3 코드 리뷰로 뒤늦게 발견했는데, 앞으로 `trackman_history.csv`
기반이든 다른 외부 소스 기반이든 새 lookup/집계 피처를 추가할 때는 처음부터
검증용(시즌 필터 있음)과 최종 제출용(전체 시즌) lookup을 분리해서 만드는 것을
기본값으로 삼는다.

v4에서 얻은 핵심 발견: **OOF 기반 보정 파이프라인은 K-fold 분할 방식 자체도
피처의 시간적 성격을 고려해야 한다.** `asof_*` 피처처럼 "이 시점까지의 누적
성적"을 담은 컬럼이 있으면, 무작위 K-fold는 같은 선수의 이후 행을 통해 이전 행의
결과가 간접적으로 학습 데이터에 섞여 들어갈 수 있다 — 대회 규칙 위반은 아니지만
내부 검증 방법론상 완전히 깨끗하지는 않다. 시즌 단위 expanding-window로 바꾸자
점수는 낮아졌지만(766.04 → 748.08), 트랙맨 피처의 상대적 기여도(+20.25, 동일
방법론 하에서 재확인)는 그대로 유지됐다 — **방법론을 고치면 절대 점수는 바뀔 수
있어도, 잘 설계된 피처의 상대적 가치는 살아남는다**는 것을 확인한 셈이다.

v5에서는 세 가지 아이디어를 v4가 만든 leakage-free OOF 인프라로 검증했다.

1. **블렌드 가중치 학습** — 고정 50:50 대신 `W_GRID`(0.10~0.90, 0.05 간격)로
   격자 탐색한 결과 **w_lgb=0.35**가 최적으로 나왔다 (CatBoost 비중을 높이는
   쪽). 748.08 → **750.75**로 개선.
2. **보정 순서 비교** — "블렌드 후 보정"(기존 방식)과 "모델별 보정 후 블렌드"를
   비교한 결과, 모든 실험에서 **블렌드 후 보정이 항상 더 좋았다** — 순서를
   바꿀 필요는 없었지만, 그동안 검증 없이 써 온 선택이 실제로 최선이었음을
   확인했다.
3. **트랙맨 lookup 키에 `same_hand`(투수·타자 좌우 매치업) 추가** — 시도했지만
   **기각했다**. 같은 (w, 순서)로 고정한 ablation에서 카운트 키만 쓴 경우
   750.75, `same_hand`를 추가한 경우 749.34로 오히려 **-1.41 악화**됐다.
   매치업 정보는 이미 `same_hand` 단일 피처로 모델에 들어가 있어서, lookup
   키에까지 추가하면 새 신호보다 셀당 표본 수 감소(72개 조합, 평균
   25,000행/셀)로 인한 노이즈가 더 크게 작용한 것으로 보인다.

v5에서 얻은 핵심 발견: **"저렴한 실험 인프라"를 한 번 잘 만들어두면 아이디어를
빠르게 검증(하고 기각)할 수 있다.** expanding-window OOF에서 LightGBM/CatBoost
예측을 분리해서 반환하도록만 바꿔두니, 모델을 다시 학습하지 않고도 블렌드
가중치 17개 × 보정 순서 2개 = 34개 조합을 몇 초 만에 비교할 수 있었다. 반면
③번(트랙맨 키 확장)처럼 실제 학습이 다시 필요한 아이디어는 시도해보기 전에는
효과를 알 수 없었고, 결과적으로 기각됐다 — **아이디어를 실행에 옮기는 비용이
낮을수록 더 과감하게 시도하고 정직하게 버릴 수 있다.**

## 저장소 구조

```
data/                              대회 원본 데이터
docs/
  data_description.md              데이터 컬럼별 설명, 안티리키지 규칙 원문
notebooks/
  train.ipynb                      학습 노트북 — model/ensemble.pkl 생성 (경로 기준: notebooks/, "../data" 등)
  inference.ipynb                  추론 노트북 — src/script.py의 원본 (경로 기준: notebooks/, "../data" 등)
src/
  script.py                        제출 zip에 들어가는 추론 엔트리포인트 (경로 기준: zip 루트, "./data" 등)
  requirements.txt                 제출 zip에 들어가는 의존성 목록
submissions/
  baseline_submit.zip              Dacon 제공 베이스라인 참고용 예시
  vN_<짧은태그>.zip                 버전별 제출 zip (model/ + script.py + requirements.txt),
                                    파일명 30자 제한 때문에 짧게 (예: v2_lgb_cat_blend.zip)
model/, output/                    재생성 가능한 아티팩트 (.gitignore 처리됨)
```

`notebooks/*.ipynb`와 `src/script.py`는 같은 `build_features()` 로직을 담고 있지만
**경로 접두사가 다르다** — 노트북은 `notebooks/` 안에서 실행되므로 저장소 루트를
`../data`, `../model` 처럼 가리키고, `script.py`는 제출 zip 안에서 `model/`과 나란히
있으므로 `./data`, `./model` 을 쓴다. 노트북에서 피처 엔지니어링을 바꾸면 반드시
`src/script.py`에도 (경로 접두사를 맞춰서) 동일하게 반영해야 한다.

## 재현 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt jupyter nbclient ipykernel
python -m ipykernel install --user --name python3

jupyter nbconvert --to notebook --execute --inplace notebooks/train.ipynb
# -> model/ensemble.pkl 생성 (notebooks/ 안에서 실행되어 "../model" 에 저장됨)

mkdir -p submit_build/model
cp model/ensemble.pkl submit_build/model/
cp src/script.py src/requirements.txt submit_build/
cd submit_build && zip -r ../submissions/vN_짧은태그.zip model script.py requirements.txt
```

## 실험 워크플로

새 실험을 시작할 때마다:

```bash
git checkout main && git pull
git checkout -b experiment/vN-설명
# ... 학습/검증/노트북 수정 ...
git add -A && git commit -m "..."
git push -u origin experiment/vN-설명
gh pr create --base main --head experiment/vN-설명 --title "..." --body "..."
gh pr merge --merge
```
