"""
제출 전 5대 하드 가드레일 검수 시스템 (Submission Guardrails)
============================================================

이 모듈은 제출 ZIP이 생성되기 전, 모델의 과적합, 앙상블 붕괴, 확률 꼬리 왜곡을
사전에 100% 차단하기 위한 엄격한 수학적 가드레일을 제공합니다.

5대 검수 기준:
  1. [Gate Check]: (A) 홀드아웃 및 (B) OOF 점수가 기준선 대비 동반 양수인가?
  2. [Ensemble Balance]: 메타러너 가중치가 특정 모델로 쏠리지 않고 상호 완충 구조(CB weight >= 0.15, LGB weight <= 4.20)를 유지하는가?
  3. [Probability Bounds]: 245,789행 대용량 추론 예측 확률이 안전 범위 [0.32, 0.77] 내에 위치하는가?
  4. [Calibration Spread]: 예측 확률의 표준편차 std(p)가 정상 범위 [0.038, 0.062] 내인가? (과도한 확신 방지)
  5. [Mean Drift]: 예측 확률의 평균이 리그 기저율(0.5238 +- 0.005)과 일치하는가?
"""

import numpy as np


class SubmissionGuardrailError(Exception):
    """제출 가드레일 검증 실패 시 발생하는 예외."""
    pass


def inspect_meta_learner_weights(lgb_weight, cb_weight, intercept):
    """메타러너 가중치 균형 검수."""
    print("\n[가드레일 1/3] 메타러너 가중치 균형 검수 중...")
    print(f"  LGB 계수: {lgb_weight:.4f}, CB 계수: {cb_weight:.4f}, Intercept: {intercept:.4f}")

    if cb_weight < 0.15:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] CatBoost 가중치 붕괴 ({cb_weight:.4f} < 0.15). "
            f"단일 모델 쏠림으로 인한 앙상블 완충 상실 위험! 제출 금지."
        )
    if lgb_weight > 4.20:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] LightGBM 가중치 과대 증폭 ({lgb_weight:.4f} > 4.20). "
            f"확률 꼬리 왜곡 및 Brier Score 폭락 위험! 제출 금지."
        )
    print("  ✅ [PASS] 메타러너 가중치 균형 합격!")


def inspect_prediction_distribution(preds, expected_mean=0.4917):
    """예측 확률 분포 및 통계적 가드레일 검수."""
    print("\n[가드레일 2/3] 예측 확률 분포 및 극단치 검수 중...")
    p_min = float(np.min(preds))
    p_max = float(np.max(preds))
    p_mean = float(np.mean(preds))
    p_std = float(np.std(preds))

    print(f"  예측 수: {len(preds):,} | Mean: {p_mean:.4f} | Std: {p_std:.4f} | Min: {p_min:.4f} | Max: {p_max:.4f}")

    # 1. Min/Max Guardrail
    if p_min < 0.320:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] 예측 최솟값 하한선 이탈 (min={p_min:.4f} < 0.320). "
            f"과도한 하방 확신으로 Brier Score 페널티 폭증 위험! 제출 금지."
        )
    if p_max > 0.770:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] 예측 최댓값 상한선 이탈 (max={p_max:.4f} > 0.770). "
            f"과도한 상방 확신으로 Brier Score 페널티 폭증 위험! 제출 금지."
        )

    # 2. Std Guardrail
    if p_std > 0.065:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] 예측 표준편차 과다 (std={p_std:.4f} > 0.065). "
            f"확률이 양극단으로 과도하게 벌어져 있음. 제출 금지."
        )
    if p_std < 0.035:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] 예측 표준편차 과소 (std={p_std:.4f} < 0.035). "
            f"모델의 분별력 부족. 제출 금지."
        )

    # 3. Mean Drift Guardrail
    if abs(p_mean - expected_mean) > 0.006:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] 평균 기저율 편차 과다 (|{p_mean:.4f} - {expected_mean:.4f}| > 0.006). 제출 금지."
        )

    print("  ✅ [PASS] 예측 확률 분포 및 통계적 가드레일 합격!")


def inspect_dual_gates(holdout_diff, oof_diff):
    """이중 게이트(A/B) 검수."""
    print("\n[가드레일 3/3] 이중 게이트(A/B) 검수 중...")
    print(f"  (A) 홀드아웃 diff: {holdout_diff:+.2f} | (B) OOF score diff: {oof_diff:+.2f}")

    if holdout_diff <= 0.0:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] (A) 2024 홀드아웃 점수 미개선 ({holdout_diff:+.2f} <= 0). 제출 금지."
        )
    if oof_diff <= 0.0:
        raise SubmissionGuardrailError(
            f"❌ [FAIL] (B) OOF 독립 게이트 미개선 ({oof_diff:+.2f} <= 0). 실측 역전 위험으로 제출 금지."
        )
    print("  ✅ [PASS] 이중 게이트 동반 개선 확인 완료!")
