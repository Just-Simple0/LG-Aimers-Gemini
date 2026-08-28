"""
Paradigm 1: Direct Brier Loss Custom Objective for GBDT (Clean & Robust)
=========================================================================

연구 목표:
  LogLoss(Cross-Entropy) 대신 Brier Score(확률 공간의 MSE)를 직접 최소화하는
  Custom Objective 함수를 설계하여 LightGBM 및 앙상블 파이프라인에 적용/검증.

수학적 정식화:
  Loss: L(y, z) = (sigma(z) - y)^2, where p = sigma(z) = 1 / (1 + exp(-z))
  Gradient: g(z) = 2 * (p - y) * p * (1 - p)
  Gauss-Newton Hessian: h_gn(z) = 2 * (p * (1 - p))^2 + 1e-5 (convex & strictly positive)

작성일: 2026-08-28
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

from docs.v63_train_and_package import (
    SeasonDecompositionEncoder,
    make_temporal_season_features,
    make_tk_lookup,
    build_base_features,
)


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def brier_obj_gn(y_true, y_pred):
    """Gauss-Newton 근사 기반 Brier Score Custom Objective."""
    p = sigmoid(y_pred)
    p_1mp = p * (1.0 - p)
    grad = 2.0 * (p - y_true) * p_1mp
    hess = 2.0 * (p_1mp ** 2) + 1e-5
    return grad, hess


def brier_obj_exact(y_true, y_pred):
    """Exact Hessian + Lower Bound Clipping 기반 Brier Score Custom Objective."""
    p = sigmoid(y_pred)
    p_1mp = p * (1.0 - p)
    grad = 2.0 * (p - y_true) * p_1mp
    exact_hess = 2.0 * p_1mp * (p_1mp + (p - y_true) * (1.0 - 2.0 * p))
    hess = np.maximum(exact_hess, 2.0 * (p_1mp ** 2) * 0.1 + 1e-5)
    return grad, hess


def main():
    print("=" * 80)
    print("🚀 [Paradigm 1] Direct Brier Custom Objective 연구 및 검증")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)

    # 1. 데이터 로드 및 피처 생성
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_VAL = train_full.loc[train_full["season"] != 2024, TARGET].mean()

    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

    train_split = train_full[train_full["season"] <= 2023]
    val_split = train_full[train_full["season"] == 2024]

    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")

    is_val_mask = fe_v14["season"] == 2024
    X_train_full = fe_v14.loc[~is_val_mask, ALL_FEATURES_V14]
    y_train_full = fe_v14.loc[~is_val_mask, TARGET].to_numpy()
    X_val_fe = fe_v14.loc[is_val_mask, ALL_FEATURES_V14]
    y_val = fe_v14.loc[is_val_mask, TARGET].to_numpy()

    X_train_cb = X_train_full.copy()
    X_val_cb = X_val_fe.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    cat_idx = [ALL_FEATURES_V14.index(c) for c in CAT_COLS]

    print(f"\n데이터 준비 완료: 훈련 {len(X_train_full):,}건, 2024 홀드아웃 {len(X_val_fe):,}건")

    # =========================================================================
    # [1] 단일 모델 성능 비교 (LogLoss vs Brier Gauss-Newton vs Brier Exact)
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Step 1] LightGBM 단독 모델 목적함수별 홀드아웃 BSS 비교")
    print("=" * 70)

    # 1-1. Standard LogLoss LGBM
    m_logloss = lgb.LGBMClassifier(
        n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary",
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    m_logloss.fit(X_train_full, y_train_full)
    p_logloss = m_logloss.predict_proba(X_val_fe)[:, 1]
    bss_logloss = brier_skill_score(y_val, p_logloss)
    print(f"  (1) Standard LogLoss LightGBM       : BSS = {bss_logloss:.4f} | [Min={p_logloss.min():.3f}, Max={p_logloss.max():.3f}]")

    # 1-2. Custom Brier Objective (Gauss-Newton)
    m_brier_gn = lgb.LGBMClassifier(
        n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective=brier_obj_gn,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    m_brier_gn.fit(X_train_full, y_train_full)
    raw_gn = m_brier_gn.predict(X_val_fe, raw_score=True)
    p_gn = sigmoid(raw_gn)
    bss_gn = brier_skill_score(y_val, p_gn)
    print(f"  (2) Direct Brier (Gauss-Newton) LGBM : BSS = {bss_gn:.4f} | [Min={p_gn.min():.3f}, Max={p_gn.max():.3f}]")

    # 1-3. Custom Brier Objective (Exact Clipped)
    m_brier_exact = lgb.LGBMClassifier(
        n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective=brier_obj_exact,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    m_brier_exact.fit(X_train_full, y_train_full)
    raw_exact = m_brier_exact.predict(X_val_fe, raw_score=True)
    p_exact = sigmoid(raw_exact)
    bss_exact = brier_skill_score(y_val, p_exact)
    print(f"  (3) Direct Brier (Exact Clipped) LGBM: BSS = {bss_exact:.4f} | [Min={p_exact.min():.3f}, Max={p_exact.max():.3f}]")

    # =========================================================================
    # [2] 5-Fold CV 기반 앙상블 스태킹 파이프라인 결합 검증
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Step 2] 5-Fold CV 앙상블 스태킹 결합 검증 (CatBoost + Direct Brier LGB)")
    print("=" * 70)

    # CatBoost Base
    cb_m = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    cb_m.fit(Pool(X_train_cb, y_train_full, cat_features=cat_idx))
    p_cb_val = cb_m.predict_proba(X_val_cb)[:, 1]

    # 5-Fold CV OOF 생성
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb_logloss = np.zeros(len(y_train_full))
    oof_lgb_brier = np.zeros(len(y_train_full))
    oof_cb = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_reset)):
        # Logloss LGB
        m_l = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
        m_l.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb_logloss[val_idx] = m_l.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        # Brier LGB
        m_b = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective=brier_obj_gn, random_state=42, n_jobs=-1, verbosity=-1)
        m_b.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb_brier[val_idx] = sigmoid(m_b.predict(X_train_reset.iloc[val_idx], raw_score=True))

        # CatBoost
        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
        m_c.fit(tp)
        oof_cb[val_idx] = m_c.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]
        print(f"  Fold {fold_i+1}/5 완료")

    # Meta Learner 1: v14 기준선 (LogLoss LGB + CB)
    stack_v14 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb_logloss, oof_cb]), y_train_full)
    p_ens_v14 = stack_v14.predict_proba(np.column_stack([p_logloss, p_cb_val]))[:, 1]
    bss_ens_v14 = brier_skill_score(y_val, p_ens_v14)
    oof_ens_v14 = brier_skill_score(y_train_full, stack_v14.predict_proba(np.column_stack([oof_lgb_logloss, oof_cb]))[:, 1])

    # Meta Learner 2: Direct Brier (Brier LGB + CB)
    stack_brier = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb_brier, oof_cb]), y_train_full)
    p_ens_brier = stack_brier.predict_proba(np.column_stack([p_gn, p_cb_val]))[:, 1]
    bss_ens_brier = brier_skill_score(y_val, p_ens_brier)
    oof_ens_brier = brier_skill_score(y_train_full, stack_brier.predict_proba(np.column_stack([oof_lgb_brier, oof_cb]))[:, 1])

    # Meta Learner 3: 3-Way Tri-Stacking (LogLoss LGB + Brier LGB + CB)
    stack_tri = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb_logloss, oof_lgb_brier, oof_cb]), y_train_full)
    p_ens_tri = stack_tri.predict_proba(np.column_stack([p_logloss, p_gn, p_cb_val]))[:, 1]
    bss_ens_tri = brier_skill_score(y_val, p_ens_tri)
    oof_ens_tri = brier_skill_score(y_train_full, stack_tri.predict_proba(np.column_stack([oof_lgb_logloss, oof_lgb_brier, oof_cb]))[:, 1])

    print("\n" + "=" * 80)
    print("🏁 [Paradigm 1] 종합 검증 결과")
    print("=" * 80)
    print(f"1. v14 기준선 (LogLoss LGB + CB)       : Holdout = {bss_ens_v14:.2f} | OOF = {oof_ens_v14:.2f} | LGB={stack_v14.coef_[0][0]:.2f}, CB={stack_v14.coef_[0][1]:.2f}")
    print(f"2. Brier Direct (Brier LGB + CB)       : Holdout = {bss_ens_brier:.2f} ({bss_ens_brier-bss_ens_v14:+.2f}) | OOF = {oof_ens_brier:.2f} ({oof_ens_brier-oof_ens_v14:+.2f}) | BrierLGB={stack_brier.coef_[0][0]:.2f}, CB={stack_brier.coef_[0][1]:.2f}")
    print(f"3. Tri-Stacking (LogLoss + Brier + CB) : Holdout = {bss_ens_tri:.2f} ({bss_ens_tri-bss_ens_v14:+.2f}) | OOF = {oof_ens_tri:.2f} ({oof_ens_tri-oof_ens_v14:+.2f}) | Coefs={stack_tri.coef_[0]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
