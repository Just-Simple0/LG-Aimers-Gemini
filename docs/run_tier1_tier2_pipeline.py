"""
Tier 1 & Tier 2 Comprehensive Autonomous Pipeline Runner
=========================================================

실행 목록:
  1. [Tier 1 - Item 2] 투수-타자 SVD 잠재 상성 임베딩 순수 격리 검증 (Raw Stacking, No Beta)
  2. [Tier 2 - Item 1] Direct Brier 목적함수 (Gauss-Newton) 3-Way 스태킹
  3. [Tier 2 - Item 2] Beta Calibration 시너지 정밀 분석

작성일: 2026-08-28
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.submission_guardrails import (
    inspect_meta_learner_weights,
    inspect_prediction_distribution,
    SubmissionGuardrailError,
)

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

BEST_LGB_ITER = 171
BEST_CB_ITER = 360

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=-1,
)

CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)

from docs.v63_train_and_package import (
    SeasonDecompositionEncoder,
    make_temporal_season_features,
    make_tk_lookup,
    build_base_features,
)
from docs.build_v70_grand_champion import (
    LatentMatchupSVDEncoder,
    make_temporal_svd_features,
)
from docs.exp3_beta_calibration import BetaCalibrator


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def brier_obj_gn(y_true, y_pred):
    p = sigmoid(y_pred)
    p_1mp = p * (1.0 - p)
    grad = 2.0 * (p - y_true) * p_1mp
    hess = 2.0 * (p_1mp ** 2) + 1e-5
    return grad, hess


def main():
    print("=" * 85)
    print("🚀 [Tier 1 & Tier 2] 종합 파이프라인 자동 실행 및 격리 검증")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 85)
    t_start = time.time()

    # 1. 데이터 로드
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

    # 기본 v14 피처
    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    # =========================================================================
    # [Step A] Tier 1 - Item 2: SVD 잠재 상성 피처 단독 격리 검증 (No Beta)
    # =========================================================================
    print("\n" + "=" * 80)
    print("▶ [Step A] Tier 1 - Item 2: SVD 잠재 상성 임베딩 순수 격리 검증")
    print("=" * 80)
    svd_train = make_temporal_svd_features(train_split, TARGET)
    svd_encoder_val = LatentMatchupSVDEncoder().fit(train_split, train_split[TARGET])
    svd_val = svd_encoder_val.transform(val_split)
    svd_all = pd.concat([svd_train, svd_val], axis=0).loc[train_full.index]

    fe_svd = pd.concat([fe_v14, svd_all], axis=1)

    NEW_COLS_SVD = [c for c in fe_svd.columns if c not in train_full.columns]
    ALL_FEATURES_SVD = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_SVD

    for c in CAT_COLS:
        fe_svd[c] = fe_svd[c].astype("category")

    is_val_mask = fe_svd["season"] == 2024
    X_train_svd = fe_svd.loc[~is_val_mask, ALL_FEATURES_SVD]
    y_train = fe_svd.loc[~is_val_mask, TARGET].to_numpy()
    X_val_svd = fe_svd.loc[is_val_mask, ALL_FEATURES_SVD]
    y_val = fe_svd.loc[is_val_mask, TARGET].to_numpy()

    X_train_svd_cb = X_train_svd.copy()
    X_val_svd_cb = X_val_svd.copy()
    for c in CAT_COLS:
        X_train_svd_cb[c] = X_train_svd_cb[c].astype(str)
        X_val_svd_cb[c] = X_val_svd_cb[c].astype(str)
    cat_idx_svd = [ALL_FEATURES_SVD.index(c) for c in CAT_COLS]

    # SVD Base Models
    lgb_svd = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective="binary", **LGB_PARAMS)
    lgb_svd.fit(X_train_svd, y_train)
    p_lgb_svd = lgb_svd.predict_proba(X_val_svd)[:, 1]

    cb_svd = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    cb_svd.fit(Pool(X_train_svd_cb, y_train, cat_features=cat_idx_svd))
    p_cb_svd = cb_svd.predict_proba(X_val_svd_cb)[:, 1]

    # 5-Fold OOF for SVD
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb_svd = np.zeros(len(y_train))
    oof_cb_svd = np.zeros(len(y_train))

    X_tr_svd_reset = X_train_svd.reset_index(drop=True)
    X_cb_svd_reset = X_train_svd_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf5.split(X_tr_svd_reset)):
        ml = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective="binary", **LGB_PARAMS)
        ml.fit(X_tr_svd_reset.iloc[tr_idx], y_train[tr_idx])
        oof_lgb_svd[val_idx] = ml.predict_proba(X_tr_svd_reset.iloc[val_idx])[:, 1]

        tp = Pool(X_cb_svd_reset.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx_svd)
        mc = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        mc.fit(tp)
        oof_cb_svd[val_idx] = mc.predict_proba(X_cb_svd_reset.iloc[val_idx])[:, 1]

    stack_svd = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb_svd, oof_cb_svd]), y_train)
    p_val_svd_stack = stack_svd.predict_proba(np.column_stack([p_lgb_svd, p_cb_svd]))[:, 1]
    p_oof_svd_stack = stack_svd.predict_proba(np.column_stack([oof_lgb_svd, oof_cb_svd]))[:, 1]
    bss_svd_val = brier_skill_score(y_val, p_val_svd_stack)
    bss_svd_oof = brier_skill_score(y_train, p_oof_svd_stack)

    print(f"  [결과] SVD 단독 (Raw 5-Fold Stacking) : Holdout = {bss_svd_val:.2f} (+{bss_svd_val-836.35:+.2f}) | OOF = {bss_svd_oof:.2f} (+{bss_svd_oof-2009.23:+.2f}) | LGB={stack_svd.coef_[0][0]:.2f}, CB={stack_svd.coef_[0][1]:.2f}")

    # =========================================================================
    # [Step B] Tier 2 - Item 1: Direct Brier 목적함수 3-Way 스태킹
    # =========================================================================
    print("\n" + "=" * 80)
    print("▶ [Step B] Tier 2 - Item 1: Direct Brier 목적함수 (Gauss-Newton) 3-Way 스태킹")
    print("=" * 80)
    # v14 피처셋 기반
    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14
    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")

    X_train_v14 = fe_v14.loc[~is_val_mask, ALL_FEATURES_V14]
    X_val_v14 = fe_v14.loc[is_val_mask, ALL_FEATURES_V14]
    X_train_v14_cb = X_train_v14.copy()
    X_val_v14_cb = X_val_v14.copy()
    for c in CAT_COLS:
        X_train_v14_cb[c] = X_train_v14_cb[c].astype(str)
        X_val_v14_cb[c] = X_val_v14_cb[c].astype(str)
    cat_idx_v14 = [ALL_FEATURES_V14.index(c) for c in CAT_COLS]

    # LogLoss LGB + Brier LGB + CB
    m_log = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective="binary", **LGB_PARAMS).fit(X_train_v14, y_train)
    p_log = m_log.predict_proba(X_val_v14)[:, 1]

    m_brier = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective=brier_obj_gn, **LGB_PARAMS).fit(X_train_v14, y_train)
    p_brier = sigmoid(m_brier.predict(X_val_v14, raw_score=True))

    m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS).fit(Pool(X_train_v14_cb, y_train, cat_features=cat_idx_v14))
    p_cb = m_cb.predict_proba(X_val_v14_cb)[:, 1]

    oof_log = np.zeros(len(y_train))
    oof_brier = np.zeros(len(y_train))
    oof_cb = np.zeros(len(y_train))

    X_tr_v14_res = X_train_v14.reset_index(drop=True)
    X_cb_v14_res = X_train_v14_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf5.split(X_tr_v14_res)):
        ml = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective="binary", **LGB_PARAMS).fit(X_tr_v14_res.iloc[tr_idx], y_train[tr_idx])
        oof_log[val_idx] = ml.predict_proba(X_tr_v14_res.iloc[val_idx])[:, 1]

        mb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective=brier_obj_gn, **LGB_PARAMS).fit(X_tr_v14_res.iloc[tr_idx], y_train[tr_idx])
        oof_brier[val_idx] = sigmoid(mb.predict(X_tr_v14_res.iloc[val_idx], raw_score=True))

        mc = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS).fit(Pool(X_cb_v14_res.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx_v14))
        oof_cb[val_idx] = mc.predict_proba(X_cb_v14_res.iloc[val_idx])[:, 1]

    stack3 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_log, oof_brier, oof_cb]), y_train)
    p_val_3w = stack3.predict_proba(np.column_stack([p_log, p_brier, p_cb]))[:, 1]
    p_oof_3w = stack3.predict_proba(np.column_stack([oof_log, oof_brier, oof_cb]))[:, 1]
    bss_3w_val = brier_skill_score(y_val, p_val_3w)
    bss_3w_oof = brier_skill_score(y_train, p_oof_3w)

    print(f"  [결과] Direct Brier 3-Way Stacking     : Holdout = {bss_3w_val:.2f} ({bss_3w_val-836.35:+.2f}) | OOF = {bss_3w_oof:.2f} ({bss_3w_oof-2009.23:+.2f}) | Coefs={stack3.coef_[0]}")

    # =========================================================================
    # [Step C] Tier 2 - Item 2: Beta Calibration 시너지 정밀 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("▶ [Step C] Tier 2 - Item 2: Beta Calibration 시너지 분석")
    print("=" * 80)
    # v14 2-Way Stacking 위에 Beta Calibration
    stack2 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_log, oof_cb]), y_train)
    p_val_2w = stack2.predict_proba(np.column_stack([p_log, p_cb]))[:, 1]
    p_oof_2w = stack2.predict_proba(np.column_stack([oof_log, oof_cb]))[:, 1]

    beta_v14 = BetaCalibrator().fit(p_oof_2w, y_train)
    p_val_beta_v14 = beta_v14.predict(p_val_2w)
    p_oof_beta_v14 = beta_v14.predict(p_oof_2w)
    bss_beta_val = brier_skill_score(y_val, p_val_beta_v14)
    bss_beta_oof = brier_skill_score(y_train, p_oof_beta_v14)

    print(f"  [결과] v14 + Beta Calibration          : Holdout = {bss_beta_val:.2f} (+{bss_beta_val-836.35:+.2f}) | OOF = {bss_beta_oof:.2f} (+{bss_beta_oof-2009.23:+.2f}) | a={beta_v14.a_:.3f}, b={beta_v14.b_:.3f}, c={beta_v14.c_:.3f}")

    print("\n" + "=" * 85)
    print("🏁 [Tier 1 & Tier 2] 종합 결과 비교표")
    print("=" * 85)
    print(f"0. v14 기준선 (5-Fold CV)               : Holdout = 836.35 | OOF = 2009.23 | Dacon 실측 = 976.51점 🏆")
    print(f"1. [Tier 1] SVD 단독 (Raw Stacking)     : Holdout = {bss_svd_val:.2f} ({bss_svd_val-836.35:+.2f}) | OOF = {bss_svd_oof:.2f} ({bss_svd_oof-2009.23:+.2f})")
    print(f"2. [Tier 2] Direct Brier 3-Way Stacking : Holdout = {bss_3w_val:.2f} ({bss_3w_val-836.35:+.2f}) | OOF = {bss_3w_oof:.2f} ({bss_3w_oof-2009.23:+.2f})")
    print(f"3. [Tier 2] Beta Calibration 보정       : Holdout = {bss_beta_val:.2f} ({bss_beta_val-836.35:+.2f}) | OOF = {bss_beta_oof:.2f} ({bss_beta_oof-2009.23:+.2f})")
    print("=" * 85)


if __name__ == "__main__":
    main()
