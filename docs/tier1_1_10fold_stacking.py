"""
Tier 1 - Item 1: 10-Fold OOF Stacking Isolation Verification (n_splits 5 -> 10)
==============================================================================

연구 목표:
  v14 챔피언 베이스라인(101개 시즌분해 피처, LGBM 171 + CB 360) 위에서
  다른 어떤 변경도 없이 오직 n_splits만 5에서 10으로 변경하여
  10-Fold OOF 스태킹의 단독 기여도(홀드아웃, OOF, 메타러너 가중치)를 격리 검증.

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
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
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


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def main():
    print("=" * 80)
    print("🔬 [Tier 1 - Item 1] 10-Fold OOF Stacking 격리 검증 (n_splits 5 -> 10)")
    print("기준선: v14 5-Fold Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)
    start_time = time.time()

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

    # 기본 v14 피처 생성
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

    print(f"데이터 준비 완료: 훈련 {len(X_train_full):,}건, 홀드아웃 {len(X_val_fe):,}건 (총 {len(ALL_FEATURES_V14)}개 피처)")

    # 1. Base Models (Full Training split)
    print("\n[1/3] 베이스 모델 학습 중 (LightGBM 171 + CatBoost 360)...")
    m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
    m_lgb.fit(X_train_full, y_train_full)
    p_lgb_val = m_lgb.predict_proba(X_val_fe)[:, 1]

    tp_full = Pool(X_train_cb, y_train_full, cat_features=cat_idx)
    m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    m_cb.fit(tp_full)
    p_cb_val = m_cb.predict_proba(X_val_cb)[:, 1]

    # 2. 10-Fold CV OOF 생성
    print("\n[2/3] 10-Fold CV OOF 생성 중 (n_splits=10)...")
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_lgb_10 = np.zeros(len(y_train_full))
    oof_cb_10 = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf10.split(X_train_reset)):
        f_t0 = time.time()
        ml = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
        ml.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb_10[val_idx] = ml.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        mc = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        mc.fit(tp)
        oof_cb_10[val_idx] = mc.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]
        print(f"  Fold {fold_i+1:2d}/10 완료 ({time.time()-f_t0:.1f}s)")

    # 3. 메타러너 학습 및 평가
    print("\n[3/3] 메타러너 스태킹 및 지표 평가...")
    stack10 = LogisticRegression(random_state=42, max_iter=1000)
    stack10.fit(np.column_stack([oof_lgb_10, oof_cb_10]), y_train_full)
    w_lgb = stack10.coef_[0][0]
    w_cb = stack10.coef_[0][1]
    intercept = stack10.intercept_[0]

    p_val_stack10 = stack10.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    p_oof_stack10 = stack10.predict_proba(np.column_stack([oof_lgb_10, oof_cb_10]))[:, 1]

    score_val = brier_skill_score(y_val, p_val_stack10)
    score_oof = brier_skill_score(y_train_full, p_oof_stack10)

    print("\n" + "=" * 80)
    print("🏁 [Tier 1 - Item 1] 10-Fold OOF 스태킹 격리 검증 결과")
    print("=" * 80)
    print(f"v14 기준선 (5-Fold CV)  : Holdout = 836.35 | OOF = 2009.23 | LGB=3.4905, CB=0.9867")
    print(f"10-Fold OOF 스태킹      : Holdout = {score_val:.2f} ({score_val-836.35:+.2f}) | OOF = {score_oof:.2f} ({score_oof-2009.23:+.2f}) | LGB={w_lgb:.4f}, CB={w_cb:.4f}, Intercept={intercept:.4f}")
    
    # 가드레일 검수
    print("\n[가드레일 검수]")
    inspect_meta_learner_weights(w_lgb, w_cb, intercept)
    inspect_prediction_distribution(p_val_stack10)
    print("=" * 80)


if __name__ == "__main__":
    main()
