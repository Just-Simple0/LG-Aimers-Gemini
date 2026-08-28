"""
Tier 2 - Item 1: Direct Brier Custom Objective 3-Way Stacking (Log LGB + Brier LGB + CB)
========================================================================================

연구 목표:
  LogLoss LightGBM + Direct Brier LightGBM (Gauss-Newton) + CatBoost를 3-Way로 결합했을 때
  메타러너 계수의 양수성 및 OOF/홀드아웃 일반화 성능을 격리 검증.

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
    print("=" * 80)
    print("🔬 [Tier 2 - Item 1] Direct Brier 3-Way Stacking 격리 검증 (Log LGB + Brier LGB + CB)")
    print("기준선: v14 5-Fold Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)

    # 1. 데이터 로드 및 베이스 피처셋 구성
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

    # 1. Base Models
    print("\n[1/3] 베이스 모델 학습 중...")
    m_log = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective="binary", **LGB_PARAMS)
    m_log.fit(X_train_full, y_train_full)
    p_log_val = m_log.predict_proba(X_val_fe)[:, 1]

    m_brier = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective=brier_obj_gn, **LGB_PARAMS)
    m_brier.fit(X_train_full, y_train_full)
    p_brier_val = sigmoid(m_brier.predict(X_val_fe, raw_score=True))

    tp_full = Pool(X_train_cb, y_train_full, cat_features=cat_idx)
    m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    m_cb.fit(tp_full)
    p_cb_val = m_cb.predict_proba(X_val_cb)[:, 1]

    # 2. 5-Fold OOF 생성
    print("\n[2/3] 5-Fold CV OOF 생성 중...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_log = np.zeros(len(y_train_full))
    oof_brier = np.zeros(len(y_train_full))
    oof_cb = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_reset)):
        f_t0 = time.time()
        ml = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective="binary", **LGB_PARAMS)
        ml.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_log[val_idx] = ml.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        mb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, objective=brier_obj_gn, **LGB_PARAMS)
        mb.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_brier[val_idx] = sigmoid(mb.predict(X_train_reset.iloc[val_idx], raw_score=True))

        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        mc = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        mc.fit(tp)
        oof_cb[val_idx] = mc.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]
        print(f"  Fold {fold_i+1}/5 완료 ({time.time()-f_t0:.1f}s)")

    # 3. 메타러너 스태킹
    print("\n[3/3] 3-Way 메타러너 스태킹...")
    # 2-way 기준선
    st2 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_log, oof_cb]), y_train_full)
    p_val_2w = st2.predict_proba(np.column_stack([p_log_val, p_cb_val]))[:, 1]
    p_oof_2w = st2.predict_proba(np.column_stack([oof_log, oof_cb]))[:, 1]
    bss_2w_val = brier_skill_score(y_val, p_val_2w)
    bss_2w_oof = brier_skill_score(y_train_full, p_oof_2w)

    # 3-way
    st3 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_log, oof_brier, oof_cb]), y_train_full)
    p_val_3w = st3.predict_proba(np.column_stack([p_log_val, p_brier_val, p_cb_val]))[:, 1]
    p_oof_3w = st3.predict_proba(np.column_stack([oof_log, oof_brier, oof_cb]))[:, 1]
    bss_3w_val = brier_skill_score(y_val, p_val_3w)
    bss_3w_oof = brier_skill_score(y_train_full, p_oof_3w)

    print("\n" + "=" * 80)
    print("🏁 [Tier 2 - Item 1] Direct Brier 3-Way 스태킹 격리 검증 결과")
    print("=" * 80)
    print(f"2-Way 기준선 (Log LGB + CB)         : Holdout = {bss_2w_val:.2f} | OOF = {bss_2w_oof:.2f} | LGB={st2.coef_[0][0]:.2f}, CB={st2.coef_[0][1]:.2f}")
    print(f"3-Way 스태킹 (Log + Brier LGB + CB) : Holdout = {bss_3w_val:.2f} ({bss_3w_val-bss_2w_val:+.2f}) | OOF = {bss_3w_oof:.2f} ({bss_3w_oof-bss_2w_oof:+.2f}) | Coefs={st3.coef_[0]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
