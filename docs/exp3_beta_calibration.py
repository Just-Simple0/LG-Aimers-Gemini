"""
Paradigm 3: Beta Calibration for Tabular Brier Optimization (Kull et al., 2017)
================================================================================

연구 목표:
  Brier Score 최적화를 위한 모수적 비대칭 캘리브레이션(Beta Calibration)을 구현하여
  5-Fold OOF 및 2024 홀드아웃 확률 분포를 정밀 보정하고 성능을 검증.

수학적 정식화:
  s in (0, 1): 모델 예측 확률
  x1 = ln(s), x2 = -ln(1 - s)
  logit(p) = a * x1 + b * x2 + c = a * ln(s) - b * ln(1 - s) + c
  (a >= 0, b >= 0, c in R)

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


class BetaCalibrator:
    """Beta Calibration (Kull et al., 2017) via Log-Odds Transformation."""

    def __init__(self, eps=1e-5):
        self.eps = eps
        self.lr = None
        self.a_ = 1.0
        self.b_ = 1.0
        self.c_ = 0.0

    def _transform_scores(self, s):
        s_safe = np.clip(np.asarray(s, dtype=float), self.eps, 1.0 - self.eps)
        x1 = np.log(s_safe)
        x2 = -np.log(1.0 - s_safe)
        return np.column_stack([x1, x2])

    def fit(self, s, y):
        X = self._transform_scores(s)
        self.lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.lr.fit(X, y)
        self.a_ = self.lr.coef_[0][0]
        self.b_ = self.lr.coef_[0][1]
        self.c_ = self.lr.intercept_[0]
        return self

    def predict(self, s):
        X = self._transform_scores(s)
        return self.lr.predict_proba(X)[:, 1]


def main():
    print("=" * 80)
    print("🚀 [Paradigm 3] Beta Calibration for Tabular Brier Optimization 연구")
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

    # 1. Base Models
    lgb_m = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
    lgb_m.fit(X_train_full, y_train_full)
    p_lgb = lgb_m.predict_proba(X_val_fe)[:, 1]

    cb_m = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    cb_m.fit(Pool(X_train_cb, y_train_full, cat_features=cat_idx))
    p_cb = cb_m.predict_proba(X_val_cb)[:, 1]

    # 2. 5-Fold OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_train_full))
    oof_cb = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_reset)):
        m_l = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
        m_l.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb[val_idx] = m_l.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
        m_c.fit(tp)
        oof_cb[val_idx] = m_c.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]

    # Standard Stacking
    stack_v14 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb, oof_cb]), y_train_full)
    p_oof_v14 = stack_v14.predict_proba(np.column_stack([oof_lgb, oof_cb]))[:, 1]
    p_val_v14 = stack_v14.predict_proba(np.column_stack([p_lgb, p_cb]))[:, 1]

    # Beta Calibration on top of Stacking
    beta_cal = BetaCalibrator().fit(p_oof_v14, y_train_full)
    p_val_beta = beta_cal.predict(p_val_v14)
    p_oof_beta = beta_cal.predict(p_oof_v14)

    bss_v14_val = brier_skill_score(y_val, p_val_v14)
    bss_v14_oof = brier_skill_score(y_train_full, p_oof_v14)
    bss_beta_val = brier_skill_score(y_val, p_val_beta)
    bss_beta_oof = brier_skill_score(y_train_full, p_oof_beta)

    print("\n" + "=" * 80)
    print("🏁 [Paradigm 3] Beta Calibration 검증 결과")
    print("=" * 80)
    print(f"1. v14 기준선 (Logistic Stacking) : Holdout = {bss_v14_val:.2f} | OOF = {bss_v14_oof:.2f}")
    print(f"2. Beta Calibration 적용         : Holdout = {bss_beta_val:.2f} ({bss_beta_val-bss_v14_val:+.2f}) | OOF = {bss_beta_oof:.2f} ({bss_beta_oof-bss_v14_oof:+.2f}) | a={beta_cal.a_:.3f}, b={beta_cal.b_:.3f}, c={beta_cal.c_:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
