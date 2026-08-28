"""
Ultimate Synergy: SVD Latent Matchup Embeddings + Direct Brier Objective + CatBoost
===================================================================================

연구 목표:
  Paradigm 1 (Direct Brier Objective)과 Paradigm 2 (SVD Latent Matchup Embeddings)를
  결합하여 5-Fold CV 앙상블 파이프라인의 최고 성능을 극대화.

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
from docs.exp2_svd_matchup_embeddings import (
    LatentMatchupSVDEncoder,
    make_temporal_svd_features,
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
    print("🚀 [Ultimate Synergy] SVD Latent Matchup + Direct Brier Objective 결합 연구")
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

    # Season Decomp + Base
    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    # SVD Latent Features
    print("[1/3] SVD Latent Matchup Features 생성 중...")
    svd_train = make_temporal_svd_features(train_split, TARGET)
    svd_encoder_val = LatentMatchupSVDEncoder().fit(train_split, train_split[TARGET])
    svd_val = svd_encoder_val.transform(val_split)
    svd_all = pd.concat([svd_train, svd_val], axis=0).loc[train_full.index]

    fe_synergy = pd.concat([fe_v14, svd_all], axis=1)
    NEW_COLS_SYN = [c for c in fe_synergy.columns if c not in train_full.columns]
    ALL_FEATURES_SYN = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_SYN

    for c in CAT_COLS:
        fe_synergy[c] = fe_synergy[c].astype("category")

    is_val_mask = fe_synergy["season"] == 2024
    X_train_full = fe_synergy.loc[~is_val_mask, ALL_FEATURES_SYN]
    y_train_full = fe_synergy.loc[~is_val_mask, TARGET].to_numpy()
    X_val_fe = fe_synergy.loc[is_val_mask, ALL_FEATURES_SYN]
    y_val = fe_synergy.loc[is_val_mask, TARGET].to_numpy()

    X_train_cb = X_train_full.copy()
    X_val_cb = X_val_fe.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    cat_idx = [ALL_FEATURES_SYN.index(c) for c in CAT_COLS]

    # 2. 베이스 모델 훈련 (LogLoss LGB + Direct Brier LGB + CatBoost)
    print("\n[2/3] 베이스 모델 전체 학습 중 (LogLoss LGB + Brier LGB + CatBoost)...")
    m_lgb_log = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
    m_lgb_log.fit(X_train_full, y_train_full)
    p_lgb_log = m_lgb_log.predict_proba(X_val_fe)[:, 1]

    m_lgb_brier = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective=brier_obj_gn, random_state=42, n_jobs=-1, verbosity=-1)
    m_lgb_brier.fit(X_train_full, y_train_full)
    p_lgb_brier = sigmoid(m_lgb_brier.predict(X_val_fe, raw_score=True))

    m_cb = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    m_cb.fit(Pool(X_train_cb, y_train_full, cat_features=cat_idx))
    p_cb = m_cb.predict_proba(X_val_cb)[:, 1]

    # 3. 5-Fold OOF 생성
    print("\n[3/3] 5-Fold CV OOF 생성 및 메타러너 스태킹...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_log = np.zeros(len(y_train_full))
    oof_brier = np.zeros(len(y_train_full))
    oof_cb = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_reset)):
        # Logloss LGB
        ml = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
        ml.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_log[val_idx] = ml.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        # Brier LGB
        mb = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective=brier_obj_gn, random_state=42, n_jobs=-1, verbosity=-1)
        mb.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_brier[val_idx] = sigmoid(mb.predict(X_train_reset.iloc[val_idx], raw_score=True))

        # CatBoost
        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        mc = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
        mc.fit(tp)
        oof_cb[val_idx] = mc.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]
        print(f"  Fold {fold_i+1}/5 완료")

    # Meta-Learner 1: SVD + LogLoss LGB + CB
    st_svd_base = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_log, oof_cb]), y_train_full)
    p_svd_base = st_svd_base.predict_proba(np.column_stack([p_lgb_log, p_cb]))[:, 1]
    bss_svd_base = brier_skill_score(y_val, p_svd_base)
    oof_svd_base = brier_skill_score(y_train_full, st_svd_base.predict_proba(np.column_stack([oof_log, oof_cb]))[:, 1])

    # Meta-Learner 2: SVD + Direct Brier LGB + CB
    st_svd_brier = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_brier, oof_cb]), y_train_full)
    p_svd_brier = st_svd_brier.predict_proba(np.column_stack([p_lgb_brier, p_cb]))[:, 1]
    bss_svd_brier = brier_skill_score(y_val, p_svd_brier)
    oof_svd_brier = brier_skill_score(y_train_full, st_svd_brier.predict_proba(np.column_stack([oof_brier, oof_cb]))[:, 1])

    # Meta-Learner 3: SVD + Tri-Stack (Log LGB + Brier LGB + CB)
    st_svd_tri = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_log, oof_brier, oof_cb]), y_train_full)
    p_svd_tri = st_svd_tri.predict_proba(np.column_stack([p_lgb_log, p_lgb_brier, p_cb]))[:, 1]
    bss_svd_tri = brier_skill_score(y_val, p_svd_tri)
    oof_svd_tri = brier_skill_score(y_train_full, st_svd_tri.predict_proba(np.column_stack([oof_log, oof_brier, oof_cb]))[:, 1])

    print("\n" + "=" * 80)
    print("🏁 [Ultimate Synergy] 종합 결합 검증 결과")
    print("=" * 80)
    print(f"0. v14 기준선                     : Holdout = 836.35 | OOF = 2009.23 | LGB=3.49, CB=0.99")
    print(f"1. SVD + Standard Stacking (LGB+CB): Holdout = {bss_svd_base:.2f} ({bss_svd_base-836.35:+.2f}) | OOF = {oof_svd_base:.2f} ({oof_svd_base-2009.23:+.2f}) | LGB={st_svd_base.coef_[0][0]:.2f}, CB={st_svd_base.coef_[0][1]:.2f}")
    print(f"2. SVD + Direct Brier Stacking    : Holdout = {bss_svd_brier:.2f} ({bss_svd_brier-836.35:+.2f}) | OOF = {oof_svd_brier:.2f} ({oof_svd_brier-2009.23:+.2f}) | BrierLGB={st_svd_brier.coef_[0][0]:.2f}, CB={st_svd_brier.coef_[0][1]:.2f}")
    print(f"3. SVD + Tri-Stacking (Log+Brier+CB): Holdout = {bss_svd_tri:.2f} ({bss_svd_tri-836.35:+.2f}) | OOF = {oof_svd_tri:.2f} ({oof_svd_tri-2009.23:+.2f}) | Coefs={st_svd_tri.coef_[0]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
