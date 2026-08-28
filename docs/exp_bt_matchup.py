"""
Step 1: Bradley-Terry Empirical Bayes Matchup Model (arXiv:1701.08055)
======================================================================

연구 목적:
  SVD의 Cold-Start 붕괴 문제를 해결하기 위해,
  희소 쌍대비교 전용 모델인 Bradley-Terry + Empirical Bayes / James-Stein 수축을 적용하여
  투수 잠재 능력(theta_p), 타자 잠재 능력(theta_b), 매치업 잔차(delta_pb), 기대 승률(P_pb)을
  시계열 누수 없이 추출하고 v14 베이스라인 위에서 홀드아웃 및 5-Fold OOF 성능 검증.

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


def logit(p, eps=1e-4):
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


class BradleyTerryEBEncoder:
    """Bradley-Terry Empirical Bayes Shrunk Matchup Encoder."""

    def __init__(self, alpha_p=50.0, alpha_b=50.0, alpha_pb=20.0):
        self.alpha_p = alpha_p
        self.alpha_b = alpha_b
        self.alpha_pb = alpha_pb
        self.mu_0 = 0.0
        self.r_0 = 0.52
        self.pitcher_theta = {}
        self.batter_theta = {}
        self.matchup_delta = {}

    def fit(self, df, target):
        df = df.copy()
        df["__target"] = np.asarray(target, dtype=float)
        self.r_0 = float(df["__target"].mean())
        self.mu_0 = logit(self.r_0)

        # 1. Pitcher Empirical Bayes Shrunk Logit
        p_agg = df.groupby("pitcher_id")["__target"].agg(["count", "mean"])
        p_n = p_agg["count"].to_numpy()
        p_rate = p_agg["mean"].to_numpy()
        p_weight = p_n / (p_n + self.alpha_p)
        p_theta_vals = p_weight * (logit(p_rate) - self.mu_0)
        self.pitcher_theta = dict(zip(p_agg.index, p_theta_vals))

        # 2. Batter Empirical Bayes Shrunk Logit
        b_agg = df.groupby("batter_id")["__target"].agg(["count", "mean"])
        b_n = b_agg["count"].to_numpy()
        b_rate = b_agg["mean"].to_numpy()
        b_weight = b_n / (b_n + self.alpha_b)
        b_theta_vals = b_weight * (logit(b_rate) - self.mu_0)
        self.batter_theta = dict(zip(b_agg.index, b_theta_vals))

        # 3. Head-to-Head Matchup Residual
        m_agg = df.groupby(["pitcher_id", "batter_id"])["__target"].agg(["count", "mean"]).reset_index()
        m_p_th = m_agg["pitcher_id"].map(self.pitcher_theta).fillna(0.0).to_numpy()
        m_b_th = m_agg["batter_id"].map(self.batter_theta).fillna(0.0).to_numpy()
        expected_logodds = self.mu_0 + m_p_th - m_b_th

        m_n = m_agg["count"].to_numpy()
        m_rate = m_agg["mean"].to_numpy()
        m_weight = m_n / (m_n + self.alpha_pb)
        m_residual = m_weight * (logit(m_rate) - expected_logodds)

        pairs = list(zip(m_agg["pitcher_id"], m_agg["batter_id"]))
        self.matchup_delta = dict(zip(pairs, m_residual))

        return self

    def transform(self, df):
        p_ids = df["pitcher_id"].to_numpy()
        b_ids = df["batter_id"].to_numpy()

        p_th = np.array([self.pitcher_theta.get(pid, 0.0) for pid in p_ids], dtype=float)
        b_th = np.array([self.batter_theta.get(bid, 0.0) for bid in b_ids], dtype=float)

        delta = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            pair = (p_ids[i], b_ids[i])
            delta[i] = self.matchup_delta.get(pair, 0.0)

        total_logodds = self.mu_0 + p_th - b_th + delta
        prob = sigmoid(total_logodds)

        return pd.DataFrame({
            "bt_pitcher_theta": p_th,
            "bt_batter_theta": b_th,
            "bt_matchup_delta": delta,
            "bt_matchup_logodds": total_logodds,
            "bt_expected_prob": prob,
        }, index=df.index)


def make_temporal_bt_features(train_df, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = BradleyTerryEBEncoder()
            dummy_target = pd.Series([0.52], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = BradleyTerryEBEncoder().fit(
                train_df.loc[past_mask], train_df.loc[past_mask, target_col]
            )
            chunks.append(encoder.transform(train_df.loc[s_mask]))

    return pd.concat(chunks, axis=0).loc[train_df.index]


def main():
    print("=" * 80)
    print("🚀 [Step 1] Bradley-Terry Empirical Bayes Matchup Model 연구")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
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

    # 기본 v14 피처
    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    # Bradley-Terry 피처 생성
    print("\n[피처 생성] Bradley-Terry Empirical Bayes 피처 계산 중...")
    bt_train = make_temporal_bt_features(train_split, TARGET)
    bt_encoder_val = BradleyTerryEBEncoder().fit(train_split, train_split[TARGET])
    bt_val = bt_encoder_val.transform(val_split)
    bt_all = pd.concat([bt_train, bt_val], axis=0).loc[train_full.index]

    fe_bt = pd.concat([fe_v14, bt_all], axis=1)

    NEW_COLS_BT = [c for c in fe_bt.columns if c not in train_full.columns]
    ALL_FEATURES_BT = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_BT

    for c in CAT_COLS:
        fe_bt[c] = fe_bt[c].astype("category")

    is_val_mask = fe_bt["season"] == 2024
    X_train_full = fe_bt.loc[~is_val_mask, ALL_FEATURES_BT]
    y_train_full = fe_bt.loc[~is_val_mask, TARGET].to_numpy()
    X_val_fe = fe_bt.loc[is_val_mask, ALL_FEATURES_BT]
    y_val = fe_bt.loc[is_val_mask, TARGET].to_numpy()

    X_train_cb = X_train_full.copy()
    X_val_cb = X_val_fe.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    cat_idx = [ALL_FEATURES_BT.index(c) for c in CAT_COLS]

    print(f"피처 구성 완료: BT 피처 5개 추가 (총 {len(ALL_FEATURES_BT)}개 피처)")

    # 1. Base Models
    print("\n[모델 훈련] LightGBM 및 CatBoost 학습 중...")
    lgb_m = lgb.LGBMClassifier(
        n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary",
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_m.fit(X_train_full, y_train_full)
    p_lgb = lgb_m.predict_proba(X_val_fe)[:, 1]

    cb_m = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    cb_m.fit(Pool(X_train_cb, y_train_full, cat_features=cat_idx))
    p_cb = cb_m.predict_proba(X_val_cb)[:, 1]

    # 2. 5-Fold OOF
    print("[OOF 생성] 5-Fold CV OOF 생성 중...")
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

    # 3. Meta-Learner
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([oof_lgb, oof_cb]), y_train_full)
    w_lgb = stack.coef_[0][0]
    w_cb = stack.coef_[0][1]

    p_val_stack = stack.predict_proba(np.column_stack([p_lgb, p_cb]))[:, 1]
    p_oof_stack = stack.predict_proba(np.column_stack([oof_lgb, oof_cb]))[:, 1]

    score_val = brier_skill_score(y_val, p_val_stack)
    score_oof = brier_skill_score(y_train_full, p_oof_stack)

    print("\n" + "=" * 80)
    print("🏁 [Step 1] Bradley-Terry Empirical Bayes 검증 결과")
    print("=" * 80)
    print(f"v14 기준선 (Without BT) : Holdout = 836.35 | OOF = 2009.23 | LGB=3.49, CB=0.99")
    print(f"BT 적용 앙상블 (With BT)   : Holdout = {score_val:.2f} ({score_val-836.35:+.2f}) | OOF = {score_oof:.2f} ({score_oof-2009.23:+.2f}) | LGB={w_lgb:.2f}, CB={w_cb:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
