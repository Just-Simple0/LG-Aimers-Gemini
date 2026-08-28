"""
Paradigm 2: Pitcher-Batter Latent Matchup Style Embeddings via SVD
===================================================================

연구 목표:
  투수와 타자의 과거 맞대결 이력 행렬(Pitcher x Batter Matrix)을
  TruncatedSVD (Rank 8)로 행렬 분해하여,
  1) 투수 잠재 스타일 벡터 U_p in R^8
  2) 타자 잠재 스타일 벡터 V_b in R^8
  3) 상성 내적 점수 dot(U_p, V_b) in R
  를 시계열 누수 없이(Temporal Freeze) 추출하고 모델 성능을 검증.

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


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


class LatentMatchupSVDEncoder:
    """투수-타자 상성 희소 행렬 SVD 분해 인코더 (Temporal Safe)."""

    def __init__(self, n_components=8, alpha=30.0, random_state=42):
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state
        self.pitcher_map = {}
        self.batter_map = {}
        self.pitcher_factors = None
        self.batter_factors = None
        self.global_prior = 0.5

    def fit(self, df, target):
        df = df.copy()
        df["__target"] = np.asarray(target, dtype=float)
        self.global_prior = float(df["__target"].mean())

        pitchers = sorted(df["pitcher_id"].dropna().unique())
        batters = sorted(df["batter_id"].dropna().unique())

        self.pitcher_map = {pid: i for i, pid in enumerate(pitchers)}
        self.batter_map = {bid: j for j, bid in enumerate(batters)}

        n_p = len(pitchers)
        n_b = len(batters)

        if n_p == 0 or n_b == 0 or len(df) < 100:
            self.pitcher_factors = np.zeros((n_p, self.n_components))
            self.batter_factors = np.zeros((n_b, self.n_components))
            return self

        # 투수 x 타자 집계
        agg = (
            df.groupby(["pitcher_id", "batter_id"])["__target"]
            .agg(n="count", s="sum")
            .reset_index()
        )

        p_idx = agg["pitcher_id"].map(self.pitcher_map).to_numpy()
        b_idx = agg["batter_id"].map(self.batter_map).to_numpy()

        smoothed_val = (agg["s"] + self.alpha * self.global_prior) / (agg["n"] + self.alpha) - self.global_prior
        data = smoothed_val.to_numpy(dtype=float)

        mat = csr_matrix((data, (p_idx, b_idx)), shape=(n_p, n_b))

        k = min(self.n_components, min(n_p, n_b) - 1)
        if k < 2:
            self.pitcher_factors = np.zeros((n_p, self.n_components))
            self.batter_factors = np.zeros((n_b, self.n_components))
            return self

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        U = svd.fit_transform(mat)  # (n_p, k)
        V = svd.components_.T       # (n_b, k)

        # 정규화
        self.pitcher_factors = np.zeros((n_p, self.n_components))
        self.batter_factors = np.zeros((n_b, self.n_components))
        self.pitcher_factors[:, :k] = U
        self.batter_factors[:, :k] = V

        return self

    def transform(self, df):
        p_ids = df["pitcher_id"].to_numpy()
        b_ids = df["batter_id"].to_numpy()

        dots = np.zeros(len(df), dtype=float)
        p_f1 = np.zeros(len(df), dtype=float)
        p_f2 = np.zeros(len(df), dtype=float)
        b_f1 = np.zeros(len(df), dtype=float)
        b_f2 = np.zeros(len(df), dtype=float)

        for i in range(len(df)):
            pid = p_ids[i]
            bid = b_ids[i]
            p_idx = self.pitcher_map.get(pid)
            b_idx = self.batter_map.get(bid)

            if p_idx is not None and b_idx is not None:
                u = self.pitcher_factors[p_idx]
                v = self.batter_factors[b_idx]
                dots[i] = np.dot(u, v)
                p_f1[i] = u[0]
                p_f2[i] = u[1]
                b_f1[i] = v[0]
                b_f2[i] = v[1]
            elif p_idx is not None:
                u = self.pitcher_factors[p_idx]
                p_f1[i] = u[0]
                p_f2[i] = u[1]
            elif b_idx is not None:
                v = self.batter_factors[b_idx]
                b_f1[i] = v[0]
                b_f2[i] = v[1]

        res = pd.DataFrame({
            "svd_matchup_dot": dots,
            "svd_pitcher_f1": p_f1,
            "svd_pitcher_f2": p_f2,
            "svd_batter_f1": b_f1,
            "svd_batter_f2": b_f2,
        }, index=df.index)
        return res


def make_temporal_svd_features(train_df, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = LatentMatchupSVDEncoder()
            dummy_target = pd.Series([0.5], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = LatentMatchupSVDEncoder().fit(
                train_df.loc[past_mask], train_df.loc[past_mask, target_col]
            )
            chunks.append(encoder.transform(train_df.loc[s_mask]))

    return pd.concat(chunks, axis=0).loc[train_df.index]


def main():
    print("=" * 80)
    print("🚀 [Paradigm 2] Pitcher-Batter SVD Latent Matchup Embeddings 연구")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)

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

    # SVD 피처 생성
    print("\n[피처 생성] Pitcher-Batter SVD 잠재 상성 피처 계산 중...")
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
    X_train_full = fe_svd.loc[~is_val_mask, ALL_FEATURES_SVD]
    y_train_full = fe_svd.loc[~is_val_mask, TARGET].to_numpy()
    X_val_fe = fe_svd.loc[is_val_mask, ALL_FEATURES_SVD]
    y_val = fe_svd.loc[is_val_mask, TARGET].to_numpy()

    X_train_cb = X_train_full.copy()
    X_val_cb = X_val_fe.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    cat_idx = [ALL_FEATURES_SVD.index(c) for c in CAT_COLS]

    print(f"피처 구성 완료: SVD 피처 5개 추가 (총 {len(ALL_FEATURES_SVD)}개 피처)")

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
    print("🏁 [Paradigm 2] SVD Latent Matchup Embeddings 검증 결과")
    print("=" * 80)
    print(f"v14 기준선 (Without SVD) : Holdout = 836.35 | OOF = 2009.23 | LGB=3.49, CB=0.99")
    print(f"SVD 적용 앙상블 (With SVD) : Holdout = {score_val:.2f} ({score_val-836.35:+.2f}) | OOF = {score_oof:.2f} ({score_oof-2009.23:+.2f}) | LGB={w_lgb:.2f}, CB={w_cb:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
