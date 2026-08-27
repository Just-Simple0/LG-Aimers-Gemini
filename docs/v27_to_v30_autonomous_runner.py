"""
Autoresearch Autonomous Runner (v27 ~ v30)
=========================================

세이버메트릭스 및 머신러닝 최신 연구 기반 4대 후보 자동 실험:
  1. [Exp 1] v27: Sabermetric Location+ 최근 경기 가중 모멘텀 및 제구 안정성/실투 추세 (3개 피처)
  2. [Exp 2] v28: Empirical Bayes 계층적 팀 베이스라인 (신인/소표본 투수 팀 사전확률)
  3. [Exp 3] v29: Platoon Split Handedness-Conditioned Trackman 편차
  4. [Exp 4] v30: Brier Score (p-y)^2 직접 최소화 SLSQP 유계 메타러너

각 실험마다 [5대 하드 가드레일]을 통과하는지 검증합니다.

작성일: 2026-08-27
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
from scipy.optimize import minimize

from src.submission_guardrails import (
    inspect_meta_learner_weights,
    inspect_prediction_distribution,
    inspect_dual_gates,
    SubmissionGuardrailError,
)

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

ENTITY_SPECS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
}

LGB_PARAMS_BASE = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS_BASE = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)


class SeasonDecompositionEncoder:
    def __init__(self, entities=("pitcher", "batter"), alpha=ALPHA, prior=None, history_tables=None):
        self.entities = list(entities)
        self.alpha = float(alpha)
        self.prior = prior
        self.history_tables = history_tables or {}

    def fit(self, df, target):
        values = np.asarray(target, dtype=float).reshape(-1)
        self.prior = float(values.mean())
        self.history_tables = {}
        for entity in self.entities:
            id_col, _, _ = ENTITY_SPECS[entity]
            working = pd.DataFrame({
                id_col: df[id_col].to_numpy(copy=False),
                "__target": values,
            })
            table = (
                working.dropna(subset=[id_col])
                .groupby(id_col, sort=False)["__target"]
                .agg(history_success="sum", history_n="count")
            )
            self.history_tables[entity] = table
        return self

    def transform(self, df):
        derived = {}
        for entity in self.entities:
            id_col, count_col, rate_col = ENTITY_SPECS[entity]
            table = self.history_tables[entity]

            h_n_series = df[id_col].map(table["history_n"])
            h_s_series = df[id_col].map(table["history_success"])
            history_n = h_n_series.fillna(0.0).to_numpy(dtype=float)
            history_success = h_s_series.fillna(0.0).to_numpy(dtype=float)

            history_rate = np.divide(
                history_success, history_n,
                out=np.full(len(df), self.prior, dtype=float),
                where=history_n > 0.0,
            )

            raw_n = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            cumulative_n = np.clip(np.rint(raw_n), 0.0, None)

            raw_rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(self.prior).to_numpy(dtype=float)
            raw_rate = np.clip(raw_rate, 0.0, 1.0)
            cumulative_success = np.rint(cumulative_n * raw_rate)

            history_n_adj = np.minimum(history_n, cumulative_n)
            history_success_adj = np.minimum(history_success, cumulative_success)

            current_n = np.maximum(0.0, cumulative_n - history_n_adj)
            current_success = np.clip(cumulative_success - history_success_adj, 0.0, current_n)

            current_posterior = (current_success + self.alpha * history_rate) / (current_n + self.alpha)
            current_history_gap = current_posterior - history_rate

            prefix = f"season_{entity}_"
            derived[prefix + "history_n"] = history_n
            derived[prefix + "history_rate"] = history_rate
            derived[prefix + "history_confidence"] = history_n / (history_n + self.alpha)
            derived[prefix + "current_n"] = current_n
            derived[prefix + "current_posterior"] = current_posterior
            derived[prefix + "current_history_gap"] = current_history_gap
            derived[prefix + "current_confidence"] = current_n / (current_n + self.alpha)
            derived[prefix + "current_cold_start"] = (current_n == 0.0).astype(np.int8)

        return pd.DataFrame(derived, index=df.index)


def make_temporal_season_features(train_df, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = SeasonDecompositionEncoder()
            dummy_target = pd.Series([0.5], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            encoder.history_tables = {
                e: pd.DataFrame({"history_n": pd.Series(dtype=float), "history_success": pd.Series(dtype=float)})
                for e in ("pitcher", "batter")
            }
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = SeasonDecompositionEncoder().fit(
                train_df.loc[past_mask], train_df.loc[past_mask, target_col]
            )
            chunks.append(encoder.transform(train_df.loc[s_mask]))

    return pd.concat(chunks, axis=0).loc[train_df.index]


def make_tk_lookup(tk_df):
    lookup = tk_df.groupby(TK_KEYS).agg(
        tk_fastball_rate=("pitch_type_group", lambda s: (s == "fastball").mean()),
        tk_breaking_rate=("pitch_type_group", lambda s: (s == "breaking").mean()),
        tk_offspeed_rate=("pitch_type_group", lambda s: (s == "offspeed").mean()),
        tk_zone_speed_mean=("zone_speed", "mean"),
        tk_rel_speed_mean=("rel_speed", "mean"),
        tk_spin_rate_mean=("spin_rate", "mean"),
        tk_horz_break_std=("horz_break", "std"),
        tk_vert_break_std=("induced_vert_break", "std"),
        tk_extension_mean=("extension", "mean"),
    ).reset_index()
    return lookup


def build_base_features_v14(df, global_mean, tk_lookup):
    df = df.copy()

    df["is_two_strike"] = (df["strikes_before"] >= 2).astype(int)
    df["is_three_ball"] = (df["balls_before"] >= 3).astype(int)
    df["is_full_count"] = ((df["balls_before"] >= 3) & (df["strikes_before"] >= 2)).astype(int)
    df["count_diff"] = df["strikes_before"] - df["balls_before"]
    df["count_total"] = df["strikes_before"] + df["balls_before"]

    df["win_exp_diff"] = df["home_win_expectancy"] - df["away_win_expectancy"]
    df["abs_score_diff_pitcher"] = df["score_diff_pitcher_team"].abs()
    df["late_and_close"] = ((df["inning"] >= 8) & (df["abs_score_diff_pitcher"] <= 1)).astype(int)

    df["is_high_leverage"] = (df["li"] >= 1.5).astype(int)
    df["li_count_diff"] = df["li"] * df["count_diff"]
    df["li_late_close"] = df["li"] * df["late_and_close"]

    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    df["pitcher_cold_start"] = df["asof_pitcher_n"].fillna(0).eq(0).astype(int)
    df["batter_cold_start"] = df["asof_batter_n"].fillna(0).eq(0).astype(int)
    df["pitchmix_cold_start"] = df["asof_pitcher_pitchmix_n"].fillna(0).eq(0).astype(int)

    def shrink(rate_col, n_col, k=30):
        n = df[n_col].fillna(0)
        r = df[rate_col].fillna(global_mean)
        return (n * r + k * global_mean) / (n + k)

    df["pitcher_success_rate_smooth"] = shrink("asof_pitcher_success_rate", "asof_pitcher_n")
    df["batter_success_rate_smooth"] = shrink("asof_batter_success_rate", "asof_batter_n")
    df["matchup_success_diff"] = df["pitcher_success_rate_smooth"] - df["batter_success_rate_smooth"]
    df["matchup_middle_diff"] = (
        df["asof_pitcher_middle_rate"].fillna(global_mean)
        - df["asof_batter_middle_rate"].fillna(global_mean)
    )

    df["pitcher_recent_trend"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    df["pitcher_recent_trend3"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]

    fb = df["asof_pitcher_fastball_rate"].fillna(0)
    br = df["asof_pitcher_breaking_rate"].fillna(0)
    os_ = df["asof_pitcher_offspeed_rate"].fillna(0)
    df["pitchmix_max_share"] = np.maximum.reduce([fb, br, os_])

    df["month_sin"] = np.sin(2 * np.pi * df["game_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["game_month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["game_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["game_dayofweek"] / 7)

    tk_cols = [c for c in tk_lookup.columns if c.startswith("tk_")]
    tk_lk = tk_lookup[TK_KEYS + tk_cols].copy()
    for k in TK_KEYS:
        tk_lk[k] = tk_lk[k].astype(df[k].dtype)
    orig_index = df.index
    df = df.merge(tk_lk, on=TK_KEYS, how="left", validate="many_to_one", sort=False)
    df.index = orig_index
    df["tk_fastball_dev"] = fb - df["tk_fastball_rate"]
    df["tk_breaking_dev"] = br - df["tk_breaking_rate"]
    df["tk_offspeed_dev"] = os_ - df["tk_offspeed_rate"]

    return df.drop(columns=[ID_COL], errors="ignore")


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def run_pipeline_and_evaluate(train_fe, val_fe, all_features, meta_mode="logistic"):
    t0 = time.time()
    is_val_mask = train_fe["season"] == 2024
    X_train_full = train_fe.loc[~is_val_mask, all_features]
    y_train_full = train_fe.loc[~is_val_mask, TARGET]
    X_val_fe = val_fe[all_features]
    y_val = val_fe[TARGET]

    cat_idx = [all_features.index(c) for c in CAT_COLS if c in all_features]

    def lgb_df(df):
        d = df.copy()
        for c in CAT_COLS:
            if c in d.columns:
                d[c] = d[c].astype("category")
        return d

    def cb_df(df):
        d = df.copy()
        for c in CAT_COLS:
            if c in d.columns:
                d[c] = d[c].astype(str)
        return d

    X_tr_lgb = lgb_df(X_train_full)
    X_tr_cb = cb_df(X_train_full)
    X_val_lgb = lgb_df(X_val_fe)
    X_val_cb = cb_df(X_val_fe)

    # 1. Early stopping for best iters
    lgb_m = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS_BASE)
    lgb_m.fit(X_tr_lgb, y_train_full, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    best_lgb_iter = lgb_m.best_iteration_
    p_lgb_val = lgb_m.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val.values, p_lgb_val)

    tp_tr = Pool(X_tr_cb, y_train_full, cat_features=cat_idx)
    tp_val = Pool(X_val_cb, y_val, cat_features=cat_idx)
    cb_m = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS_BASE)
    cb_m.fit(tp_tr, eval_set=tp_val, use_best_model=True)
    best_cb_iter = cb_m.get_best_iteration() + 1
    p_cb_val = cb_m.predict_proba(X_val_cb)[:, 1]
    score_cb = brier_skill_score(y_val.values, p_cb_val)

    # 2. 5-Fold OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_lgb_r = X_tr_lgb.reset_index(drop=True)
    X_cb_r = X_tr_cb.reset_index(drop=True)
    y_r = y_train_full.reset_index(drop=True)

    lgb_oof = np.zeros(len(y_r))
    cb_oof = np.zeros(len(y_r))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_lgb_r)):
        m_l = lgb.LGBMClassifier(n_estimators=best_lgb_iter, **LGB_PARAMS_BASE)
        m_l.fit(X_lgb_r.iloc[tr_idx], y_r.iloc[tr_idx])
        lgb_oof[oof_idx] = m_l.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_cb_r.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=best_cb_iter, **CB_PARAMS_BASE)
        m_c.fit(tp_f)
        cb_oof[oof_idx] = m_c.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]

    y_oof_arr = y_r.to_numpy()

    # 3. Meta-learner
    if meta_mode == "logistic":
        stack = LogisticRegression(random_state=42, max_iter=1000)
        stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof_arr)
        w_lgb = stack.coef_[0][0]
        w_cb = stack.coef_[0][1]
        intercept = stack.intercept_[0]
        preds_val = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
        preds_oof = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    elif meta_mode == "brier_slsqp":
        # Direct Brier Score minimization with bounds: w1 in [1.0, 3.5], w2 in [0.3, 2.0]
        def brier_obj(weights):
            w1, w2, b = weights
            logit = w1 * lgb_oof + w2 * cb_oof + b
            p = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
            return np.mean((p - y_oof_arr) ** 2)

        init_w = [3.0, 1.0, -2.1]
        bounds = [(1.0, 3.5), (0.3, 2.0), (-3.0, -1.0)]
        res_opt = minimize(brier_obj, init_w, method="SLSQP", bounds=bounds)
        w_lgb, w_cb, intercept = res_opt.x

        def predict_slsqp(p1, p2):
            logit = w_lgb * p1 + w_cb * p2 + intercept
            return 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

        preds_val = predict_slsqp(p_lgb_val, p_cb_val)
        preds_oof = predict_slsqp(lgb_oof, cb_oof)

    score_val = brier_skill_score(y_val.values, preds_val)
    score_oof = brier_skill_score(y_oof_arr, preds_oof)
    elapsed = time.time() - t0

    # 4. Guardrail inspection
    guardrail_status = "PASS ✅"
    guardrail_msg = ""
    try:
        inspect_meta_learner_weights(w_lgb, w_cb, intercept)
        inspect_prediction_distribution(preds_val)
        inspect_dual_gates(score_val - 836.35, score_oof - 2009.23)
    except SubmissionGuardrailError as e:
        guardrail_status = "FAIL ❌"
        guardrail_msg = str(e)

    return {
        "n_features": len(all_features),
        "lgb_iter": best_lgb_iter,
        "cb_iter": best_cb_iter,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "w_lgb": w_lgb,
        "w_cb": w_cb,
        "intercept": intercept,
        "val_min": float(np.min(preds_val)),
        "val_max": float(np.max(preds_val)),
        "val_std": float(np.std(preds_val)),
        "holdout": score_val,
        "oof": score_oof,
        "diff_holdout": score_val - 836.35,
        "diff_oof": score_oof - 2009.23,
        "guardrail_status": guardrail_status,
        "guardrail_msg": guardrail_msg,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 80)
    print("🚀 Autoresearch Autonomous Runner (v27 ~ v30)")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)
    total_start = time.time()

    # 데이터 로드
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_VAL = train_full.loc[train_full["season"] != 2024, TARGET].mean()

    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed", "pitcher_hand", "batter_hand"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

    train_split = train_full[train_full["season"] <= 2023]
    val_split = train_full[train_full["season"] == 2024]

    # 기본 v14 Season Decomposition
    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe_v14 = build_base_features_v14(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe_v14, season_decomp_all], axis=1)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")

    val_fe_v14 = fe_v14[fe_v14["season"] == 2024].copy()

    batch_results = []

    # =========================================================================
    # [Exp 1] v27: Location+ 최근 경기 가중 모멘텀 & 제구 변동성 & 실투 추세 (3개 피처)
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 1/4] v27: Location+ Style Recency Momentum & Stability")
    print("=" * 70)
    fe_v27 = fe_v14.copy()
    p1 = fe_v27["asof_pitcher_prev1_game_success_rate"].fillna(GLOBAL_MEAN_VAL)
    p3 = fe_v27["asof_pitcher_prev3_game_success_rate"].fillna(GLOBAL_MEAN_VAL)
    p5 = fe_v27["asof_pitcher_prev5_game_success_rate"].fillna(GLOBAL_MEAN_VAL)
    career_succ = fe_v27["asof_pitcher_success_rate"].fillna(GLOBAL_MEAN_VAL)

    # 1. 지수 가중 최근 모멘텀 vs 커리어 평균
    fe_v27["pitcher_recency_momentum"] = (0.5 * p1 + 0.3 * p3 + 0.2 * p5) - career_succ
    # 2. 최근 등판 제구 안정성/변동성 (절댓값 차분)
    fe_v27["pitcher_control_stability"] = (p1 - p3).abs()
    # 3. 최근 실투율 증가 추세
    m1 = fe_v27["asof_pitcher_prev1_game_middle_rate"].fillna(GLOBAL_MEAN_VAL)
    m5 = fe_v27["asof_pitcher_prev5_game_middle_rate"].fillna(GLOBAL_MEAN_VAL)
    fe_v27["pitcher_recent_danger_trend"] = m1 - m5

    NEW_V27 = ["pitcher_recency_momentum", "pitcher_control_stability", "pitcher_recent_danger_trend"]
    ALL_FEATURES_V27 = ALL_FEATURES_V14 + NEW_V27
    val_fe_v27 = fe_v27[fe_v27["season"] == 2024].copy()

    res1 = run_pipeline_and_evaluate(fe_v27, val_fe_v27, ALL_FEATURES_V27)
    res1["experiment"] = "Exp 1 (v27-recency-momentum)"
    batch_results.append(res1)

    # =========================================================================
    # [Exp 2] v28: Empirical Bayes 계층적 팀 베이스라인 (신인/소표본 투수 팀 사전확률)
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 2/4] v28: Hierarchical Team Pitching Prior for Cold Starts")
    print("=" * 70)
    fe_v28 = fe_v14.copy()
    # 과거 시즌 기준 팀별 투수 제구 성공률 산출
    team_priors = train_split.groupby("pitcher_team_id")[TARGET].mean().to_dict()
    fe_v28["pitcher_team_control_prior"] = fe_v28["pitcher_team_id"].map(team_priors).fillna(GLOBAL_MEAN_VAL)
    fe_v28["pitcher_vs_team_prior_diff"] = fe_v28["pitcher_success_rate_smooth"] - fe_v28["pitcher_team_control_prior"]

    NEW_V28 = ["pitcher_team_control_prior", "pitcher_vs_team_prior_diff"]
    ALL_FEATURES_V28 = ALL_FEATURES_V14 + NEW_V28
    val_fe_v28 = fe_v28[fe_v28["season"] == 2024].copy()

    res2 = run_pipeline_and_evaluate(fe_v28, val_fe_v28, ALL_FEATURES_V28)
    res2["experiment"] = "Exp 2 (v28-hierarchical-team-prior)"
    batch_results.append(res2)

    # =========================================================================
    # [Exp 3] v29: Platoon Handedness-Conditioned Trackman Physics
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 3/4] v29: Platoon Split Handedness-Conditioned Trackman Lookup")
    print("=" * 70)
    PLATOON_KEYS = TK_KEYS + ["same_hand"]
    tk_split_raw = tk_raw[tk_raw["season"] <= 2023].copy()
    tk_split_raw["same_hand"] = (tk_split_raw["pitcher_hand"] == tk_split_raw["batter_hand"]).astype(int)
    platoon_lookup = tk_split_raw.groupby(PLATOON_KEYS).agg(
        tk_platoon_fastball_rate=("pitch_type_group", lambda s: (s == "fastball").mean()),
        tk_platoon_zone_speed_mean=("zone_speed", "mean"),
    ).reset_index()

    fe_v29 = fe_v14.copy()
    orig_idx = fe_v29.index
    fe_v29 = fe_v29.merge(platoon_lookup, on=PLATOON_KEYS, how="left", validate="many_to_one", sort=False)
    fe_v29.index = orig_idx
    fe_v29["tk_platoon_fastball_rate"] = fe_v29["tk_platoon_fastball_rate"].fillna(fe_v29["tk_fastball_rate"])
    fe_v29["tk_platoon_zone_speed_mean"] = fe_v29["tk_platoon_zone_speed_mean"].fillna(fe_v29["tk_zone_speed_mean"])
    fe_v29["tk_platoon_fastball_dev"] = fe_v29["asof_pitcher_fastball_rate"].fillna(0) - fe_v29["tk_platoon_fastball_rate"]

    NEW_V29 = ["tk_platoon_fastball_rate", "tk_platoon_fastball_dev"]
    ALL_FEATURES_V29 = ALL_FEATURES_V14 + NEW_V29
    val_fe_v29 = fe_v29[fe_v29["season"] == 2024].copy()

    res3 = run_pipeline_and_evaluate(fe_v29, val_fe_v29, ALL_FEATURES_V29)
    res3["experiment"] = "Exp 3 (v29-platoon-trackman-dev)"
    batch_results.append(res3)

    # =========================================================================
    # [Exp 4] v30: Brier Score (p-y)^2 직접 최소화 SLSQP 유계 메타러너
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 4/4] v30: Brier Score Direct Minimization SLSQP Meta-Learner")
    print("=" * 70)
    res4 = run_pipeline_and_evaluate(fe_v14, val_fe_v14, ALL_FEATURES_V14, meta_mode="brier_slsqp")
    res4["experiment"] = "Exp 4 (v30-brier-slsqp-metalearner)"
    batch_results.append(res4)

    # =========================================================================
    # 종합 요약
    # =========================================================================
    print("\n" + "=" * 90)
    print("🏁 4-Batch Autoresearch 종합 결과 및 5대 가드레일 판정")
    print("=" * 90)
    print(f"{'실험명':<33} {'(A)홀드아웃':>11} {'(B)OOF':>11} {'LGB coef':>9} {'CB coef':>9} {'[Min, Max]':>15} {'가드레일':>9}")
    print(f"{'v14 기준선':<33} {836.35:>11.2f} {2009.23:>11.2f} {3.49:>9.2f} {0.99:>9.2f} {'[0.327, 0.752]':>15} {'PASS ✅':>9}")

    for r in batch_results:
        bounds_str = f"[{r['val_min']:.3f}, {r['val_max']:.3f}]"
        print(f"{r['experiment']:<33} {r['holdout']:>7.2f}({r['diff_holdout']:>+4.1f}) {r['oof']:>7.2f}({r['diff_oof']:>+4.1f}) {r['w_lgb']:>9.2f} {r['w_cb']:>9.2f} {bounds_str:>15} {r['guardrail_status']:>9}")

    total_min = (time.time() - total_start) / 60
    print(f"\n총 소요 시간: {total_min:.1f}분")
    print("=" * 90)

    out = "docs/v27_to_v30_batch_results.csv"
    pd.DataFrame(batch_results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
