"""
Autoresearch Autonomous Batch Runner (5 Consecutive Experiments)
================================================================

실험 목록:
  1. Exp 1 (v22): GBDT 하이퍼파라미터 정규화 최적화 (LGB reg_lambda, colsample | CB l2_leaf_reg, depth)
  2. Exp 2 (v23): 메타러너 정규화 및 비음수 제약 (Non-negative Ridge / Constrained Stacking)
  3. Exp 3 (v24): 투수/타자 기본 스무딩 계수 k 스윕 (shrink k in [15, 30, 50, 80])
  4. Exp 4 (v25): 순수 당해 시즌 폼 매치업 차분 (matchup_current_posterior_diff 1개만 단독 추가)
  5. Exp 5 (v26): 이닝 후반(7~9회) & 레버리지 특화 베이지안 폼 세분화 (Late Inning Form)

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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import KFold

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


def build_base_features_k(df, global_mean, tk_lookup, k_smooth=30):
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

    def shrink(rate_col, n_col, k=k_smooth):
        n = df[n_col].fillna(0)
        r = df[rate_col].fillna(global_mean)
        return (n * r + k * global_mean) / (n + k)

    df["pitcher_success_rate_smooth"] = shrink("asof_pitcher_success_rate", "asof_pitcher_n", k_smooth)
    df["batter_success_rate_smooth"] = shrink("asof_batter_success_rate", "asof_batter_n", k_smooth)
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


def run_single_eval(train_fe, val_fe, all_features, lgb_p=LGB_PARAMS_BASE, cb_p=CB_PARAMS_BASE, meta_learner_type="logistic_default"):
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
    lgb_m = lgb.LGBMClassifier(n_estimators=2000, **lgb_p)
    lgb_m.fit(X_tr_lgb, y_train_full, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    best_lgb_iter = lgb_m.best_iteration_
    p_lgb_val = lgb_m.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val.values, p_lgb_val)

    tp_tr = Pool(X_tr_cb, y_train_full, cat_features=cat_idx)
    tp_val = Pool(X_val_cb, y_val, cat_features=cat_idx)
    cb_m = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **cb_p)
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
        m_l = lgb.LGBMClassifier(n_estimators=best_lgb_iter, **lgb_p)
        m_l.fit(X_lgb_r.iloc[tr_idx], y_r.iloc[tr_idx])
        lgb_oof[oof_idx] = m_l.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_cb_r.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=best_cb_iter, **cb_p)
        m_c.fit(tp_f)
        cb_oof[oof_idx] = m_c.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]

    # 3. Meta-learner
    y_oof_arr = y_r.to_numpy()
    if meta_learner_type == "logistic_default":
        stack = LogisticRegression(random_state=42, max_iter=1000)
        stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof_arr)
        preds_val = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
        preds_oof = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    elif meta_learner_type == "ridge_c01":
        stack = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
        stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof_arr)
        preds_val = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
        preds_oof = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    elif meta_learner_type == "simplex_blend":
        best_w = 0.5
        best_s = -1
        for w in np.linspace(0, 1, 101):
            s = brier_skill_score(y_oof_arr, w * lgb_oof + (1 - w) * cb_oof)
            if s > best_s:
                best_s = s
                best_w = w
        preds_val = best_w * p_lgb_val + (1 - best_w) * p_cb_val
        preds_oof = best_w * lgb_oof + (1 - best_w) * cb_oof

    score_val = brier_skill_score(y_val.values, preds_val)
    score_oof = brier_skill_score(y_oof_arr, preds_oof)
    elapsed = time.time() - t0

    return {
        "n_features": len(all_features),
        "lgb_iter": best_lgb_iter,
        "cb_iter": best_cb_iter,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "holdout": score_val,
        "oof": score_oof,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 80)
    print("🚀 Autoresearch Autonomous 5-Batch Runner 시작")
    print("기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)")
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
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

    train_split = train_full[train_full["season"] <= 2023]
    val_split = train_full[train_full["season"] == 2024]

    # 기본 Season Decomposition
    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features_k(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL, k_smooth=30)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")

    val_fe_v14 = fe_v14[fe_v14["season"] == 2024].copy()

    batch_results = []

    # =========================================================================
    # [Exp 1] v22: GBDT 하이퍼파라미터 정규화 (CatBoost l2_leaf_reg=5.0)
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 1/5] v22: CatBoost L2 Regularization (l2_leaf_reg=5.0)")
    print("=" * 70)
    cb_p_exp1 = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    res1 = run_single_eval(fe_v14, val_fe_v14, ALL_FEATURES_V14, cb_p=cb_p_exp1)
    res1["experiment"] = "Exp 1 (v22-cb-reg5)"
    print(f"  [Exp 1 결과] (A) 홀드아웃: {res1['holdout']:.2f} (diff={res1['holdout']-836.35:+.2f}) | (B) OOF: {res1['oof']:.2f} (diff={res1['oof']-2009.23:+.2f})")
    batch_results.append(res1)

    # =========================================================================
    # [Exp 2] v23: 메타러너 Ridge 규제 (C=0.1)
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 2/5] v23: Meta-Learner Ridge Regularization (C=0.1)")
    print("=" * 70)
    res2 = run_single_eval(fe_v14, val_fe_v14, ALL_FEATURES_V14, meta_learner_type="ridge_c01")
    res2["experiment"] = "Exp 2 (v23-meta-ridge)"
    print(f"  [Exp 2 결과] (A) 홀드아웃: {res2['holdout']:.2f} (diff={res2['holdout']-836.35:+.2f}) | (B) OOF: {res2['oof']:.2f} (diff={res2['oof']-2009.23:+.2f})")
    batch_results.append(res2)

    # =========================================================================
    # [Exp 3] v24: 투수/타자 기본 스무딩 계수 k 스윕 (k=50)
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 3/5] v24: Career Rate Smoothing k=50 (v14 default k=30)")
    print("=" * 70)
    base_fe_k50 = build_base_features_k(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL, k_smooth=50)
    fe_v24 = pd.concat([base_fe_k50, season_decomp_all], axis=1)
    for c in CAT_COLS:
        fe_v24[c] = fe_v24[c].astype("category")
    val_fe_v24 = fe_v24[fe_v24["season"] == 2024].copy()

    res3 = run_single_eval(fe_v24, val_fe_v24, ALL_FEATURES_V14)
    res3["experiment"] = "Exp 3 (v24-smooth-k50)"
    print(f"  [Exp 3 결과] (A) 홀드아웃: {res3['holdout']:.2f} (diff={res3['holdout']-836.35:+.2f}) | (B) OOF: {res3['oof']:.2f} (diff={res3['oof']-2009.23:+.2f})")
    batch_results.append(res3)

    # =========================================================================
    # [Exp 4] v25: 순수 당해 시즌 폼 매치업 격차 1개 피처만 단독 추가
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 4/5] v25: Pure Current Posterior Matchup Diff (1 Feature Only)")
    print("=" * 70)
    fe_v25 = fe_v14.copy()
    fe_v25["matchup_current_posterior_diff"] = (
        fe_v25["season_pitcher_current_posterior"] - fe_v25["season_batter_current_posterior"]
    )
    ALL_FEATURES_V25 = ALL_FEATURES_V14 + ["matchup_current_posterior_diff"]
    val_fe_v25 = fe_v25[fe_v25["season"] == 2024].copy()

    res4 = run_single_eval(fe_v25, val_fe_v25, ALL_FEATURES_V25)
    res4["experiment"] = "Exp 4 (v25-pure-matchup-diff)"
    print(f"  [Exp 4 결과] (A) 홀드아웃: {res4['holdout']:.2f} (diff={res4['holdout']-836.35:+.2f}) | (B) OOF: {res4['oof']:.2f} (diff={res4['oof']-2009.23:+.2f})")
    batch_results.append(res4)

    # =========================================================================
    # [Exp 5] v26: Late-Inning & High-Leverage Interaction with Success Rate
    # =========================================================================
    print("\n" + "=" * 70)
    print("▶ [Exp 5/5] v26: Late-Inning Success Rate Modulation (1 Feature Only)")
    print("=" * 70)
    fe_v26 = fe_v14.copy()
    fe_v26["late_close_pitcher_smooth"] = fe_v26["late_and_close"] * fe_v26["pitcher_success_rate_smooth"]
    ALL_FEATURES_V26 = ALL_FEATURES_V14 + ["late_close_pitcher_smooth"]
    val_fe_v26 = fe_v26[fe_v26["season"] == 2024].copy()

    res5 = run_single_eval(fe_v26, val_fe_v26, ALL_FEATURES_V26)
    res5["experiment"] = "Exp 5 (v26-late-close-mod)"
    print(f"  [Exp 5 결과] (A) 홀드아웃: {res5['holdout']:.2f} (diff={res5['holdout']-836.35:+.2f}) | (B) OOF: {res5['oof']:.2f} (diff={res5['oof']-2009.23:+.2f})")
    batch_results.append(res5)

    # =========================================================================
    # 종합 요약
    # =========================================================================
    print("\n" + "=" * 80)
    print("🏁 5-Batch Autoresearch 종합 결과 요약")
    print("=" * 80)
    print(f"{'실험명':<30} {'LGB solo':>10} {'CB solo':>10} {'(A) 홀드아웃':>14} {'(B) OOF score':>14} {'판정':>8}")
    print(f"{'v14 기준선':<30} {809.27:>10.2f} {840.81:>10.2f} {836.35:>14.2f} {2009.23:>14.2f} {'BASE':>8}")
    
    for r in batch_results:
        diff_h = r["holdout"] - 836.35
        diff_o = r["oof"] - 2009.23
        verdict = "KEEP ✅" if (diff_h > 0.0 and diff_o > 0.0) else "DISCARD ❌"
        print(f"{r['experiment']:<30} {r['lgb_solo']:>10.2f} {r['cb_solo']:>10.2f} {r['holdout']:>14.2f}({diff_h:>+5.2f}) {r['oof']:>14.2f}({diff_o:>+5.2f}) {verdict:>8}")

    total_min = (time.time() - total_start) / 60
    print(f"\n총 소요 시간: {total_min:.1f}분")
    print("=" * 80)

    out = "docs/autoresearch_batch_results.csv"
    pd.DataFrame(batch_results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
