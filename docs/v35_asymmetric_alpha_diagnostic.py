"""
v35 후보 실험: 세이버메트릭스 표본 신뢰도 기반 비대칭 시즌 분해 (Asymmetric Alpha Decomposition)
=============================================================================================

배경 및 야구 통계학적 이론:
  - Russell Carleton(Baseball Prospectus)의 표본 크기 신뢰도(Sample Size Reliability) 연구:
    - 투수의 제구력(Walk rate / Strikeout rate)은 약 70~100 타자 상대 시 신뢰도 50%에 도달 (빠른 안정화).
    - 타자의 제구 반응/볼넷 유도율은 약 200~300 타석이 지나야 안정화 (느린 안정화, 높은 변동성).
  - 현재 v14 모델은 투수와 타자 모두 대칭적으로 alpha = 50.0을 사용함.
  - 가설:
    - 투수는 표본 안정화가 빠르므로 작은 수축 계수 (alpha_pitcher in [20, 30, 40])
    - 타자는 표본 노이즈가 크므로 큰 수축 계수 (alpha_batter in [60, 80, 100])
    를 적용하는 '비대칭 베이지안 수축(Asymmetric Empirical Bayes Shrinkage)'을 적용하면
    과거 커리어와 당해 시즌 폼의 최적 결합이 이루어져 홀드아웃 및 OOF가 동반 개선될 것임.

기준선: v14 대칭 alpha=50.0 (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)

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
    inspect_dual_gates,
    SubmissionGuardrailError,
)

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)

ENTITY_SPECS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
}


class AsymmetricSeasonDecompositionEncoder:
    def __init__(self, alpha_pitcher=50.0, alpha_batter=50.0, prior=None, history_tables=None):
        self.alphas = {"pitcher": float(alpha_pitcher), "batter": float(alpha_batter)}
        self.prior = prior
        self.history_tables = history_tables or {}

    def fit(self, df, target):
        values = np.asarray(target, dtype=float).reshape(-1)
        self.prior = float(values.mean())
        self.history_tables = {}
        for entity in ("pitcher", "batter"):
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
        for entity in ("pitcher", "batter"):
            id_col, count_col, rate_col = ENTITY_SPECS[entity]
            table = self.history_tables[entity]
            alpha = self.alphas[entity]

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

            current_posterior = (current_success + alpha * history_rate) / (current_n + alpha)
            current_history_gap = current_posterior - history_rate

            prefix = f"season_{entity}_"
            derived[prefix + "history_n"] = history_n
            derived[prefix + "history_rate"] = history_rate
            derived[prefix + "history_confidence"] = history_n / (history_n + alpha)
            derived[prefix + "current_n"] = current_n
            derived[prefix + "current_posterior"] = current_posterior
            derived[prefix + "current_history_gap"] = current_history_gap
            derived[prefix + "current_confidence"] = current_n / (current_n + alpha)
            derived[prefix + "current_cold_start"] = (current_n == 0.0).astype(np.int8)

        return pd.DataFrame(derived, index=df.index)


def make_temporal_season_features(train_df, alpha_p=50.0, alpha_b=50.0, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = AsymmetricSeasonDecompositionEncoder(alpha_pitcher=alpha_p, alpha_batter=alpha_b)
            dummy_target = pd.Series([0.5], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            encoder.history_tables = {
                e: pd.DataFrame({"history_n": pd.Series(dtype=float), "history_success": pd.Series(dtype=float)})
                for e in ("pitcher", "batter")
            }
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = AsymmetricSeasonDecompositionEncoder(alpha_pitcher=alpha_p, alpha_batter=alpha_b).fit(
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


def evaluate_alpha_config(train_full, base_fe, GLOBAL_MEAN_VAL, alpha_p, alpha_b, FEATURES_BASE):
    label = f"alpha_p={alpha_p}, alpha_b={alpha_b}"
    t0 = time.time()

    train_split = train_full[train_full["season"] <= 2023]
    val_split = train_full[train_full["season"] == 2024]

    season_decomp_train = make_temporal_season_features(train_split, alpha_p=alpha_p, alpha_b=alpha_b, target_col=TARGET)
    encoder_val = AsymmetricSeasonDecompositionEncoder(alpha_pitcher=alpha_p, alpha_batter=alpha_b).fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    fe = pd.concat([base_fe, season_decomp_all], axis=1)

    NEW_COLS = [c for c in fe.columns if c not in train_full.columns]
    ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS

    for c in CAT_COLS:
        fe[c] = fe[c].astype("category")

    val_fe = fe[fe["season"] == 2024].copy()

    is_val_mask = fe["season"] == 2024
    X_train_full = fe.loc[~is_val_mask, ALL_FEATURES]
    y_train_full = fe.loc[~is_val_mask, TARGET]
    X_val_fe = val_fe[ALL_FEATURES]
    y_val = val_fe[TARGET]

    cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS if c in ALL_FEATURES]

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
    lgb_m = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    lgb_m.fit(X_tr_lgb, y_train_full, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    best_lgb_iter = lgb_m.best_iteration_
    p_lgb_val = lgb_m.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val.values, p_lgb_val)

    cb_m = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS)
    cb_m.fit(Pool(X_tr_cb, y_train_full, cat_features=cat_idx), eval_set=Pool(X_val_cb, y_val, cat_features=cat_idx), use_best_model=True)
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
        m_l = lgb.LGBMClassifier(n_estimators=best_lgb_iter, **LGB_PARAMS)
        m_l.fit(X_lgb_r.iloc[tr_idx], y_r.iloc[tr_idx])
        lgb_oof[oof_idx] = m_l.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_cb_r.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=best_cb_iter, **CB_PARAMS)
        m_c.fit(tp_f)
        cb_oof[oof_idx] = m_c.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]

    y_oof_arr = y_r.to_numpy()

    # 3. Meta-learner
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof_arr)
    w_lgb = stack.coef_[0][0]
    w_cb = stack.coef_[0][1]
    intercept = stack.intercept_[0]

    preds_val = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    preds_oof = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]

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

    diff_A = score_val - 836.35
    diff_B = score_oof - 2009.23

    print(f"  [{label}] Holdout={score_val:.2f}({diff_A:>+5.2f}) | OOF={score_oof:.2f}({diff_B:>+5.2f}) | LGB={w_lgb:.2f}, CB={w_cb:.2f} | {guardrail_status}")

    return {
        "alpha_p": alpha_p,
        "alpha_b": alpha_b,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "w_lgb": w_lgb,
        "w_cb": w_cb,
        "intercept": intercept,
        "p_min": float(np.min(preds_val)),
        "p_max": float(np.max(preds_val)),
        "p_std": float(np.std(preds_val)),
        "holdout": score_val,
        "oof": score_oof,
        "diff_A": diff_A,
        "diff_B": diff_B,
        "guardrail_status": guardrail_status,
        "guardrail_msg": guardrail_msg,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 80)
    print("v35 후보 실험: 세이버메트릭스 표본 신뢰도 기반 비대칭 시즌 분해 탐색")
    print("기준선: v14 대칭 alpha=50.0 (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)
    t_start = time.time()

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
    base_fe = build_base_features_v14(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)

    # 비대칭 알파 후보군 탐색
    # 야구 이론: 투수는 안정화가 빠르므로 작은 alpha(25, 35, 50), 타자는 변동이 크므로 큰 alpha(50, 75, 100)
    alpha_configs = [
        (50.0, 50.0),   # v14 기준선
        (30.0, 70.0),   # 투수 30, 타자 70
        (25.0, 75.0),   # 투수 25, 타자 75
        (35.0, 80.0),   # 투수 35, 타자 80
        (40.0, 60.0),   # 투수 40, 타자 60
    ]

    results = []
    for ap, ab in alpha_configs:
        res = evaluate_alpha_config(train_full, base_fe, GLOBAL_MEAN_VAL, ap, ab, FEATURES_BASE)
        results.append(res)

    print("\n" + "=" * 90)
    print("v35 비대칭 시즌 분해 탐색 종합 요약")
    print("=" * 90)
    print(f"{'구성':<25} {'LGB solo':>10} {'CB solo':>10} {'(A) 홀드아웃':>14} {'(B) OOF score':>14} {'가드레일':>9}")
    for r in results:
        cfg_str = f"P={r['alpha_p']:.0f}, B={r['alpha_b']:.0f}"
        print(f"{cfg_str:<25} {r['lgb_solo']:>10.2f} {r['cb_solo']:>10.2f} {r['holdout']:>14.2f}({r['diff_A']:>+5.2f}) {r['oof']:>14.2f}({r['diff_B']:>+5.2f}) {r['guardrail_status']:>9}")

    print(f"\n총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print("=" * 90)

    out = "docs/v35_asymmetric_alpha_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
