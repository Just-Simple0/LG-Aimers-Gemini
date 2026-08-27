"""
v17 후보 실험: Season Decomposition 베이지안 수축 계수(alpha) 스윕
===================================================================

배경 및 가설:
  - v14에서 Season Decomposition(alpha=50.0 고정)으로 976.51점을 달성함.
  - alpha=50.0은 사전 지식(history_rate)과 당해 시즌 관측(current_success) 간의 가중치 균형을 결정하는 핵심 하이퍼파라미터.
    - alpha가 작을수록 (예: 15, 30): 당해 시즌 폼의 작은 표본 변화에도 빠르게 반응.
    - alpha가 클수록 (예: 75, 100, 150): 커리어 통계로 더 강하게 수축하여 노이즈 억제.
  - 탐색 후보: alpha in [15.0, 25.0, 35.0, 50.0, 75.0, 100.0, 150.0]
  - 이중 게이트:
    (A) 2024 홀드아웃 점수
    (B) OOF score (독립 게이트)

작성일: 2026-08-27
"""

import os
import sys
import time

# 프로젝트 루트를 sys.path에 추가 (임포트 에러 방지)
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


class SeasonDecompositionEncoder:
    def __init__(self, entities=("pitcher", "batter"), alpha=50.0, prior=None, history_tables=None):
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
        if self.prior is None:
            raise RuntimeError("Encoder must be fitted before transform")

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


def make_temporal_season_features(train_df, alpha=50.0, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = SeasonDecompositionEncoder(alpha=alpha)
            dummy_target = pd.Series([0.5], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            encoder.history_tables = {
                e: pd.DataFrame({"history_n": pd.Series(dtype=float), "history_success": pd.Series(dtype=float)})
                for e in ("pitcher", "batter")
            }
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = SeasonDecompositionEncoder(alpha=alpha).fit(
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


def build_base_features(df, global_mean, tk_lookup):
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


def evaluate_alpha(alpha, base_fe, train_full, val_split, train_split, GLOBAL_MEAN_VAL, FEATURES_BASE):
    t0 = time.time()
    season_decomp_train = make_temporal_season_features(train_split, alpha=alpha, target_col=TARGET)
    encoder_val = SeasonDecompositionEncoder(alpha=alpha).fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    train_fe = pd.concat([base_fe, season_decomp_all], axis=1)
    NEW_COLS = [c for c in train_fe.columns if c not in train_full.columns]
    ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS

    for c in CAT_COLS:
        train_fe[c] = train_fe[c].astype("category")

    is_val = train_fe["season"] == 2024
    X_train_df = train_fe.loc[~is_val, ALL_FEATURES]
    y_train = train_fe.loc[~is_val, TARGET].to_numpy()
    X_val_df = train_fe.loc[is_val, ALL_FEATURES]
    y_val = train_fe.loc[is_val, TARGET].to_numpy()

    cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]

    X_tr_lgb = X_train_df.copy()
    X_val_lgb = X_val_df.copy()

    X_tr_cb = X_train_df.copy()
    X_val_cb = X_val_df.copy()
    for c in CAT_COLS:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    # 1. Early stopping for best iteration
    lgb_model = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    lgb_model.fit(X_tr_lgb, y_train, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    BEST_LGB = lgb_model.best_iteration_
    p_lgb_val = lgb_model.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val, p_lgb_val)

    tp_tr = Pool(X_tr_cb, y_train, cat_features=cat_idx)
    tp_val = Pool(X_val_cb, y_val, cat_features=cat_idx)
    cb_model = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS)
    cb_model.fit(tp_tr, eval_set=tp_val, use_best_model=True)
    BEST_CB = cb_model.get_best_iteration() + 1
    p_cb_val = cb_model.predict_proba(X_val_cb)[:, 1]
    score_cb = brier_skill_score(y_val, p_cb_val)

    # 2. 5-Fold KFold OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y_train))
    cb_oof = np.zeros(len(y_train))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train_df)):
        m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB, **LGB_PARAMS)
        m_lgb.fit(X_tr_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[oof_idx] = m_lgb.predict_proba(X_tr_lgb.iloc[oof_idx])[:, 1]

        tp_fold = Pool(X_tr_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=BEST_CB, **CB_PARAMS)
        m_cb.fit(tp_fold)
        cb_oof[oof_idx] = m_cb.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]

    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([lgb_oof, cb_oof]), y_train)
    preds_val = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    preds_oof = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    score_val = brier_skill_score(y_val, preds_val)
    score_oof = brier_skill_score(y_train, preds_oof)

    elapsed = time.time() - t0
    print(f"  alpha={alpha:<6.1f} | LGB={score_lgb:.2f}(iter={BEST_LGB}) CB={score_cb:.2f}(iter={BEST_CB}) | (A)Holdout={score_val:.2f} | (B)OOF={score_oof:.2f} | {elapsed/60:.1f}분")

    return {
        "alpha": alpha,
        "lgb_iter": BEST_LGB,
        "cb_iter": BEST_CB,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "holdout": score_val,
        "oof": score_oof,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 70)
    print("v17 후보 실험: Season Decomposition 베이지안 수축 계수 (alpha) 스윕")
    print("기준선: alpha=50.0 (v14: 홀드아웃 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 70)
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
    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)

    train_split = train_full[train_full["season"] <= 2023]
    val_split = train_full[train_full["season"] == 2024]

    ALPHA_CANDIDATES = [20.0, 35.0, 50.0, 75.0, 100.0]
    results = []

    print(f"\n총 {len(ALPHA_CANDIDATES)}개 alpha 후보 탐색 시작: {ALPHA_CANDIDATES}...")
    for alpha in ALPHA_CANDIDATES:
        res = evaluate_alpha(alpha, base_fe, train_full, val_split, train_split, GLOBAL_MEAN_VAL, FEATURES_BASE)
        results.append(res)

    print("\n" + "=" * 70)
    print("최종 결과 요약 (Alpha Sweep)")
    print("=" * 70)
    print(f"{'alpha':<10} {'LGB solo':>12} {'CB solo':>12} {'(A) 홀드아웃':>15} {'(B) OOF score':>15}")
    for r in results:
        marker = " (v14 기준)" if r["alpha"] == 50.0 else ""
        print(f"{r['alpha']:<10.1f} {r['lgb_solo']:>12.2f} {r['cb_solo']:>12.2f} {r['holdout']:>15.2f} {r['oof']:>15.2f}{marker}")

    best_r = max(results, key=lambda x: x["holdout"])
    diff_holdout = best_r["holdout"] - [r for r in results if r["alpha"] == 50.0][0]["holdout"]
    diff_oof = best_r["oof"] - [r for r in results if r["alpha"] == 50.0][0]["oof"]

    print(f"\n최적 alpha: {best_r['alpha']} (홀드아웃 diff={diff_holdout:+.2f}, OOF diff={diff_oof:+.2f})")
    print(f"총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print("=" * 70)

    out = "docs/v17_alpha_sweep_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
