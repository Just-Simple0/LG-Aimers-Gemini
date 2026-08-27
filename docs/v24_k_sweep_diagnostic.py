"""
v24 k-sweep diagnostic: Career Rate Smoothing k in [30, 40, 50, 60, 70, 80]
=============================================================================
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

from src.script import SeasonDecompositionEncoder, ALPHA

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]

LGB_PARAMS_BASE = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS_BASE = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)


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


def main():
    print("=" * 70)
    print("v24 k-Smoothing Sweep: k in [30, 40, 50, 60, 70, 80]")
    print("=" * 70)
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

    K_LIST = [30, 40, 50, 60, 70, 80]
    results = []

    for k in K_LIST:
        t0 = time.time()
        base_fe = build_base_features_k(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL, k_smooth=k)
        fe = pd.concat([base_fe, season_decomp_all], axis=1)

        NEW_COLS = [c for c in fe.columns if c not in train_full.columns]
        ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS

        for c in CAT_COLS:
            fe[c] = fe[c].astype("category")

        val_fe = fe[fe["season"] == 2024].copy()

        # eval
        is_val_mask = fe["season"] == 2024
        X_train_full = fe.loc[~is_val_mask, ALL_FEATURES]
        y_train_full = fe.loc[~is_val_mask, TARGET]
        X_val_fe = val_fe[ALL_FEATURES]
        y_val = val_fe[TARGET]

        cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS if c in ALL_FEATURES]

        def lgb_df(d):
            d_ = d.copy()
            for c in CAT_COLS:
                if c in d_.columns:
                    d_[c] = d_[c].astype("category")
            return d_

        def cb_df(d):
            d_ = d.copy()
            for c in CAT_COLS:
                if c in d_.columns:
                    d_[c] = d_[c].astype(str)
            return d_

        X_tr_lgb = lgb_df(X_train_full)
        X_tr_cb = cb_df(X_train_full)
        X_val_lgb = lgb_df(X_val_fe)
        X_val_cb = cb_df(X_val_fe)

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

            tp_f = Pool(X_tr_cb.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
            m_c = CatBoostClassifier(iterations=best_cb_iter, **CB_PARAMS_BASE)
            m_c.fit(tp_f)
            cb_oof[oof_idx] = m_c.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]

        stack = LogisticRegression(random_state=42, max_iter=1000)
        y_oof_arr = y_r.to_numpy()
        stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof_arr)

        score_val = brier_skill_score(y_val.values, stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1])
        score_oof = brier_skill_score(y_oof_arr, stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1])
        elapsed = time.time() - t0

        print(f"k={k:<3} | LGB={score_lgb:.2f}(iter={best_lgb_iter}) CB={score_cb:.2f}(iter={best_cb_iter}) | (A)Holdout={score_val:.2f} | (B)OOF={score_oof:.2f} | {elapsed/60:.1f}분")
        results.append({
            "k": k,
            "lgb_solo": score_lgb,
            "cb_solo": score_cb,
            "holdout": score_val,
            "oof": score_oof,
            "elapsed_min": elapsed / 60,
        })

    print("\n" + "=" * 70)
    print("최종 요약 (k-Smoothing Sweep)")
    print("=" * 70)
    for r in results:
        diff_h = r["holdout"] - 836.35
        diff_o = r["oof"] - 2009.23
        print(f"k={r['k']:<3} | LGB={r['lgb_solo']:>7.2f} CB={r['cb_solo']:>7.2f} | (A)Holdout={r['holdout']:>7.2f}({diff_h:>+6.2f}) | (B)OOF={r['oof']:>7.2f}({diff_o:>+6.2f})")
    
    best_k = max(results, key=lambda x: x["holdout"])
    print(f"\n최적 k: {best_k['k']} (Holdout={best_k['holdout']:.2f}, OOF={best_k['oof']:.2f})")


if __name__ == "__main__":
    main()
