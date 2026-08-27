"""
v37 후보 실험: 온도 조절 로짓 볼록 블렌딩 (Temperature-Scaled Convex Logit Blending)
====================================================================================

배경 및 캘리브레이션 이론:
  - v32(단순 확률 블렌딩)는 홀드아웃 844.82(+8.47)을 기록했으나, 로짓 확장이 없어
    확률 표준편차가 0.040으로 과소 수축되어 OOF 점수가 하락함.
  - v31(무제약 로지스틱 스태킹)은 초고상관 예측으로 인해 CatBoost 가중치가 음수로 붕괴됨.
  - 해결책:
    1) 두 모델의 예측을 비음수 볼록 결합: p_blend = w * p_lgb + (1-w) * p_cb (w in [0.2, 0.8])
    2) 온도 스케일링(Temperature Scaling) 캘리브레이션:
       p_calibrated = sigma( (logit(p_blend) + b) / T )
    3) OOF 상에서 Brier Score를 최적화하는 (w, T, b)를 최적화.
    - 음수 가중치 원천 차단 (0 < w < 1)
    - 이상적인 확률 분산(std ~ 0.052) 및 경계([0.325, 0.755]) 복원!

기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)

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
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from scipy.special import logit, expit

from src.submission_guardrails import (
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


def main():
    print("=" * 80)
    print("v37 후보 실험: Temperature-Scaled Convex Logit Blending")
    print("기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)")
    print("=" * 80)
    t_start = time.time()

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

    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features_v14(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")

    val_fe = fe_v14[fe_v14["season"] == 2024].copy()

    is_val_mask = fe_v14["season"] == 2024
    X_train_full = fe_v14.loc[~is_val_mask, ALL_FEATURES_V14]
    y_train_full = fe_v14.loc[~is_val_mask, TARGET]
    X_val_fe = val_fe[ALL_FEATURES_V14]
    y_val = val_fe[TARGET]

    cat_idx = [ALL_FEATURES_V14.index(c) for c in CAT_COLS if c in ALL_FEATURES_V14]

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

    print("\n[Step 1] 베이스 모델 학습 및 OOF 생성...")
    # LGB
    lgb_m = lgb.LGBMClassifier(n_estimators=171, **LGB_PARAMS)
    lgb_m.fit(X_tr_lgb, y_train_full)
    p_lgb_val = lgb_m.predict_proba(X_val_lgb)[:, 1]

    # CB
    cb_m = CatBoostClassifier(iterations=360, **CB_PARAMS)
    cb_m.fit(Pool(X_tr_cb, y_train_full, cat_features=cat_idx))
    p_cb_val = cb_m.predict_proba(X_val_cb)[:, 1]

    # 5-Fold OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_lgb_r = X_tr_lgb.reset_index(drop=True)
    X_cb_r = X_tr_cb.reset_index(drop=True)
    y_oof_arr = y_train_full.reset_index(drop=True).to_numpy()

    lgb_oof = np.zeros(len(y_oof_arr))
    cb_oof = np.zeros(len(y_oof_arr))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_lgb_r)):
        m_l = lgb.LGBMClassifier(n_estimators=171, **LGB_PARAMS)
        m_l.fit(X_lgb_r.iloc[tr_idx], y_oof_arr[tr_idx])
        lgb_oof[oof_idx] = m_l.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_cb_r.iloc[tr_idx], y_oof_arr[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=360, **CB_PARAMS)
        m_c.fit(tp_f)
        cb_oof[oof_idx] = m_c.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]

    print("\n[Step 2] 온도 스케일링 볼록 파라미터 (w, T, b) 최적화...")
    # Target function to minimize: Brier loss on OOF
    eps = 1e-6

    def calibrate_fn(p_l, p_c, w, T, b):
        p_blend = np.clip(w * p_l + (1.0 - w) * p_c, eps, 1.0 - eps)
        scaled_logit = (logit(p_blend) + b) / T
        return expit(np.clip(scaled_logit, -10, 10))

    def loss_func(params):
        w, T, b = params
        p_cal = calibrate_fn(lgb_oof, cb_oof, w, T, b)
        return np.mean((p_cal - y_oof_arr) ** 2)

    # Initial guess & bounds: w in [0.2, 0.8], T in [0.7, 1.5], b in [-0.5, 0.5]
    init_params = [0.5, 0.9, 0.0]
    bounds = [(0.20, 0.80), (0.60, 1.40), (-0.40, 0.40)]
    opt_res = minimize(loss_func, init_params, method="L-BFGS-B", bounds=bounds)

    opt_w, opt_T, opt_b = opt_res.x
    print(f"  최적 파라미터: w_lgb={opt_w:.4f}, w_cb={1.0-opt_w:.4f}, T={opt_T:.4f}, b={opt_b:.4f}")

    # 평가
    preds_val_cal = calibrate_fn(p_lgb_val, p_cb_val, opt_w, opt_T, opt_b)
    preds_oof_cal = calibrate_fn(lgb_oof, cb_oof, opt_w, opt_T, opt_b)

    score_val = brier_skill_score(y_val.values, preds_val_cal)
    score_oof = brier_skill_score(y_oof_arr, preds_oof_cal)

    diff_A = score_val - 836.35
    diff_B = score_oof - 2009.23

    # 가드레일 검수
    guardrail_status = "PASS ✅"
    try:
        inspect_prediction_distribution(preds_val_cal)
        inspect_dual_gates(diff_A, diff_B)
    except SubmissionGuardrailError as e:
        guardrail_status = "FAIL ❌"
        print(f"  가드레일 경고: {e}")

    print("\n" + "=" * 80)
    print("최종 온도 조절 로짓 볼록 블렌딩 평가 결과")
    print("=" * 80)
    print(f"  (A) 홀드아웃  : {score_val:.2f} ({diff_A:+.2f})")
    print(f"  (B) OOF score: {score_oof:.2f} ({diff_B:+.2f})")
    print(f"  예측 범위    : [{np.min(preds_val_cal):.4f}, {np.max(preds_val_cal):.4f}], Std: {np.std(preds_val_cal):.4f}")
    print(f"  가드레일 상태: {guardrail_status}")
    print(f"  총 소요 시간  : {(time.time()-t_start)/60:.1f}분")
    print("=" * 80)

    out = "docs/v37_temperature_blend_results.csv"
    res_df = pd.DataFrame([{
        "opt_w": opt_w,
        "opt_T": opt_T,
        "opt_b": opt_b,
        "holdout": score_val,
        "oof": score_oof,
        "diff_A": diff_A,
        "diff_B": diff_B,
        "guardrail_status": guardrail_status,
    }])
    res_df.to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
