"""
v36 후보 실험: 3대 트리 패러다임 삼각 앙상블 스태킹 (LightGBM + CatBoost + HistGB 3-Way Stacking)
====================================================================================================

배경 및 머신러닝 스태킹 이론:
  - 2개 모델(LGB + CB)만 사용할 경우 두 모델의 예측 상관계수(r > 0.99)로 인해
    메타러너의 다중공선성(Collinearity) 문제가 발생하기 쉬움.
  - 진정한 스태킹 안정성을 위해서는 '구조적 모델 다양성(Architectural Diversity)'이 필수적임:
    1) LightGBM: Leaf-wise (Best-first) 비대칭 트리 분할
    2) CatBoost: Oblivious (Symmetric) 대칭 트리 분할
    3) HistGradientBoosting: Scikit-Learn C++ 깊이 우선 균등 히스토그램 분할
  - 가설:
    v14 검증된 101개 피처셋 상에서 서로 다른 3대 트리 아키텍처(LGB, CB, HistGB)를 동시에 학습시키고
    5-Fold OOF 기반으로 3-Way 메타러너 스태킹을 구성하면:
    1) 3개 모델이 상호 보완적인 양수 가중치(w_lgb > 0, w_cb > 0, w_hgb > 0)로 안정화됨.
    2) 단일 모델의 편향이 3방향으로 상쇄되어 Brier Skill Score가 역대 최고치를 갱신할 것임.

기준선: v14 2-Way (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

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
HGB_PARAMS = dict(
    learning_rate=0.03, max_iter=200, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0,
    random_state=42, early_stopping=False,
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
    print("v36 후보 실험: 3대 트리 패러다임 삼각 앙상블 (LGB + CB + HistGB 3-Way Stacking)")
    print("기준선: v14 2-Way (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)")
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
    cat_mask = [c in CAT_COLS for c in ALL_FEATURES_V14]

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

    def hgb_df(df):
        d = df.copy()
        for c in CAT_COLS:
            if c in d.columns:
                d[c] = d[c].astype("category").cat.codes
        return d

    X_tr_lgb = lgb_df(X_train_full)
    X_tr_cb = cb_df(X_train_full)
    X_tr_hgb = hgb_df(X_train_full)

    X_val_lgb = lgb_df(X_val_fe)
    X_val_cb = cb_df(X_val_fe)
    X_val_hgb = hgb_df(X_val_fe)

    print("\n[Step 1] 3대 베이스 모델 조기 종료 및 검증 예측...")
    # 1. LGB
    t0 = time.time()
    lgb_m = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    lgb_m.fit(X_tr_lgb, y_train_full, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    best_lgb = lgb_m.best_iteration_
    p_lgb_val = lgb_m.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val.values, p_lgb_val)
    print(f"  LGB best_iter={best_lgb} | Solo={score_lgb:.2f} :: {time.time()-t0:.1f}s")

    # 2. CB
    t0 = time.time()
    cb_m = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS)
    cb_m.fit(Pool(X_tr_cb, y_train_full, cat_features=cat_idx), eval_set=Pool(X_val_cb, y_val, cat_features=cat_idx), use_best_model=True)
    best_cb = cb_m.get_best_iteration() + 1
    p_cb_val = cb_m.predict_proba(X_val_cb)[:, 1]
    score_cb = brier_skill_score(y_val.values, p_cb_val)
    print(f"  CB  best_iter={best_cb} | Solo={score_cb:.2f} :: {time.time()-t0:.1f}s")

    # 3. HistGB
    t0 = time.time()
    hgb_m = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB_PARAMS)
    hgb_m.fit(X_tr_hgb, y_train_full)
    p_hgb_val = hgb_m.predict_proba(X_val_hgb)[:, 1]
    score_hgb = brier_skill_score(y_val.values, p_hgb_val)
    print(f"  HistGB max_iter=200 | Solo={score_hgb:.2f} :: {time.time()-t0:.1f}s")

    print("\n[Step 2] 3개 모델 5-Fold OOF 생성 중...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_lgb_r = X_tr_lgb.reset_index(drop=True)
    X_cb_r = X_tr_cb.reset_index(drop=True)
    X_hgb_r = X_tr_hgb.reset_index(drop=True)
    y_r = y_train_full.reset_index(drop=True)
    y_oof_arr = y_r.to_numpy()

    lgb_oof = np.zeros(len(y_r))
    cb_oof = np.zeros(len(y_r))
    hgb_oof = np.zeros(len(y_r))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_lgb_r)):
        t_f = time.time()
        # LGB
        m_l = lgb.LGBMClassifier(n_estimators=best_lgb, **LGB_PARAMS)
        m_l.fit(X_lgb_r.iloc[tr_idx], y_r.iloc[tr_idx])
        lgb_oof[oof_idx] = m_l.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

        # CB
        tp_f = Pool(X_cb_r.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=best_cb, **CB_PARAMS)
        m_c.fit(tp_f)
        cb_oof[oof_idx] = m_c.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]

        # HistGB
        m_h = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB_PARAMS)
        m_h.fit(X_hgb_r.iloc[tr_idx], y_r.iloc[tr_idx])
        hgb_oof[oof_idx] = m_h.predict_proba(X_hgb_r.iloc[oof_idx])[:, 1]

        print(f"  Fold {fold_i+1}/5 완료 :: {time.time()-t_f:.1f}s")

    print("\n[Step 3] 3-Way 메타러너 스태킹 적합...")
    # (A) 2-Way Base (LGB + CB)
    stack_2way = LogisticRegression(random_state=42, max_iter=1000)
    stack_2way.fit(np.column_stack([lgb_oof, cb_oof]), y_oof_arr)
    w_lgb_2w, w_cb_2w = stack_2way.coef_[0]
    preds_2w_val = stack_2way.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    preds_2w_oof = stack_2way.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    score_2w_val = brier_skill_score(y_val.values, preds_2w_val)
    score_2w_oof = brier_skill_score(y_oof_arr, preds_2w_oof)

    # (B) 3-Way (LGB + CB + HistGB)
    stack_3way = LogisticRegression(random_state=42, max_iter=1000)
    stack_3way.fit(np.column_stack([lgb_oof, cb_oof, hgb_oof]), y_oof_arr)
    w_lgb_3w, w_cb_3w, w_hgb_3w = stack_3way.coef_[0]
    intercept_3w = stack_3way.intercept_[0]

    preds_3w_val = stack_3way.predict_proba(np.column_stack([p_lgb_val, p_cb_val, p_hgb_val]))[:, 1]
    preds_3w_oof = stack_3way.predict_proba(np.column_stack([lgb_oof, cb_oof, hgb_oof]))[:, 1]
    score_3w_val = brier_skill_score(y_val.values, preds_3w_val)
    score_3w_oof = brier_skill_score(y_oof_arr, preds_3w_oof)

    diff_3w_A = score_3w_val - 836.35
    diff_3w_B = score_3w_oof - 2009.23

    # 가드레일 검수
    guardrail_status = "PASS ✅"
    try:
        inspect_prediction_distribution(preds_3w_val)
        inspect_dual_gates(diff_3w_A, diff_3w_B)
        if min(w_lgb_3w, w_cb_3w, w_hgb_3w) < 0.10:
            guardrail_status = "FAIL ❌"
    except SubmissionGuardrailError as e:
        guardrail_status = "FAIL ❌"
        print(f"  가드레일 경고: {e}")

    print("\n" + "=" * 80)
    print("최종 3-Way 스태킹 결과 비교")
    print("=" * 80)
    print(f"{'구성':<25} {'LGB coef':>9} {'CB coef':>9} {'HGB coef':>9} {'(A) 홀드아웃':>14} {'(B) OOF score':>14} {'가드레일':>9}")
    print(f"{'v14 (2-Way 기준선)':<25} {w_lgb_2w:>9.2f} {w_cb_2w:>9.2f} {'-':>9} {score_2w_val:>14.2f} {score_2w_oof:>14.2f} {'PASS ✅':>9}")
    print(f"{'v36 (3-Way 삼각 앙상블)':<25} {w_lgb_3w:>9.2f} {w_cb_3w:>9.2f} {w_hgb_3w:>9.2f} {score_3w_val:>14.2f}({diff_3w_A:>+5.2f}) {score_3w_oof:>14.2f}({diff_3w_B:>+5.2f}) {guardrail_status:>9}")
    print(f"예측 범위: [{np.min(preds_3w_val):.4f}, {np.max(preds_3w_val):.4f}], Intercept: {intercept_3w:.4f}")
    print(f"총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print("=" * 80)

    out = "docs/v36_3way_results.csv"
    res_df = pd.DataFrame([{
        "mode": "3-Way (LGB+CB+HistGB)",
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "hgb_solo": score_hgb,
        "w_lgb": w_lgb_3w,
        "w_cb": w_cb_3w,
        "w_hgb": w_hgb_3w,
        "intercept": intercept_3w,
        "holdout": score_3w_val,
        "oof": score_3w_oof,
        "diff_A": diff_3w_A,
        "diff_B": diff_3w_B,
        "guardrail_status": guardrail_status,
    }])
    res_df.to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
