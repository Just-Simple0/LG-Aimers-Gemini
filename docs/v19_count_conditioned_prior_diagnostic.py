"""
v19 후보 실험: Count-Conditioned Empirical Bayesian Prior (xCTRL 기대 제구율)
================================================================================

배경 및 가설:
  - 현재 v14 모델은 전역 평균(global_mean ~0.53)을 prior로 삼아
    투수/타자의 당해 시즌 사후확률(current_posterior)을 계산함.
  - 하지만 야구에서 볼카운트(Balls, Strikes)에 따른 기저 제구율(Zone/Control rate)은 크게 다름:
    - 3-0, 3-1 카운트: 스트라이크를 잡아야 하므로 제구율 기저가 높음 (~0.60+).
    - 0-2, 1-2 카운트: 유인구를 던져 헛스윙을 유도하므로 제구율 기저가 낮음 (~0.45).
  - 가설:
    1) 과거 시즌(<= 2023) 기준 12개 볼카운트(balls in 0~3, strikes in 0~2)별 리그 제구 성공률(count_expected_control)을 사전 룩업 테이블로 산출.
    2) xCTRL 스타일 제구 초과 기여도 피처 생성:
       - pitcher_control_over_expected = season_pitcher_current_posterior - count_expected_control
       - batter_control_over_expected = season_batter_current_posterior - count_expected_control
       - matchup_xctrl_diff = pitcher_control_over_expected - batter_control_over_expected
       - count_expected_control (자체 피처)
    3) 곱셈 상호작용 없이 순수 선형 차분(diff)으로 구성하여 CatBoost의 분할 효율을 보존하고
       이중 게이트((A) 홀드아웃, (B) OOF)를 동반 개선.

기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)

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

from src.script import SeasonDecompositionEncoder, ALPHA

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
COUNT_KEYS = ["balls_before", "strikes_before"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS = dict(
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


def make_count_expected_control_lookup(train_df):
    """과거 시즌 train_df로부터 볼카운트별 기대 제구 성공률 산출."""
    lookup = train_df.groupby(COUNT_KEYS)[TARGET].agg(
        count_expected_control="mean"
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


def add_count_conditioned_prior_features(df, count_lookup, global_mean):
    """v19 후보: 볼카운트별 기대 제구율 룩업 및 xCTRL 초과 기여도 피처 생성."""
    df = df.copy()
    orig_index = df.index

    # 1. count_expected_control 매핑
    clk = count_lookup.copy()
    for k in COUNT_KEYS:
        clk[k] = clk[k].astype(df[k].dtype)

    df = df.merge(clk, on=COUNT_KEYS, how="left", validate="many_to_one", sort=False)
    df.index = orig_index
    df["count_expected_control"] = df["count_expected_control"].fillna(global_mean)

    # 2. xCTRL 스타일 기대 대비 초과 기여도 (선형 차분)
    df["pitcher_control_over_expected"] = (
        df["season_pitcher_current_posterior"] - df["count_expected_control"]
    )
    df["batter_control_over_expected"] = (
        df["season_batter_current_posterior"] - df["count_expected_control"]
    )
    df["matchup_xctrl_diff"] = (
        df["pitcher_control_over_expected"] - df["batter_control_over_expected"]
    )

    return df


def kfold_oof(X_lgb, X_cb, y, lgb_n_est, cb_n_est, cat_idx, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_lgb_r = X_lgb.reset_index(drop=True)
    X_cb_r = X_cb.reset_index(drop=True)
    y_r = y.reset_index(drop=True)

    lgb_oof = np.zeros(len(y_r))
    cb_oof = np.zeros(len(y_r))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_lgb_r)):
        ft = time.time()
        m_lgb = lgb.LGBMClassifier(n_estimators=lgb_n_est, **LGB_PARAMS)
        m_lgb.fit(X_lgb_r.iloc[tr_idx], y_r.iloc[tr_idx])
        lgb_oof[oof_idx] = m_lgb.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

        tp = Pool(X_cb_r.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=cb_n_est, **CB_PARAMS)
        m_cb.fit(tp)
        cb_oof[oof_idx] = m_cb.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]
        print(f"    KFold {fold_i+1}/{n_splits} :: {time.time()-ft:.1f}s")

    return lgb_oof, cb_oof, y_r.to_numpy()


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def run_one(label, train_fe, val_fe, all_features):
    print(f"\n{'='*60}")
    print(f"실험: {label}")
    print(f"  피처 수: {len(all_features)}")
    print(f"{'='*60}")
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

    # Step 1: early stopping
    t = time.time()
    lgb_val_model = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    lgb_val_model.fit(
        X_tr_lgb, y_train_full,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    BEST_LGB = lgb_val_model.best_iteration_
    print(f"  LGB best_iteration={BEST_LGB} :: {time.time()-t:.1f}s")

    t = time.time()
    cb_val_model = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS)
    cb_val_model.fit(
        Pool(X_tr_cb, y_train_full, cat_features=cat_idx),
        eval_set=Pool(X_val_cb, y_val, cat_features=cat_idx),
        use_best_model=True,
    )
    BEST_CB = cb_val_model.get_best_iteration() + 1
    print(f"  CB  best_iteration={BEST_CB} :: {time.time()-t:.1f}s")

    # Step 2: 단독 점수
    p_lgb_val = lgb_val_model.predict_proba(X_val_lgb)[:, 1]
    p_cb_val = cb_val_model.predict_proba(X_val_cb)[:, 1]
    score_lgb = brier_skill_score(y_val.values, p_lgb_val)
    score_cb = brier_skill_score(y_val.values, p_cb_val)
    print(f"  LGB solo: {score_lgb:.2f}  CB solo: {score_cb:.2f}")

    # Step 3: KFold OOF
    print("  KFold OOF 생성 중...")
    lgb_oof, cb_oof, y_oof = kfold_oof(
        X_tr_lgb, X_tr_cb, y_train_full, BEST_LGB, BEST_CB, cat_idx
    )

    # Step 4: 메타러너
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof)

    preds_A = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    score_A = brier_skill_score(y_val.values, preds_A)

    preds_B = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    score_B = brier_skill_score(y_oof, preds_B)

    elapsed = time.time() - t0
    print(f"\n  [결과] {label}")
    print(f"    (A) 홀드아웃  : {score_A:.2f}")
    print(f"    (B) OOF score: {score_B:.2f}")
    print(f"    소요시간      : {elapsed/60:.1f}분")

    return {
        "label": label,
        "n_features": len(all_features),
        "lgb_best_iter": BEST_LGB,
        "cb_best_iter": BEST_CB,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "score_A_holdout": score_A,
        "score_B_oof": score_B,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 70)
    print("v19 후보 실험: Count-Conditioned Empirical Bayesian Prior (xCTRL 기대 제구율)")
    print("기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)")
    print("=" * 70)

    # 1. 데이터 로드
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_VAL = train_full.loc[train_full["season"] != 2024, TARGET].mean()
    print(f"train: {train_full.shape}  GLOBAL_MEAN_VAL={GLOBAL_MEAN_VAL:.6f}")

    # 2. Trackman lookup (검증용, season <= 2023)
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

    # 3. 볼카운트별 기대 제구율 룩업 테이블 (검증용, season <= 2023)
    train_split = train_full[train_full["season"] <= 2023]
    COUNT_EXPECTED_LOOKUP_VAL = make_count_expected_control_lookup(train_split)
    print("볼카운트별 기대 제구율 테이블 (<=2023):")
    print(COUNT_EXPECTED_LOOKUP_VAL.to_string(index=False))

    # 4. v14 기준 피처 생성
    print("\nv14 기준 피처 생성 중...")
    season_decomp_train = make_temporal_season_features(train_split, TARGET)

    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    val_split = train_full[train_full["season"] == 2024]
    season_decomp_val = encoder_val.transform(val_split)

    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]
    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    # 5. v19 WITH 피처: 볼카운트 조건부 xCTRL 피처 추가
    print("\nv19 xCTRL 볼카운트 기대 제구 피처 추가 중...")
    fe_v19 = add_count_conditioned_prior_features(fe_v14, COUNT_EXPECTED_LOOKUP_VAL, GLOBAL_MEAN_VAL)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    NEW_COLS_V19 = [c for c in fe_v19.columns if c not in train_full.columns]
    ALL_FEATURES_V19 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V19

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")
        fe_v19[c] = fe_v19[c].astype("category")

    val_fe_v14 = fe_v14[fe_v14["season"] == 2024].copy()
    val_fe_v19 = fe_v19[fe_v19["season"] == 2024].copy()

    # 6. 실험 실행
    results = []
    res_v14 = run_one(
        "WITHOUT (v14 기준선, 101개 피처)",
        fe_v14, val_fe_v14, ALL_FEATURES_V14,
    )
    results.append(res_v14)

    res_v19 = run_one(
        "WITH (v19 xCTRL 볼카운트 기대 제구 4개 추가, 105개 피처)",
        fe_v19, val_fe_v19, ALL_FEATURES_V19,
    )
    results.append(res_v19)

    # 7. 최종 요약
    diff_A = res_v19["score_A_holdout"] - res_v14["score_A_holdout"]
    diff_B = res_v19["score_B_oof"] - res_v14["score_B_oof"]

    print("\n" + "=" * 70)
    print("최종 결과 요약 (Count-Conditioned xCTRL Prior)")
    print("=" * 70)
    print(f"{'항목':<35} {'v14(WITHOUT)':>12} {'v19(WITH)':>12} {'diff':>10}")
    print(f"{'피처 수':<35} {res_v14['n_features']:>12} {res_v19['n_features']:>12}")
    print(f"{'LGB solo':<35} {res_v14['lgb_solo']:>12.2f} {res_v19['lgb_solo']:>12.2f}")
    print(f"{'CB solo':<35} {res_v14['cb_solo']:>12.2f} {res_v19['cb_solo']:>12.2f}")
    print(f"{'(A) 홀드아웃':<35} {res_v14['score_A_holdout']:>12.2f} {res_v19['score_A_holdout']:>12.2f} {diff_A:>+10.2f}")
    print(f"{'(B) OOF score':<35} {res_v14['score_B_oof']:>12.2f} {res_v19['score_B_oof']:>12.2f} {diff_B:>+10.2f}")

    if diff_A > 0.0 and diff_B > 0.0:
        verdict = "채택 후보 ✅  이중 게이트(A/B) 동반 개선 확인 (Dacon 재제출 권장)"
    elif diff_B > 0.0:
        verdict = "보류 ⚠️   OOF만 소폭 개선, 홀드아웃 미개선"
    else:
        verdict = "기각 ❌  독립 게이트 음수"

    print(f"\n판정: {verdict}")
    print("=" * 70)

    out = "docs/v19_count_conditioned_results.csv"
    df_res = pd.DataFrame(results)
    df_res["diff_A"] = diff_A
    df_res["diff_B"] = diff_B
    df_res["verdict"] = verdict
    df_res.to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
