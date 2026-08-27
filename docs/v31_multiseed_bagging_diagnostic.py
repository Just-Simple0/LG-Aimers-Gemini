"""
v31 후보 실험: 5-Seed Multi-Bagging 앙상블 (LightGBM 5-Seed + CatBoost 5-Seed)
=============================================================================

배경 및 가설:
  - v14(101개 피처셋, k=30, alpha=50.0)는 Dacon 976.51점으로 프로젝트 역대 최고 점수를 기록함.
  - 30번의 실험을 통해 새로운 피처를 무리하게 추가하면 GBDT 간 다중공선성이 증가하여
    메타러너가 단일 모델로 쏠리거나 음수 가중치가 발생하는 현상이 확인됨.
  - 가설:
    피처셋을 변경하지 않고 완벽히 고정한 상태에서,
    서로 다른 5개 랜덤 시드(42, 1004, 2024, 777, 999)로 학습된 LightGBM과 CatBoost의 확률을
    시드 평균(Seed-Averaging Bagging)하여 메타러너에 입력하면:
    1) 트리 분할의 확률적 샘플링 분산(Variance)이 감소하여 Brier 오차 제곱이 수학적으로 축소됨.
    2) 앙상블의 상호 완충력이 극대화되어 [0.32, 0.77] 안전 구간 및 균형 잡힌 메타러너 가중치가 완벽히 유지됨.
    3) 5대 하드 가드레일을 100% 만족하며 이중 게이트(A/B)를 동반 개선.

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
ALPHA = 50.0

SEEDS = [42, 1004, 2024, 777, 999]
BEST_LGB_ITER = 171
BEST_CB_ITER = 360

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


def run_multiseed_bagging(train_fe, val_fe, all_features, seed_list):
    label = f"Multi-Seed Bagging ({len(seed_list)} Seeds: {seed_list})"
    print(f"\n{'='*70}")
    print(f"실험: {label}")
    print(f"{'='*70}")
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

    n_samples = len(y_train_full)
    n_val = len(y_val)

    lgb_oof_seeds = np.zeros((len(seed_list), n_samples))
    cb_oof_seeds = np.zeros((len(seed_list), n_samples))
    lgb_val_seeds = np.zeros((len(seed_list), n_val))
    cb_val_seeds = np.zeros((len(seed_list), n_val))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_lgb_r = X_tr_lgb.reset_index(drop=True)
    X_cb_r = X_tr_cb.reset_index(drop=True)
    y_r = y_train_full.reset_index(drop=True)
    y_oof_arr = y_r.to_numpy()

    for s_idx, s in enumerate(seed_list):
        t_seed = time.time()
        print(f"\n  [Seed {s_idx+1}/{len(seed_list)}] Seed={s} 학습 시작...")

        lgb_p = dict(
            learning_rate=0.03, num_leaves=63, min_child_samples=200,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary", random_state=s, n_jobs=-1, verbosity=-1,
        )
        cb_p = dict(
            learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
            loss_function="Logloss", random_seed=s, verbose=False, thread_count=-1,
        )

        # 1. Full training for val prediction
        m_lgb_full = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **lgb_p)
        m_lgb_full.fit(X_tr_lgb, y_train_full)
        lgb_val_seeds[s_idx] = m_lgb_full.predict_proba(X_val_lgb)[:, 1]

        tp_full = Pool(X_tr_cb, y_train_full, cat_features=cat_idx)
        m_cb_full = CatBoostClassifier(iterations=BEST_CB_ITER, **cb_p)
        m_cb_full.fit(tp_full)
        cb_val_seeds[s_idx] = m_cb_full.predict_proba(X_val_cb)[:, 1]

        # 2. 5-Fold OOF
        for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_lgb_r)):
            m_l = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **lgb_p)
            m_l.fit(X_lgb_r.iloc[tr_idx], y_train_full.iloc[tr_idx])
            lgb_oof_seeds[s_idx, oof_idx] = m_l.predict_proba(X_lgb_r.iloc[oof_idx])[:, 1]

            tp_f = Pool(X_cb_r.iloc[tr_idx], y_train_full.iloc[tr_idx], cat_features=cat_idx)
            m_c = CatBoostClassifier(iterations=BEST_CB_ITER, **cb_p)
            m_c.fit(tp_f)
            cb_oof_seeds[s_idx, oof_idx] = m_c.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]

        score_lgb_s = brier_skill_score(y_val.values, lgb_val_seeds[s_idx])
        score_cb_s = brier_skill_score(y_val.values, cb_val_seeds[s_idx])
        print(f"    Seed={s} 완료 :: {time.time()-t_seed:.1f}s | LGB={score_lgb_s:.2f} CB={score_cb_s:.2f}")

    # 3. Seed-Averaging Bagging
    p_lgb_val_avg = np.mean(lgb_val_seeds, axis=0)
    p_cb_val_avg = np.mean(cb_val_seeds, axis=0)
    lgb_oof_avg = np.mean(lgb_oof_seeds, axis=0)
    cb_oof_avg = np.mean(cb_oof_seeds, axis=0)

    score_lgb_bagged = brier_skill_score(y_val.values, p_lgb_val_avg)
    score_cb_bagged = brier_skill_score(y_val.values, p_cb_val_avg)
    print(f"\n  [Seed Averaging 단독 성과]")
    print(f"    LGB Bagged Solo: {score_lgb_bagged:.2f} (단일시드 809.27 대비 {score_lgb_bagged-809.27:+.2f})")
    print(f"    CB  Bagged Solo: {score_cb_bagged:.2f} (단일시드 840.81 대비 {score_cb_bagged-840.81:+.2f})")

    # 4. Meta-Learner Stacking
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([lgb_oof_avg, cb_oof_avg]), y_oof_arr)
    w_lgb = stack.coef_[0][0]
    w_cb = stack.coef_[0][1]
    intercept = stack.intercept_[0]

    preds_val = stack.predict_proba(np.column_stack([p_lgb_val_avg, p_cb_val_avg]))[:, 1]
    preds_oof = stack.predict_proba(np.column_stack([lgb_oof_avg, cb_oof_avg]))[:, 1]

    score_val = brier_skill_score(y_val.values, preds_val)
    score_oof = brier_skill_score(y_oof_arr, preds_oof)
    elapsed = time.time() - t0

    # 5. 5대 하드 가드레일 검수
    print("\n" + "=" * 50)
    print("5대 하드 가드레일 검수 시작")
    print("=" * 50)
    guardrail_status = "PASS ✅"
    try:
        inspect_meta_learner_weights(w_lgb, w_cb, intercept)
        inspect_prediction_distribution(preds_val)
        inspect_dual_gates(score_val - 836.35, score_oof - 2009.23)
    except SubmissionGuardrailError as e:
        guardrail_status = "FAIL ❌"
        print(f"  {e}")

    diff_A = score_val - 836.35
    diff_B = score_oof - 2009.23

    print(f"\n  [최종 평가 결과: {label}]")
    print(f"    (A) 홀드아웃  : {score_val:.2f} ({diff_A:+.2f})")
    print(f"    (B) OOF score: {score_oof:.2f} ({diff_B:+.2f})")
    print(f"    메타러너 계수: LGB={w_lgb:.4f}, CB={w_cb:.4f}, intercept={intercept:.4f}")
    print(f"    예측 범위    : [{np.min(preds_val):.4f}, {np.max(preds_val):.4f}]")
    print(f"    가드레일 상태: {guardrail_status}")
    print(f"    소요시간      : {elapsed/60:.1f}분")

    return {
        "seeds": len(seed_list),
        "lgb_bagged_solo": score_lgb_bagged,
        "cb_bagged_solo": score_cb_bagged,
        "w_lgb": w_lgb,
        "w_cb": w_cb,
        "intercept": intercept,
        "val_min": float(np.min(preds_val)),
        "val_max": float(np.max(preds_val)),
        "holdout": score_val,
        "oof": score_oof,
        "diff_A": diff_A,
        "diff_B": diff_B,
        "guardrail_status": guardrail_status,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 80)
    print("v31 후보 실험: Multi-Seed Bagging 앙상블 (LightGBM + CatBoost)")
    print("기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)")
    print("=" * 80)
    t_start = time.time()

    # 1. 데이터 로드 및 피처 생성
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

    val_fe_v14 = fe_v14[fe_v14["season"] == 2024].copy()

    # 2. 3-Seed vs 5-Seed 평가
    results = []

    # (1) 3-Seed Bagging
    res_3seed = run_multiseed_bagging(fe_v14, val_fe_v14, ALL_FEATURES_V14, seed_list=[42, 1004, 2024])
    results.append(res_3seed)

    # (2) 5-Seed Bagging
    res_5seed = run_multiseed_bagging(fe_v14, val_fe_v14, ALL_FEATURES_V14, seed_list=[42, 1004, 2024, 777, 999])
    results.append(res_5seed)

    # 3. 요약
    print("\n" + "=" * 80)
    print("최종 Multi-Seed Bagging 요약 및 5대 가드레일 판정")
    print("=" * 80)
    print(f"{'구성':<25} {'LGB solo':>10} {'CB solo':>10} {'(A) 홀드아웃':>14} {'(B) OOF score':>14} {'가드레일':>9}")
    print(f"{'v14 (1-Seed 기준선)':<25} {809.27:>10.2f} {840.81:>10.2f} {836.35:>14.2f} {2009.23:>14.2f} {'PASS ✅':>9}")
    
    for r in results:
        lbl = f"{r['seeds']}개 시드 배깅 앙상블"
        print(f"{lbl:<25} {r['lgb_bagged_solo']:>10.2f} {r['cb_bagged_solo']:>10.2f} {r['holdout']:>14.2f}({r['diff_A']:>+5.2f}) {r['oof']:>14.2f}({r['diff_B']:>+5.2f}) {r['guardrail_status']:>9}")

    print(f"\n총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print("=" * 80)

    out = "docs/v31_multiseed_bagging_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
