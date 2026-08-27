"""
v33 후보 실험: Pairwise Head-to-Head Bayesian Familiarity (투타 맞대결 상대전적 베이지안 분해)
=============================================================================================

배경 및 가설:
  - 현재 v14 모델은 투수 개인(`pitcher_id`)과 타자 개인(`batter_id`)의 시즌 분해만 수행함.
  - 야구 세이버메트릭스에서 투수-타자 간의 맞대결 친숙도(Familiarity & Head-to-Head History):
    1) 통산 맞대결 투구 수(history_pair_n): 과거 시즌 동안 해당 타자를 상대해 본 경험.
    2) 맞대결 제구 성공 수(history_pair_success) 및 베이지안 맞대결 사후확률:
       - alpha_pair = 20.0 (소표본 수축)
       - prior = matchup_success_diff 또는 global_mean
    3) 초면 맞대결 여부(is_first_time_matchup): 통산 맞대결이 0구인 낯선 상대 상황.
  - 가설:
    투타 1:1 상대전적을 과거 시즌 동결 누적치 기반으로 무누수 산출하여
    투수-타자 간 맞대결 친숙도 및 상대 전적 사후확률 피처를 추가하면
    선수 간 상성(Matchup Chemistry)을 직접 포착하여 점수를 개선할 수 있을 것임.

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


class PairwiseMatchupEncoder:
    """투수-타자 맞대결 통산 이력 무누수 인코더."""

    def __init__(self, alpha=20.0, global_prior=0.5238):
        self.alpha = float(alpha)
        self.global_prior = float(global_prior)
        self.pair_table = None

    def fit(self, df, target):
        values = np.asarray(target, dtype=float).reshape(-1)
        pairs = df["pitcher_id"].astype(str) + "_" + df["batter_id"].astype(str)
        working = pd.DataFrame({
            "pair_key": pairs.to_numpy(copy=False),
            "__target": values,
        })
        self.pair_table = (
            working.groupby("pair_key", sort=False)["__target"]
            .agg(history_pair_success="sum", history_pair_n="count")
        )
        return self

    def transform(self, df):
        pairs = df["pitcher_id"].astype(str) + "_" + df["batter_id"].astype(str)
        if self.pair_table is None:
            h_n = np.zeros(len(df), dtype=float)
            h_succ = np.zeros(len(df), dtype=float)
        else:
            h_n = pairs.map(self.pair_table["history_pair_n"]).fillna(0.0).to_numpy(dtype=float)
            h_succ = pairs.map(self.pair_table["history_pair_success"]).fillna(0.0).to_numpy(dtype=float)

        # 베이지안 사후 맞대결 성공률
        pair_posterior = (h_succ + self.alpha * self.global_prior) / (h_n + self.alpha)
        is_first_matchup = (h_n == 0.0).astype(np.int8)

        return pd.DataFrame({
            "pair_history_n": np.log1p(h_n),
            "pair_history_posterior": pair_posterior,
            "pair_is_first_matchup": is_first_matchup,
        }, index=df.index)


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


def make_temporal_pairwise_features(train_df, target_col=TARGET, global_mean=0.5238):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = PairwiseMatchupEncoder(alpha=20.0, global_prior=global_mean)
            dummy_target = pd.Series([0.5], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = PairwiseMatchupEncoder(alpha=20.0, global_prior=global_mean).fit(
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
    w_lgb = stack.coef_[0][0]
    w_cb = stack.coef_[0][1]
    intercept = stack.intercept_[0]

    preds_A = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    score_A = brier_skill_score(y_val.values, preds_A)

    preds_B = stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    score_B = brier_skill_score(y_oof, preds_B)

    elapsed = time.time() - t0

    # Step 5: 5대 가드레일 검수
    guardrail_status = "PASS ✅"
    try:
        inspect_meta_learner_weights(w_lgb, w_cb, intercept)
        inspect_prediction_distribution(preds_A)
        inspect_dual_gates(score_A - 836.35, score_B - 2009.23)
    except SubmissionGuardrailError as e:
        guardrail_status = "FAIL ❌"
        print(f"  가드레일 경고: {e}")

    print(f"\n  [결과] {label}")
    print(f"    (A) 홀드아웃  : {score_A:.2f} ({score_A-836.35:+.2f})")
    print(f"    (B) OOF score: {score_B:.2f} ({score_B-2009.23:+.2f})")
    print(f"    메타러너 계수: LGB={w_lgb:.4f}, CB={w_cb:.4f}, intercept={intercept:.4f}")
    print(f"    가드레일 상태: {guardrail_status}")
    print(f"    소요시간      : {elapsed/60:.1f}분")

    return {
        "label": label,
        "n_features": len(all_features),
        "lgb_best_iter": BEST_LGB,
        "cb_best_iter": BEST_CB,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "w_lgb": w_lgb,
        "w_cb": w_cb,
        "score_A_holdout": score_A,
        "score_B_oof": score_B,
        "diff_A": score_A - 836.35,
        "diff_B": score_B - 2009.23,
        "guardrail_status": guardrail_status,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 70)
    print("v33 후보 실험: Pairwise Head-to-Head Bayesian Familiarity")
    print("기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)")
    print("=" * 70)

    # 1. 데이터 로드
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_VAL = train_full.loc[train_full["season"] != 2024, TARGET].mean()
    print(f"train: {train_full.shape}  GLOBAL_MEAN_VAL={GLOBAL_MEAN_VAL:.6f}")

    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

    # 2. v14 기준 Season Decomposition
    print("\nv14 Season Decomposition 피처 생성 중...")
    train_split = train_full[train_full["season"] <= 2023]
    season_decomp_train = make_temporal_season_features(train_split, TARGET)

    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    val_split = train_full[train_full["season"] == 2024]
    season_decomp_val = encoder_val.transform(val_split)

    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]
    base_fe = build_base_features_v14(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    # 3. v33 맞대결 상대전적 피처 생성 (Pairwise Matchup History)
    print("\nv33 맞대결 상대전적 (Pairwise Matchup History) 피처 생성 중...")
    pairwise_decomp_train = make_temporal_pairwise_features(train_split, TARGET, GLOBAL_MEAN_VAL)
    pair_encoder_val = PairwiseMatchupEncoder(alpha=20.0, global_prior=GLOBAL_MEAN_VAL).fit(train_split, train_split[TARGET])
    pairwise_decomp_val = pair_encoder_val.transform(val_split)
    pairwise_decomp_all = pd.concat([pairwise_decomp_train, pairwise_decomp_val], axis=0).loc[train_full.index]

    fe_v33 = pd.concat([fe_v14, pairwise_decomp_all], axis=1)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    NEW_COLS_V33 = [c for c in fe_v33.columns if c not in train_full.columns]
    ALL_FEATURES_V33 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V33

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")
        fe_v33[c] = fe_v33[c].astype("category")

    val_fe_v14 = fe_v14[fe_v14["season"] == 2024].copy()
    val_fe_v33 = fe_v33[fe_v33["season"] == 2024].copy()

    # 4. 실험 실행
    results = []
    res_v14 = run_one("WITHOUT (v14 기준선, 101개 피처)", fe_v14, val_fe_v14, ALL_FEATURES_V14)
    results.append(res_v14)

    res_v33 = run_one("WITH (v33 맞대결 상대전적 3개 추가, 104개 피처)", fe_v33, val_fe_v33, ALL_FEATURES_V33)
    results.append(res_v33)

    # 5. 요약
    diff_A = res_v33["score_A_holdout"] - res_v14["score_A_holdout"]
    diff_B = res_v33["score_B_oof"] - res_v14["score_B_oof"]

    print("\n" + "=" * 70)
    print("최종 결과 요약 (Pairwise Matchup Bayesian Familiarity)")
    print("=" * 70)
    print(f"{'항목':<35} {'v14(WITHOUT)':>12} {'v33(WITH)':>12} {'diff':>10}")
    print(f"{'피처 수':<35} {res_v14['n_features']:>12} {res_v33['n_features']:>12}")
    print(f"{'LGB solo':<35} {res_v14['lgb_solo']:>12.2f} {res_v33['lgb_solo']:>12.2f}")
    print(f"{'CB solo':<35} {res_v14['cb_solo']:>12.2f} {res_v33['cb_solo']:>12.2f}")
    print(f"{'LGB coef':<35} {res_v14['w_lgb']:>12.4f} {res_v33['w_lgb']:>12.4f}")
    print(f"{'CB coef':<35} {res_v14['w_cb']:>12.4f} {res_v33['w_cb']:>12.4f}")
    print(f"{'(A) 홀드아웃':<35} {res_v14['score_A_holdout']:>12.2f} {res_v33['score_A_holdout']:>12.2f} {diff_A:>+10.2f}")
    print(f"{'(B) OOF score':<35} {res_v14['score_B_oof']:>12.2f} {res_v33['score_B_oof']:>12.2f} {diff_B:>+10.2f}")
    print(f"{'가드레일 상태':<35} {res_v14['guardrail_status']:>12} {res_v33['guardrail_status']:>12}")
    print("=" * 70)

    out = "docs/v33_pairwise_matchup_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
