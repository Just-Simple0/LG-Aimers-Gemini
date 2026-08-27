"""
v14 후보 실험: Season Decomposition (R18 스타일 베이지안 당해 시즌 분해)
========================================================================

배경 및 문헌 근거:
  - asof_pitcher_success_rate 및 asof_batter_success_rate는 커리어 전체 누적치라
    당해 시즌의 폼 변화(breakout/slump)와 커리어 평균이 혼합되어 있음.
  - Phase2 R18에서 확립된 SeasonDecompositionEncoder:
    과거 시즌에서 동결된 누적치(history_n, history_success, history_rate)를 기준으로
    현재 행의 cumulative에서 차감해 당해 시즌의 투구 수(current_n)와 성공 수(current_success)를 복원.
    여기에 베이지안 수축(alpha=50.0)을 적용해 당해 시즌 사후 확률(current_posterior) 및
    커리어 대비 폼 괴리도(current_history_gap) 등 13개 파생 피처를 생성.
  - 투수(pitcher) 및 타자(batter) 모두에 적용.

이중 게이트 절차 (PR#10 이후 확립):
  (A) 기존 방식: 2024 홀드아웃 직접 최댓값
  (B) 독립 게이트: 가중치는 OOF로만 선택, 홀드아웃은 평가 전용

기각 기준: (B)에서 without 대비 차이가 -0.5점 이하 또는 음수

작성일: 2026-08-27
"""

import os
import time

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

ALPHA = 50.0
SUCCESS_ROUND_ATOL = 0.01

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)


# ──────────────────────────────────────────────
# SeasonDecompositionEncoder 구현 (누수 방지 시간 분해)
# ──────────────────────────────────────────────

ENTITY_SPECS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
}


class SeasonDecompositionEncoder:
    """과거 시즌의 선수별 통계를 동결하고, 현재 행에서 당해 시즌 증거를 복원하는 인코더."""

    def __init__(self, entities=("pitcher", "batter"), alpha=ALPHA):
        self.entities = entities
        self.alpha = float(alpha)
        self.prior = None
        self.history_tables = {}

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
            history_seen = h_n_series.notna().to_numpy(dtype=np.int8)
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

            # history_n보다 cumulative_n이 작으면 (예: 신규 선수 또는 이전 시즌 이력이 현재보다 큰 경우) clip
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
    """
    학습 데이터 내에서 각 시즌(S)은 strictly 이전 시즌(<S)의 데이터로만 인코더를 fit.
    첫 시즌(2019)은 neutral prior(0.5)로 빈 history를 반환.
    """
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            # 첫 시즌 (2019): 과거 데이터가 없으므로 빈 이력으로 변환
            encoder = SeasonDecompositionEncoder()
            # 임의의 빈 더미 fit
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


# ──────────────────────────────────────────────
# 기본 피처 엔지니어링 (v10 기준선)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# kfold_oof & scoring
# ──────────────────────────────────────────────

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

    # ── Step 1: early stopping으로 best_iteration 결정 ──
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

    # ── Step 2: 단독 홀드아웃 점수 ──
    p_lgb_val = lgb_val_model.predict_proba(X_val_lgb)[:, 1]
    p_cb_val = cb_val_model.predict_proba(X_val_cb)[:, 1]
    score_lgb = brier_skill_score(y_val.values, p_lgb_val)
    score_cb = brier_skill_score(y_val.values, p_cb_val)
    print(f"  LGB solo: {score_lgb:.2f}  CB solo: {score_cb:.2f}")

    # ── Step 3: KFold OOF ──
    print("  KFold OOF 생성 중...")
    lgb_oof, cb_oof, y_oof = kfold_oof(
        X_tr_lgb, X_tr_cb, y_train_full, BEST_LGB, BEST_CB, cat_idx
    )

    # ── Step 4: 로지스틱 메타러너 ──
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([lgb_oof, cb_oof]), y_oof)

    # (A) 홀드아웃 점수
    preds_A = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    score_A = brier_skill_score(y_val.values, preds_A)

    # (B) OOF 점수 (독립 게이트)
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
    print("v14 후보 실험: Season Decomposition (R18 스타일 베이지안 당해 시즌 분해)")
    print("기준: logistic_stack_diagnostic.py (v10 실측 904.93)")
    print("=" * 70)

    # ── 데이터 로드 ──
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                             encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_VAL = train_full.loc[train_full["season"] != 2024, TARGET].mean()
    print(f"train: {train_full.shape}  GLOBAL_MEAN_VAL={GLOBAL_MEAN_VAL:.6f}")

    # ── Trackman lookup (검증용, season <= 2023) ──
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

    # ── WITHOUT 기본 피처 ──
    print("\n기본 피처 엔지니어링 (WITHOUT)...")
    fe_without = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)

    # ── WITH Season Decomposition 피처 ──
    print("Season Decomposition 피처 생성 중 (WITH)...")
    t0 = time.time()
    # 훈련 데이터 (2019~2023): 시간순 temporal decomposition (< season 기준)
    train_split = train_full[train_full["season"] <= 2023]
    season_decomp_train = make_temporal_season_features(train_split, TARGET)

    # 검증 데이터 (2024): 2019~2023 전체를 history로 하여 fit 후 transform
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    val_split = train_full[train_full["season"] == 2024]
    season_decomp_val = encoder_val.transform(val_split)

    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]
    fe_with = pd.concat([fe_without, season_decomp_all], axis=1)
    print(f"  Season Decomposition 완료 :: {time.time()-t0:.1f}s | 추가 피처: {season_decomp_all.shape[1]}개")

    # ALL_FEATURES 구성
    NEW_COLS_WITHOUT = [c for c in fe_without.columns if c not in train_full.columns]
    ALL_FEATURES_WITHOUT = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_WITHOUT

    NEW_COLS_WITH = [c for c in fe_with.columns if c not in train_full.columns]
    ALL_FEATURES_WITH = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_WITH

    # LGB category dtype
    for c in CAT_COLS:
        fe_without[c] = fe_without[c].astype("category")
        fe_with[c] = fe_with[c].astype("category")

    val_fe_without = fe_without[fe_without["season"] == 2024].copy()
    val_fe_with = fe_with[fe_with["season"] == 2024].copy()

    # ── 실험 실행 ──
    results = []

    res_wo = run_one(
        "WITHOUT (기존 v10 기준선)",
        fe_without, val_fe_without, ALL_FEATURES_WITHOUT,
    )
    results.append(res_wo)

    res_wi = run_one(
        "WITH (Season Decomposition 추가)",
        fe_with, val_fe_with, ALL_FEATURES_WITH,
    )
    results.append(res_wi)

    # ── 최종 요약 ──
    diff_A = res_wi["score_A_holdout"] - res_wo["score_A_holdout"]
    diff_B = res_wi["score_B_oof"] - res_wo["score_B_oof"]

    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)
    print(f"{'항목':<25} {'WITHOUT':>10} {'WITH':>10} {'diff':>10}")
    print(f"{'피처 수':<25} {res_wo['n_features']:>10} {res_wi['n_features']:>10}")
    print(f"{'LGB solo':<25} {res_wo['lgb_solo']:>10.2f} {res_wi['lgb_solo']:>10.2f}")
    print(f"{'CB solo':<25} {res_wo['cb_solo']:>10.2f} {res_wi['cb_solo']:>10.2f}")
    print(f"{'(A) 홀드아웃':<25} {res_wo['score_A_holdout']:>10.2f} {res_wi['score_A_holdout']:>10.2f} {diff_A:>+10.2f}")
    print(f"{'(B) OOF score':<25} {res_wo['score_B_oof']:>10.2f} {res_wi['score_B_oof']:>10.2f} {diff_B:>+10.2f}")

    if diff_B > 0.5:
        verdict = "채택 후보 ✅  Dacon 재제출 권장"
    elif diff_B > 0.0:
        verdict = "보류 ⚠️   노이즈 수준, 추가 검토 필요"
    else:
        verdict = "기각 ❌  독립 게이트 음수"

    print(f"\n판정: {verdict}")
    print("=" * 70)

    # ── CSV 저장 ──
    out = "docs/v14_season_decomposition_results.csv"
    df_res = pd.DataFrame(results)
    df_res["diff_A"] = diff_A
    df_res["diff_B"] = diff_B
    df_res["verdict"] = verdict
    df_res.to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
