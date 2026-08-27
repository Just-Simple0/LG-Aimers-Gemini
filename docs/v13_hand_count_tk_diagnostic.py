"""
v13 후보 실험: 손 조합 × 카운트 Trackman lookup 확장
=======================================================

기준 스크립트: docs/logistic_stack_diagnostic.py (v10 실측 782.47 재현 확정)
수정사항:
  - LGB_PARAMS, CB_PARAMS를 logistic_stack_diagnostic.py와 동일하게 맞춤
  - early_stopping으로 BEST_ITERATION 먼저 결정 후 OOF에 재사용
  - pitcher_hand/batter_hand dtype 캐스팅 버그 수정

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

DATA_DIR = "data"   # 프로젝트 루트에서 실행 기준
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
TK_KEYS_HAND = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]

# ── logistic_stack_diagnostic.py와 완전히 동일한 파라미터 ──
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
# Trackman lookup 생성
# ──────────────────────────────────────────────

def make_tk_lookup(tk_df):
    """기존 (balls, strikes, outs) 3-key lookup — logistic_stack_diagnostic.py와 동일."""
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
    assert not lookup.duplicated(TK_KEYS).any()
    return lookup


def make_tk_lookup_hand(tk_df):
    """추가: (pitcher_hand, batter_hand, balls, strikes, outs) 5-key lookup."""
    lookup = tk_df.groupby(TK_KEYS_HAND).agg(
        tk_ph_bh_fastball_rate=("pitch_type_group", lambda s: (s == "fastball").mean()),
        tk_ph_bh_breaking_rate=("pitch_type_group", lambda s: (s == "breaking").mean()),
        tk_ph_bh_zone_speed_mean=("zone_speed", "mean"),
        tk_ph_bh_spin_rate_mean=("spin_rate", "mean"),
        tk_ph_bh_horz_break_mean=("horz_break", "mean"),
        tk_ph_bh_vert_break_mean=("induced_vert_break", "mean"),
    ).reset_index()
    assert not lookup.duplicated(TK_KEYS_HAND).any()
    return lookup


# ──────────────────────────────────────────────
# build_features: logistic_stack_diagnostic.py와 동일 (WITHOUT)
# ──────────────────────────────────────────────

def build_features(df, global_mean, tk_lookup, tk_lookup_hand=None):
    """
    기존 build_features (logistic_stack_diagnostic.py 문자단위 동일) +
    tk_lookup_hand가 주어지면 손 조합 피처 추가 (WITH 모드).
    """
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

    # 기존 3-key lookup (logistic_stack_diagnostic.py와 동일)
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

    # ── WITH 모드: 손 조합 5-key lookup 추가 ──
    if tk_lookup_hand is not None:
        hand_cols = [c for c in tk_lookup_hand.columns if c.startswith("tk_ph_bh_")]
        tk_h = tk_lookup_hand[TK_KEYS_HAND + hand_cols].copy()
        # ★ 버그 수정: TK_KEYS_HAND의 모든 키를 df 쪽 dtype에 맞춰 캐스팅
        for k in TK_KEYS_HAND:
            tk_h[k] = tk_h[k].astype(df[k].dtype)
        orig_index = df.index
        df = df.merge(tk_h, on=TK_KEYS_HAND, how="left", sort=False)
        df.index = orig_index
        # 카운트 기준 리그 전체 대비 손 조합별 구속 이탈량 (추가 파생)
        df["tk_ph_bh_speed_dev"] = df["tk_ph_bh_zone_speed_mean"] - df["tk_zone_speed_mean"]

    return df.drop(columns=[ID_COL], errors="ignore")


# ──────────────────────────────────────────────
# kfold_oof: logistic_stack_diagnostic.py와 동일한 구조
# (best_iteration 외부에서 받아서 고정)
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


# ──────────────────────────────────────────────
# 단일 실험 실행
# ──────────────────────────────────────────────

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

    # LGB/CB용 dtype 변환 (category / str)
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

    # ── Step 3: KFold OOF (best_iteration 고정) ──
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


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("v13 후보 실험 v2: 손 조합 × 카운트 Trackman lookup 확장")
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

    # ── Trackman lookup ──
    HAND_MAP = {"Right": 2, "Left": 1}
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS_HAND + ["pitch_type_group", "rel_speed", "spin_rate",
                                              "induced_vert_break", "horz_break",
                                              "extension", "zone_speed"],
    )
    tk_raw["pitcher_hand"] = tk_raw["pitcher_hand"].map(HAND_MAP)
    tk_raw["batter_hand"] = tk_raw["batter_hand"].map(HAND_MAP)
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])
    TK_LOOKUP_HAND_VAL = make_tk_lookup_hand(tk_raw[tk_raw["season"] <= 2023])
    print(f"tk_lookup (val): {TK_LOOKUP_VAL.shape}")
    print(f"tk_lookup_hand (val): {TK_LOOKUP_HAND_VAL.shape}")

    # ── 피처 엔지니어링 ──
    print("\n피처 엔지니어링 (WITHOUT)...")
    fe_without = build_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL, tk_lookup_hand=None)

    print("피처 엔지니어링 (WITH)...")
    fe_with = build_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL, tk_lookup_hand=TK_LOOKUP_HAND_VAL)

    # 손 조합 lookup 커버리지 확인
    cov = fe_with["tk_ph_bh_fastball_rate"].notna().mean()
    print(f"  손 조합 lookup 커버리지: {cov:.4%}")

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
        "WITH (손 조합×카운트 lookup 추가)",
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
    import csv
    out = "v13_hand_count_tk_results_v2.csv"
    df_res = pd.DataFrame(results)
    df_res["diff_A"] = diff_A
    df_res["diff_B"] = diff_B
    df_res["verdict"] = verdict
    df_res.to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
