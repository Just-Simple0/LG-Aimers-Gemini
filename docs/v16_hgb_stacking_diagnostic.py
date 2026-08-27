"""
v16 후보 실험: HistGradientBoosting(HGB) 3번째 베이스 모델 추가 (3-Way Stacking)
================================================================================

배경 및 가설:
  - 현재 파이프라인은 LightGBM + CatBoost 2개 모델의 OOF 예측을 LogisticRegression으로 스태킹 중.
  - Phase2 R18/R26/R35에서 sklearn의 HistGradientBoostingClassifier(HGB)는
    LightGBM/CatBoost와 다른 히스토그램 분위수 binning 및 트리 분할 구현을 가짐으로써
    높은 앙상블 다양성을 제공하고 블렌드 가중치의 50% 이상을 차지한 바 있음.
  - scikit-learn 표준 내장 모델이므로 추가 라이브러리 설치 없이 즉시 사용 가능.
  - 가설:
    LightGBM + CatBoost + HGB 3개 모델의 5-Fold OOF 예측 [p_lgb, p_cb, p_hgb]를
    LogisticRegression 메타러너에 입력하면 2-way 스태킹 대비 앙상블 다양성이 증가하여
    홀드아웃 및 OOF 점수가 동반 상승할 것임.

이중 게이트 절차:
  (A) 2024 홀드아웃 점수
  (B) OOF score (독립 게이트)

기준선: v14 2-way Stacking (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.script import SeasonDecompositionEncoder, build_features, ALPHA

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
HGB_PARAMS = dict(
    learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
    l2_regularization=1.0, random_state=42, early_stopping=True,
    n_iter_no_change=20, max_iter=1000,
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


def main():
    print("=" * 70)
    print("v16 후보 실험: HistGradientBoosting (HGB) 3-Way Stacking")
    print("기준선: v14 2-Way Stacking (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 70)
    t_start = time.time()

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드 및 피처 생성 (v14 피처셋)...")
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
    season_decomp_train = make_temporal_season_features(train_split, TARGET)

    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    val_split = train_full[train_full["season"] == 2024]
    season_decomp_val = encoder_val.transform(val_split)

    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]
    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
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

    # 모델별 입력 데이터 준비
    X_tr_lgb = X_train_df.copy()
    X_val_lgb = X_val_df.copy()

    X_tr_cb = X_train_df.copy()
    X_val_cb = X_val_df.copy()
    for c in CAT_COLS:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    # HGB는 categorical_features를 지원 (정수/카테고리 인코딩 필요)
    X_tr_hgb = X_train_df.copy()
    X_val_hgb = X_val_df.copy()
    for c in CAT_COLS:
        X_tr_hgb[c] = X_tr_hgb[c].cat.codes
        X_val_hgb[c] = X_val_hgb[c].cat.codes

    # 2. 베이스 모델 홀드아웃 학습 (best_iteration 탐색)
    print("\n[2/4] 베이스 모델 학습 및 단독 홀드아웃 검증...")
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    lgb_model.fit(X_tr_lgb, y_train, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    BEST_LGB = lgb_model.best_iteration_
    p_lgb_val = lgb_model.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val, p_lgb_val)
    print(f"  LightGBM 완료 :: {time.time()-t0:.1f}s | best_iter={BEST_LGB} | solo score={score_lgb:.2f}")

    t0 = time.time()
    tp_tr = Pool(X_tr_cb, y_train, cat_features=cat_idx)
    tp_val = Pool(X_val_cb, y_val, cat_features=cat_idx)
    cb_model = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS)
    cb_model.fit(tp_tr, eval_set=tp_val, use_best_model=True)
    BEST_CB = cb_model.get_best_iteration() + 1
    p_cb_val = cb_model.predict_proba(X_val_cb)[:, 1]
    score_cb = brier_skill_score(y_val, p_cb_val)
    print(f"  CatBoost 완료 :: {time.time()-t0:.1f}s | best_iter={BEST_CB} | solo score={score_cb:.2f}")

    t0 = time.time()
    hgb_model = HistGradientBoostingClassifier(categorical_features=cat_idx, **HGB_PARAMS)
    hgb_model.fit(X_tr_hgb, y_train)
    BEST_HGB = hgb_model.n_iter_
    p_hgb_val = hgb_model.predict_proba(X_val_hgb)[:, 1]
    score_hgb = brier_skill_score(y_val, p_hgb_val)
    print(f"  HistGradientBoosting 완료 :: {time.time()-t0:.1f}s | best_iter={BEST_HGB} | solo score={score_hgb:.2f}")

    # 3. 5-Fold KFold OOF 생성
    print("\n[3/4] 5-Fold KFold OOF 생성 (LGB + CB + HGB)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y_train))
    cb_oof = np.zeros(len(y_train))
    hgb_oof = np.zeros(len(y_train))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train_df)):
        ft = time.time()
        # LGB
        m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB, **LGB_PARAMS)
        m_lgb.fit(X_tr_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[oof_idx] = m_lgb.predict_proba(X_tr_lgb.iloc[oof_idx])[:, 1]

        # CB
        tp_fold = Pool(X_tr_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=BEST_CB, **CB_PARAMS)
        m_cb.fit(tp_fold)
        cb_oof[oof_idx] = m_cb.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]

        # HGB
        m_hgb = HistGradientBoostingClassifier(max_iter=BEST_HGB, categorical_features=cat_idx, **{k: v for k, v in HGB_PARAMS.items() if k not in ("max_iter", "early_stopping")}, early_stopping=False)
        m_hgb.fit(X_tr_hgb.iloc[tr_idx], y_train[tr_idx])
        hgb_oof[oof_idx] = m_hgb.predict_proba(X_tr_hgb.iloc[oof_idx])[:, 1]

        print(f"    KFold {fold_i+1}/5 :: {time.time()-ft:.1f}s")

    # 4. 2-Way vs 3-Way 스태킹 비교
    print("\n[4/4] 2-Way Stacking (LGB+CB) vs 3-Way Stacking (LGB+CB+HGB) 평가...")

    # 2-Way
    stack_2w = LogisticRegression(random_state=42, max_iter=1000)
    stack_2w.fit(np.column_stack([lgb_oof, cb_oof]), y_train)
    preds_2w_val = stack_2w.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    preds_2w_oof = stack_2w.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    score_2w_val = brier_skill_score(y_val, preds_2w_val)
    score_2w_oof = brier_skill_score(y_train, preds_2w_oof)

    # 3-Way
    stack_3w = LogisticRegression(random_state=42, max_iter=1000)
    stack_3w.fit(np.column_stack([lgb_oof, cb_oof, hgb_oof]), y_train)
    preds_3w_val = stack_3w.predict_proba(np.column_stack([p_lgb_val, p_cb_val, p_hgb_val]))[:, 1]
    preds_3w_oof = stack_3w.predict_proba(np.column_stack([lgb_oof, cb_oof, hgb_oof]))[:, 1]
    score_3w_val = brier_skill_score(y_val, preds_3w_val)
    score_3w_oof = brier_skill_score(y_train, preds_3w_oof)

    print(f"\n  2-Way 메타러너 계수: LGB={stack_2w.coef_[0][0]:.4f}, CB={stack_2w.coef_[0][1]:.4f}, intercept={stack_2w.intercept_[0]:.4f}")
    print(f"  3-Way 메타러너 계수: LGB={stack_3w.coef_[0][0]:.4f}, CB={stack_3w.coef_[0][1]:.4f}, HGB={stack_3w.coef_[0][2]:.4f}, intercept={stack_3w.intercept_[0]:.4f}")

    diff_val = score_3w_val - score_2w_val
    diff_oof = score_3w_oof - score_2w_oof

    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)
    print(f"{'항목':<35} {'2-Way (LGB+CB)':>15} {'3-Way (+HGB)':>15} {'diff':>10}")
    print(f"{'LightGBM solo':<35} {score_lgb:>15.2f} {score_lgb:>15.2f}")
    print(f"{'CatBoost solo':<35} {score_cb:>15.2f} {score_cb:>15.2f}")
    print(f"{'HGB solo':<35} {'-':>15} {score_hgb:>15.2f}")
    print(f"{'(A) 2024 홀드아웃':<35} {score_2w_val:>15.2f} {score_3w_val:>15.2f} {diff_val:>+10.2f}")
    print(f"{'(B) OOF score (독립 게이트)':<35} {score_2w_oof:>15.2f} {score_3w_oof:>15.2f} {diff_oof:>+10.2f}")

    if diff_val > 0.0 and diff_oof > 0.0:
        verdict = "채택 후보 ✅  3-Way Stacking 성능 향상 확인 (Dacon 재제출 권장)"
    elif diff_oof > 0.0:
        verdict = "보류 ⚠️   OOF만 소폭 개선, 홀드아웃 미개선"
    else:
        verdict = "기각 ❌  개선 신호 부재"

    print(f"\n판정: {verdict}")
    print(f"총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print("=" * 70)

    out = "docs/v16_hgb_stacking_results.csv"
    res_df = pd.DataFrame([
        {"label": "2-Way Stacking (v14)", "holdout": score_2w_val, "oof": score_2w_oof},
        {"label": "3-Way Stacking (v16)", "holdout": score_3w_val, "oof": score_3w_oof},
    ])
    res_df["diff_holdout"] = diff_val
    res_df["diff_oof"] = diff_oof
    res_df["verdict"] = verdict
    res_df.to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
