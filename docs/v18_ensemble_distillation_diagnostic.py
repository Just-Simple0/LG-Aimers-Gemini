"""
v18 심화 실험: Teacher (LGB, CB) + Student (Distilled LGB) 3-Model Stacking / Blending
=====================================================================================

배경 및 가설:
  - v18 Soft Distillation에서 Student LGB 단독 성능이 809.27 -> 814.94로 향상되고,
    CatBoost와의 2-way 스태킹 홀드아웃 점수가 836.35 -> 846.75 (+10.40)로 역대 최고를 달성함.
  - 가설:
    기존 교사(Hard LGB, Hard CB)와 학생(Soft Distilled LGB)을 모두 보존하고
    3개 모델을 Ridge / Non-negative Stacking 또는 최적 가중 블렌딩하면
    앙상블 다양성이 극대화되어 850+ 돌파가 가능할 것임.

작성일: 2026-08-27
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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

BEST_LGB_ITER = 171
BEST_CB_ITER = 360


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
    print("v18 심화 실험: Teacher (LGB, CB) + Student (Distilled) 앙상블 아키텍처")
    print("기준선: v14 (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 70)
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

    X_tr_lgb = X_train_df.copy()
    X_val_lgb = X_val_df.copy()

    X_tr_cb = X_train_df.copy()
    X_val_cb = X_val_df.copy()
    for c in CAT_COLS:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    # 2. 교사 모델 훈련 (LGB & CB)
    print("\n[1/3] 교사 모델 학습 및 OOF 생성...")
    lgb_teacher = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
    lgb_teacher.fit(X_tr_lgb, y_train)
    p_lgb_val = lgb_teacher.predict_proba(X_val_lgb)[:, 1]

    tp_tr = Pool(X_tr_cb, y_train, cat_features=cat_idx)
    cb_teacher = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    cb_teacher.fit(tp_tr)
    p_cb_val = cb_teacher.predict_proba(X_val_cb)[:, 1]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y_train))
    cb_oof = np.zeros(len(y_train))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train_df)):
        m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
        m_lgb.fit(X_tr_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[oof_idx] = m_lgb.predict_proba(X_tr_lgb.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_tr_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        m_cb.fit(tp_f)
        cb_oof[oof_idx] = m_cb.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]

    p_teacher_oof = 0.5 * (lgb_oof + cb_oof)

    # 3. Student (w_soft=0.30 & 0.45) 학습 및 OOF
    print("\n[2/3] Student 증류 모델 학습...")
    W_BEST = 0.30  # 균형잡힌 0.30 기준
    y_soft_train = (1.0 - W_BEST) * y_train + W_BEST * p_teacher_oof

    student_lgb = lgb.LGBMRegressor(
        n_estimators=BEST_LGB_ITER,
        **{k: v for k, v in LGB_PARAMS.items() if k != "objective"},
        objective="cross_entropy",
    )
    student_lgb.fit(X_tr_lgb, y_soft_train)
    p_student_val = student_lgb.predict(X_val_lgb)

    student_oof = np.zeros(len(y_train))
    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train_df)):
        s_m = lgb.LGBMRegressor(
            n_estimators=BEST_LGB_ITER,
            **{k: v for k, v in LGB_PARAMS.items() if k != "objective"},
            objective="cross_entropy",
        )
        s_m.fit(X_tr_lgb.iloc[tr_idx], y_soft_train[tr_idx])
        student_oof[oof_idx] = s_m.predict(X_tr_lgb.iloc[oof_idx])

    # 4. 다양한 앙상블 조합 비교
    print("\n[3/3] 앙상블 아키텍처 비교...")
    # (1) v14 기준선: [p_lgb, p_cb] 로지스틱 스태킹
    st_v14 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([lgb_oof, cb_oof]), y_train)
    s14_val = brier_skill_score(y_val, st_v14.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1])
    s14_oof = brier_skill_score(y_train, st_v14.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1])

    # (2) Student + CB 2-way 스태킹
    st_st_cb = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([student_oof, cb_oof]), y_train)
    s_sc_val = brier_skill_score(y_val, st_st_cb.predict_proba(np.column_stack([p_student_val, p_cb_val]))[:, 1])
    s_sc_oof = brier_skill_score(y_train, st_st_cb.predict_proba(np.column_stack([student_oof, cb_oof]))[:, 1])

    # (3) 3-Way 스태킹: [p_lgb, p_cb, p_student] with L2 Ridge Penalty (C=0.1)
    st_3w = LogisticRegression(C=0.1, random_state=42, max_iter=1000).fit(np.column_stack([lgb_oof, cb_oof, student_oof]), y_train)
    s_3w_val = brier_skill_score(y_val, st_3w.predict_proba(np.column_stack([p_lgb_val, p_cb_val, p_student_val]))[:, 1])
    s_3w_oof = brier_skill_score(y_train, st_3w.predict_proba(np.column_stack([lgb_oof, cb_oof, student_oof]))[:, 1])

    # (4) 3-Model 최적 가중 블렌드 (OOF 최적화: w_lgb * p_lgb + w_cb * p_cb + w_st * p_st)
    best_blend_score = -1
    best_weights = None
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-5:
                continue
            oof_blend = w1 * lgb_oof + w2 * cb_oof + w3 * student_oof
            score = brier_skill_score(y_train, oof_blend)
            if score > best_blend_score:
                best_blend_score = score
                best_weights = (w1, w2, w3)

    w1, w2, w3 = best_weights
    p_blend_val = w1 * p_lgb_val + w2 * p_cb_val + w3 * p_student_val
    s_blend_val = brier_skill_score(y_val, p_blend_val)
    s_blend_oof = best_blend_score

    print("\n" + "=" * 70)
    print("최종 아키텍처 비교 요약")
    print("=" * 70)
    print(f"{'앙상블 아키텍처':<40} {'(A) 2024 홀드아웃':>15} {'(B) OOF score':>15} {'diff (A)':>10}")
    print(f"{'1. v14 기준선 (Hard LGB + CB Stack)':<40} {s14_val:>15.2f} {s14_oof:>15.2f} {'+0.00':>10}")
    print(f"{'2. Distilled (Student LGB + CB Stack)':<40} {s_sc_val:>15.2f} {s_sc_oof:>15.2f} {s_sc_val-s14_val:>+10.2f}")
    print(f"{'3. 3-Way Ridge Stack [LGB+CB+Student]':<40} {s_3w_val:>15.2f} {s_3w_oof:>15.2f} {s_3w_val-s14_val:>+10.2f}")
    print(f"{'4. 3-Way Simplex Blend (w={w1:.2f},{w2:.2f},{w3:.2f})':<40} {s_blend_val:>15.2f} {s_blend_oof:>15.2f} {s_blend_val-s14_val:>+10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
