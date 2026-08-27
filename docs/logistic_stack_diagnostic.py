"""v10-logistic-stacking 독립 진단 스크립트.

main(eea05e5=v7+li) 기준 notebooks/train.ipynb 섹션 1~6(2024 홀드아웃 검증 경로)을
문자단위로 재현한 뒤, 섹션 6의 "W_GRID 격자 탐색 + idxmax() 블렌드"를 로지스틱 회귀
메타러너(스태킹)로 교체했을 때 2024 홀드아웃 Brier Skill Score가 어떻게 바뀌는지 비교한다.

비교 대상 3가지 (모두 동일 kfold_oof() OOF/val_pred_lgb/val_pred_cb/y_oof/y_val 사용):
  (1) 기존 파이프라인: W_GRID 격자탐색 + isotonic 보정 블렌드 (재현 기준점: 774.07,
      calibrate_then_blend/w_lgb=0.45 — README.md/results.tsv에 기록된 실측 확정 수치)
  (2) 로지스틱 스태킹, 보정 없음(raw): LogisticRegression([lgb_oof, cb_oof] -> y_oof)의
      predict_proba를 그대로 2024 홀드아웃에 적용
  (3) 로지스틱 스태킹 + isotonic 보정: (2)의 OOF 예측에 isotonic을 한 번 더 fit해서
      (blend_then_calibrate와 동일한 패턴으로) 2024 홀드아웃에 적용

OOF 재사용에 대한 판단: 메타러너 학습에 쓰는 [lgb_oof, cb_oof] -> y_oof 는 기존
calibration에 쓰던 것과 동일한 KFold(shuffle=True) OOF다. 오늘 리서치 근거대로, 피처가
2개뿐인 선형 로지스틱 회귀는 표현력이 낮아(사실상 non-negative/sum-to-1 제약이 없는
가중 블렌드의 일반화판) 과적합 위험이 낮다고 판단해 별도의 중첩 KFold를 두지 않았다 —
GBM 등 고용량 메타러너였다면 이 판단은 달랐을 것이다. 대신 이 프로젝트의 표준 관행대로
"홀드아웃에서 직접 격자탐색 최댓값을 고르는" 방식과 달리, 로지스틱 회귀 자체는 OOF에서
1회 fit되고(하이퍼파라미터 탐색 없음) 홀드아웃은 평가 전용으로만 쓰인다는 점도 원래
이슈#4/v8-nn/season-workload 계열에서 지적된 "홀드아웃 직접 최적화" 과적합 패턴과는
성격이 다르다.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DATA_DIR = "../data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
W_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)


def make_tk_lookup(tk_df, tk_keys):
    lookup = tk_df.groupby(tk_keys).agg(
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
    assert not lookup.duplicated(tk_keys).any(), "tk_lookup에 중복 카운트 키가 있음"
    return lookup


def build_features(df, global_mean, tk_lookup):
    """train.ipynb/inference.ipynb/script.py의 build_features()와 문자단위 동일."""
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

    tk_keys = ["balls_before", "strikes_before", "outs_before"]
    tk_cols = [c for c in tk_lookup.columns if c.startswith("tk_")]
    tk_lookup = tk_lookup[tk_keys + tk_cols].copy()
    for k in tk_keys:
        tk_lookup[k] = tk_lookup[k].astype(df[k].dtype)
    orig_index = df.index
    df = df.merge(
        tk_lookup, on=tk_keys, how="left",
        validate="many_to_one", sort=False,
    )
    df.index = orig_index
    df["tk_fastball_dev"] = fb - df["tk_fastball_rate"]
    df["tk_breaking_dev"] = br - df["tk_breaking_rate"]
    df["tk_offspeed_dev"] = os_ - df["tk_offspeed_rate"]

    return df.drop(columns=[ID_COL], errors="ignore")


def kfold_oof(X, X_cb, y, lgb_n_estimators, cb_iterations, cat_idx,
              n_splits=5, random_state=42):
    """train.ipynb 섹션 3.5의 kfold_oof()와 문자단위 동일."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_r = X.reset_index(drop=True)
    X_cb_r = X_cb.reset_index(drop=True)
    y_r = y.reset_index(drop=True)

    lgb_oof = np.zeros(len(X_r))
    cb_oof = np.zeros(len(X_r))
    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_r)):
        ft = time.time()
        m_lgb = lgb.LGBMClassifier(n_estimators=lgb_n_estimators, **LGB_PARAMS)
        m_lgb.fit(X_r.iloc[tr_idx], y_r.iloc[tr_idx])
        lgb_oof[oof_idx] = m_lgb.predict_proba(X_r.iloc[oof_idx])[:, 1]

        tp = Pool(X_cb_r.iloc[tr_idx], y_r.iloc[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=cb_iterations, **CB_PARAMS)
        m_cb.fit(tp)
        cb_oof[oof_idx] = m_cb.predict_proba(X_cb_r.iloc[oof_idx])[:, 1]
        print(f"  KFold {fold_i + 1}/{n_splits} 완료 :: {time.time() - ft:.1f}s")

    return lgb_oof, cb_oof, y_r.to_numpy()


def brier_score(y_true, p):
    r = y_true.mean()
    brier = ((p - y_true) ** 2).mean()
    baseline_brier = r * (1 - r)
    return max(0, 100000 * (1 - brier / baseline_brier)), brier


def main():
    t0 = time.time()
    # ---- 섹션 2: 데이터 로드 ----
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    FEATURES = [c for c in test_cols if c != ID_COL]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=FEATURES + [TARGET])
    print("train:", train.shape, "| 원본 피처:", len(FEATURES))
    GLOBAL_MEAN_VAL = train.loc[train["season"] != 2024, TARGET].mean()

    # ---- 섹션 2.5: trackman count-state lookup (검증용, season<=2023) ----
    TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                 "induced_vert_break", "horz_break", "extension", "zone_speed"],
    )
    TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023], TK_KEYS)
    print("TK_LOOKUP_VAL:", TK_LOOKUP_VAL.shape)

    # ---- 섹션 3: 피처 엔지니어링 ----
    train_fe = build_features(train, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    NEW_COLS = [c for c in train_fe.columns if c not in train.columns]
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS] + NEW_COLS
    ALL_FEATURES = CAT_COLS + NUM_COLS
    for c in CAT_COLS:
        train_fe[c] = train_fe[c].astype("category")
    print("전체 피처 수:", len(ALL_FEATURES))

    # ---- 섹션 4: 학습/검증 분리 + 모델 학습 (2024 홀드아웃) ----
    is_val = train_fe["season"] == 2024
    X_train, y_train = train_fe.loc[~is_val, ALL_FEATURES], train_fe.loc[~is_val, TARGET]
    X_val, y_val = train_fe.loc[is_val, ALL_FEATURES], train_fe.loc[is_val, TARGET]
    print("train:", len(X_train), "| val:", len(X_val))

    cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]
    X_train_cb = X_train.copy(); X_val_cb = X_val.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    t = time.time()
    val_model = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    val_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    BEST_ITERATION_LGB = val_model.best_iteration_
    print(f"LightGBM 학습 완료 :: {time.time() - t:.1f}s | best_iteration={BEST_ITERATION_LGB}")

    t = time.time()
    train_pool = Pool(X_train_cb, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val_cb, y_val, cat_features=cat_idx)
    cb_val_model = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **CB_PARAMS)
    cb_val_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    BEST_ITERATION_CB = cb_val_model.get_best_iteration() + 1
    print(f"CatBoost 학습 완료 :: {time.time() - t:.1f}s | best_iteration={BEST_ITERATION_CB}")

    # ---- 섹션 5: 단독 검증 ----
    val_pred_lgb = val_model.predict_proba(X_val)[:, 1]
    val_pred_cb = cb_val_model.predict_proba(X_val_cb)[:, 1]
    s_lgb, _ = brier_score(y_val, val_pred_lgb)
    s_cb, _ = brier_score(y_val, val_pred_cb)
    print(f"LightGBM 단독: score={s_lgb:.2f}")
    print(f"CatBoost 단독:  score={s_cb:.2f}")

    # ---- 섹션 3.5/6: KFold OOF ----
    t = time.time()
    lgb_oof, cb_oof, y_oof = kfold_oof(
        X_train, X_train_cb, y_train, BEST_ITERATION_LGB, BEST_ITERATION_CB, cat_idx,
    )
    print(f"OOF 생성 완료 :: {time.time() - t:.1f}s")

    # ==========================================================
    # 비교 (1): 기존 파이프라인 — W_GRID 격자탐색 + isotonic 보정 블렌드
    # (main의 sanity-check 재현 대상: 774.07, calibrate_then_blend/w_lgb=0.45)
    # ==========================================================
    iso_lgb_val = IsotonicRegression(out_of_bounds="clip").fit(lgb_oof, y_oof)
    iso_cb_val = IsotonicRegression(out_of_bounds="clip").fit(cb_oof, y_oof)
    cal_val_pred_lgb = iso_lgb_val.predict(val_pred_lgb)
    cal_val_pred_cb = iso_cb_val.predict(val_pred_cb)

    search_rows = []
    for w in W_GRID:
        oof_blend_w = w * lgb_oof + (1 - w) * cb_oof
        cal_btc = IsotonicRegression(out_of_bounds="clip").fit(oof_blend_w, y_oof)
        pred_btc = cal_btc.predict(w * val_pred_lgb + (1 - w) * val_pred_cb)
        s_btc, _ = brier_score(y_val, pred_btc)
        search_rows.append(("blend_then_calibrate", w, s_btc))

        pred_ctb = w * cal_val_pred_lgb + (1 - w) * cal_val_pred_cb
        s_ctb, _ = brier_score(y_val, pred_ctb)
        search_rows.append(("calibrate_then_blend", w, s_ctb))

    search_df = pd.DataFrame(search_rows, columns=["order", "w_lgb", "score"])
    best_row = search_df.loc[search_df["score"].idxmax()]
    print("\n" + "=" * 60)
    print("(1) 기존 파이프라인: W_GRID 격자탐색 + isotonic 보정 블렌드")
    print("=" * 60)
    print(search_df.sort_values("score", ascending=False).head(5).to_string(index=False))
    print(f"선택: order={best_row['order']}, w_lgb={best_row['w_lgb']:.2f}, "
          f"score={best_row['score']:.2f}  (main 기록된 재현 기준: 774.07)")

    # ==========================================================
    # 비교 (2)/(3): 로지스틱 회귀 메타러너 스태킹
    # ==========================================================
    print("\n" + "=" * 60)
    print("(2)/(3) 로지스틱 회귀 메타러너 스태킹")
    print("=" * 60)

    X_meta_oof = np.column_stack([lgb_oof, cb_oof])
    X_meta_val = np.column_stack([val_pred_lgb, val_pred_cb])

    stack_model = LogisticRegression()
    stack_model.fit(X_meta_oof, y_oof)
    print(f"stack_model.coef_={stack_model.coef_}, intercept_={stack_model.intercept_}")

    # (2) 보정 없이 raw 출력
    stack_pred_val_raw = stack_model.predict_proba(X_meta_val)[:, 1]
    s_stack_raw, b_stack_raw = brier_score(y_val, stack_pred_val_raw)
    print(f"(2) 로지스틱 스태킹, 보정 없음(raw):        score={s_stack_raw:.2f} (brier={b_stack_raw:.6f})")

    # (3) 메타러너의 OOF 출력에 isotonic 한 번 더 fit (blend_then_calibrate와 동일 패턴)
    stack_pred_oof = stack_model.predict_proba(X_meta_oof)[:, 1]
    stack_calibrator = IsotonicRegression(out_of_bounds="clip").fit(stack_pred_oof, y_oof)
    stack_pred_val_cal = stack_calibrator.predict(stack_pred_val_raw)
    s_stack_cal, b_stack_cal = brier_score(y_val, stack_pred_val_cal)
    print(f"(3) 로지스틱 스태킹 + isotonic 보정:         score={s_stack_cal:.2f} (brier={b_stack_cal:.6f})")

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"(1) 기존(W_GRID 격자탐색+isotonic 블렌드) 재현: {best_row['score']:.2f}  (기준 774.07)")
    print(f"(2) 로지스틱 스태킹, raw:                     {s_stack_raw:.2f}")
    print(f"(3) 로지스틱 스태킹 + isotonic 보정:            {s_stack_cal:.2f}")
    print(f"\n총 소요 시간: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
