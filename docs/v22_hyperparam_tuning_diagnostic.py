"""
v22 후보 실험: GBDT 하이퍼파라미터 정밀 최적화 (v14 101개 피처셋 기준)
========================================================================

배경 및 가설:
  - 현재 LGBM(num_leaves=63, lambda=1.0)과 CatBoost(depth=8, reg=3.0) 파라미터는
    v1~v3(초기 38개 피처 시절)에 고정된 기본값임.
  - v14에서 Season Decomposition(101개 피처)이 도입되면서 피처의 연속형/정보량 구조가 근본적으로 변화함.
  - 가설:
    1) CatBoost: 101개 피처 환경에서 depth=7 또는 depth=8에 l2_leaf_reg(3.0, 5.0, 8.0) 정규화를 강화하여 일반화 능력 향상.
    2) LightGBM: colsample_bytree(0.6, 0.8) 및 reg_lambda(1.0, 5.0, 10.0) 스윕으로 과적합 방지.
    3) 최적 베이스 파라미터 조합으로 (A) 홀드아웃 및 (B) OOF 동반 개선 달성.

기준선: v14 기본 파라미터 (홀드아웃 836.35, OOF 2009.23, Dacon 실측 976.51)

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

from src.script import SeasonDecompositionEncoder, build_features, ALPHA

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]


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


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def evaluate_pipeline(X_tr_lgb, X_val_lgb, X_tr_cb, X_val_cb, y_train, y_val, lgb_params, cb_params, cat_idx):
    t0 = time.time()
    
    # 1. Early stopping
    lgb_m = lgb.LGBMClassifier(n_estimators=2000, **lgb_params)
    lgb_m.fit(X_tr_lgb, y_train, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    best_lgb_iter = lgb_m.best_iteration_
    p_lgb_val = lgb_m.predict_proba(X_val_lgb)[:, 1]
    score_lgb = brier_skill_score(y_val, p_lgb_val)

    tp_tr = Pool(X_tr_cb, y_train, cat_features=cat_idx)
    tp_val = Pool(X_val_cb, y_val, cat_features=cat_idx)
    cb_m = CatBoostClassifier(iterations=3000, early_stopping_rounds=50, **cb_params)
    cb_m.fit(tp_tr, eval_set=tp_val, use_best_model=True)
    best_cb_iter = cb_m.get_best_iteration() + 1
    p_cb_val = cb_m.predict_proba(X_val_cb)[:, 1]
    score_cb = brier_skill_score(y_val, p_cb_val)

    # 2. 5-Fold OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y_train))
    cb_oof = np.zeros(len(y_train))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_tr_lgb)):
        m_l = lgb.LGBMClassifier(n_estimators=best_lgb_iter, **lgb_params)
        m_l.fit(X_tr_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[oof_idx] = m_l.predict_proba(X_tr_lgb.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_tr_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=best_cb_iter, **cb_params)
        m_c.fit(tp_f)
        cb_oof[oof_idx] = m_c.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]

    # 3. Logistic Stacking
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([lgb_oof, cb_oof]), y_train)

    score_val = brier_skill_score(y_val, stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1])
    score_oof = brier_skill_score(y_train, stack.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1])
    elapsed = time.time() - t0

    return {
        "lgb_iter": best_lgb_iter,
        "cb_iter": best_cb_iter,
        "lgb_solo": score_lgb,
        "cb_solo": score_cb,
        "holdout": score_val,
        "oof": score_oof,
        "elapsed_min": elapsed / 60,
    }


def main():
    print("=" * 70)
    print("v22 후보 실험: GBDT 하이퍼파라미터 정밀 최적화 (v14 101개 피처셋 기준)")
    print("기준선: v14 기본 파라미터 (홀드아웃 836.35, OOF 2009.23, Dacon 976.51)")
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
    
    # build_features_base
    from docs.v14_season_decomposition_diagnostic import build_base_features
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

    # 2. 파라미터 후보군 정의
    CONFIGS = [
        ("v14 기준 (LGB: leaves=63, reg=1.0 | CB: depth=8, reg=3.0)",
         dict(learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1),
         dict(learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)),

        ("후보 1 (CatBoost L2 정규화 강화: reg=5.0)",
         dict(learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1),
         dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)),

        ("후보 2 (LightGBM colsample_bytree=0.7, reg_lambda=3.0)",
         dict(learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1),
         dict(learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)),

        ("후보 3 (LGB & CB 동시 정규화 강화: LGB reg=3.0, CB reg=5.0)",
         dict(learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1),
         dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)),
    ]

    results = []
    print(f"\n총 {len(CONFIGS)}개 하이퍼파라미터 조합 평가 시작...")

    for label, lgb_p, cb_p in CONFIGS:
        print(f"\n--- {label} ---")
        res = evaluate_pipeline(X_tr_lgb, X_val_lgb, X_tr_cb, X_val_cb, y_train, y_val, lgb_p, cb_p, cat_idx)
        res["label"] = label
        print(f"  LGB={res['lgb_solo']:.2f}(iter={res['lgb_iter']}) CB={res['cb_solo']:.2f}(iter={res['cb_iter']}) | (A)Holdout={res['holdout']:.2f} | (B)OOF={res['oof']:.2f} | {res['elapsed_min']:.1f}분")
        results.append(res)

    # 3. 요약
    print("\n" + "=" * 70)
    print("최종 결과 요약 (Hyperparameter Optimization)")
    print("=" * 70)
    print(f"{'설정':<45} {'LGB solo':>10} {'CB solo':>10} {'(A) 홀드아웃':>14} {'(B) OOF score':>14}")
    base_holdout = results[0]["holdout"]
    base_oof = results[0]["oof"]
    for r in results:
        diff_h = r["holdout"] - base_holdout
        diff_o = r["oof"] - base_oof
        print(f"{r['label']:<45} {r['lgb_solo']:>10.2f} {r['cb_solo']:>10.2f} {r['holdout']:>14.2f}({diff_h:>+5.2f}) {r['oof']:>14.2f}({diff_o:>+5.2f})")

    best_r = max(results, key=lambda x: (x["holdout"] + x["oof"]))
    print(f"\n최적 설정: {best_r['label']}")
    print(f"총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print("=" * 70)

    out = "docs/v22_hyperparam_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
