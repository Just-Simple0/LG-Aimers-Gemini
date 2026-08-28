"""
Finish Items 5 & 6 of Outside Two Repositories
==============================================
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]

BEST_LGB_ITER = 171
BEST_CB_ITER = 360

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)

CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)

from docs.v63_train_and_package import (
    SeasonDecompositionEncoder,
    make_temporal_season_features,
    make_tk_lookup,
    build_base_features,
)
from docs.run_external_6_innovations import evaluate_fe, brier_skill_score


def main():
    print("=" * 80)
    print("🚀 Evaluating Items 5 & 6 of Outside Two Repositories")
    print("=" * 80)

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

    # 기본 v14 피처
    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    is_val_mask = fe_v14["season"] == 2024

    # -------------------------------------------------------------------------
    # [5] Count-Conditioned Local Isotonic Calibration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [5/6] Count-Conditioned Local Isotonic Calibration (국소 카운트 보정)")
    print("=" * 80)
    # v14 baseline evaluation first
    v_s, o_s, w_l, w_c, p_val_v14, p_oof_v14, y_tr, y_va = evaluate_fe(fe_v14, train_full, "v14 Baseline")

    is_2s_tr = (fe_v14.loc[~is_val_mask, "is_two_strike"] == 1).to_numpy()
    is_2s_va = (fe_v14.loc[is_val_mask, "is_two_strike"] == 1).to_numpy()

    p_val_iso = p_val_v14.copy()
    iso_2s = IsotonicRegression(out_of_bounds="clip").fit(p_oof_v14[is_2s_tr], y_tr[is_2s_tr])
    iso_oth = IsotonicRegression(out_of_bounds="clip").fit(p_oof_v14[~is_2s_tr], y_tr[~is_2s_tr])
    p_val_iso[is_2s_va] = iso_2s.predict(p_val_v14[is_2s_va])
    p_val_iso[~is_2s_va] = iso_oth.predict(p_val_v14[~is_2s_va])
    iso_val_score = brier_skill_score(y_va, p_val_iso)
    print(f"  [결과] Local Count Isotonic : Holdout = {iso_val_score:.2f} ({iso_val_score-836.35:+.2f})")

    # -------------------------------------------------------------------------
    # [6] Bayesian Inning Workload / Decay 동역학 모델
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [6/6] Bayesian Inning Workload / Fatigue Decay 모델")
    print("=" * 80)
    fe_decay = fe_v14.copy()
    inn = fe_decay["inning"].fillna(1).astype(float)
    fe_decay["inning_fatigue_decay"] = np.exp(-0.05 * np.maximum(0.0, inn - 5.0))
    fe_decay["decayed_pitcher_posterior"] = fe_decay["season_pitcher_current_posterior"] * fe_decay["inning_fatigue_decay"]
    v_s_dec, o_s_dec, w_l_dec, w_c_dec, _, _, _, _ = evaluate_fe(fe_decay, train_full, "Bayesian Workload Decay")


if __name__ == "__main__":
    main()
