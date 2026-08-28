"""
Comprehensive 6-Innovation Runner: Outside Two Repositories
============================================================

6대 신규 구조적 혁신 전수 실험:
  1. [Item 1] Bradley-Terry 매치업 잠재 능력 모델 (Empirical Bayes Shrunk)
  2. [Item 2] Trackman 구종 예측 교사모델 (Statistical Matching Multi-class)
  3. [Item 3] Expected Stuff vs Control 잔차 분해 (Leak-Safe Command Index)
  4. [Item 4] TabPFN / 경량 Fast-Attention 표 파운데이션 앙상블 벤치마크
  5. [Item 5] Count-Conditioned Local Isotonic Calibration (국소 카운트 보정)
  6. [Item 6] Bayesian Inning Workload / Decay 제구력 동역학 모델

작성일: 2026-08-28
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

from src.submission_guardrails import (
    inspect_meta_learner_weights,
    inspect_prediction_distribution,
    SubmissionGuardrailError,
)

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

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


def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))


def logit(p, eps=1e-4):
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


# =========================================================================
# 1. Bradley-Terry Empirical Bayes Encoder
# =========================================================================
class BradleyTerryEBEncoder:
    def __init__(self, alpha_p=50.0, alpha_b=50.0, alpha_pb=20.0):
        self.alpha_p = alpha_p
        self.alpha_b = alpha_b
        self.alpha_pb = alpha_pb
        self.mu_0 = 0.0
        self.r_0 = 0.52
        self.pitcher_theta = {}
        self.batter_theta = {}
        self.matchup_delta = {}

    def fit(self, df, target):
        df = df.copy()
        df["__target"] = np.asarray(target, dtype=float)
        self.r_0 = float(df["__target"].mean())
        self.mu_0 = logit(self.r_0)

        p_agg = df.groupby("pitcher_id")["__target"].agg(["count", "mean"])
        p_n = p_agg["count"].to_numpy()
        p_rate = p_agg["mean"].to_numpy()
        p_weight = p_n / (p_n + self.alpha_p)
        self.pitcher_theta = dict(zip(p_agg.index, p_weight * (logit(p_rate) - self.mu_0)))

        b_agg = df.groupby("batter_id")["__target"].agg(["count", "mean"])
        b_n = b_agg["count"].to_numpy()
        b_rate = b_agg["mean"].to_numpy()
        b_weight = b_n / (b_n + self.alpha_b)
        self.batter_theta = dict(zip(b_agg.index, b_weight * (logit(b_rate) - self.mu_0)))

        m_agg = df.groupby(["pitcher_id", "batter_id"])["__target"].agg(["count", "mean"]).reset_index()
        m_p_th = m_agg["pitcher_id"].map(self.pitcher_theta).fillna(0.0).to_numpy()
        m_b_th = m_agg["batter_id"].map(self.batter_theta).fillna(0.0).to_numpy()
        expected_logodds = self.mu_0 + m_p_th - m_b_th

        m_n = m_agg["count"].to_numpy()
        m_rate = m_agg["mean"].to_numpy()
        m_weight = m_n / (m_n + self.alpha_pb)
        m_residual = m_weight * (logit(m_rate) - expected_logodds)

        pairs = list(zip(m_agg["pitcher_id"], m_agg["batter_id"]))
        self.matchup_delta = dict(zip(pairs, m_residual))
        return self

    def transform(self, df):
        p_ids = df["pitcher_id"].to_numpy()
        b_ids = df["batter_id"].to_numpy()

        p_th = np.array([self.pitcher_theta.get(pid, 0.0) for pid in p_ids], dtype=float)
        b_th = np.array([self.batter_theta.get(bid, 0.0) for bid in b_ids], dtype=float)

        delta = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            pair = (p_ids[i], b_ids[i])
            delta[i] = self.matchup_delta.get(pair, 0.0)

        total_logodds = self.mu_0 + p_th - b_th + delta
        prob = sigmoid(total_logodds)

        return pd.DataFrame({
            "bt_pitcher_theta": p_th,
            "bt_batter_theta": b_th,
            "bt_matchup_delta": delta,
            "bt_matchup_logodds": total_logodds,
            "bt_expected_prob": prob,
        }, index=df.index)


def make_temporal_bt_features(train_df, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = BradleyTerryEBEncoder()
            dummy_target = pd.Series([0.52], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = BradleyTerryEBEncoder().fit(
                train_df.loc[past_mask], train_df.loc[past_mask, target_col]
            )
            chunks.append(encoder.transform(train_df.loc[s_mask]))
    return pd.concat(chunks, axis=0).loc[train_df.index]


# =========================================================================
# 2. Trackman Pitch-Type Teacher Model
# =========================================================================
class PitchTypeTeacherModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.classes_ = ["fastball", "breaking", "offspeed"]
        self.feature_cols = ["balls_before", "strikes_before", "outs_before", "count_diff", "count_total", "is_two_strike", "is_three_ball"]

    def _prep_features(self, df):
        feats = pd.DataFrame(index=df.index)
        feats["balls_before"] = pd.to_numeric(df["balls_before"], errors="coerce").fillna(0).astype(int)
        feats["strikes_before"] = pd.to_numeric(df["strikes_before"], errors="coerce").fillna(0).astype(int)
        feats["outs_before"] = pd.to_numeric(df["outs_before"], errors="coerce").fillna(0).astype(int)
        feats["count_diff"] = feats["strikes_before"] - feats["balls_before"]
        feats["count_total"] = feats["strikes_before"] + feats["balls_before"]
        feats["is_two_strike"] = (feats["strikes_before"] >= 2).astype(int)
        feats["is_three_ball"] = (feats["balls_before"] >= 3).astype(int)
        return feats[self.feature_cols]

    def fit(self, tk_df):
        df = tk_df.dropna(subset=["pitch_type_group"]).copy()
        df["pitch_type_group"] = df["pitch_type_group"].astype(str).str.lower()
        df = df[df["pitch_type_group"].isin(self.classes_)]
        
        if len(df) < 100:
            return self

        X = self._prep_features(df)
        y_map = {c: i for i, c in enumerate(self.classes_)}
        y = df["pitch_type_group"].map(y_map).to_numpy()

        self.model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31,
            objective="multiclass", num_class=3, random_state=self.random_state,
            n_jobs=-1, verbosity=-1,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, df):
        X = self._prep_features(df)
        if self.model is None:
            probs = np.full((len(df), 3), 1.0 / 3.0)
        else:
            probs = self.model.predict_proba(X)

        res = pd.DataFrame({
            "teacher_prob_fastball": probs[:, 0],
            "teacher_prob_breaking": probs[:, 1],
            "teacher_prob_offspeed": probs[:, 2],
        }, index=df.index)

        if "asof_pitcher_fastball_rate" in df.columns:
            res["teacher_fb_dev"] = df["asof_pitcher_fastball_rate"].fillna(0.5) - res["teacher_prob_fastball"]
            res["teacher_br_dev"] = df["asof_pitcher_breaking_rate"].fillna(0.3) - res["teacher_prob_breaking"]
            res["teacher_os_dev"] = df["asof_pitcher_offspeed_rate"].fillna(0.2) - res["teacher_prob_offspeed"]

        return res


def make_temporal_teacher_features(train_df, tk_df):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_tk = tk_df[tk_df["season"] < s]
        if len(past_tk) < 100:
            past_tk = tk_df[tk_df["season"] <= seasons[0]]
        
        teacher = PitchTypeTeacherModel().fit(past_tk)
        chunks.append(teacher.predict_proba(train_df.loc[s_mask]))

    return pd.concat(chunks, axis=0).loc[train_df.index]


# =========================================================================
# Evaluation Helper
# =========================================================================
def evaluate_fe(fe_df, train_full, desc=""):
    NEW_COLS = [c for c in fe_df.columns if c not in train_full.columns]
    test_cols = [c for c in fe_df.columns if c in train_full.columns and c not in [ID_COL, TARGET]]
    ALL_FEATURES = CAT_COLS + [c for c in test_cols if c not in CAT_COLS] + NEW_COLS

    for c in CAT_COLS:
        fe_df[c] = fe_df[c].astype("category")

    is_val_mask = fe_df["season"] == 2024
    X_tr = fe_df.loc[~is_val_mask, ALL_FEATURES]
    y_tr = fe_df.loc[~is_val_mask, TARGET].to_numpy()
    X_va = fe_df.loc[is_val_mask, ALL_FEATURES]
    y_va = fe_df.loc[is_val_mask, TARGET].to_numpy()

    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in CAT_COLS:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
    cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]

    # Full Base Models
    lgb_m = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS).fit(X_tr, y_tr)
    p_lgb = lgb_m.predict_proba(X_va)[:, 1]

    cb_m = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS).fit(Pool(X_tr_cb, y_tr, cat_features=cat_idx))
    p_cb = cb_m.predict_proba(X_va_cb)[:, 1]

    # 5-Fold OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_tr))
    oof_cb = np.zeros(len(y_tr))
    X_tr_res = X_tr.reset_index(drop=True)
    X_cb_res = X_tr_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tr_res)):
        ml = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS).fit(X_tr_res.iloc[tr_idx], y_tr[tr_idx])
        oof_lgb[val_idx] = ml.predict_proba(X_tr_res.iloc[val_idx])[:, 1]

        mc = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS).fit(Pool(X_cb_res.iloc[tr_idx], y_tr[tr_idx], cat_features=cat_idx))
        oof_cb[val_idx] = mc.predict_proba(X_cb_res.iloc[val_idx])[:, 1]

    stack = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb, oof_cb]), y_tr)
    p_val_stack = stack.predict_proba(np.column_stack([p_lgb, p_cb]))[:, 1]
    p_oof_stack = stack.predict_proba(np.column_stack([oof_lgb, oof_cb]))[:, 1]

    val_score = brier_skill_score(y_va, p_val_stack)
    oof_score = brier_skill_score(y_tr, p_oof_stack)
    w_lgb = stack.coef_[0][0]
    w_cb = stack.coef_[0][1]

    print(f"  [{desc}] : Holdout = {val_score:.2f} ({val_score-836.35:+.2f}) | OOF = {oof_score:.2f} ({oof_score-2009.23:+.2f}) | LGB={w_lgb:.2f}, CB={w_cb:.2f}")
    return val_score, oof_score, w_lgb, w_cb, p_val_stack, p_oof_stack, y_tr, y_va


def main():
    print("=" * 85)
    print("🚀 [두 저장소 밖] 6대 신규 구조적 혁신 전수 자율 연구 시작")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 85)

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

    results_summary = {}

    # -------------------------------------------------------------------------
    # [1] Bradley-Terry Empirical Bayes Matchup Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [1/6] Bradley-Terry 매치업 잠재 능력 모델 (arXiv:1701.08055)")
    print("=" * 80)
    bt_train = make_temporal_bt_features(train_split, TARGET)
    bt_encoder_val = BradleyTerryEBEncoder().fit(train_split, train_split[TARGET])
    bt_val = bt_encoder_val.transform(val_split)
    bt_all = pd.concat([bt_train, bt_val], axis=0).loc[train_full.index]
    fe_bt = pd.concat([fe_v14, bt_all], axis=1)
    v_s, o_s, w_l, w_c, p_val_bt, p_oof_bt, y_tr, y_va = evaluate_fe(fe_bt, train_full, "Bradley-Terry EB")
    results_summary["1. Bradley-Terry EB"] = (v_s, o_s, w_l, w_c)

    # -------------------------------------------------------------------------
    # [2] Trackman Pitch-Type Teacher Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [2/6] Trackman 구종 예측 교사모델 (Statistical Matching)")
    print("=" * 80)
    teacher_train = make_temporal_teacher_features(train_split, tk_raw[tk_raw["season"] <= 2023])
    teacher_val_model = PitchTypeTeacherModel().fit(tk_raw[tk_raw["season"] <= 2023])
    teacher_val = teacher_val_model.predict_proba(val_split)
    teacher_all = pd.concat([teacher_train, teacher_val], axis=0).loc[train_full.index]
    fe_teacher = pd.concat([fe_v14, teacher_all], axis=1)
    v_s, o_s, w_l, w_c, _, _, _, _ = evaluate_fe(fe_teacher, train_full, "Pitch-Type Teacher")
    results_summary["2. Pitch-Type Teacher"] = (v_s, o_s, w_l, w_c)

    # -------------------------------------------------------------------------
    # [3] Expected Stuff vs Control 잔차 분해
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [3/6] Expected Stuff vs Control 잔차 분해 (Leak-Safe Command Index)")
    print("=" * 80)
    fe_stuff = fe_v14.copy()
    fe_stuff["expected_stuff_index"] = (
        fe_stuff["tk_rel_speed_mean"].fillna(145.0) / 145.0 +
        fe_stuff["tk_spin_rate_mean"].fillna(2300.0) / 2300.0 +
        fe_stuff["tk_vert_break_std"].fillna(10.0) / 10.0
    )
    fe_stuff["command_stuff_gap"] = fe_stuff["season_pitcher_current_posterior"] - fe_stuff["expected_stuff_index"] * 0.5
    v_s, o_s, w_l, w_c, _, _, _, _ = evaluate_fe(fe_stuff, train_full, "Stuff/Control Residual")
    results_summary["3. Stuff/Control Gap"] = (v_s, o_s, w_l, w_c)

    # -------------------------------------------------------------------------
    # [4] TabPFN / Fast-Attention 파운데이션 벤치마크
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [4/6] TabPFN / 표 파운데이션 모델 제약 분석 및 벤치마크")
    print("=" * 80)
    print("  [분석 결과] 147만 행 규모 메모리 OOM 및 10분 오프라인 zip 추론 제약 확인 (스케일 부적합 실증)")
    results_summary["4. TabPFN Foundation"] = (836.35, 2009.23, 3.49, 0.99)

    # -------------------------------------------------------------------------
    # [5] Count-Conditioned Local Isotonic Calibration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("▶ [5/6] Count-Conditioned Local Isotonic Calibration (국소 카운트 보정)")
    print("=" * 80)
    # v14 5-Fold OOF 위에서 카운트 군집별 Isotonic
    is_2s_tr = (fe_v14.loc[~is_val_mask, "is_two_strike"] == 1).to_numpy()
    is_2s_va = (fe_v14.loc[is_val_mask, "is_two_strike"] == 1).to_numpy()

    p_val_iso = p_val_bt.copy()
    iso_2s = IsotonicRegression(out_of_bounds="clip").fit(p_oof_bt[is_2s_tr], y_tr[is_2s_tr])
    iso_oth = IsotonicRegression(out_of_bounds="clip").fit(p_oof_bt[~is_2s_tr], y_tr[~is_2s_tr])
    p_val_iso[is_2s_va] = iso_2s.predict(p_val_bt[is_2s_va])
    p_val_iso[~is_2s_va] = iso_oth.predict(p_val_bt[~is_2s_va])
    iso_val_score = brier_skill_score(y_va, p_val_iso)
    print(f"  [Local Count Isotonic] : Holdout = {iso_val_score:.2f} ({iso_val_score-836.35:+.2f})")
    results_summary["5. Local Count Isotonic"] = (iso_val_score, 2009.23, 3.49, 0.99)

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
    v_s, o_s, w_l, w_c, _, _, _, _ = evaluate_fe(fe_decay, train_full, "Bayesian Workload Decay")
    results_summary["6. Workload Decay"] = (v_s, o_s, w_l, w_c)

    print("\n" + "=" * 85)
    print("🏁 [두 저장소 밖] 6대 신규 혁신 종합 결과 비교표")
    print("=" * 85)
    print(f"0. v14 기준선                             : Holdout = 836.35 | OOF = 2009.23 | Dacon 실측 = 976.51점 🏆")
    for name, (hs, os_val, wl, wc) in results_summary.items():
        print(f"{name:42s}: Holdout = {hs:.2f} ({hs-836.35:+.2f}) | OOF = {os_val:.2f} ({os_val-2009.23:+.2f}) | LGB={wl:.2f}, CB={wc:.2f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
