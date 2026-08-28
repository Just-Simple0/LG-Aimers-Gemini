"""
두 저장소 밖 - Item 2: Pitch-Type 사전 구종 예측 교사모델 (Statistical Matching)
=============================================================================

연구 목적:
  trackman_history의 120만+ 투구 데이터로부터 투구 직전 상황 맥락
  (카운트, 볼/스트라이크 차이, 이닝, 좌우 손 매칭, 점수차 등)만으로
  다음 구종(Fastball, Breaking, Offspeed) 다중분류 교사모델(LightGBM Multi-class)을 학습하고,
  사전 예상 구종 확률 분포를 train 및 test에 시계열 누수 없이 주입하여 모델 성능 검증.

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


class PitchTypeTeacherModel:
    """Trackman 기반 사전 구종 예측 교사모델."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.classes_ = ["Fastball", "Breaking", "Offspeed"]
        self.feature_cols = [
            "balls_before", "strikes_before", "outs_before",
            "count_diff", "count_total", "is_two_strike", "is_three_ball",
        ]

    def _prep_features(self, df):
        feats = pd.DataFrame(index=df.index)
        feats["balls_before"] = df["balls_before"].fillna(0).astype(int)
        feats["strikes_before"] = df["strikes_before"].fillna(0).astype(int)
        feats["outs_before"] = df["outs_before"].fillna(0).astype(int)
        feats["count_diff"] = feats["strikes_before"] - feats["balls_before"]
        feats["count_total"] = feats["strikes_before"] + feats["balls_before"]
        feats["is_two_strike"] = (feats["strikes_before"] >= 2).astype(int)
        feats["is_three_ball"] = (feats["balls_before"] >= 3).astype(int)
        return feats[self.feature_cols]

    def fit(self, tk_df):
        df = tk_df.dropna(subset=["pitch_type_group"]).copy()
        df = df[df["pitch_type_group"].isin(self.classes_)]
        
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

        # 투수 본인 피치믹스와의 괴리(Deviation)
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
            past_tk = tk_df[tk_df["season"] == seasons[0]]
        
        teacher = PitchTypeTeacherModel().fit(past_tk)
        chunks.append(teacher.predict_proba(train_df.loc[s_mask]))

    return pd.concat(chunks, axis=0).loc[train_df.index]


def main():
    print("=" * 80)
    print("🚀 [두 저장소 밖 - Item 2] Pitch-Type 사전 구종 예측 교사모델 연구")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
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

    # 교사모델 피처 생성
    print("\n[피처 생성] Trackman 구종 다중분류 교사모델 학습 및 피처 추출 중...")
    teacher_train = make_temporal_teacher_features(train_split, tk_raw[tk_raw["season"] <= 2023])
    teacher_val_model = PitchTypeTeacherModel().fit(tk_raw[tk_raw["season"] <= 2023])
    teacher_val = teacher_val_model.predict_proba(val_split)
    teacher_all = pd.concat([teacher_train, teacher_val], axis=0).loc[train_full.index]

    fe_teacher = pd.concat([fe_v14, teacher_all], axis=1)

    NEW_COLS_T = [c for c in fe_teacher.columns if c not in train_full.columns]
    ALL_FEATURES_T = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_T

    for c in CAT_COLS:
        fe_teacher[c] = fe_teacher[c].astype("category")

    is_val_mask = fe_teacher["season"] == 2024
    X_train_full = fe_teacher.loc[~is_val_mask, ALL_FEATURES_T]
    y_train_full = fe_teacher.loc[~is_val_mask, TARGET].to_numpy()
    X_val_fe = fe_teacher.loc[is_val_mask, ALL_FEATURES_T]
    y_val = fe_teacher.loc[is_val_mask, TARGET].to_numpy()

    X_train_cb = X_train_full.copy()
    X_val_cb = X_val_fe.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    cat_idx = [ALL_FEATURES_T.index(c) for c in CAT_COLS]

    print(f"피처 구성 완료: 교사모델 피처 6개 추가 (총 {len(ALL_FEATURES_T)}개 피처)")

    # 1. Base Models
    print("\n[1/3] 베이스 모델 학습 중...")
    lgb_m = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
    lgb_m.fit(X_train_full, y_train_full)
    p_lgb_val = lgb_m.predict_proba(X_val_fe)[:, 1]

    tp_full = Pool(X_train_cb, y_train_full, cat_features=cat_idx)
    cb_m = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    cb_m.fit(tp_full)
    p_cb_val = cb_m.predict_proba(X_val_cb)[:, 1]

    # 2. 5-Fold OOF
    print("\n[2/3] 5-Fold CV OOF 생성 중...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_train_full))
    oof_cb = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_reset)):
        f_t0 = time.time()
        m_l = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
        m_l.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb[val_idx] = m_l.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        m_c.fit(tp)
        oof_cb[val_idx] = m_c.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]
        print(f"  Fold {fold_i+1}/5 완료 ({time.time()-f_t0:.1f}s)")

    # 3. Meta-Learner
    print("\n[3/3] 메타러너 스태킹...")
    stack = LogisticRegression(random_state=42, max_iter=1000)
    stack.fit(np.column_stack([oof_lgb, oof_cb]), y_train_full)
    w_lgb = stack.coef_[0][0]
    w_cb = stack.coef_[0][1]
    intercept = stack.intercept_[0]

    p_val_stack = stack.predict_proba(np.column_stack([p_lgb_val, p_cb_val]))[:, 1]
    p_oof_stack = stack.predict_proba(np.column_stack([oof_lgb, oof_cb]))[:, 1]

    score_val = brier_skill_score(y_val, p_val_stack)
    score_oof = brier_skill_score(y_train_full, p_oof_stack)

    print("\n" + "=" * 80)
    print("🏁 [두 저장소 밖 - Item 2] Pitch-Type 교사모델 검증 결과")
    print("=" * 80)
    print(f"v14 기준선 (Without Teacher) : Holdout = 836.35 | OOF = 2009.23 | LGB=3.4905, CB=0.9867")
    print(f"교사모델 적용 앙상블 (With Teacher) : Holdout = {score_val:.2f} ({score_val-836.35:+.2f}) | OOF = {score_oof:.2f} ({score_oof-2009.23:+.2f}) | LGB={w_lgb:.4f}, CB={w_cb:.4f}")
    
    print("\n[가드레일 검수]")
    inspect_meta_learner_weights(w_lgb, w_cb, intercept)
    inspect_prediction_distribution(p_val_stack)
    print("=" * 80)


if __name__ == "__main__":
    main()
