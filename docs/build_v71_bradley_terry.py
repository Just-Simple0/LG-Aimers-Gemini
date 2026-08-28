"""
v71 Packaging: Bradley-Terry Empirical Bayes + 5-Fold Stacking (100% Pure Raw Logistic)
========================================================================================

작성일: 2026-08-29
"""

import os
import sys
import time
import zipfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
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


def logit(p, eps=1e-4):
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


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

    def to_dict(self):
        return {
            "alpha_p": self.alpha_p,
            "alpha_b": self.alpha_b,
            "alpha_pb": self.alpha_pb,
            "mu_0": self.mu_0,
            "r_0": self.r_0,
            "pitcher_theta": self.pitcher_theta,
            "batter_theta": self.batter_theta,
            "matchup_delta": self.matchup_delta,
        }

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


def main():
    print("=" * 80)
    print("🏆 Training and Packaging v71 Bradley-Terry Submission")
    print("Architecture: Bradley-Terry Empirical Bayes + 5-Fold Pure Logistic Stacking")
    print("=" * 80)
    start_time = time.time()

    # 1. Full 데이터 로드
    print("\n[1/7] Full 훈련 데이터 로드...")
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_FULL = float(train_full[TARGET].mean())
    print(f"  전체 데이터 행 수: {len(train_full):,} | Global Mean: {GLOBAL_MEAN_FULL:.4f}")

    # 2. Trackman Lookup 생성
    print("\n[2/7] Trackman Lookup 생성...")
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_FULL = make_tk_lookup(tk_raw)
    print(f"  TK Lookup 크기: {TK_LOOKUP_FULL.shape}")

    # 3. Season Decomposition & Bradley-Terry Feature Engineering
    print("\n[3/7] Season Decomposition & Bradley-Terry 피처 엔지니어링...")
    t0 = time.time()
    final_season_encoder = SeasonDecompositionEncoder().fit(train_full, train_full[TARGET])
    season_decomp_train = make_temporal_season_features(train_full, TARGET)
    base_fe_train = build_base_features(train_full, GLOBAL_MEAN_FULL, TK_LOOKUP_FULL)

    final_bt_encoder = BradleyTerryEBEncoder().fit(train_full, train_full[TARGET])
    bt_train = make_temporal_bt_features(train_full, TARGET)

    train_fe = pd.concat([base_fe_train, season_decomp_train, bt_train], axis=1)

    NEW_COLS = [c for c in train_fe.columns if c not in train_full.columns]
    ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS
    print(f"  피처 엔지니어링 완료 ({time.time()-t0:.1f}s) | 총 피처 수: {len(ALL_FEATURES)}개")

    # 4. 5-Fold CV OOF 생성 및 메타러너 학습
    print("\n[4/7] 5-Fold CV OOF 생성 및 메타러너 학습...")
    t0 = time.time()
    X_train = train_fe[ALL_FEATURES].copy()
    y_train = train_fe[TARGET].to_numpy()

    X_train_lgb = X_train.copy()
    for c in CAT_COLS:
        X_train_lgb[c] = X_train_lgb[c].astype("category")

    X_train_cb = X_train.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
    cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y_train))
    cb_oof = np.zeros(len(y_train))

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        f_t0 = time.time()
        m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
        m_lgb.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[val_idx] = m_lgb.predict_proba(X_train_lgb.iloc[val_idx])[:, 1]

        tp = Pool(X_train_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        m_cb.fit(tp)
        cb_oof[val_idx] = m_cb.predict_proba(X_train_cb.iloc[val_idx])[:, 1]
        print(f"    Fold {fold_i+1:2d}/5 완료 ({time.time()-f_t0:.1f}s)")

    stack_model = LogisticRegression(random_state=42, max_iter=1000)
    stack_model.fit(np.column_stack([lgb_oof, cb_oof]), y_train)
    w_lgb = stack_model.coef_[0][0]
    w_cb = stack_model.coef_[0][1]
    intercept = stack_model.intercept_[0]
    print(f"  메타러너 학습 완료: LGB={w_lgb:.4f}, CB={w_cb:.4f}, intercept={intercept:.4f}")

    # 가드레일 1 검수
    inspect_meta_learner_weights(w_lgb, w_cb, intercept)

    # 5. 전체 데이터 최종 베이스 모델 학습
    print("\n[5/7] 전체 데이터 최종 모델 학습...")
    t0 = time.time()
    final_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
    final_lgb.fit(X_train_lgb, y_train)
    print(f"  LightGBM 전체 학습 완료 ({time.time()-t0:.1f}s)")

    t0 = time.time()
    tp_full = Pool(X_train_cb, y_train, cat_features=cat_idx)
    final_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    final_cb.fit(tp_full)
    print(f"  CatBoost 전체 학습 완료 ({time.time()-t0:.1f}s)")

    # 6. 아티팩트 저장 (model/ensemble.pkl) - 순수 딕셔너리 직렬화
    print("\n[6/7] 아티팩트 저장 (model/ensemble.pkl)...")
    os.makedirs("model", exist_ok=True)
    artifact = {
        "lgbm_model": final_lgb,
        "catboost_model": final_cb,
        "stack_model": stack_model,
        "cat_cols": CAT_COLS,
        "all_features": ALL_FEATURES,
        "global_mean": GLOBAL_MEAN_FULL,
        "tk_lookup": TK_LOOKUP_FULL,
        "season_encoder_data": final_season_encoder.to_dict(),
        "bt_encoder_data": final_bt_encoder.to_dict(),
    }
    MODEL_PKL_PATH = "model/ensemble.pkl"
    joblib.dump(artifact, MODEL_PKL_PATH)
    pkl_size_mb = os.path.getsize(MODEL_PKL_PATH) / (1024 * 1024)
    print(f"  저장 완료: {MODEL_PKL_PATH} ({pkl_size_mb:.2f} MB)")

    # 7. script.py 업데이트 & zip 패키징
    print("\n[7/7] 제출용 ZIP 패키징...")
    os.makedirs("submissions", exist_ok=True)
    ZIP_PATH = "submissions/v71_bradley_terry.zip"
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write("src/script.py", arcname="script.py")
        zf.write("src/requirements.txt", arcname="requirements.txt")
        zf.write("model/ensemble.pkl", arcname="model/ensemble.pkl")

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"  ZIP 패키징 완료: {ZIP_PATH} ({zip_size_mb:.2f} MB)")

    total_sec = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 v71 Bradley-Terry 빌드 완료! ({total_sec/60:.1f}분) -> {ZIP_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
