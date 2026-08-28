"""
v70 Grand Champion: SVD Matchup Embeddings + 5-Fold Stacking + Beta Calibration
================================================================================

순수 딕셔너리 직렬화 기반 100% 독립 환경 무결성 보장 빌드 스크립트.

작성일: 2026-08-28
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
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
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


class LatentMatchupSVDEncoder:
    def __init__(self, n_components=8, alpha=30.0, random_state=42):
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state
        self.pitcher_map = {}
        self.batter_map = {}
        self.pitcher_factors = None
        self.batter_factors = None
        self.global_prior = 0.5

    def fit(self, df, target):
        df = df.copy()
        df["__target"] = np.asarray(target, dtype=float)
        self.global_prior = float(df["__target"].mean())

        pitchers = sorted(df["pitcher_id"].dropna().unique())
        batters = sorted(df["batter_id"].dropna().unique())

        self.pitcher_map = {pid: i for i, pid in enumerate(pitchers)}
        self.batter_map = {bid: j for j, bid in enumerate(batters)}

        n_p = len(pitchers)
        n_b = len(batters)

        if n_p == 0 or n_b == 0 or len(df) < 100:
            self.pitcher_factors = np.zeros((n_p, self.n_components))
            self.batter_factors = np.zeros((n_b, self.n_components))
            return self

        agg = (
            df.groupby(["pitcher_id", "batter_id"])["__target"]
            .agg(n="count", s="sum")
            .reset_index()
        )

        p_idx = agg["pitcher_id"].map(self.pitcher_map).to_numpy()
        b_idx = agg["batter_id"].map(self.batter_map).to_numpy()

        smoothed_val = (agg["s"] + self.alpha * self.global_prior) / (agg["n"] + self.alpha) - self.global_prior
        data = smoothed_val.to_numpy(dtype=float)

        mat = csr_matrix((data, (p_idx, b_idx)), shape=(n_p, n_b))
        k = min(self.n_components, min(n_p, n_b) - 1)
        if k < 2:
            self.pitcher_factors = np.zeros((n_p, self.n_components))
            self.batter_factors = np.zeros((n_b, self.n_components))
            return self

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        U = svd.fit_transform(mat)
        V = svd.components_.T

        self.pitcher_factors = np.zeros((n_p, self.n_components))
        self.batter_factors = np.zeros((n_b, self.n_components))
        self.pitcher_factors[:, :k] = U
        self.batter_factors[:, :k] = V
        return self

    def to_dict(self):
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "pitcher_map": self.pitcher_map,
            "batter_map": self.batter_map,
            "pitcher_factors": self.pitcher_factors,
            "batter_factors": self.batter_factors,
            "global_prior": self.global_prior,
        }

    def transform(self, df):
        p_ids = df["pitcher_id"].to_numpy()
        b_ids = df["batter_id"].to_numpy()

        dots = np.zeros(len(df), dtype=float)
        p_f1 = np.zeros(len(df), dtype=float)
        p_f2 = np.zeros(len(df), dtype=float)
        b_f1 = np.zeros(len(df), dtype=float)
        b_f2 = np.zeros(len(df), dtype=float)

        for i in range(len(df)):
            pid = p_ids[i]
            bid = b_ids[i]
            p_idx = self.pitcher_map.get(pid)
            b_idx = self.batter_map.get(bid)

            if p_idx is not None and b_idx is not None:
                u = self.pitcher_factors[p_idx]
                v = self.batter_factors[b_idx]
                dots[i] = np.dot(u, v)
                p_f1[i] = u[0]
                p_f2[i] = u[1]
                b_f1[i] = v[0]
                b_f2[i] = v[1]
            elif p_idx is not None:
                u = self.pitcher_factors[p_idx]
                p_f1[i] = u[0]
                p_f2[i] = u[1]
            elif b_idx is not None:
                v = self.batter_factors[b_idx]
                b_f1[i] = v[0]
                b_f2[i] = v[1]

        return pd.DataFrame({
            "svd_matchup_dot": dots,
            "svd_pitcher_f1": p_f1,
            "svd_pitcher_f2": p_f2,
            "svd_batter_f1": b_f1,
            "svd_batter_f2": b_f2,
        }, index=df.index)


def make_temporal_svd_features(train_df, target_col=TARGET):
    seasons = sorted(train_df["season"].unique())
    chunks = []
    for s in seasons:
        s_mask = train_df["season"] == s
        past_mask = train_df["season"] < s
        if past_mask.sum() == 0:
            encoder = LatentMatchupSVDEncoder()
            dummy_target = pd.Series([0.5], index=[0])
            dummy_df = pd.DataFrame({"pitcher_id": [-999], "batter_id": [-999]})
            encoder.fit(dummy_df, dummy_target)
            chunks.append(encoder.transform(train_df.loc[s_mask]))
        else:
            encoder = LatentMatchupSVDEncoder().fit(
                train_df.loc[past_mask], train_df.loc[past_mask, target_col]
            )
            chunks.append(encoder.transform(train_df.loc[s_mask]))

    return pd.concat(chunks, axis=0).loc[train_df.index]


class BetaCalibrator:
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.lr = None
        self.a_ = 1.0
        self.b_ = 1.0
        self.c_ = 0.0

    def fit(self, s, y):
        s_safe = np.clip(np.asarray(s, dtype=float), self.eps, 1.0 - self.eps)
        x1 = np.log(s_safe)
        x2 = -np.log(1.0 - s_safe)
        X = np.column_stack([x1, x2])
        self.lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.lr.fit(X, y)
        self.a_ = float(self.lr.coef_[0][0])
        self.b_ = float(self.lr.coef_[0][1])
        self.c_ = float(self.lr.intercept_[0])
        return self

    def to_dict(self):
        return {
            "a_": self.a_,
            "b_": self.b_,
            "c_": self.c_,
            "eps": self.eps,
        }


def main():
    print("=" * 80)
    print("🏆 Training and Packaging v70 Grand Champion Submission")
    print("Architecture: SVD Matchup Embeddings + 5-Fold Stacking + Beta Calibration")
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

    # 3. Season Decomposition & SVD Feature Engineering
    print("\n[3/7] Season Decomposition & SVD Latent 피처 엔지니어링...")
    t0 = time.time()
    final_season_encoder = SeasonDecompositionEncoder().fit(train_full, train_full[TARGET])
    season_decomp_train = make_temporal_season_features(train_full, TARGET)
    base_fe_train = build_base_features(train_full, GLOBAL_MEAN_FULL, TK_LOOKUP_FULL)

    final_svd_encoder = LatentMatchupSVDEncoder().fit(train_full, train_full[TARGET])
    svd_train = make_temporal_svd_features(train_full, TARGET)

    train_fe = pd.concat([base_fe_train, season_decomp_train, svd_train], axis=1)

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

    # Beta Calibration 적합
    p_oof_stacked = stack_model.predict_proba(np.column_stack([lgb_oof, cb_oof]))[:, 1]
    beta_calibrator = BetaCalibrator().fit(p_oof_stacked, y_train)
    print(f"  Beta Calibrator 적합 완료: a={beta_calibrator.a_:.4f}, b={beta_calibrator.b_:.4f}, c={beta_calibrator.c_:.4f}")

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

    # 6. 아티팩트 저장 (model/ensemble.pkl) - 순수 내장 딕셔너리로 직렬화
    print("\n[6/7] 아티팩트 저장 (model/ensemble.pkl)...")
    os.makedirs("model", exist_ok=True)
    artifact = {
        "lgbm_model": final_lgb,
        "catboost_model": final_cb,
        "stack_model": stack_model,
        "beta_calibrator_data": beta_calibrator.to_dict(),
        "cat_cols": CAT_COLS,
        "all_features": ALL_FEATURES,
        "global_mean": GLOBAL_MEAN_FULL,
        "tk_lookup": TK_LOOKUP_FULL,
        "season_encoder_data": final_season_encoder.to_dict(),
        "svd_encoder_data": final_svd_encoder.to_dict(),
    }
    MODEL_PKL_PATH = "model/ensemble.pkl"
    joblib.dump(artifact, MODEL_PKL_PATH)
    pkl_size_mb = os.path.getsize(MODEL_PKL_PATH) / (1024 * 1024)
    print(f"  저장 완료: {MODEL_PKL_PATH} ({pkl_size_mb:.2f} MB)")

    # 7. script.py 업데이트 & zip 패키징
    print("\n[7/7] script.py 업데이트 및 제출용 ZIP 패키징...")
    os.makedirs("submissions", exist_ok=True)
    ZIP_PATH = "submissions/v70_grand_champion.zip"
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
    print(f"🎉 v70 Grand Champion 빌드 완료! ({total_sec/60:.1f}분) -> {ZIP_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
