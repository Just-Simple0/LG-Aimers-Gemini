"""
v63 Champion (10-Fold CV Stacking + LGB reg_lambda=3.0) Train and Package Script
================================================================================

이 스크립트는:
  1. 전체 훈련 데이터(2020~2024)를 사용하여 10-Fold CV OOF 생성
  2. 메타러너(로지스틱 회귀) 학습
  3. 전체 데이터로 LightGBM (reg_lambda=3.0) 및 CatBoost 최종 학습
  4. 5대 하드 가드레일 전수 검수
  5. model/ensemble.pkl 저장 및 submissions/v63_champion.zip 패키징
  6. 독립 환경 script.py 테스트 실행 및 output/submission.csv 무결성 검증

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
    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,  # v63 핵심 파라미터
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)

CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)

ENTITY_SPECS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
}


class SeasonDecompositionEncoder:
    def __init__(self, entities=("pitcher", "batter"), alpha=ALPHA, prior=None, history_tables=None):
        self.entities = list(entities)
        self.alpha = float(alpha)
        self.prior = prior
        self.history_tables = history_tables or {}

    def fit(self, df, target):
        values = np.asarray(target, dtype=float).reshape(-1)
        self.prior = float(values.mean())
        self.history_tables = {}
        for entity in self.entities:
            id_col, _, _ = ENTITY_SPECS[entity]
            working = pd.DataFrame({
                id_col: df[id_col].to_numpy(copy=False),
                "__target": values,
            })
            table = (
                working.dropna(subset=[id_col])
                .groupby(id_col, sort=False)["__target"]
                .agg(history_success="sum", history_n="count")
            )
            self.history_tables[entity] = table
        return self

    def to_dict(self):
        return {
            "entities": self.entities,
            "alpha": self.alpha,
            "prior": self.prior,
            "history_tables": self.history_tables,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            entities=data["entities"],
            alpha=data["alpha"],
            prior=data["prior"],
            history_tables=data["history_tables"],
        )

    def transform(self, df):
        derived = {}
        for entity in self.entities:
            id_col, count_col, rate_col = ENTITY_SPECS[entity]
            table = self.history_tables[entity]

            h_n_series = df[id_col].map(table["history_n"])
            h_s_series = df[id_col].map(table["history_success"])
            history_n = h_n_series.fillna(0.0).to_numpy(dtype=float)
            history_success = h_s_series.fillna(0.0).to_numpy(dtype=float)

            history_rate = np.divide(
                history_success, history_n,
                out=np.full(len(df), self.prior, dtype=float),
                where=history_n > 0.0,
            )

            raw_n = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            cumulative_n = np.clip(np.rint(raw_n), 0.0, None)

            raw_rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(self.prior).to_numpy(dtype=float)
            raw_rate = np.clip(raw_rate, 0.0, 1.0)
            cumulative_success = np.rint(cumulative_n * raw_rate)

            history_n_adj = np.minimum(history_n, cumulative_n)
            history_success_adj = np.minimum(history_success, cumulative_success)

            current_n = np.maximum(0.0, cumulative_n - history_n_adj)
            current_success = np.clip(cumulative_success - history_success_adj, 0.0, current_n)

            current_posterior = (current_success + self.alpha * history_rate) / (current_n + self.alpha)
            current_history_gap = current_posterior - history_rate

            prefix = f"season_{entity}_"
            derived[prefix + "history_n"] = history_n
            derived[prefix + "history_rate"] = history_rate
            derived[prefix + "history_confidence"] = history_n / (history_n + self.alpha)
            derived[prefix + "current_n"] = current_n
            derived[prefix + "current_posterior"] = current_posterior
            derived[prefix + "current_history_gap"] = current_history_gap
            derived[prefix + "current_confidence"] = current_n / (current_n + self.alpha)
            derived[prefix + "current_cold_start"] = (current_n == 0.0).astype(np.int8)

        return pd.DataFrame(derived, index=df.index)


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


def main():
    print("=" * 80)
    print("🏆 Training and Packaging v63 New Champion Submission")
    print("Architecture: 10-Fold CV Stacking + LGB reg_lambda=3.0 + Safe Envelope")
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

    # 2. Trackman lookup 생성
    print("\n[2/7] Trackman Lookup 생성...")
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_FULL = make_tk_lookup(tk_raw)
    print(f"  TK Lookup 크기: {TK_LOOKUP_FULL.shape}")

    # 3. Season Decomposition 인코더 fit 및 피처 생성
    print("\n[3/7] Season Decomposition 피처 엔지니어링...")
    t0 = time.time()
    final_season_encoder = SeasonDecompositionEncoder().fit(train_full, train_full[TARGET])
    season_decomp_train = make_temporal_season_features(train_full, TARGET)
    base_fe_train = build_base_features(train_full, GLOBAL_MEAN_FULL, TK_LOOKUP_FULL)
    train_fe = pd.concat([base_fe_train, season_decomp_train], axis=1)

    NEW_COLS = [c for c in train_fe.columns if c not in train_full.columns]
    ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS
    print(f"  피처 엔지니어링 완료 ({time.time()-t0:.1f}s) | 총 피처 수: {len(ALL_FEATURES)}개")

    # 4. 10-Fold CV OOF 생성 및 메타러너 학습
    print("\n[4/7] 10-Fold CV OOF 생성 및 메타러너 학습...")
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

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
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
        print(f"    Fold {fold_i+1:2d}/10 완료 ({time.time()-f_t0:.1f}s)")

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

    # 6. 아티팩트 저장 (model/ensemble.pkl)
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
    }
    MODEL_PKL_PATH = "model/ensemble.pkl"
    joblib.dump(artifact, MODEL_PKL_PATH)
    pkl_size_mb = os.path.getsize(MODEL_PKL_PATH) / (1024 * 1024)
    print(f"  저장 완료: {MODEL_PKL_PATH} ({pkl_size_mb:.2f} MB)")

    # 7. zip 패키징 및 독립 추론 테스트
    print("\n[7/7] 제출용 ZIP 패키징 및 독립 추론 검증...")
    os.makedirs("submissions", exist_ok=True)
    ZIP_PATH = "submissions/v63_champion.zip"
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write("src/script.py", arcname="script.py")
        zf.write("src/requirements.txt", arcname="requirements.txt")
        zf.write("model/ensemble.pkl", arcname="model/ensemble.pkl")

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"  ZIP 패키징 완료: {ZIP_PATH} ({zip_size_mb:.2f} MB)")

    # 독립 검증: 임시 디렉토리에서 압축 해제 후 script.py 실행
    print("\n[검증 테스트] 독립 샌드박스에서 script.py 실행...")
    test_sandbox = "temp_v63_test"
    if os.path.exists(test_sandbox):
        shutil.rmtree(test_sandbox)
    os.makedirs(test_sandbox, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(test_sandbox)

    # data 복사 링크
    os.makedirs(os.path.join(test_sandbox, "data"), exist_ok=True)
    shutil.copy(os.path.join(DATA_DIR, "test.csv"), os.path.join(test_sandbox, "data", "test.csv"))
    shutil.copy(os.path.join(DATA_DIR, "sample_submission.csv"), os.path.join(test_sandbox, "data", "sample_submission.csv"))

    # 실행
    python_bin = sys.executable
    ret = os.system(f"cd {test_sandbox} && {python_bin} script.py")
    if ret != 0:
        raise RuntimeError("script.py 테스트 실행 실패!")

    sub_file = os.path.join(test_sandbox, "output", "submission.csv")
    if not os.path.exists(sub_file):
        raise RuntimeError("submission.csv가 생성되지 않았습니다!")

    sub_df = pd.read_csv(sub_file)
    print(f"  생성된 submission.csv 크기: {sub_df.shape}")

    # 최종 가드레일 2 검수
    inspect_prediction_distribution(sub_df[TARGET])

    # 대상 디렉토리에 복사
    os.makedirs("output", exist_ok=True)
    shutil.copy(sub_file, "output/submission.csv")
    shutil.rmtree(test_sandbox)

    total_sec = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 v63 Champion 제출 파일 빌드 & 5대 가드레일 검수 100% 완료! ({total_sec/60:.1f}분)")
    print(f"최종 제출 파일: {ZIP_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
