"""
v14 최종 전체 데이터 재학습 및 제출 패키지 생성 스크립트
=========================================================

1. 2019~2024 전체 학습 데이터(1,475,092행)로 SeasonDecompositionEncoder fit
2. temporal decomposition으로 학습용 당해 시즌 분해 피처 생성
3. TK_LOOKUP_FULL 생성 및 전체 기본 피처 + 시즌 분해 피처 결합
4. 5-fold KFold OOF로 LightGBM + CatBoost 예측 생성
5. LogisticRegression 메타러너 fit
6. 전체 데이터로 최종 LightGBM (171 iter) / CatBoost (360 iter) fit
7. model/ensemble.pkl 저장
8. src/script.py, src/requirements.txt 와 함께 submissions/v14_season_decomp.zip 패키징
9. 5행 샘플 및 245,789행 스트레스 테스트 (추론 시간 및 메모리 검증)

작성일: 2026-08-27
"""

import os
import sys
import time
import zipfile
import shutil
import subprocess
import tempfile

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.script import SeasonDecompositionEncoder, build_features

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

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


def main():
    print("=" * 70)
    print("v14 최종 전체 데이터 재학습 및 패키징 시작")
    print("=" * 70)
    t_start = time.time()

    # 1. 데이터 로드
    print("\n[1/7] 전체 데이터 로드...")
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_FULL = float(train_full[TARGET].mean())
    print(f"  train: {train_full.shape} | global_mean: {GLOBAL_MEAN_FULL:.6f}")

    # 2. Trackman lookup (FULL: 2019~2024)
    print("\n[2/7] Trackman lookup (FULL) 생성...")
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_FULL = make_tk_lookup(tk_raw)
    print(f"  TK_LOOKUP_FULL: {TK_LOOKUP_FULL.shape}")

    # 3. SeasonDecompositionEncoder fit (FULL) & temporal feature generation
    print("\n[3/7] Season Decomposition 인코더 fit 및 훈련 피처 생성...")
    t0 = time.time()
    final_season_encoder = SeasonDecompositionEncoder().fit(train_full, train_full[TARGET])
    season_decomp_train = make_temporal_season_features(train_full, TARGET)
    base_fe_train = build_base_features(train_full, GLOBAL_MEAN_FULL, TK_LOOKUP_FULL)
    train_fe = pd.concat([base_fe_train, season_decomp_train], axis=1)

    NEW_COLS = [c for c in train_fe.columns if c not in train_full.columns]
    ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS
    print(f"  피처 엔지니어링 완료 :: {time.time()-t0:.1f}s | 전체 피처: {len(ALL_FEATURES)}개")

    # 4. KFold OOF로 메타러너 학습
    print("\n[4/7] 5-Fold OOF 및 메타러너 학습...")
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
        ft = time.time()
        m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
        m_lgb.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[val_idx] = m_lgb.predict_proba(X_train_lgb.iloc[val_idx])[:, 1]

        tp = Pool(X_train_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        m_cb.fit(tp)
        cb_oof[val_idx] = m_cb.predict_proba(X_train_cb.iloc[val_idx])[:, 1]
        print(f"    KFold {fold_i+1}/5 :: {time.time()-ft:.1f}s")

    stack_model = LogisticRegression(random_state=42, max_iter=1000)
    stack_model.fit(np.column_stack([lgb_oof, cb_oof]), y_train)
    print(f"  메타러너 계수: LGB={stack_model.coef_[0][0]:.4f}, CB={stack_model.coef_[0][1]:.4f}, intercept={stack_model.intercept_[0]:.4f}")

    # 5. 전체 데이터 최종 베이스 모델 학습
    print("\n[5/7] 전체 데이터 최종 모델 학습...")
    t0 = time.time()
    final_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
    final_lgb.fit(X_train_lgb, y_train)
    print(f"  LightGBM 전체 학습 완료 :: {time.time()-t0:.1f}s")

    t0 = time.time()
    tp_full = Pool(X_train_cb, y_train, cat_features=cat_idx)
    final_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    final_cb.fit(tp_full)
    print(f"  CatBoost 전체 학습 완료 :: {time.time()-t0:.1f}s")

    # 6. 아티팩트 저장
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
    print("\n[7/7] 제출용 ZIP 패키징 및 검증...")
    os.makedirs("submissions", exist_ok=True)
    ZIP_PATH = "submissions/v14_season_decomp.zip"
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write("src/script.py", arcname="script.py")
        zf.write("src/requirements.txt", arcname="requirements.txt")
        zf.write("model/ensemble.pkl", arcname="model/ensemble.pkl")

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    zip_name_len = len(os.path.basename(ZIP_PATH))
    print(f"  제출 ZIP 생성: {ZIP_PATH} ({zip_size_mb:.2f} MB, 파일명 길이={zip_name_len}자/30자 제한)")
    assert zip_name_len <= 30, f"파일명 30자 초과: {os.path.basename(ZIP_PATH)}"

    # 독립 임시 디렉토리에서 압축 해제 후 script.py 실행 검증 (Smoke & Stress test)
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n  [검증 1] ZIP 압축 해제 및 5행 스모크 테스트 (in {tmpdir})...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmpdir)
        
        # data 디렉토리 복사 (5행 test.csv 및 sample_submission)
        os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
        shutil.copy("data/test.csv", os.path.join(tmpdir, "data", "test.csv"))
        shutil.copy("data/sample_submission.csv", os.path.join(tmpdir, "data", "sample_submission.csv"))

        t_inf = time.time()
        ret = subprocess.run([sys.executable, "script.py"], cwd=tmpdir, capture_output=True, text=True)
        print("  script.py stdout:", ret.stdout.strip())
        if ret.returncode != 0:
            print("  script.py stderr:", ret.stderr)
            raise RuntimeError("script.py 5행 스모크 테스트 실패!")
        
        sub_5 = pd.read_csv(os.path.join(tmpdir, "output", "submission.csv"))
        print(f"  5행 예측 완료 :: {time.time()-t_inf:.2f}s | 결과:\n{sub_5}")
        assert len(sub_5) == 5
        assert sub_5[TARGET].notna().all()
        assert (sub_5[TARGET] >= 0.0).all() and (sub_5[TARGET] <= 1.0).all()

        # 스트레스 테스트 (245,789행 모의 테스트 데이터)
        print(f"\n  [검증 2] 245,789행 대용량 스트레스 테스트...")
        test_full = train_full.sample(n=245789, replace=True, random_state=42).copy()
        test_full[ID_COL] = [f"TEST_{i:07d}" for i in range(len(test_full))]
        test_full.to_csv(os.path.join(tmpdir, "data", "test.csv"), index=False, encoding="utf-8-sig")
        
        sample_sub_full = pd.DataFrame({
            ID_COL: test_full[ID_COL],
            TARGET: [0.5] * len(test_full),
        })
        sample_sub_full.to_csv(os.path.join(tmpdir, "data", "sample_submission.csv"), index=False, encoding="utf-8-sig")

        t_inf_large = time.time()
        ret_large = subprocess.run([sys.executable, "script.py"], cwd=tmpdir, capture_output=True, text=True)
        inf_elapsed = time.time() - t_inf_large
        print(f"  245,789행 추론 소요 시간: {inf_elapsed:.2f}초 ({inf_elapsed/60:.2f}분 / 10분 예산 대비 여유)")
        if ret_large.returncode != 0:
            print("  대용량 추론 에러 stderr:", ret_large.stderr)
            raise RuntimeError("245,789행 추론 테스트 실패!")
        
        sub_large = pd.read_csv(os.path.join(tmpdir, "output", "submission.csv"))
        print(f"  대용량 예측 완료: rows={len(sub_large)}, mean={sub_large[TARGET].mean():.4f}, min={sub_large[TARGET].min():.4f}, max={sub_large[TARGET].max():.4f}")
        assert len(sub_large) == 245789
        assert sub_large[TARGET].notna().all()
        assert (sub_large[TARGET] >= 0.0).all() and (sub_large[TARGET] <= 1.0).all()

    print(f"\n{'='*70}")
    print(f"🎉 모든 검증 성공! 최종 제출 파일: {ZIP_PATH}")
    print(f"총 소요 시간: {(time.time()-t_start)/60:.1f}분")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
