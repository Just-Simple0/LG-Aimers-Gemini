"""
v18 최종 전체 데이터 재학습 및 제출 패키징 스크립트
======================================================================

파이프라인:
  1. 전체 2019~2024 train.csv(1.47M행) 및 Trackman(FULL) 로드
  2. SeasonDecompositionEncoder fit 및 101개 피처 생성
  3. 교사 모델 5-Fold OOF 생성 (Hard LGB + Hard CB) -> p_teacher_oof
  4. 소프트 타겟 생성: y_soft = 0.70 * y + 0.30 * p_teacher_oof
  5. 학생 모델 5-Fold OOF 생성 (Soft Distilled LGB)
  6. 3-Way Ridge Stacking 메타러너 학습: LogisticRegression(C=0.1) on [p_lgb, p_cb, p_student]
  7. 전체 데이터 최종 학습:
     - lgb_teacher (171 iter)
     - cb_teacher (360 iter)
     - lgb_student (171 iter on y_soft_full)
  8. 아티팩트 저장: model/ensemble.pkl (커스텀 클래스 없이 순수 dict/pandas 직렬화)
  9. 제출 ZIP 생성: submissions/v18_soft_distillation.zip (27자 <= 30자 규정)
  10. 독립 임시 환경에서 5행 스모크 테스트 + 245,789행 대용량 스트레스 테스트

작성일: 2026-08-27
"""

import os
import sys
import time
import zipfile
import shutil
import subprocess
import tempfile

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

BEST_LGB_ITER = 171
BEST_CB_ITER = 360
W_SOFT = 0.30  # Teacher Consensus 가중치

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
)
CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1,
)


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


def main():
    print("=" * 70)
    print("v18 최종 전체 데이터 재학습 및 패키징 시작 (Soft Distilled Stacking)")
    print("=" * 70)
    t_start = time.time()

    # 1. 데이터 로드
    print("\n[1/8] 전체 데이터 로드...")
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    FEATURES_BASE = [c for c in test_cols if c != ID_COL]

    train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=FEATURES_BASE + [TARGET])
    GLOBAL_MEAN_FULL = float(train_full[TARGET].mean())
    print(f"  train: {train_full.shape} | global_mean: {GLOBAL_MEAN_FULL:.6f}")

    # 2. Trackman lookup (FULL 2019-2024)
    print("\n[2/8] Trackman lookup (FULL) 생성...")
    tk_raw = pd.read_csv(
        os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate",
                                        "induced_vert_break", "horz_break",
                                        "extension", "zone_speed"],
    )
    TK_LOOKUP_FULL = make_tk_lookup(tk_raw)
    print(f"  TK_LOOKUP_FULL: {TK_LOOKUP_FULL.shape}")

    # 3. Season Decomposition Fit 및 피처 생성
    print("\n[3/8] Season Decomposition 인코더 fit 및 훈련 피처 생성...")
    t0 = time.time()
    final_season_encoder = SeasonDecompositionEncoder().fit(train_full, train_full[TARGET])
    train_fe = build_features(train_full, GLOBAL_MEAN_FULL, TK_LOOKUP_FULL, final_season_encoder)

    NEW_COLS = [c for c in train_fe.columns if c not in train_full.columns]
    ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS

    X_train_df = train_fe[ALL_FEATURES].copy()
    y_train = train_full[TARGET].to_numpy()

    for c in CAT_COLS:
        X_train_df[c] = X_train_df[c].astype("category")

    cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]
    print(f"  피처 엔지니어링 완료 :: {time.time()-t0:.1f}s | 전체 피처: {len(ALL_FEATURES)}개")

    X_train_lgb = X_train_df.copy()
    X_train_cb = X_train_df.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)

    # 4. 교사 모델 5-Fold OOF 생성
    print("\n[4/8] 교사 모델 5-Fold OOF 생성 (LGB & CB)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y_train))
    cb_oof = np.zeros(len(y_train))

    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train_df)):
        t_fold = time.time()
        m_lgb = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
        m_lgb.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx])
        lgb_oof[oof_idx] = m_lgb.predict_proba(X_train_lgb.iloc[oof_idx])[:, 1]

        tp_f = Pool(X_train_cb.iloc[tr_idx], y_train[tr_idx], cat_features=cat_idx)
        m_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
        m_cb.fit(tp_f)
        cb_oof[oof_idx] = m_cb.predict_proba(X_train_cb.iloc[oof_idx])[:, 1]
        print(f"    교사 KFold {fold_i+1}/5 :: {time.time()-t_fold:.1f}s")

    p_teacher_oof = 0.5 * (lgb_oof + cb_oof)
    y_soft_train = (1.0 - W_SOFT) * y_train + W_SOFT * p_teacher_oof

    # 5. 학생 모델 5-Fold OOF 생성
    print("\n[5/8] 학생 모델 5-Fold OOF 생성 (Soft Distilled LGB)...")
    student_oof = np.zeros(len(y_train))
    for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train_df)):
        t_fold = time.time()
        s_m = lgb.LGBMRegressor(
            n_estimators=BEST_LGB_ITER,
            **{k: v for k, v in LGB_PARAMS.items() if k != "objective"},
            objective="cross_entropy",
        )
        s_m.fit(X_train_lgb.iloc[tr_idx], y_soft_train[tr_idx])
        student_oof[oof_idx] = s_m.predict(X_train_lgb.iloc[oof_idx])
        print(f"    학생 KFold {fold_i+1}/5 :: {time.time()-t_fold:.1f}s")

    # 6. 3-Way Ridge Stacking 메타러너 학습
    print("\n[6/8] 3-Way Ridge Stacking 메타러너 학습...")
    stack_model = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    X_meta_oof = np.column_stack([lgb_oof, cb_oof, student_oof])
    stack_model.fit(X_meta_oof, y_train)
    print(f"  메타러너 계수: LGB={stack_model.coef_[0][0]:.4f}, CB={stack_model.coef_[0][1]:.4f}, Student={stack_model.coef_[0][2]:.4f}, intercept={stack_model.intercept_[0]:.4f}")

    # 7. 전체 데이터 최종 학습 (교사 LGB, 교사 CB, 학생 LGB)
    print("\n[7/8] 전체 데이터 최종 모델 학습...")
    t0 = time.time()
    final_lgb_teacher = lgb.LGBMClassifier(n_estimators=BEST_LGB_ITER, **LGB_PARAMS)
    final_lgb_teacher.fit(X_train_lgb, y_train)
    print(f"  Teacher LightGBM 전체 학습 완료 :: {time.time()-t0:.1f}s")

    t0 = time.time()
    tp_full = Pool(X_train_cb, y_train, cat_features=cat_idx)
    final_cb = CatBoostClassifier(iterations=BEST_CB_ITER, **CB_PARAMS)
    final_cb.fit(tp_full)
    print(f"  Teacher CatBoost 전체 학습 완료 :: {time.time()-t0:.1f}s")

    t0 = time.time()
    final_lgb_student = lgb.LGBMRegressor(
        n_estimators=BEST_LGB_ITER,
        **{k: v for k, v in LGB_PARAMS.items() if k != "objective"},
        objective="cross_entropy",
    )
    final_lgb_student.fit(X_train_lgb, y_soft_train)
    print(f"  Student LightGBM 전체 학습 완료 :: {time.time()-t0:.1f}s")

    # 8. 아티팩트 저장
    print("\n[8/8] 아티팩트 저장 (model/ensemble.pkl)...")
    os.makedirs("model", exist_ok=True)
    artifact = {
        "lgb_teacher": final_lgb_teacher,
        "lgb_student": final_lgb_student,
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

    # 9. ZIP 패키징 및 독립 추론 테스트
    os.makedirs("submissions", exist_ok=True)
    ZIP_PATH = "submissions/v18_soft_distillation.zip"
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

    # 독립 임시 디렉토리에서 스모크 및 스트레스 테스트
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n  [검증 1] ZIP 압축 해제 및 5행 스모크 테스트 (in {tmpdir})...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmpdir)
        
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
    print("=" * 70)


if __name__ == "__main__":
    main()
