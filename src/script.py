# script.py
import os
import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

ENTITY_SPECS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


# =======================
# 1. SeasonDecompositionEncoder (순수 딕셔너리 직렬화)
# =======================

class SeasonDecompositionEncoder:
    def __init__(self, entities=("pitcher", "batter"), alpha=ALPHA, prior=None, history_tables=None):
        self.entities = list(entities)
        self.alpha = float(alpha)
        self.prior = prior
        self.history_tables = history_tables or {}

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


# =======================
# 2. LatentMatchupSVDEncoder (순수 딕셔너리 직렬화)
# =======================

class LatentMatchupSVDEncoder:
    def __init__(self, n_components=8, alpha=30.0, pitcher_map=None, batter_map=None, pitcher_factors=None, batter_factors=None, global_prior=0.5):
        self.n_components = n_components
        self.alpha = alpha
        self.pitcher_map = pitcher_map or {}
        self.batter_map = batter_map or {}
        self.pitcher_factors = pitcher_factors
        self.batter_factors = batter_factors
        self.global_prior = global_prior

    @classmethod
    def from_dict(cls, data):
        return cls(
            n_components=data["n_components"],
            alpha=data["alpha"],
            pitcher_map=data["pitcher_map"],
            batter_map=data["batter_map"],
            pitcher_factors=data["pitcher_factors"],
            batter_factors=data["batter_factors"],
            global_prior=data["global_prior"],
        )

    def transform(self, df):
        p_ids = df["pitcher_id"].to_numpy()
        b_ids = df["batter_id"].to_numpy()

        dots = np.zeros(len(df), dtype=float)
        p_f1 = np.zeros(len(df), dtype=float)
        p_f2 = np.zeros(len(df), dtype=float)
        b_f1 = np.zeros(len(df), dtype=float)
        b_f2 = np.zeros(len(df), dtype=float)

        if self.pitcher_factors is not None and self.batter_factors is not None:
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


# =======================
# 3. BetaCalibrator (순수 수식 계산)
# =======================

class BetaCalibrator:
    def __init__(self, a=1.0, b=1.0, c=0.0, eps=1e-5):
        self.a = a
        self.b = b
        self.c = c
        self.eps = eps

    @classmethod
    def from_dict(cls, data):
        return cls(
            a=data["a_"],
            b=data["b_"],
            c=data["c_"],
            eps=data.get("eps", 1e-5),
        )

    def predict(self, s):
        s_safe = np.clip(np.asarray(s, dtype=float), self.eps, 1.0 - self.eps)
        x1 = np.log(s_safe)
        x2 = -np.log(1.0 - s_safe)
        logit_p = self.a * x1 + self.b * x2 + self.c
        return sigmoid(logit_p)


# =======================
# 데이터 로드 & 피처 엔지니어링
# =======================

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님")
    return df


def build_features(df, global_mean, tk_lookup, season_encoder, svd_encoder=None):
    df = df.copy()

    # 1. 기본 파생 피처
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

    # 2. Season Decomposition 피처
    season_fe = season_encoder.transform(df)
    res = pd.concat([df, season_fe], axis=1)

    # 3. SVD Latent Matchup 피처
    if svd_encoder is not None:
        svd_fe = svd_encoder.transform(df)
        res = pd.concat([res, svd_fe], axis=1)

    return res.drop(columns=[ID_COL], errors="ignore")


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측 누락 {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "ensemble.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 아티팩트 로드 ----
    print("Load model...")
    artifact = joblib.load(MODEL_PATH)
    lgbm_model = artifact["lgbm_model"]
    cb_model = artifact["catboost_model"]
    stack_model = artifact["stack_model"]
    cat_cols = artifact["cat_cols"]
    all_features = artifact["all_features"]
    global_mean = artifact["global_mean"]
    tk_lookup = artifact["tk_lookup"]
    
    season_encoder = SeasonDecompositionEncoder.from_dict(artifact["season_encoder_data"])
    svd_encoder = LatentMatchupSVDEncoder.from_dict(artifact["svd_encoder_data"]) if "svd_encoder_data" in artifact else None
    beta_calibrator = BetaCalibrator.from_dict(artifact["beta_calibrator_data"]) if "beta_calibrator_data" in artifact else None

    print(f" OK. n_features={len(all_features)}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 & 피처 엔지니어링 ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, global_mean, tk_lookup, season_encoder, svd_encoder=svd_encoder)
    X = X[all_features]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    X_cb = X.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
    print(f" features={X.shape[1]}")

    # ---- 예측 (LightGBM/CatBoost) + 로지스틱 메타러너 스태킹 + Beta Calibration ----
    print("Inference model...")
    if len(X):
        p_lgb = lgbm_model.predict_proba(X)[:, 1]
        p_cb = cb_model.predict_proba(X_cb)[:, 1]
        X_meta = np.column_stack([p_lgb, p_cb])
        raw_stack = stack_model.predict_proba(X_meta)[:, 1]
        if beta_calibrator is not None:
            preds = beta_calibrator.predict(raw_stack)
        else:
            preds = raw_stack
    else:
        preds = []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
