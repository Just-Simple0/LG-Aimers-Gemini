import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DATA_DIR = "data"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

def brier_skill_score(y_true, y_pred):
    bs = np.mean((y_pred - y_true) ** 2)
    r = y_true.mean()
    return max(0, 100000 * (1 - bs / (r * (1 - r))))

# Load data
from docs.v63_train_and_package import (
    SeasonDecompositionEncoder, make_temporal_season_features,
    make_tk_lookup, build_base_features
)

test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
FEATURES_BASE = [c for c in test_cols if c != "row_id"]
train_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=FEATURES_BASE + [TARGET])
GLOBAL_MEAN_VAL = train_full.loc[train_full["season"] != 2024, TARGET].mean()

tk_raw = pd.read_csv(
    os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
    usecols=["season"] + TK_KEYS + ["pitch_type_group", "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "zone_speed"]
)
TK_LOOKUP_VAL = make_tk_lookup(tk_raw[tk_raw["season"] <= 2023])

train_split = train_full[train_full["season"] <= 2023]
val_split = train_full[train_full["season"] == 2024]

season_decomp_train = make_temporal_season_features(train_split, TARGET)
encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
season_decomp_val = encoder_val.transform(val_split)
season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
fe_all = pd.concat([base_fe, season_decomp_all], axis=1)

NEW_COLS = [c for c in fe_all.columns if c not in train_full.columns]
ALL_FEATURES = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS

for c in CAT_COLS:
    fe_all[c] = fe_all[c].astype("category")

is_val = fe_all["season"] == 2024
X_tr = fe_all.loc[~is_val, ALL_FEATURES].copy()
y_tr = fe_all.loc[~is_val, TARGET].to_numpy()
X_va = fe_all.loc[is_val, ALL_FEATURES].copy()
y_va = fe_all.loc[is_val, TARGET].to_numpy()

X_tr_cb = X_tr.copy()
X_va_cb = X_va.copy()
for c in CAT_COLS:
    X_tr_cb[c] = X_tr_cb[c].astype(str)
    X_va_cb[c] = X_va_cb[c].astype(str)
cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]

# 1. Fit Base Models
print("Fitting LGB (lambda=1.0 and lambda=3.0)...")
LGB_1 = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
LGB_1.fit(X_tr, y_tr)
p_lgb_1 = LGB_1.predict_proba(X_va)[:, 1]

LGB_3 = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
LGB_3.fit(X_tr, y_tr)
p_lgb_3 = LGB_3.predict_proba(X_va)[:, 1]

print("Fitting CB (depth=8, l2=3.0)...")
CB = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
CB.fit(Pool(X_tr_cb, y_tr, cat_features=cat_idx))
p_cb = CB.predict_proba(X_va_cb)[:, 1]

# 2. 5-Fold OOF vs 10-Fold OOF
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)

# 5-fold OOF for LGB1 and CB
lgb1_oof_5 = np.zeros(len(y_tr))
cb_oof_5 = np.zeros(len(y_tr))
for tr_idx, val_idx in kf5.split(X_tr):
    m = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
    lgb1_oof_5[val_idx] = m.predict_proba(X_tr.iloc[val_idx])[:, 1]

    m_c = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    m_c.fit(Pool(X_tr_cb.iloc[tr_idx], y_tr[tr_idx], cat_features=cat_idx))
    cb_oof_5[val_idx] = m_c.predict_proba(X_tr_cb.iloc[val_idx])[:, 1]

# 10-fold OOF for LGB3 and CB
lgb3_oof_10 = np.zeros(len(y_tr))
cb_oof_10 = np.zeros(len(y_tr))
for tr_idx, val_idx in kf10.split(X_tr):
    m = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
    lgb3_oof_10[val_idx] = m.predict_proba(X_tr.iloc[val_idx])[:, 1]

    m_c = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    m_c.fit(Pool(X_tr_cb.iloc[tr_idx], y_tr[tr_idx], cat_features=cat_idx))
    cb_oof_10[val_idx] = m_c.predict_proba(X_tr_cb.iloc[val_idx])[:, 1]

# Meta-learner fits
st5 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([lgb1_oof_5, cb_oof_5]), y_tr)
p_v14_raw = st5.predict_proba(np.column_stack([p_lgb_1, p_cb]))[:, 1]
p_v14_clip = np.clip(p_v14_raw, 0.325, 0.755)

st10 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([lgb3_oof_10, cb_oof_10]), y_tr)
p_v63_raw = st10.predict_proba(np.column_stack([p_lgb_3, p_cb]))[:, 1]
p_v63_clip = np.clip(p_v63_raw, 0.325, 0.755)

print("\n=======================================================")
print("DIAGNOSTIC COMPARISON ON 2024 HOLDOUT:")
print("=======================================================")
print(f"v14 Raw (Dacon 976.51 baseline) : BSS = {brier_skill_score(y_va, p_v14_raw):.4f} | Min={p_v14_raw.min():.4f}, Max={p_v14_raw.max():.4f}")
print(f"v14 Clipped [0.325, 0.755]      : BSS = {brier_skill_score(y_va, p_v14_clip):.4f} | Min={p_v14_clip.min():.4f}, Max={p_v14_clip.max():.4f}")
print(f"v63 Raw (Unclipped)             : BSS = {brier_skill_score(y_va, p_v63_raw):.4f} | Min={p_v63_raw.min():.4f}, Max={p_v63_raw.max():.4f}")
print(f"v63 Clipped (Dacon 970.87)      : BSS = {brier_skill_score(y_va, p_v63_clip):.4f} | Min={p_v63_clip.min():.4f}, Max={p_v63_clip.max():.4f}")
print("=======================================================")
