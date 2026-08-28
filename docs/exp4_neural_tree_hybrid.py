"""
Paradigm 4: Heterogeneous Neural-Tree Hybrid (PyTorch Tabular MLP + GBDT)
========================================================================

연구 목표:
  GBDT(LightGBM, CatBoost)의 직교 분할 한계를 극복하기 위해,
  투수/타자 Entity-Embedding과 연속형 피처를 학습하는 PyTorch Tabular Neural Net(MLP)을
  구축하여 5-Fold OOF를 생성하고 3종 이종 앙상블(GBDT + Neural Net)을 검증.

작성일: 2026-08-28
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DATA_DIR = "data"
ID_COL = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
TK_KEYS = ["balls_before", "strikes_before", "outs_before"]
ALPHA = 50.0

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


class TabularMLP(nn.Module):
    def __init__(self, num_pitchers, num_batters, emb_dim=16, num_cont=90, hidden_dim=128):
        super().__init__()
        self.pitcher_emb = nn.Embedding(num_pitchers + 1, emb_dim, padding_idx=0)
        self.batter_emb = nn.Embedding(num_batters + 1, emb_dim, padding_idx=0)
        self.bn_cont = nn.BatchNorm1d(num_cont)

        in_dim = emb_dim * 2 + num_cont
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, p_ids, b_ids, x_cont):
        p_vec = self.pitcher_emb(p_ids)
        b_vec = self.batter_emb(b_ids)
        x_c = self.bn_cont(x_cont)
        x = torch.cat([p_vec, b_vec, x_c], dim=1)
        return self.net(x).squeeze(-1)


def train_mlp_fold(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    best_loss = float("inf")
    best_preds = None

    for epoch in range(epochs):
        model.train()
        for p_b, b_b, x_b, y_b in train_loader:
            p_b, b_b, x_b, y_b = p_b.to(device), b_b.to(device), x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(p_b, b_b, x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    # Inference on val
    model.eval()
    val_preds = []
    with torch.no_grad():
        for p_b, b_b, x_b, _ in val_loader:
            p_b, b_b, x_b = p_b.to(device), b_b.to(device), x_b.to(device)
            logits = model(p_b, b_b, x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.append(probs)

    return np.concatenate(val_preds)


def main():
    print("=" * 80)
    print("🚀 [Paradigm 4] Heterogeneous Neural-Tree Hybrid (PyTorch MLP + GBDT)")
    print("기준선: v14 Champion (Holdout 836.35, OOF 2009.23, Dacon 976.51)")
    print("=" * 80)

    # 1. 데이터 로드 및 피처 생성
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

    season_decomp_train = make_temporal_season_features(train_split, TARGET)
    encoder_val = SeasonDecompositionEncoder().fit(train_split, train_split[TARGET])
    season_decomp_val = encoder_val.transform(val_split)
    season_decomp_all = pd.concat([season_decomp_train, season_decomp_val], axis=0).loc[train_full.index]

    base_fe = build_base_features(train_full, GLOBAL_MEAN_VAL, TK_LOOKUP_VAL)
    fe_v14 = pd.concat([base_fe, season_decomp_all], axis=1)

    NEW_COLS_V14 = [c for c in fe_v14.columns if c not in train_full.columns]
    ALL_FEATURES_V14 = CAT_COLS + [c for c in FEATURES_BASE if c not in CAT_COLS] + NEW_COLS_V14

    for c in CAT_COLS:
        fe_v14[c] = fe_v14[c].astype("category")

    is_val_mask = fe_v14["season"] == 2024
    X_train_full = fe_v14.loc[~is_val_mask, ALL_FEATURES_V14]
    y_train_full = fe_v14.loc[~is_val_mask, TARGET].to_numpy()
    X_val_fe = fe_v14.loc[is_val_mask, ALL_FEATURES_V14]
    y_val = fe_v14.loc[is_val_mask, TARGET].to_numpy()

    # GBDT용 데이터
    X_train_cb = X_train_full.copy()
    X_val_cb = X_val_fe.copy()
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    cat_idx = [ALL_FEATURES_V14.index(c) for c in CAT_COLS]

    # 1. GBDT 학습
    print("\n[1/3] GBDT (LightGBM & CatBoost) 베이스 예측 생성...")
    lgb_m = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
    lgb_m.fit(X_train_full, y_train_full)
    p_lgb = lgb_m.predict_proba(X_val_fe)[:, 1]

    cb_m = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
    cb_m.fit(Pool(X_train_cb, y_train_full, cat_features=cat_idx))
    p_cb = cb_m.predict_proba(X_val_cb)[:, 1]

    # 2. PyTorch 데이터 전처리 (Entity IDs + Scaled Continuous)
    print("\n[2/3] PyTorch Tabular Neural Net 데이터 구성 및 5-Fold OOF 학습...")
    pitchers = sorted(train_full["pitcher_id"].dropna().unique())
    batters = sorted(train_full["batter_id"].dropna().unique())
    p_map = {pid: i + 1 for i, pid in enumerate(pitchers)}
    b_map = {bid: i + 1 for i, bid in enumerate(batters)}

    cont_cols = [c for c in ALL_FEATURES_V14 if c not in CAT_COLS and c not in ["pitcher_id", "batter_id"]]
    scaler = StandardScaler()
    X_tr_cont = scaler.fit_transform(X_train_full[cont_cols].fillna(0.0))
    X_va_cont = scaler.transform(X_val_fe[cont_cols].fillna(0.0))

    p_tr_ids = X_train_full["pitcher_id"].map(p_map).fillna(0).to_numpy(dtype=np.int64)
    b_tr_ids = X_train_full["batter_id"].map(b_map).fillna(0).to_numpy(dtype=np.int64)
    p_va_ids = X_val_fe["pitcher_id"].map(p_map).fillna(0).to_numpy(dtype=np.int64)
    b_va_ids = X_val_fe["batter_id"].map(b_map).fillna(0).to_numpy(dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Neural Net Device: {device} | 연속형 피처: {len(cont_cols)}개")

    # 5-Fold OOF for GBDT and Neural Net
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_train_full))
    oof_cb = np.zeros(len(y_train_full))
    oof_nn = np.zeros(len(y_train_full))

    X_train_reset = X_train_full.reset_index(drop=True)
    X_cb_reset = X_train_cb.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_reset)):
        # GBDT OOF
        m_l = lgb.LGBMClassifier(n_estimators=171, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary", random_state=42, n_jobs=-1, verbosity=-1)
        m_l.fit(X_train_reset.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb[val_idx] = m_l.predict_proba(X_train_reset.iloc[val_idx])[:, 1]

        tp = Pool(X_cb_reset.iloc[tr_idx], y_train_full[tr_idx], cat_features=cat_idx)
        m_c = CatBoostClassifier(iterations=360, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function="Logloss", random_seed=42, verbose=False, thread_count=-1)
        m_c.fit(tp)
        oof_cb[val_idx] = m_c.predict_proba(X_cb_reset.iloc[val_idx])[:, 1]

        # Neural Net OOF
        tr_ds = TensorDataset(
            torch.tensor(p_tr_ids[tr_idx]),
            torch.tensor(b_tr_ids[tr_idx]),
            torch.tensor(X_tr_cont[tr_idx], dtype=torch.float32),
            torch.tensor(y_train_full[tr_idx], dtype=torch.float32),
        )
        va_ds = TensorDataset(
            torch.tensor(p_tr_ids[val_idx]),
            torch.tensor(b_tr_ids[val_idx]),
            torch.tensor(X_tr_cont[val_idx], dtype=torch.float32),
            torch.tensor(y_train_full[val_idx], dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=2048, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=4096, shuffle=False)

        mlp_m = TabularMLP(num_pitchers=len(pitchers), num_batters=len(batters), num_cont=len(cont_cols))
        oof_nn[val_idx] = train_mlp_fold(mlp_m, tr_loader, va_loader, epochs=6, lr=2e-3, device=device)
        print(f"  Fold {fold_i+1}/5 완료 (GBDT + PyTorch MLP)")

    # Full Train Neural Net for Validation Inference
    full_tr_ds = TensorDataset(
        torch.tensor(p_tr_ids),
        torch.tensor(b_tr_ids),
        torch.tensor(X_tr_cont, dtype=torch.float32),
        torch.tensor(y_train_full, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(p_va_ids),
        torch.tensor(b_va_ids),
        torch.tensor(X_va_cont, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    full_tr_loader = DataLoader(full_tr_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    full_mlp = TabularMLP(num_pitchers=len(pitchers), num_batters=len(batters), num_cont=len(cont_cols))
    p_nn = train_mlp_fold(full_mlp, full_tr_loader, val_loader, epochs=6, lr=2e-3, device=device)

    # 3. 앙상블 스태킹 비교
    print("\n[3/3] 이종 앙상블 스태킹 메타러너 학습 및 검증...")
    # v14 기준선 (LGB + CB)
    stack_v14 = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb, oof_cb]), y_train_full)
    p_ens_v14 = stack_v14.predict_proba(np.column_stack([p_lgb, p_cb]))[:, 1]
    bss_v14 = brier_skill_score(y_val, p_ens_v14)
    oof_v14 = brier_skill_score(y_train_full, stack_v14.predict_proba(np.column_stack([oof_lgb, oof_cb]))[:, 1])

    # 3종 이종 결합 (LGB + CB + PyTorch Neural Net)
    stack_hybrid = LogisticRegression(random_state=42, max_iter=1000).fit(np.column_stack([oof_lgb, oof_cb, oof_nn]), y_train_full)
    p_ens_hybrid = stack_hybrid.predict_proba(np.column_stack([p_lgb, p_cb, p_nn]))[:, 1]
    bss_hybrid = brier_skill_score(y_val, p_ens_hybrid)
    oof_hybrid = brier_skill_score(y_train_full, stack_hybrid.predict_proba(np.column_stack([oof_lgb, oof_cb, oof_nn]))[:, 1])

    print("\n" + "=" * 80)
    print("🏁 [Paradigm 4] Heterogeneous Neural-Tree Hybrid 검증 결과")
    print("=" * 80)
    print(f"1. v14 기준선 (LGB + CB)             : Holdout = {bss_v14:.2f} | OOF = {oof_v14:.2f} | LGB={stack_v14.coef_[0][0]:.2f}, CB={stack_v14.coef_[0][1]:.2f}")
    print(f"2. 이종 앙상블 (LGB + CB + PyTorch) : Holdout = {bss_hybrid:.2f} ({bss_hybrid-bss_v14:+.2f}) | OOF = {oof_hybrid:.2f} ({oof_hybrid-oof_v14:+.2f}) | Coefs={stack_hybrid.coef_[0]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
