# -*- coding: utf-8 -*-
"""
XGBoost 训练（对应 xgb_train.ipynb）。

建模细节原样保留：
- 特征 = symbol_id + time_id + 79 基础特征 + 9 滞后特征（91 维）
- train+valid 合并训练（Kaggle LB boosting trick）
- 参数：lr=0.05, max_depth=6, n_estimators=200, subsample=0.8, colsample=0.8,
        reg_alpha=1, reg_lambda=5, GPU hist
- 训练 4 个不同种子的模型（推理时等权 0.25 平均），对应报告“4个XGBoost不同种子”
- 在 validation 上计算整体加权 R² 与分 symbol 的 CV 明细
"""
import argparse
import gc
import pickle
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import config as C
from utils import seed_everything, to_xgb_device


def load_data():
    train = pl.read_parquet(C.PROCESSED_TRAIN).to_pandas()
    valid = pl.read_parquet(C.PROCESSED_VALID).to_pandas()
    import os as _os
    if _os.environ.get("HONEST_TRAIN"):
        # 无泄露模式：不合并 validation，仅用 date_id<=1577 训练，在 date_id>1577 评估
        print("[XGB] HONEST_TRAIN 模式：不合并 validation（无数据泄露）")
        return train, valid
    # Trick of boosting LB score (原 notebook: 0.45->0.49)
    train = pd.concat([train, valid]).reset_index(drop=True)
    return train, valid


def get_model(seed):
    params = to_xgb_device(C.XGBConfig.params)
    params["random_state"] = seed
    return XGBRegressor(**params)


def train_one(seed, train, valid):
    print(f"\n[XGB] ===== seed={seed} =====")
    X_train = train[C.XGBConfig.feature_cols]
    y_train = train[C.TARGET_COL]
    w_train = train[C.WEIGHT_COL]
    X_valid = valid[C.XGBConfig.feature_cols]
    y_valid = valid[C.TARGET_COL]
    w_valid = valid[C.WEIGHT_COL]

    model = get_model(seed)
    model.fit(X_train, y_train, sample_weight=w_train)

    # 训练集 R²（分半预测，原 notebook 写法）
    half = X_train.shape[0] // 2
    y_pred_train = np.concatenate(
        [model.predict(X_train.iloc[:half]), model.predict(X_train.iloc[half:])], axis=0
    )
    train_score = r2_score(y_train, y_pred_train, sample_weight=w_train)

    y_pred_valid = model.predict(X_valid)
    valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid)
    print(f"[XGB] seed={seed} train_r2={train_score:.5f} valid_r2={valid_score:.5f}")
    return model, valid_score


def per_symbol_cv(model, valid):
    """分 symbol 计算 CV 明细（原 notebook cell 14）。"""
    cv_detail = {sid: 0 for sid in range(39)}
    for symbol_id, gdf in valid.groupby("symbol_id"):
        X_v = gdf[C.XGBConfig.feature_cols]
        y_v = gdf[C.TARGET_COL]
        w_v = gdf[C.WEIGHT_COL]
        score = r2_score(y_v, model.predict(X_v), sample_weight=w_v)
        cv_detail[symbol_id] = score
    return cv_detail


def y_means(train):
    """各 symbol 的历史 y 均值（原 notebook cell 13）。"""
    ym = {sid: -1 for sid in range(39)}
    for symbol_id, gdf in train[["symbol_id", C.TARGET_COL]].groupby("symbol_id"):
        ym[symbol_id] = float(gdf[C.TARGET_COL].mean())
    return ym


def main():
    seed_everything(C.SEED)
    train, valid = load_data()
    print(f"train.shape={train.shape}, valid.shape={valid.shape}")

    cv_details_all = []
    valid_scores = []
    last_model = None
    for seed in C.XGBConfig.seeds:
        model, valid_score = train_one(seed, train, valid)
        last_model = model
        valid_scores.append(valid_score)
        cv_detail = per_symbol_cv(model, valid)
        cv_details_all.append(cv_detail)

        result = {
            "model": model,
            "cv": valid_score,
            "cv_detail": cv_detail,
            "y_mean": y_means(train),
            "seed": seed,
        }
        out = C.MODELS_DIR / f"{C.XGBConfig.result_prefix}_seed{seed}.pkl"
        with open(out, "wb") as fp:
            pickle.dump(result, fp)
        print(f"[XGB] 已保存 -> {out}")
        del model
        gc.collect()

    # ---- 可视化：分 symbol 的 CV 得分（用第一个种子模型展示）----
    cv_detail = cv_details_all[0]
    sids = list(cv_detail.keys())
    plt.figure(figsize=(12, 4))
    plt.bar(sids, [cv_detail[s] for s in sids])
    plt.grid(axis="y")
    plt.xlabel("symbol_id")
    plt.ylabel("CV score (weighted R²)")
    plt.title(f"XGBoost per-symbol validation R² (seed={C.XGBConfig.seeds[0]})")
    plt.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGURES_DIR / "xgb_per_symbol_r2.png", dpi=150)
    plt.close()

    print("\n[XGB] 全部种子 valid_r2:", [f"{s:.5f}" for s in valid_scores])
    print(f"[XGB] 平均 valid_r2={np.mean(valid_scores):.5f}")


if __name__ == "__main__":
    main()
