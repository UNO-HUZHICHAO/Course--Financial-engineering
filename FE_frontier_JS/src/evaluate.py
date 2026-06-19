# -*- coding: utf-8 -*-
"""
单模型对比评估（无集成）：在诚实验证集（date_id>1577）上分别计算
Ridge / XGBoost / TabM / NN 四个单模型的加权 R²，并存逐样本预测供回测。

NN 仅作扩展实验，输出 R² 供报告引用，不参与策略收益回测（训练规模受限）。

复用 ensemble.py 中的 predict_ridge / predict_xgb / predict_tabm / predict_nn
（子进程隔离推理），本脚本只做编排与指标聚合，不修改任何建模细节。
"""
import gc
import numpy as np
import pandas as pd
import polars as pl

import config as C
from utils import r2_val
import ensemble as E

# 进入对比的模型；NN 标记为扩展（仅R²）
MODELS = ["ridge", "xgb"]                  # 主对比（TabM 已弃用）
MODELS_EXT = ["nn"]                        # 扩展（仅R²，不回测收益）

PREDICTORS = {
    "ridge": E.predict_ridge,
    "xgb": E.predict_xgb,
    "tabm": E.predict_tabm,
    "nn": E.predict_nn,
}


def main():
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("[eval] 读取诚实验证集 (date_id>1577) ...")
    valid = E.load_valid()
    y = valid[C.TARGET_COL].values
    w = valid[C.WEIGHT_COL].values
    print(f"[eval] valid n={len(valid)}, days={valid['date_id'].nunique()}")

    preds = {}
    # NN 模型可能未就绪（扩展实验，5折训练中）；缺 ckpt 则跳过
    nn_ready = all((C.MODELS_DIR / f"{C.NNConfig.model_prefix}_{f}.ckpt").exists()
                   for f in range(C.NNConfig.n_fold))
    run_list = list(MODELS) + (MODELS_EXT if nn_ready else [])
    if not nn_ready:
        print("[eval] NN 模型未就绪，本次仅评估主模型（ridge/xgb）；NN 跑完可单独补。")
    for name in run_list:
        print(f"\n[eval] === {name} 推理 ===")
        preds[name] = E._run_in_subprocess(name, PREDICTORS[name].__name__)
        r2 = r2_val(y, preds[name], w)
        print(f"[eval] {name} weighted_r2 = {r2:.6f}")
        gc.collect()

    # ---- 整体 R² 表 ----
    rows = []
    for name in MODELS:
        rows.append({"model": name, "weighted_r2": r2_val(y, preds[name], w),
                     "role": "main"})
    for name in MODELS_EXT:
        rows.append({"model": name, "weighted_r2": r2_val(y, preds[name], w),
                     "role": "extension (R2 only)"})
    overall = pd.DataFrame(rows)
    overall.to_csv(C.METRICS_DIR / "overall_r2.csv", index=False)
    print("\n========== Overall weighted R² ==========")
    print(overall.to_string(index=False))

    # ---- 分 symbol R²（主模型）----
    per_sym_rows = []
    for sid, g in valid.groupby("symbol_id"):
        idx = g.index
        yy = g[C.TARGET_COL].values
        ww = g[C.WEIGHT_COL].values
        row = {"symbol_id": int(sid)}
        for name in MODELS:
            row[name] = r2_val(yy, preds[name][idx], ww)
        per_sym_rows.append(row)
    per_sym = pd.DataFrame(per_sym_rows).sort_values("symbol_id").reset_index(drop=True)
    per_sym.to_csv(C.METRICS_DIR / "per_symbol_r2.csv", index=False)

    # ---- 模型间相关性（主模型）----
    cols = MODELS
    P = np.vstack([preds[n] for n in cols])
    pear = np.corrcoef(P)
    pd.DataFrame(pear, index=cols, columns=cols).to_csv(C.METRICS_DIR / "corr_pearson.csv")
    from scipy.stats import spearmanr
    spear = np.zeros_like(pear)
    for i in range(len(cols)):
        for j in range(len(cols)):
            spear[i, j] = spearmanr(preds[cols[i]], preds[cols[j]]).correlation
    pd.DataFrame(spear, index=cols, columns=cols).to_csv(C.METRICS_DIR / "corr_spearman.csv")

    # ---- 逐日 R²（XGB 为主模型代表）----
    daily_rows = []
    for d, g in valid.groupby("date_id"):
        idx = g.index
        daily_rows.append({"date_id": int(d), "r2": r2_val(
            g[C.TARGET_COL].values, preds["xgb"][idx], g[C.WEIGHT_COL].values), "n": len(g)})
    pd.DataFrame(daily_rows).sort_values("date_id").to_csv(
        C.METRICS_DIR / "daily_r2.csv", index=False)

    # ---- 存逐样本预测（供回测；NN 也存，但回测只用主模型）----
    base = valid[["date_id", "time_id", "symbol_id"]].copy() \
        if "row_id" not in valid else valid[["date_id", "time_id", "symbol_id", "row_id"]].copy()
    base["y_true"] = y
    base["weight"] = w
    for name in MODELS + MODELS_EXT:
        base[f"pred_{name}"] = preds[name]
    base.to_parquet(C.PREDICTIONS_DIR / "valid_predictions.parquet", index=False)
    print(f"\n[eval] 预测已保存 -> {C.PREDICTIONS_DIR/'valid_predictions.parquet'}")

    _plot_all(valid, preds, overall, per_sym, cols, pear, spear, y, w)
    print("[eval] 完成。接下来运行 backtest.py 做单模型策略对比。")


def _plot_all(valid, preds, overall, per_sym, cols, pear, spear, y, w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import r2_val
    FD, MD = C.FIGURES_DIR, C.METRICS_DIR
    all_models = MODELS + [m for m in MODELS_EXT if m in preds]

    # overall R2
    plt.figure(figsize=(8, 4.5))
    colors = sns.color_palette("viridis", len(overall))
    plt.bar(overall["model"], overall["weighted_r2"], color=colors)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("Weighted R²"); plt.title("Single-model weighted R² (honest validation)")
    plt.xticks(rotation=30); plt.grid(axis="y"); plt.tight_layout()
    plt.savefig(FD / "overall_r2.png", dpi=150); plt.close()

    # per-symbol heatmap
    heat_cols = [c for c in all_models if c in per_sym.columns]
    mat = per_sym[heat_cols].T.values
    plt.figure(figsize=(14, 3.2))
    sns.heatmap(mat, xticklabels=per_sym["symbol_id"].values, yticklabels=heat_cols,
                center=0, cmap="RdYlGn", cbar_kws={"label": "weighted R²"})
    plt.xlabel("symbol_id"); plt.title("Per-symbol weighted R² by model")
    plt.tight_layout(); plt.savefig(FD / "per_symbol_r2_heatmap.png", dpi=150); plt.close()

    # correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(pear, ax=axes[0], xticklabels=cols, yticklabels=cols, annot=True,
                fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1); axes[0].set_title("Pearson")
    sns.heatmap(spear, ax=axes[1], xticklabels=cols, yticklabels=cols, annot=True,
                fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1); axes[1].set_title("Spearman rank")
    plt.tight_layout(); plt.savefig(FD / "model_correlation_heatmap.png", dpi=150); plt.close()

    # daily R2 (xgb)
    daily_rows = []
    for d, g in valid.groupby("date_id"):
        idx = g.index
        daily_rows.append({"date_id": int(d), "r2": r2_val(
            g[C.TARGET_COL].values, preds["xgb"][idx], g[C.WEIGHT_COL].values)})
    daily = pd.DataFrame(daily_rows).sort_values("date_id")
    plt.figure(figsize=(12, 4))
    plt.plot(daily["date_id"], daily["r2"], lw=0.8)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("date_id"); plt.ylabel("daily weighted R²")
    plt.title("XGBoost daily weighted R² (honest validation)"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FD / "daily_r2_timeseries.png", dpi=150); plt.close()

    # prediction distribution
    plt.figure(figsize=(9, 4.5))
    for n in all_models:
        plt.hist(np.clip(preds[n], -5, 5), bins=80, alpha=0.45, label=n, density=True)
    plt.xlabel("prediction (clipped to [-5,5])"); plt.ylabel("density")
    plt.title("Prediction distribution by model"); plt.legend()
    plt.tight_layout(); plt.savefig(FD / "prediction_distribution.png", dpi=150); plt.close()

    # scatter pred vs true (xgb)
    plt.figure(figsize=(5, 5))
    n = len(y); k = min(n, 200000)
    idx = np.random.RandomState(42).choice(n, k, replace=False)
    plt.scatter(y[idx], np.clip(preds["xgb"], -5, 5)[idx], s=1, alpha=0.15)
    lim = (-5, 5); plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("y_true"); plt.ylabel("XGBoost prediction")
    plt.title("XGBoost: prediction vs truth (subsampled)")
    plt.tight_layout(); plt.savefig(FD / "scatter_pred_vs_true.png", dpi=150); plt.close()
    print("[eval] 图表已更新至 result/figures")


if __name__ == "__main__":
    main()
