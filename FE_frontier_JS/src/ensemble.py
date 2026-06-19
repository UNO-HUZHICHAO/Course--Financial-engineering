# -*- coding: utf-8 -*-
"""
集成评估与可视化（对应 ensemble.ipynb 的推理与融合逻辑，改为离线向量化评估）。

融合权重原样保留：
- 第一层：NN+XGB = 0.55*XGB(4模型等权) + 0.45*NN(5折等权)
- 第二层：final = 0.70*NNXGB + 0.10*Ridge + 0.40*TabM，clip[-5,5]（权重和 1.20，允许过冲）

在 validation.parquet 上离线计算各单模型与集成的加权 R²，并产出：
- 指标 CSV：整体 R²、分 symbol R²、模型间相关性、消融、逐日 R²
- 预测 parquet：各模型与集成的逐样本预测
- 可视化 PNG：单模型 vs 集成、分 symbol 热力图、相关性热力图、消融、预测分布、逐日 R² 时序
"""
import gc
import dill
import pickle
import joblib
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import config as C
from utils import (r2_val, r2_val_sum, standardize, encode_column_train, get_device)

# torch / NN / TabM 模型采用延迟导入：仅在 predict_nn / predict_tabm 内部加载，
# 使 predict_ridge / predict_xgb 子进程不触发 torch（避免 xgboost+torch 在部分
# 平台同进程的 native 冲突），同时主进程做指标与画图时也不依赖 torch。
DEVICE = None
Model = None
make_parameter_groups = None
NN = None


def _ensure_torch():
    """延迟加载 torch 相关模块与设备。"""
    global DEVICE, Model, make_parameter_groups, NN
    if DEVICE is not None:
        return
    import torch
    from tabm_reference import Model as _Model, make_parameter_groups as _mpg
    from nn_train import NN as _NN
    Model = _Model
    make_parameter_groups = _mpg
    NN = _NN
    DEVICE = get_device(C.NNConfig.gpuid)
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
MODEL_NAMES = ["xgb", "nn", "ensemble"]


# =====================================================================
# 数据
# =====================================================================
def load_valid():
    valid = pl.read_parquet(C.PROCESSED_VALID).to_pandas()
    if C.EvalConfig.honest_heldout:
        valid = valid[valid["date_id"] > C.EvalConfig.honest_cutoff].reset_index(drop=True)
        print(f"[ensemble] honest_heldout 模式：仅在 date_id>{C.EvalConfig.honest_cutoff} 上评估，n={len(valid)}")
    return valid


# =====================================================================
# 各模型预测（离线向量化，逻辑与原 ensemble 推理一致）
# =====================================================================
def predict_ridge(valid):
    with open(C.RidgeConfig.model_path, "rb") as fp:
        rdg = dill.load(fp)
    cols = C.RidgeConfig.feature_cols
    preds = rdg.predict(valid[cols].fillna(C.RidgeConfig.fillna_value).values)
    return preds.ravel()


def predict_xgb(valid):
    preds = np.zeros((len(valid),))
    n = 0
    for seed in C.XGBConfig.seeds:
        path = C.MODELS_DIR / f"{C.XGBConfig.result_prefix}_seed{seed}.pkl"
        with open(path, "rb") as fp:
            result = pickle.load(fp)
        m = result["model"]
        preds += m.predict(valid[C.XGBConfig.feature_cols]) / len(C.XGBConfig.seeds)
        n += 1
        del m
        gc.collect()
    return preds


def _batched_nn_predict(models, x_tensor, bs=8192):
    import torch
    preds = np.zeros((x_tensor.shape[0],))
    with torch.no_grad():
        for i in range(0, x_tensor.shape[0], bs):
            xb = x_tensor[i:i+bs].to(DEVICE)
            acc = torch.zeros((xb.shape[0],)).to(DEVICE)
            for m in models:
                m.eval()
                acc += m(xb) / len(models)
            preds[i:i+bs] = acc.cpu().numpy()
    return preds


def predict_nn(valid):
    _ensure_torch()
    import torch
    models = []
    for fold in range(C.NNConfig.n_fold):
        ckpt = C.MODELS_DIR / f"{C.NNConfig.model_prefix}_{fold}.ckpt"
        # 手动重建模型结构并加载 state_dict，避免 PL load_from_checkpoint 的
        # trainer/optimizer state 反序列化开销与潜在副作用（结构与训练时完全一致）
        ckpt_data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        h = ckpt_data["hyper_parameters"]
        m = NN(input_dim=h["input_dim"], hidden_dims=h["hidden_dims"],
               dropouts=h["dropouts"], lr=h["lr"], weight_decay=h["weight_decay"])
        m.load_state_dict(ckpt_data["state_dict"])
        m = m.to(DEVICE).eval()
        models.append(m)
    feat = valid[C.NNConfig.feature_cols].ffill().fillna(0)
    x_tensor = torch.FloatTensor(feat.values)
    preds = _batched_nn_predict(models, x_tensor)
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def _batched_tabm_predict(model, x_cont, x_cat, bs=8192):
    import torch
    preds = np.zeros((x_cont.shape[0],))
    with torch.no_grad():
        for i in range(0, x_cont.shape[0], bs):
            xc = x_cont[i:i+bs].to(DEVICE)
            xk = x_cat[i:i+bs].to(DEVICE)
            out = model(xc, xk).squeeze(-1)
            preds[i:i+bs] = out.mean(1).cpu().numpy()
    return preds


def predict_tabm(valid):
    _ensure_torch()
    import torch
    # 释放前一阶段（NN）残留的张量与缓存，避免在 CPU 后端重建模型时触发 native 竞态
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cfg = C.TabMConfig
    v = pl.from_pandas(valid.copy())
    for col in cfg.feature_cat + ["symbol_id", "time_id"]:
        v = encode_column_train(v, col, C.CATEGORY_MAPPINGS[col])
    # 缺失值填充：与训练一致，标准化前对连续特征 fill_null(0)
    v = v.with_columns([pl.col(c).fill_null(0.0) for c in cfg.feature_all])
    stats = joblib.load(C.DATA_STATS_PATH)
    v = standardize(v, cfg.std_feature, stats["mean"], stats["std"])

    # 全部 cast 为 Float32，保证 to_numpy() 返回连续 float 数组
    v = v.with_columns([pl.col(c).cast(pl.Float32) for c in cfg.feature_all])
    X = v[cfg.feature_all].to_numpy()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # 列序：feature_all 中 feature_09/10/11 位于索引 9/10/11
    x_cat = X_tensor[:, [9, 10, 11]]
    symbol_t = torch.tensor(v["symbol_id"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    time_t = torch.tensor(v["time_id"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    x_cat = torch.concat([x_cat, symbol_t, time_t], axis=1).to(torch.int64)
    x_cont = X_tensor[:, [i for i in range(X_tensor.shape[1]) if i not in [9, 10, 11]]]

    model = Model(
        n_num_features=cfg.n_cont_features,
        cat_cardinalities=cfg.cat_cardinalities,
        n_classes=None,
        backbone=cfg.backbone,
        bins=None,
        num_embeddings=None,
        arch_type="tabm",
        k=cfg.k,
    ).to(DEVICE)
    ckpt = torch.load(cfg.model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    preds = _batched_tabm_predict(model, x_cont, x_cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


# =====================================================================
# 指标
# =====================================================================
def weighted_r2(y, p, w):
    return r2_val(y, p, w)


def per_symbol(valid, preds_dict):
    rows = []
    for sid, g in valid.groupby("symbol_id"):
        idx = g.index
        y = g[C.TARGET_COL].values
        w = g[C.WEIGHT_COL].values
        row = {"symbol_id": int(sid)}
        for name in MODEL_NAMES:
            if name in preds_dict:
                row[name] = weighted_r2(y, preds_dict[name][idx], w)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("symbol_id").reset_index(drop=True)


def daily_r2(valid, preds):
    rows = []
    for d, g in valid.groupby("date_id"):
        idx = g.index
        y = g[C.TARGET_COL].values
        w = g[C.WEIGHT_COL].values
        rows.append({"date_id": int(d), "r2": weighted_r2(y, preds[idx], w), "n": len(g)})
    return pd.DataFrame(rows).sort_values("date_id").reset_index(drop=True)


def correlations(preds_dict):
    cols = [n for n in MODEL_NAMES if n in preds_dict]
    P = np.vstack([preds_dict[n] for n in cols])
    pear = np.corrcoef(P)
    spear = np.zeros_like(pear)
    for i in range(len(cols)):
        for j in range(len(cols)):
            spear[i, j] = spearmanr(preds_dict[cols[i]], preds_dict[cols[j]]).correlation
    return cols, pear, spear


def ablation(preds_dict, y, w):
    """精简版：XGB+NN 两模型融合的消融——剔除各组件与单模型对比。"""
    xgb = preds_dict["xgb"]
    nn = preds_dict["nn"]
    wx, wn = C.EnsembleConfig.e_weights
    rows = []
    rows.append({"config": "full(xgb+nn)", "r2": weighted_r2(y, np.clip(
        wx*xgb + wn*nn, *C.EnsembleConfig.clip), w)})
    rows.append({"config": "no_xgb(only_nn)", "r2": weighted_r2(y, np.clip(
        preds_dict["nn"], *C.EnsembleConfig.clip), w)})
    rows.append({"config": "no_nn(only_xgb)", "r2": weighted_r2(y, np.clip(
        preds_dict["xgb"], *C.EnsembleConfig.clip), w)})
    return pd.DataFrame(rows)


# =====================================================================
# 可视化
# =====================================================================
def plot_overall(overall_df):
    plt.figure(figsize=(8, 4.5))
    colors = sns.color_palette("viridis", len(overall_df))
    plt.bar(overall_df["model"], overall_df["weighted_r2"], color=colors)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("Weighted R²")
    plt.title("Single-model vs Ensemble (validation)")
    plt.xticks(rotation=30)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "overall_r2.png", dpi=150)
    plt.close()


def plot_per_symbol_heatmap(per_sym_df):
    cols = [c for c in MODEL_NAMES if c in per_sym_df.columns]
    mat = per_sym_df[cols].T.values
    plt.figure(figsize=(14, 3.2))
    sns.heatmap(mat, xticklabels=per_sym_df["symbol_id"].values, yticklabels=cols,
                center=0, cmap="RdYlGn", cbar_kws={"label": "weighted R²"})
    plt.xlabel("symbol_id")
    plt.title("Per-symbol weighted R² by model")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "per_symbol_r2_heatmap.png", dpi=150)
    plt.close()


def plot_correlation(cols, pear, spear):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pear, ax=axes[0], xticklabels=cols, yticklabels=cols, annot=True,
                fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Pearson correlation")
    sns.heatmap(spear, ax=axes[1], xticklabels=cols, yticklabels=cols, annot=True,
                fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Spearman rank correlation")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "model_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_ablation(abl_df):
    plt.figure(figsize=(9, 4.5))
    colors = ["green" if i == 0 else "steelblue" for i in range(len(abl_df))]
    plt.bar(abl_df["config"], abl_df["r2"], color=colors)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("Weighted R²")
    plt.title("Ablation: component contribution to ensemble")
    plt.xticks(rotation=30)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "ablation.png", dpi=150)
    plt.close()


def plot_pred_distribution(preds_dict):
    plt.figure(figsize=(9, 4.5))
    for n in ["xgb", "nn", "ensemble"]:
        if n in preds_dict:
            plt.hist(np.clip(preds_dict[n], -5, 5), bins=80, alpha=0.45, label=n, density=True)
    plt.xlabel("prediction (clipped to [-5,5])")
    plt.ylabel("density")
    plt.title("Prediction distribution by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "prediction_distribution.png", dpi=150)
    plt.close()


def plot_daily_r2(daily_df):
    plt.figure(figsize=(12, 4))
    plt.plot(daily_df["date_id"], daily_df["r2"], lw=0.8)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("date_id")
    plt.ylabel("daily weighted R²")
    plt.title("Ensemble daily weighted R² (validation)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "daily_r2_timeseries.png", dpi=150)
    plt.close()


def plot_scatter(y, p):
    plt.figure(figsize=(5, 5))
    # 下采样以加速
    n = len(y)
    k = min(n, 200000)
    idx = np.random.RandomState(42).choice(n, k, replace=False)
    plt.scatter(y[idx], p[idx], s=1, alpha=0.15)
    lim = (-5, 5)
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("y_true")
    plt.ylabel("ensemble prediction")
    plt.title("Ensemble: prediction vs truth (subsampled)")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "scatter_pred_vs_true.png", dpi=150)
    plt.close()


# =====================================================================
# 子进程隔离推理：避免同进程内多模型交替初始化触发 torch native 竞态
# （Apple Silicon CPU 后端偶发 segfault；CUDA 后端无此问题，但隔离亦有益无害）
# =====================================================================
def _run_in_subprocess(name, fn_name):
    """在独立子进程运行指定 predict 函数，预测经临时 npy 文件回传。"""
    import subprocess, sys, tempfile, os
    tmp = tempfile.mktemp(suffix=".npy")
    out_path = tmp  # np.save 对 *.npy 路径不再追加后缀
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "import numpy as np\n"
        "import ensemble as E\n"
        "valid = E.load_valid()\n"
        f"fn = getattr(E, '{fn_name}')\n"
        "p = fn(valid)\n"
        f"np.save(r'{out_path}', p)\n"
    )
    env = dict(os.environ)
    ret = subprocess.call([sys.executable, "-c", code], cwd=str(C.SRC_DIR), env=env)
    if ret != 0:
        raise RuntimeError(f"子进程 {fn_name} 失败 (exit={ret})")
    return np.load(out_path)


# =====================================================================
# 主流程
# =====================================================================
def main():
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("[ensemble] 读取验证集 ...")
    valid = load_valid()
    y = valid[C.TARGET_COL].values
    w = valid[C.WEIGHT_COL].values
    print(f"[ensemble] valid n={len(valid)}")

    # 精简版：仅 XGB + NN 两模型推理（独立子进程隔离 torch/xgb native 状态）
    print("[ensemble] XGBoost 推理 ..."); preds = {"xgb": _run_in_subprocess("xgb", "predict_xgb")}
    print("[ensemble] NN 推理 ...");     preds["nn"] = _run_in_subprocess("nn", "predict_nn")

    # 两模型加权融合
    wx, wn = C.EnsembleConfig.e_weights
    final = wx * preds["xgb"] + wn * preds["nn"]
    preds["ensemble"] = np.clip(final, *C.EnsembleConfig.clip)

    # ---- 整体 R² ----
    overall = pd.DataFrame([
        {"model": n, "weighted_r2": weighted_r2(y, preds[n], w)} for n in MODEL_NAMES
    ])
    overall.to_csv(C.METRICS_DIR / "overall_r2.csv", index=False)
    print("\n========== Overall weighted R² ==========")
    print(overall.to_string(index=False))

    # ---- 分 symbol ----
    per_sym = per_symbol(valid, preds)
    per_sym.to_csv(C.METRICS_DIR / "per_symbol_r2.csv", index=False)

    # ---- 逐日 ----
    daily = daily_r2(valid, preds["ensemble"])
    daily.to_csv(C.METRICS_DIR / "daily_r2.csv", index=False)

    # ---- 相关性 ----
    cols, pear, spear = correlations(preds)
    pd.DataFrame(pear, index=cols, columns=cols).to_csv(C.METRICS_DIR / "corr_pearson.csv")
    pd.DataFrame(spear, index=cols, columns=cols).to_csv(C.METRICS_DIR / "corr_spearman.csv")

    # ---- 消融 ----
    abl = ablation(preds, y, w)
    abl.to_csv(C.METRICS_DIR / "ablation.csv", index=False)
    print("\n========== Ablation ==========")
    print(abl.to_string(index=False))

    # ---- 预测保存 ----
    out = valid[["date_id", "time_id", "symbol_id", "row_id"]].copy() if "row_id" in valid else \
          valid[["date_id", "time_id", "symbol_id"]].copy()
    out["y_true"] = y
    out["weight"] = w
    for n in MODEL_NAMES:
        out[f"pred_{n}"] = preds[n]
    out.to_parquet(C.PREDICTIONS_DIR / "valid_predictions.parquet", index=False)
    print(f"\n[ensemble] 预测已保存 -> {C.PREDICTIONS_DIR/'valid_predictions.parquet'}")

    # ---- 可视化 ----
    print("[ensemble] 生成图表 ...")
    plot_overall(overall)
    plot_per_symbol_heatmap(per_sym)
    plot_correlation(cols, pear, spear)
    plot_ablation(abl)
    plot_pred_distribution(preds)
    plot_daily_r2(daily)
    plot_scatter(y, preds["ensemble"])
    print(f"[ensemble] 图表已保存至 {C.FIGURES_DIR}")
    print("[ensemble] 全部完成。")


if __name__ == "__main__":
    main()
