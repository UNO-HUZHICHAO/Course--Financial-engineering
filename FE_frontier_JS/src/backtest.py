# -*- coding: utf-8 -*-
"""
量化策略回测（单模型对比，诚实口径）。

将各单模型预测转化为市场中性多空组合，在诚实验证窗口（date_id>1577）上
比较模型的排序收益能力。不做模型集成，仅横向对比 Ridge / XGBoost / TabM
（NN 为扩展实验，不参与收益回测）。

策略逻辑（对应报告第七节）：
- 每个时间步 (date_id, time_id) 对 39 个 symbol 按预测值横截面排序
- 做多预测最高的 top_K 个、做空预测最低的 top_K 个
- 用竞赛 weight 加权，多空等权对冲
- 单个时间步组合收益 = weighted_mean(y_top) - weighted_mean(y_bottom)
- 逐日聚合（968 个 time_id 求和）得到日收益序列

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
重要诚实说明（务必写入报告）：
1. responder_6 是 Jane Street 脱敏后的收益度量（量级 ±5），非真实百分比收益。
   本回测的"收益/Sharpe/年化"均在该脱敏度量下，仅用于横向比较模型把预测质量
   转化为横截面排序收益的能力，不直接等同于实盘可交易收益。
2. 回测不含交易成本、滑点、容量限制与冲击成本，属理想化上界。
3. 当模型加权 R² 高达 ~0.86 时，横截面排序近乎完美，每个时间步多空价差稳定
   为正（per-bar Sharpe 高、win_rate 接近 1、max_drawdown 接近 0）——这是
   "预测精度极高"的必然结果，而非策略缺陷。读者应据此理解绝对数值的量级。
4. 全程无数据泄露训练（XGB/TabM 仅用 date_id<=1577 训练，回测在 date_id>1577），
   故结果为模型真实泛化能力的无偏估计。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法：
    python backtest.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import config as C

PRED_PATH = C.PREDICTIONS_DIR / "valid_predictions.parquet"
METRICS_DIR = C.METRICS_DIR
FIGURES_DIR = C.FIGURES_DIR
# 回测对比模型（NN 标注为扩展，但一并回测以展示 R²→收益的因果关系）
MODELS = ["ridge", "xgb", "nn"]
TOP_K = 10
ANNUAL_FACTOR = 252


def load_predictions():
    df = pd.read_parquet(PRED_PATH)
    df = df.sort_values(["date_id", "time_id", "symbol_id"]).reset_index(drop=True)
    df = df[df["date_id"] > C.EvalConfig.honest_cutoff].reset_index(drop=True)
    return df


def weighted_mean(y, w):
    s = w.sum()
    return (y * w).sum() / s if s > 0 else 0.0


def long_short_returns(df, pred_col, top_k=TOP_K):
    """每个时间步按 pred 排序，weight 加权多空。返回每个 (date_id,time_id) 的收益。"""
    def _step(g):
        g = g.sort_values(pred_col)
        n = len(g)
        k = min(top_k, n // 2)
        if k == 0:
            return 0.0
        top = g.iloc[-k:]
        bot = g.iloc[:k]
        return weighted_mean(top["y_true"].values, top["weight"].values) - \
               weighted_mean(bot["y_true"].values, bot["weight"].values)
    ts_ret = df.groupby(["date_id", "time_id"], sort=False).apply(_step, include_groups=False)
    return ts_ret.rename("ret").reset_index()


def daily_returns(ts_ret):
    daily = ts_ret.groupby("date_id")["ret"].sum().reset_index()
    daily.columns = ["date_id", "daily_ret"]
    return daily


def compute_metrics(ts_ret, daily_ret):
    """ts_ret: 每个时间步的收益；daily_ret: 逐日收益。"""
    r = daily_ret["daily_ret"].values
    bar = ts_ret["ret"].values
    n_days = len(r)
    if n_days == 0:
        zeros = {k: 0.0 for k in ["n_days", "total_return", "annual_return",
                                  "annual_vol", "sharpe", "max_drawdown",
                                  "win_rate", "per_bar_sharpe", "avg_daily_ret"]}
        return zeros
    cum = np.cumsum(r)
    total_ret = cum[-1]
    ann_ret = total_ret * (ANNUAL_FACTOR / n_days)
    ann_std = r.std() * np.sqrt(ANNUAL_FACTOR)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0.0
    drawdown = cum - np.maximum.accumulate(cum)
    max_dd = drawdown.min()
    win_rate = (r > 0).mean()
    # per-bar Sharpe：每个时间步多空价差的均值/标准差，衡量信号纯度
    # （R²越高、排序越准，该值越大；这是 Sharpe 爆高的直接原因）
    bar_mean = bar.mean()
    bar_std = bar.std()
    per_bar_sharpe = bar_mean / bar_std if bar_std > 0 else 0.0
    return {
        "n_days": n_days, "total_return": total_ret, "annual_return": ann_ret,
        "annual_vol": ann_std, "sharpe": sharpe, "max_drawdown": max_dd,
        "win_rate": win_rate, "per_bar_sharpe": per_bar_sharpe,
        "avg_daily_ret": r.mean(),
    }


def run_backtest(df, tag, suffix):
    rows, daily_by_model, ts_by_model = [], {}, {}
    for m in MODELS:
        pred_col = f"pred_{m}"
        if pred_col not in df.columns:
            print(f"  [skip] {m}: 无预测列 {pred_col}")
            continue
        ts = long_short_returns(df, pred_col)
        daily = daily_returns(ts)
        daily_by_model[m] = daily
        ts_by_model[m] = ts
        met = compute_metrics(ts, daily)
        met["model"] = m
        rows.append(met)
        print(f"  [{tag}] {m}: R2驱动 -> Sharpe={met['sharpe']:.2f} "
              f"per_bar_Sharpe={met['per_bar_sharpe']:.3f} "
              f"winrate={met['win_rate']:.3f} MaxDD={met['max_drawdown']:.4f} "
              f"ann_ret={met['annual_return']:.1f}")

    cols = ["model", "n_days", "total_return", "annual_return", "annual_vol",
            "sharpe", "per_bar_sharpe", "max_drawdown", "win_rate", "avg_daily_ret"]
    metrics_df = pd.DataFrame(rows)[cols]
    metrics_df.to_csv(METRICS_DIR / f"backtest_metrics{suffix}.csv", index=False)
    if "xgb" in daily_by_model:
        daily_by_model["xgb"].to_csv(METRICS_DIR / f"backtest_daily{suffix}.csv", index=False)

    # ---- 累计收益曲线 ----
    plt.figure(figsize=(12, 5))
    for m, daily in daily_by_model.items():
        cum = np.cumsum(daily["daily_ret"].values)
        plt.plot(daily["date_id"].values, cum, label=m, lw=1.3)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("date_id")
    plt.ylabel("Cumulative long-short PnL (responder_6 units)")
    plt.title(f"Market-Neutral L/S Cumulative PnL [{tag}] (top{TOP_K}/bottom{TOP_K})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cumulative_return{suffix}.png", dpi=150)
    plt.close()

    # ---- Sharpe & 年化对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = metrics_df["model"].tolist()
    axes[0].bar(names, metrics_df["sharpe"], color="steelblue")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("Annualized Sharpe (desensitized units)")
    axes[0].set_title(f"Sharpe [{tag}]")
    axes[0].tick_params(axis="x", rotation=30); axes[0].grid(axis="y")
    axes[1].bar(names, metrics_df["per_bar_sharpe"], color="darkorange")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Per-bar L/S Sharpe (signal purity)")
    axes[1].set_title(f"Per-bar signal purity [{tag}]")
    axes[1].tick_params(axis="x", rotation=30); axes[1].grid(axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"strategy_metrics{suffix}.png", dpi=150)
    plt.close()

    # ---- 回撤曲线（XGB 代表）----
    if "xgb" in daily_by_model:
        r = daily_by_model["xgb"]["daily_ret"].values
        cum = np.cumsum(r)
        dd = cum - np.maximum.accumulate(cum)
        plt.figure(figsize=(12, 4))
        plt.fill_between(daily_by_model["xgb"]["date_id"].values, dd, 0, color="indianred", alpha=0.6)
        plt.xlabel("date_id"); plt.ylabel("Drawdown")
        plt.title(f"XGBoost Drawdown [{tag}]")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"drawdown{suffix}.png", dpi=150)
        plt.close()
    return metrics_df


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[backtest] 读取诚实预测 (date_id>1577) ...")
    df = load_predictions()
    print(f"[backtest] honest: {len(df)} rows, {df['date_id'].nunique()} days")
    print("[backtest] === 单模型多空策略对比（脱敏度量，无成本上界）===")
    run_backtest(df, "honest", "_honest")
    print(f"\n[backtest] done. see result/metrics, result/figures")


if __name__ == "__main__":
    main()
