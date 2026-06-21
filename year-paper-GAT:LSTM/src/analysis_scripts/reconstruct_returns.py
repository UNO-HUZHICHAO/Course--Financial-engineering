"""
策略日收益率重建模块
======================
从 optimal_weights.csv + Daily_Returns.csv 重建每日策略收益序列

核心公式（与 BacktestReporter._compute_strategy_nav() 一致）:
  r_p[t] = Σ(w_i[t-1] × r_i[t])    组合毛收益
  cost[t] = 0.002 × Σ|w_i[t] - w_i[t-1]|  双边0.2%交易成本
  r_net[t] = r_p[t] - cost[t]      组合净收益
  r_excess[t] = r_net[t] - r_bench[t]  超额收益

注意：权重在 t-1 日确定，收益在 t 日实现（隔夜持仓）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "学年论文实证部分"


def load_daily_returns() -> pd.DataFrame:
    """加载日个股收益率矩阵 (pivot: date × stkcd)"""
    path = DATA_ROOT / "data/processed/market/Daily_Returns.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    # 确保列名是字符串（股票代码）
    df.columns = [str(c).zfill(6) for c in df.columns]
    return df


def load_benchmark_returns() -> pd.Series:
    """
    加载科创50基准日收益率

    返回:
        Series, index=date, values=日收益率（小数制）
    """
    path = DATA_ROOT / "data/processed/market/KCB50_Index_Daily.csv"
    df = pd.read_csv(path)
    # trade_date 是 YYYYMMDD 整数格式（如 20251231），需显式指定格式
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    # pct_chg 是百分比制（如 -1.1525 表示 -1.1525%），转为小数
    df['ret'] = df['pct_chg'] / 100.0
    df = df.sort_values('trade_date')
    return df.set_index('trade_date')['ret']


def load_optimal_weights(weights_path: str) -> pd.DataFrame:
    """
    加载最优权重文件

    参数:
        weights_path: CSV路径 (trade_date, stkcd, weight, is_constituent)

    返回:
        DataFrame with trade_date as datetime, stkcd as string
    """
    df = pd.read_csv(weights_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['stkcd'] = df['stkcd'].astype(str).str.zfill(6)
    return df


def reconstruct_strategy_returns(
    weights_path: str,
    daily_returns: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    cost_rate: float = 0.002,
    verbose: bool = True
) -> Dict[str, pd.Series]:
    """
    从权重和股票收益重建策略日收益序列

    参数:
        weights_path: optimal_weights.csv 路径
        daily_returns: 日个股收益率矩阵（lazy load）
        benchmark_returns: 基准收益率序列（lazy load）
        cost_rate: 双边交易成本率 (默认0.2%)
        verbose: 是否打印进度信息

    返回:
        dict with keys:
            'gross_ret': 组合毛收益（费前）
            'cost': 日交易成本
            'net_ret': 组合净收益（费后）
            'excess_ret': 超额收益（净收益-基准）
            'turnover': 日换手率
            'nav': 累计净值
    """
    # 加载数据
    if daily_returns is None:
        daily_returns = load_daily_returns()
    if benchmark_returns is None:
        benchmark_returns = load_benchmark_returns()

    weights = load_optimal_weights(weights_path)

    # 获取所有调仓日
    trade_dates = sorted(weights['trade_date'].unique())

    if verbose:
        print(f"  权重日期数: {len(trade_dates)}")
        print(f"  日期范围: {trade_dates[0].date()} ~ {trade_dates[-1].date()}")

    # 构建日期到下一个交易日期的映射
    all_return_dates = sorted(daily_returns.index)
    date_to_next = {}
    for i, d in enumerate(all_return_dates[:-1]):
        date_to_next[d] = all_return_dates[i + 1]

    # 遍历每个调仓日，计算组合收益
    records = []
    prev_weights = None
    skipped = 0

    for t_date in trade_dates:
        # 获取当日权重
        w_t = weights[weights['trade_date'] == t_date].set_index('stkcd')['weight']

        # 确定收益实现日（下一个交易日）
        if t_date not in date_to_next:
            skipped += 1
            continue
        ret_date = date_to_next[t_date]

        # 获取收益日当天的个股收益
        if ret_date not in daily_returns.index:
            skipped += 1
            continue
        returns_t = daily_returns.loc[ret_date]

        # 对齐股票代码（取交集）
        common_stocks = w_t.index.intersection(returns_t.index)
        if len(common_stocks) == 0:
            skipped += 1
            continue

        w_aligned = w_t.loc[common_stocks]
        r_aligned = returns_t.loc[common_stocks]

        # 计算组合毛收益
        gross_ret = (w_aligned * r_aligned).sum()

        # 计算交易成本（换手率 × 双边成本率）
        if prev_weights is not None:
            prev_aligned = prev_weights.reindex(w_t.index, fill_value=0)
            turnover = (w_t - prev_aligned).abs().sum()
        else:
            # 首日：初始建仓成本
            turnover = w_t.abs().sum()

        cost = turnover * cost_rate

        records.append({
            'trade_date': t_date,
            'ret_date': ret_date,
            'gross_ret': gross_ret,
            'cost': cost,
            'turnover': turnover
        })

        prev_weights = w_t.copy()

    if verbose:
        print(f"  成功计算 {len(records)} 个交易日收益（跳过 {skipped} 个）")

    # 构建结果DataFrame
    results = pd.DataFrame(records)
    results['ret_date'] = pd.to_datetime(results['ret_date'])
    results = results.set_index('ret_date').sort_index()

    # 计算净收益
    results['net_ret'] = results['gross_ret'] - results['cost']

    # 对齐基准收益（仅计算有基准数据的时期）
    common_dates = results.index.intersection(benchmark_returns.index)
    results['benchmark_ret'] = np.nan
    results.loc[common_dates, 'benchmark_ret'] = benchmark_returns.loc[common_dates]
    results['excess_ret'] = results['net_ret'] - results['benchmark_ret']

    # 计算累计净值
    results['nav'] = (1 + results['net_ret']).cumprod()
    results['benchmark_nav'] = (1 + results['benchmark_ret'].fillna(0)).cumprod()

    return {
        'gross_ret': results['gross_ret'],
        'cost': results['cost'],
        'net_ret': results['net_ret'],
        'excess_ret': results['excess_ret'].dropna(),
        'turnover': results['turnover'],
        'nav': results['nav'],
        'benchmark_nav': results['benchmark_nav'],
        'benchmark_ret': results['benchmark_ret'].dropna(),
        'full_df': results
    }


def compute_annual_stats(
    daily_excess: pd.Series,
    daily_net: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    从日收益率计算年化统计指标

    参数:
        daily_excess: 日超额收益序列
        daily_net: 日净收益序列
        risk_free_rate: 年化无风险利率
        periods_per_year: 年交易日数

    返回:
        dict with: annual_return, annual_excess, volatility, tracking_error,
                   information_ratio, sharpe_ratio, max_drawdown
    """
    n = len(daily_excess)
    if n == 0:
        return {}

    # 年化收益
    annual_return = (1 + daily_net).prod() ** (periods_per_year / n) - 1
    annual_excess = daily_excess.mean() * periods_per_year

    # 波动率
    volatility = daily_net.std() * np.sqrt(periods_per_year)
    tracking_error = daily_excess.std() * np.sqrt(periods_per_year)

    # 信息比率
    ir = annual_excess / tracking_error if tracking_error > 0 else 0

    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

    # 最大回撤
    nav = (1 + daily_net).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()

    return {
        'annual_return': annual_return,
        'annual_excess': annual_excess,
        'volatility': volatility,
        'tracking_error': tracking_error,
        'information_ratio': ir,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


if __name__ == '__main__':
    # 测试：重建主结果的分层等权策略收益
    print("=" * 60)
    print("测试：重建分层等权策略日收益")
    print("=" * 60)

    weights_path = DATA_ROOT / "result/layered_equal_weight/optimal_weights.csv"
    result = reconstruct_strategy_returns(str(weights_path))

    print(f"\n日收益统计:")
    print(f"  交易日数: {len(result['net_ret'])}")
    print(f"  日均净收益: {result['net_ret'].mean():.6f}")
    print(f"  日均超额收益: {result['excess_ret'].mean():.6f}")
    print(f"  日均换手率: {result['turnover'].mean():.4f}")
    print(f"  日均成本: {result['cost'].mean():.6f}")

    stats = compute_annual_stats(result['excess_ret'], result['net_ret'])
    print(f"\n年化统计:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n最终净值: {result['nav'].iloc[-1]:.4f}")
    print(f"基准净值: {result['benchmark_nav'].iloc[-1]:.4f}")
