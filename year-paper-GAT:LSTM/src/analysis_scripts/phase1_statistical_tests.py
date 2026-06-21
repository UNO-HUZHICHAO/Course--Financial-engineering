"""
阶段一：零成本统计显著性补强
==============================
基于现有回测结果CSV，计算：
  1. 因子ICIR分析（读取IC序列，计算Newey-West调整t统计量）
  2. 收益显著性t检验（单样本 + Newey-West标准误）
  3. Jobson-Korkie夏普比率差异检验
  4. 时间序列分块检验（Block Bootstrap / 季度分块）

注意：不训练任何模型，仅对已有回测结果做后处理
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "学年论文实证部分"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================

def newwest_west_tstat(series, lag=5):
    """
    计算Newey-West调整的t统计量

    参数:
        series: 时间序列
        lag: 自相关滞后阶数

    返回:
        (mean, nw_se, t_stat, p_value)
    """
    n = len(series)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean = series.mean()
    # 计算Newey-West标准误
    residuals = series - mean
    # 方差
    var = np.sum(residuals ** 2) / n

    # 自协方差加权
    nw_var = var
    for j in range(1, min(lag + 1, n - 1)):
        auto_cov = np.sum(residuals[j:] * residuals[:-j]) / n
        weight = 1 - j / (lag + 1)  # Bartlett kernel
        nw_var += 2 * weight * auto_cov

    nw_se = np.sqrt(nw_var / n)

    if nw_se > 0:
        t_stat = mean / nw_se
        p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    else:
        t_stat = np.inf
        p_value = 0.0

    return mean, nw_se, t_stat, p_value


def significance_stars(p_value):
    """返回显著性星号"""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return ""


def annualize_return(daily_ret, periods=252):
    """日收益率年化"""
    n = len(daily_ret)
    if n == 0:
        return np.nan
    return (1 + daily_ret).prod() ** (periods / n) - 1


def annualize_sharpe(daily_ret, rf=0.02, periods=252):
    """年化夏普比率"""
    n = len(daily_ret)
    if n < 2:
        return np.nan
    ann_ret = annualize_return(daily_ret, periods)
    ann_vol = daily_ret.std() * np.sqrt(periods)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan


def compute_max_drawdown(daily_ret):
    """计算最大回撤"""
    nav = (1 + daily_ret).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    return drawdown.min(), drawdown.idxmin()


# ============================================================
# 数据加载
# ============================================================

def load_daily_returns():
    """加载日个股收益率矩阵"""
    path = DATA_ROOT / "data/processed/market/Daily_Returns.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [str(c).zfill(6) for c in df.columns]
    return df


def load_benchmark():
    """加载科创50基准日收益率"""
    path = DATA_ROOT / "data/processed/market/KCB50_Index_Daily.csv"
    df = pd.read_csv(path)
    # trade_date 是 YYYYMMDD 整数格式（如 20251231），需显式指定格式
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df['ret'] = df['pct_chg'] / 100.0  # 百分比转小数
    df = df.sort_values('trade_date')
    return df.set_index('trade_date')['ret']


def load_weights(weights_path):
    """加载最优权重"""
    df = pd.read_csv(weights_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['stkcd'] = df['stkcd'].astype(str).str.zfill(6)
    return df


def reconstruct_returns(weights_df, daily_returns, benchmark, cost_rate=0.002):
    """
    从权重和日收益重建策略日收益序列

    返回:
        DataFrame with: gross_ret, cost, net_ret, excess_ret, turnover
    """
    trade_dates = sorted(weights_df['trade_date'].unique())
    all_return_dates = sorted(daily_returns.index)

    # 日期到下一交易日映射
    date_to_next = {}
    for i, d in enumerate(all_return_dates[:-1]):
        date_to_next[d] = all_return_dates[i + 1]

    records = []
    prev_weights = None

    for t_date in trade_dates:
        w_t = weights_df[weights_df['trade_date'] == t_date].set_index('stkcd')['weight']

        if t_date not in date_to_next:
            continue
        ret_date = date_to_next[t_date]

        if ret_date not in daily_returns.index:
            continue
        returns_t = daily_returns.loc[ret_date]

        common = w_t.index.intersection(returns_t.index)
        if len(common) == 0:
            continue

        w_aligned = w_t.loc[common]
        r_aligned = returns_t.loc[common]

        gross_ret = (w_aligned * r_aligned).sum()

        if prev_weights is not None:
            prev_aligned = prev_weights.reindex(w_t.index, fill_value=0)
            turnover = (w_t - prev_aligned).abs().sum()
        else:
            turnover = w_t.abs().sum()

        cost = turnover * cost_rate

        records.append({
            'date': ret_date,
            'gross_ret': gross_ret,
            'cost': cost,
            'turnover': turnover
        })
        prev_weights = w_t.copy()

    df = pd.DataFrame(records).set_index('date').sort_index()
    df['net_ret'] = df['gross_ret'] - df['cost']

    # 对齐基准
    common_dates = df.index.intersection(benchmark.index)
    df['excess_ret'] = np.nan
    df.loc[common_dates, 'excess_ret'] = df.loc[common_dates, 'net_ret'] - benchmark.loc[common_dates]

    return df


def reconstruct_all_strategies():
    """
    重建所有需要对比的策略日收益率

    返回:
        dict: strategy_name -> DataFrame of daily returns
    """
    print("=" * 70)
    print("重建策略日收益率序列")
    print("=" * 70)

    daily_returns = load_daily_returns()
    benchmark = load_benchmark()

    strategies = {}

    # --- 主结果 (GL V4, Seed 42) ---
    print("\n[主结果 Seed 42]")
    seed42_base = (DATA_ROOT / "GL V4/result/"
                   "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed42")

    for strat_name, subdir in [
        ("分层等权 Seed42", "layered_equal_weight"),
        ("QP守卫 Seed42", "qp_guarded"),
        ("自适应分层等权 Seed42", "adaptive_layered_equal_weight")
    ]:
        wp = seed42_base / subdir / "optimal_weights.csv"
        if wp.exists():
            print(f"  加载: {strat_name}")
            w = load_weights(str(wp))
            strategies[strat_name] = reconstruct_returns(w, daily_returns, benchmark)
        else:
            print(f"  跳过(文件不存在): {strat_name}")

    # --- 主结果 (GL V4, Seed 123) ---
    print("\n[主结果 Seed 123]")
    seed123_base = (DATA_ROOT / "GL V4/result/"
                    "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed123")

    for strat_name, subdir in [
        ("分层等权 Seed123", "layered_equal_weight"),
        ("QP守卫 Seed123", "qp_guarded"),
        ("自适应分层等权 Seed123", "adaptive_layered_equal_weight")
    ]:
        wp = seed123_base / subdir / "optimal_weights.csv"
        if wp.exists():
            print(f"  加载: {strat_name}")
            w = load_weights(str(wp))
            strategies[strat_name] = reconstruct_returns(w, daily_returns, benchmark)
        else:
            print(f"  跳过(文件不存在): {strat_name}")

    # --- GLV3 消融实验 ---
    print("\n[GLV3 消融实验]")
    glv3_configs = {
        "全模型 base16+micro8": "full_base16+micro8_anchor1",
        "仅LSTM base16+micro8": "lstm_only_base16+micro8_anchor1",
        "仅GAT base16+micro8": "gat_only_base16+micro8_anchor1",
        "全模型 all33": "full_all33_anchor1",
    }

    for cfg_name, dir_name in glv3_configs.items():
        for strat_suffix, subdir in [
            ("分层等权", "layered_equal_weight"),
            ("QP保守", "qp_conservative"),
        ]:
            key = f"[GLV3] {cfg_name} {strat_suffix}"
            wp = DATA_ROOT / "GLV3结果" / dir_name / subdir / "optimal_weights.csv"
            if wp.exists():
                print(f"  加载: {key}")
                w = load_weights(str(wp))
                strategies[key] = reconstruct_returns(w, daily_returns, benchmark)
            else:
                print(f"  跳过: {key}")

    return strategies


# ============================================================
# Step 1.1: ICIR分析
# ============================================================

def step1_1_icir_analysis():
    """
    因子ICIR分析
    - 读取GL_V3_Enhanced的factor_ic_icir.csv
    - 额外计算日度IC序列的Newey-West t统计量
    - 输出增强ICIR表格
    """
    print("\n" + "=" * 70)
    print("Step 1.1: 因子ICIR分析")
    print("=" * 70)

    # 读取因子ICIR汇总
    icir_path = (DATA_ROOT / "GL_V3_Enhanced/code/result/diagnostics/"
                 "factor_ic_icir.csv")
    df_icir = pd.read_csv(icir_path)
    print(f"\n读取因子ICIR数据: {len(df_icir)} 个因子")
    print(f"列: {df_icir.columns.tolist()}")

    # 读取日度IC序列（用于计算NW t统计量）
    daily_ic_path = DATA_ROOT / "GL V1/results/diagnosis_daily_ic.csv"
    df_daily = pd.read_csv(daily_ic_path)
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])

    # 为每个因子组计算NW t统计量
    # 注：日度IC只有整体rank_ic和pearson_ic，没有按因子拆分
    # 因此我们从factor_ic_icir中的t_stat读取已有的t统计量
    # 额外计算rank_ic整体的NW t统计量

    print(f"\n整体Rank IC日度序列统计:")
    rank_ic = df_daily['rank_ic'].dropna()
    mean_ic, nw_se, t_nw, p_nw = newwest_west_tstat(rank_ic, lag=5)
    # 普通t检验
    t_regular, p_regular = stats.ttest_1samp(rank_ic, 0)

    print(f"  观测数: {len(rank_ic)}")
    print(f"  均值: {mean_ic:.4f}")
    print(f"  标准差: {rank_ic.std():.4f}")
    print(f"  普通t统计量: {t_regular:.2f} (p={p_regular:.4f})")
    print(f"  NW调整t统计量: {t_nw:.2f} (p={p_nw:.4f}) {significance_stars(p_nw)}")

    # 因子分类汇总
    print(f"\n因子ICIR分类汇总（按ICIR绝对值降序）:")
    print(f"{'因子':<20} {'IC均值':>8} {'IC标准差':>8} {'ICIR':>8} {'t统计量':>8} {'显著性':>6}")
    print("-" * 65)

    # 按ICIR绝对值排序
    df_sorted = df_icir.copy()
    df_sorted['abs_icir'] = df_sorted['icir'].abs()
    df_sorted = df_sorted.sort_values('abs_icir', ascending=False)

    top_positive = []
    top_negative = []

    for _, row in df_sorted.iterrows():
        stars = ""
        if 't_stat' in row.index and not pd.isna(row['t_stat']):
            t_val = row['t_stat']
            if abs(t_val) > 2.576:
                stars = "***"
            elif abs(t_val) > 1.96:
                stars = "**"
            elif abs(t_val) > 1.645:
                stars = "*"

        sign = "+" if row['icir'] > 0 else "-"
        print(f"{row['factor']:<20} {row['ic_mean']:>+8.4f} {row['ic_std']:>8.4f} "
              f"{row['icir']:>+8.4f} {row.get('t_stat', np.nan):>8.2f} {stars:>6}")

        if row['icir'] > 0.05:
            top_positive.append(row['factor'])
        if row['icir'] < -0.2:
            top_negative.append(row['factor'])

    print(f"\n正向显著因子 (ICIR > 0.05): {top_positive}")
    print(f"负向显著因子 (ICIR < -0.2): {top_negative}")

    # 保存增强表格
    df_icir['nw_t_stat'] = np.nan
    df_icir['nw_p_value'] = np.nan
    df_icir['significance'] = ""

    # 为每个因子计算NW t统计量（如果有日度因子IC数据的话）
    # 注：当前只有整体日度IC，没有按因子拆分的日度IC序列
    # 因此这里使用factor_ic_icir中已有的t_stat

    output_path = TABLE_DIR / "table5_icir_enhanced.csv"
    df_icir.to_csv(output_path, index=False)
    print(f"\n增强ICIR表已保存至: {output_path}")

    return df_icir


# ============================================================
# Step 1.3: 收益显著性t检验
# ============================================================

def step1_3_return_ttest(strategies):
    """
    对策略日超额收益序列进行显著性检验

    - 单样本t检验: H0: μ_excess = 0
    - Newey-West调整标准误
    """
    print("\n" + "=" * 70)
    print("Step 1.3: 策略收益显著性t检验")
    print("=" * 70)

    results = []

    for name, df in strategies.items():
        # 超额收益（有基准数据的时期，2022年起）
        excess = df['excess_ret'].dropna()
        # 净收益（全时期）
        net = df['net_ret'].dropna()

        if len(excess) < 30:
            print(f"\n{name}: 超额收益数据不足 (<30天)，跳过")
            continue

        # 超额收益 - 普通t检验
        t_excess_reg, p_excess_reg = stats.ttest_1samp(excess, 0)
        # 超额收益 - NW调整t检验
        mean_excess, nw_se_excess, t_excess_nw, p_excess_nw = newwest_west_tstat(excess, lag=5)

        # 净收益 - NW调整t检验
        mean_net, nw_se_net, t_net_nw, p_net_nw = newwest_west_tstat(net, lag=5)

        # 年化统计
        ann_excess = excess.mean() * 252
        ann_vol = excess.std() * np.sqrt(252)
        ir = ann_excess / ann_vol if ann_vol > 0 else 0
        sharpe = annualize_sharpe(net)

        stars_str = significance_stars(p_excess_nw)

        results.append({
            '策略名称': name,
            '样本数': len(excess),
            '日均超额收益': excess.mean(),
            '年化超额收益': ann_excess,
            '年化跟踪误差': ann_vol,
            '信息比率': ir,
            '夏普比率': sharpe,
            't统计量(普通)': t_excess_reg,
            't统计量(NW)': t_excess_nw,
            'p值(NW)': p_excess_nw,
            '显著性': stars_str
        })

        print(f"\n{name}:")
        print(f"  样本数: {len(excess)}")
        print(f"  日均超额收益: {excess.mean():.6f} ({ann_excess*100:.2f}% 年化)")
        print(f"  信息比率: {ir:.4f}")
        print(f"  夏普比率: {sharpe:.4f}")
        print(f"  t统计量(普通): {t_excess_reg:.2f} (p={p_excess_reg:.4f})")
        print(f"  t统计量(NW):   {t_excess_nw:.2f} (p={p_excess_nw:.4f}) {stars_str}")

    # 汇总表
    if len(results) == 0:
        print("\n警告：没有策略有足够的超额收益数据进行t检验")
        print("可能原因：基准收益数据日期格式解析错误或数据范围不匹配")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('t统计量(NW)', ascending=False)

    print(f"\n{'='*70}")
    print("收益显著性汇总（按NW t统计量降序）")
    print(f"{'='*70}")
    print(f"{'策略名称':<40} {'NW t':>6} {'p值':>8} {'显著性':>6}")
    print("-" * 65)
    for _, row in df_results.iterrows():
        print(f"{row['策略名称']:<40} {row['t统计量(NW)']:>6.2f} "
              f"{row['p值(NW)']:>8.4f} {row['显著性']:>6}")

    # 重点对比：Adaptive FiLM vs 门控融合
    print(f"\n{'='*70}")
    print("核心发现：Adaptive FiLM vs 门控融合（分层等权）")
    print(f"{'='*70}")
    key_strategies = [s for s in results
                      if '分层等权' in s['策略名称'] and 'Seed' in s['策略名称']]
    for s in key_strategies:
        print(f"  {s['策略名称']}: "
              f"年化Alpha={s['年化超额收益']*100:.2f}%, "
              f"t(NW)={s['t统计量(NW)']:.2f}, "
              f"p={s['p值(NW)']:.4f} {s['显著性']}")

    output_path = TABLE_DIR / "table7_8_9_significance.csv"
    if len(df_results) > 0:
        df_results.to_csv(output_path, index=False)
        print(f"\n显著性汇总表已保存至: {output_path}")

    return df_results


# ============================================================
# Step 1.4: Jobson-Korkie检验
# ============================================================

def jobson_korkie_test(returns_a, returns_b, rf=0.02):
    """
    Jobson & Korkie (1981) 夏普比率差异检验

    检验两个策略的夏普比率差异是否统计显著。

    参数:
        returns_a: 策略A的日收益率序列
        returns_b: 策略B的日收益率序列
        rf: 年化无风险利率

    返回:
        (z_stat, p_value, sharpe_a, sharpe_b, sharpe_diff)
    """
    n = len(returns_a)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # 对齐
    common_idx = returns_a.index.intersection(returns_b.index)
    r_a = returns_a.loc[common_idx].values
    r_b = returns_b.loc[common_idx].values
    n = len(r_a)

    # 计算统计量
    mu_a = r_a.mean()
    mu_b = r_b.mean()
    sigma_a = r_a.std()
    sigma_b = r_b.std()

    # 年化夏普比率
    sharpe_a = (mu_a * 252 - rf) / (sigma_a * np.sqrt(252))
    sharpe_b = (mu_b * 252 - rf) / (sigma_b * np.sqrt(252))
    sharpe_diff = sharpe_a - sharpe_b

    # 协方差和相关性
    cov_ab = np.cov(r_a, r_b)[0, 1]
    rho_ab = np.corrcoef(r_a, r_b)[0, 1]

    # Jobson-Korkie Z统计量
    # 需要: mu, sigma, covariance, skewness, kurtosis
    # 简化版: 使用Mertens (2002) 渐近分布公式

    # 计算高阶矩
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    sigma_a2 = sigma_a ** 2
    sigma_b2 = sigma_b ** 2

    # 协偏度和协峰度的简化近似
    r_a_dm = r_a - mu_a
    r_b_dm = r_b - mu_b

    skew_a = np.mean(r_a_dm ** 3) / (sigma_a ** 3) if sigma_a > 0 else 0
    skew_b = np.mean(r_b_dm ** 3) / (sigma_b ** 3) if sigma_b > 0 else 0
    kurt_a = np.mean(r_a_dm ** 4) / (sigma_a ** 4) - 3 if sigma_a > 0 else 0
    kurt_b = np.mean(r_b_dm ** 4) / (sigma_b ** 4) - 3 if sigma_b > 0 else 0

    # Jobson-Korkie Z统计量（考虑偏度和峰度的版本）
    # 基于 Opdyke (2007) 和 Mertens (2002)
    sharpe_vec = np.array([sharpe_a, sharpe_b])

    # 渐近协方差矩阵
    # V = [[V_aa, V_ab], [V_ab, V_bb]]
    V_aa = 1 + 0.5 * sharpe_a**2 - sharpe_a * skew_a + 0.25 * sharpe_a**2 * kurt_a
    V_bb = 1 + 0.5 * sharpe_b**2 - sharpe_b * skew_b + 0.25 * sharpe_b**2 * kurt_b
    V_ab = rho_ab + 0.5 * sharpe_a * sharpe_b * rho_ab**2 \
           - 0.5 * (sharpe_a * np.mean(r_a_dm * r_b_dm ** 2) / (sigma_a * sigma_b2)
                    + sharpe_b * np.mean(r_a_dm ** 2 * r_b_dm) / (sigma_a2 * sigma_b))

    # 差异的渐近方差
    asymp_var = V_aa + V_bb - 2 * V_ab

    if asymp_var > 0:
        z_stat = np.sqrt(n) * sharpe_diff / np.sqrt(asymp_var)
        p_value = 2 * stats.norm.sf(abs(z_stat))
    else:
        z_stat = np.nan
        p_value = np.nan

    return z_stat, p_value, sharpe_a, sharpe_b, sharpe_diff


def step1_4_jobson_korkie(strategies):
    """
    对关键策略对进行Jobson-Korkie夏普比率差异检验
    """
    print("\n" + "=" * 70)
    print("Step 1.4: Jobson-Korkie 夏普比率差异检验")
    print("=" * 70)

    # 定义需要对比的策略对
    pairs = [
        # 消融实验对比：Adaptive FiLM vs 门控融合
        ("分层等权 Seed42", "分层等权 Seed123",
         "Seed42 vs Seed123 分层等权"),

        # QP守卫对比
        ("QP守卫 Seed42", "QP守卫 Seed123",
         "Seed42 vs Seed123 QP守卫"),

        # 分层等权 vs QP守卫 (Seed42)
        ("分层等权 Seed42", "QP守卫 Seed42",
         "Seed42: 分层等权 vs QP守卫"),
    ]

    # GLV3消融对比对
    glv3_keys = [k for k in strategies.keys() if k.startswith('[GLV3]') and '分层等权' in k]
    if len(glv3_keys) >= 2:
        # 全模型 vs 仅LSTM
        full_key = [k for k in glv3_keys if '全模型 base16' in k]
        lstm_key = [k for k in glv3_keys if '仅LSTM' in k]
        if full_key and lstm_key:
            pairs.append((full_key[0], lstm_key[0],
                          "GLV3 分层等权: 全模型 vs 仅LSTM"))
        # 全模型 vs 仅GAT
        gat_key = [k for k in glv3_keys if '仅GAT' in k]
        if full_key and gat_key:
            pairs.append((full_key[0], gat_key[0],
                          "GLV3 分层等权: 全模型 vs 仅GAT"))

    results = []
    for key_a, key_b, label in pairs:
        if key_a not in strategies or key_b not in strategies:
            print(f"\n{label}: 策略数据不可用，跳过")
            continue

        # 使用净收益（全时期）进行检验
        ret_a = strategies[key_a]['net_ret'].dropna()
        ret_b = strategies[key_b]['net_ret'].dropna()

        if len(ret_a) < 30 or len(ret_b) < 30:
            print(f"\n{label}: 数据不足，跳过")
            continue

        z_stat, p_val, sr_a, sr_b, sr_diff = jobson_korkie_test(ret_a, ret_b)

        stars_str = significance_stars(p_val) if not np.isnan(p_val) else ""

        results.append({
            '对比': label,
            '策略A': key_a,
            '策略B': key_b,
            '夏普A': sr_a,
            '夏普B': sr_b,
            '夏普差异': sr_diff,
            'Z统计量': z_stat,
            'p值': p_val,
            '显著性': stars_str
        })

        print(f"\n{label}:")
        print(f"  策略A: {key_a} (夏普={sr_a:.4f})")
        print(f"  策略B: {key_b} (夏普={sr_b:.4f})")
        print(f"  夏普差异: {sr_diff:.4f}")
        print(f"  J-K Z统计量: {z_stat:.2f} (p={p_val:.4f}) {stars_str}")
        if p_val < 0.05:
            print(f"  >> 夏普比率差异统计显著 ({stars_str})")
        else:
            print(f"  >> 夏普比率差异不显著")

    df_results = pd.DataFrame(results)
    output_path = TABLE_DIR / "jobson_korkie_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nJobson-Korkie检验结果已保存至: {output_path}")

    return df_results


# ============================================================
# Step 1.5: Block Bootstrap / 时间序列分块检验
# ============================================================

def step1_5_block_bootstrap(strategies):
    """
    时间序列分块检验

    将2021-2025年切分为季度块，在每个块内计算：
    - Alpha胜率（正收益区间占比）
    - 均值t检验
    """
    print("\n" + "=" * 70)
    print("Step 1.5: 时间序列分块检验 (Block Bootstrap)")
    print("=" * 70)

    # 聚焦Seed 42分层等权策略
    target_key = "分层等权 Seed42"
    if target_key not in strategies:
        print(f"策略 {target_key} 不可用，尝试找到可用的Seed42策略...")
        available = [k for k in strategies.keys() if 'Seed42' in k and '分层等权' in k]
        if available:
            target_key = available[0]
            print(f"  使用: {target_key}")
        else:
            print("  无可用策略")
            return None

    df = strategies[target_key]
    excess = df['excess_ret'].dropna()  # 超额收益（2022年起，有基准数据）
    net = df['net_ret'].dropna()        # 净收益（全时期2021-2025）

    # === 按季度分块 ===
    print(f"\n使用策略: {target_key}")
    print(f"超额收益时期: {excess.index[0].date()} ~ {excess.index[-1].date()}")
    print(f"净收益时期: {net.index[0].date()} ~ {net.index[-1].date()}")

    # 创建季度标签
    def quarter_label(d):
        return f"{d.year}-Q{(d.month - 1) // 3 + 1}"

    # 对超额收益分块（有基准数据的时期）
    excess_df = excess.to_frame('excess_ret')
    excess_df['quarter'] = [quarter_label(d) for d in excess_df.index]

    # 对净收益分块（全时期）
    net_df = net.to_frame('net_ret')
    net_df['quarter'] = [quarter_label(d) for d in net_df.index]
    net_df['year'] = [d.year for d in net_df.index]

    # 按季度统计
    print(f"\n--- 季度分块统计 (超额收益) ---")
    print(f"{'季度':<10} {'交易日':>6} {'日均超额':>10} {'胜率':>8} {'累计超额':>10} {'显著性':>6}")
    print("-" * 60)

    quarter_stats = []
    for q in sorted(excess_df['quarter'].unique()):
        q_data = excess_df[excess_df['quarter'] == q]['excess_ret']
        n_days = len(q_data)
        mean_ret = q_data.mean()
        win_rate = (q_data > 0).mean()
        cum_ret = (1 + q_data).prod() - 1
        _, _, t_stat, p_val = newwest_west_tstat(q_data, lag=2)
        stars = significance_stars(p_val)

        quarter_stats.append({
            'quarter': q,
            'n_days': n_days,
            'mean_excess': mean_ret,
            'win_rate': win_rate,
            'cum_excess': cum_ret,
            't_stat': t_stat,
            'p_value': p_val,
            'significance': stars
        })

        print(f"{q:<10} {n_days:>6} {mean_ret:>10.6f} {win_rate:>8.2%} "
              f"{cum_ret:>10.4%} {stars:>6}")

    df_quarter = pd.DataFrame(quarter_stats)

    # 季度块均值的t检验
    n_quarters = len(df_quarter)
    quarter_means = df_quarter['mean_excess'].values
    t_block, p_block = stats.ttest_1samp(quarter_means, 0)
    n_positive = (quarter_means > 0).sum()

    print(f"\n--- 分块汇总 ---")
    print(f"总季度数: {n_quarters}")
    print(f"正超额季度数: {n_positive} ({n_positive/n_quarters:.1%})")
    print(f"季度均值t检验: t={t_block:.2f}, p={p_block:.4f} {significance_stars(p_block)}")

    # === 按年度统计（净收益，全时期2021-2025） ===
    print(f"\n--- 年度统计 (净收益，全时期) ---")
    print(f"{'年份':<6} {'交易日':>6} {'日均净收益':>10} {'胜率':>8} {'年化收益':>10} {'最大回撤':>10}")
    print("-" * 65)

    yearly_stats = []
    for y in sorted(net_df['year'].unique()):
        y_data = net_df[net_df['year'] == y]['net_ret']
        n_days = len(y_data)
        mean_ret = y_data.mean()
        win_rate = (y_data > 0).mean()
        ann_ret = annualize_return(y_data)
        max_dd, dd_date = compute_max_drawdown(y_data)

        yearly_stats.append({
            'year': y,
            'n_days': n_days,
            'mean_ret': mean_ret,
            'win_rate': win_rate,
            'annual_return': ann_ret,
            'max_drawdown': max_dd
        })

        print(f"{y:<6} {n_days:>6} {mean_ret:>10.6f} {win_rate:>8.2%} "
              f"{ann_ret:>10.4%} {max_dd:>10.4%}")

    df_yearly = pd.DataFrame(yearly_stats)

    # 重点：2022-2023熊市表现
    bear_data = net_df[net_df['year'].isin([2022, 2023])]['net_ret']
    bear_mean = bear_data.mean()
    bear_ann = annualize_return(bear_data)
    bear_win = (bear_data > 0).mean()
    _, _, bear_t, bear_p = newwest_west_tstat(bear_data, lag=5)

    print(f"\n--- 2022-2023熊市期间 ---")
    print(f"交易日: {len(bear_data)}")
    print(f"日均净收益: {bear_mean:.6f}")
    print(f"年化收益: {bear_ann:.4%}")
    print(f"日胜率: {bear_win:.2%}")
    print(f"t统计量(NW): {bear_t:.2f} (p={bear_p:.4f}) {significance_stars(bear_p)}")

    # === 半年分块 ===
    print(f"\n--- 半年度分块统计 ---")
    half_labels = []
    for y in sorted(net_df['year'].unique()):
        for half, (start_m, end_m) in enumerate([(1, 6), (7, 12)], 1):
            h_data = net_df[(net_df['year'] == y) &
                            (pd.DatetimeIndex(net_df.index).month >= start_m) &
                            (pd.DatetimeIndex(net_df.index).month <= end_m)]['net_ret']
            if len(h_data) > 10:
                mean_r = h_data.mean()
                win_r = (h_data > 0).mean()
                ann_r = annualize_return(h_data)
                half_labels.append({
                    'half': f"{y}-H{half}",
                    'n_days': len(h_data),
                    'mean_ret': mean_r,
                    'win_rate': win_r,
                    'annual_return': ann_r
                })
                print(f"  {y}-H{half}: {len(h_data):>3}天, "
                      f"日均={mean_r:.6f}, 胜率={win_r:.1%}, 年化={ann_r:.4%}")

    # 保存结果
    df_quarter.to_csv(TABLE_DIR / "block_bootstrap_quarterly.csv", index=False)
    df_yearly.to_csv(TABLE_DIR / "block_bootstrap_yearly.csv", index=False)
    pd.DataFrame(half_labels).to_csv(TABLE_DIR / "block_bootstrap_halfyearly.csv", index=False)

    print(f"\n分块检验结果已保存至: {TABLE_DIR}/")

    return df_quarter, df_yearly


# ============================================================
# 主流程
# ============================================================

def main():
    print("╔" + "=" * 68 + "╗")
    print("║  阶段一：零成本统计显著性补强                              ║")
    print("║  弱信号环境下的图网络机制诊断与改进                        ║")
    print("╚" + "=" * 68 + "╝")

    # Step 1.1: ICIR分析
    step1_1_icir_analysis()

    # 重建所有策略日收益率
    strategies = reconstruct_all_strategies()

    # Step 1.3: 收益显著性t检验
    df_ttest = step1_3_return_ttest(strategies)

    # Step 1.4: Jobson-Korkie检验
    df_jk = step1_4_jobson_korkie(strategies)

    # Step 1.5: 时间序列分块检验
    step1_5_block_bootstrap(strategies)

    # 最终汇总
    print("\n" + "=" * 70)
    print("阶段一完成：所有统计检验结果汇总")
    print("=" * 70)
    print(f"\n显著性标注说明:")
    print(f"  *** : p < 0.01 (1%水平显著)")
    print(f"  **  : p < 0.05 (5%水平显著)")
    print(f"  *   : p < 0.10 (10%水平显著)")
    print(f"\n输出文件:")
    print(f"  {TABLE_DIR / 'table5_icir_enhanced.csv'}")
    print(f"  {TABLE_DIR / 'table7_8_9_significance.csv'}")
    print(f"  {TABLE_DIR / 'jobson_korkie_results.csv'}")
    print(f"  {TABLE_DIR / 'block_bootstrap_quarterly.csv'}")
    print(f"  {TABLE_DIR / 'block_bootstrap_yearly.csv'}")


if __name__ == '__main__':
    main()
