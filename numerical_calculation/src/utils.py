"""utils.py
=========================================================
公共工具函数：Black-Scholes IV 反推、Carr–Madan 离散化、
delta 网格 PCHIP 插值、put-call parity 校验、期限匹配无风险利率、
Kadan-Tang 期望收益下界、Newey–West HAC 标准误、月度汇总。

复刻 Neuhierl, Tang, Varneskov & Zhou (2024) 中的关键计算单元。
=========================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, interp1d


# =====================================================
# 1. Black–Scholes 公式与隐含波动率反推
# =====================================================

def bs_price(S, K, T, r, sigma, q=0.0, option_type='C'):
    """Black-Scholes 期权理论价格（含连续股息率 q）。"""
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        if option_type.upper() == 'C':
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.upper() == 'C':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, q=0.0, option_type='C'):
    """Black-Scholes delta（含连续股息率）。"""
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.upper() == 'C':
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, q=0.0, option_type='C',
                lo=1e-4, hi=5.0):
    """对欧式期权按 Black-Scholes 反推隐含波动率（brentq）。失败返回 NaN。"""
    if price is None or np.isnan(price) or price <= 0 or T <= 0:
        return np.nan
    intrinsic = max((S * np.exp(-q * T) - K * np.exp(-r * T)) if option_type.upper() == 'C'
                    else (K * np.exp(-r * T) - S * np.exp(-q * T)), 0.0)
    if price < intrinsic - 1e-8:
        return np.nan
    f = lambda sigma: bs_price(S, K, T, r, sigma, q, option_type) - price
    try:
        if f(lo) * f(hi) > 0:
            hi2 = 10.0
            if f(lo) * f(hi2) > 0:
                return np.nan
            return brentq(f, lo, hi2, xtol=1e-8, maxiter=200)
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


# =====================================================
# 1b. 向量化 BS 价格与 IV 反推（批量，数十倍加速）
# =====================================================

def bs_price_vec(S, K, T, r, sigma, q=0.0, is_call=True):
    """向量化 Black-Scholes 价格。输入均为 numpy 数组。"""
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float); r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float); q = np.asarray(q, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    price = np.full_like(S, np.nan, dtype=float)
    valid = (T > 0) & (sigma > 0) & np.isfinite(S) & np.isfinite(K) & np.isfinite(sigma)
    if not valid.any():
        return price
    Sv, Kv, Tv, rv, sv, qv = S[valid], K[valid], T[valid], r[valid], sigma[valid], q[valid]
    cv = is_call[valid]
    sqT = np.sqrt(Tv)
    d1 = (np.log(Sv / Kv) + (rv - qv + 0.5 * sv ** 2) * Tv) / (sv * sqT)
    d2 = d1 - sv * sqT
    n_c_d1 = Sv * np.exp(-qv * Tv) * norm.cdf(d1)
    n_c_d2 = Kv * np.exp(-rv * Tv) * norm.cdf(d2)
    call_p = n_c_d1 - n_c_d2
    put_p = Kv * np.exp(-rv * Tv) * norm.cdf(-d2) - Sv * np.exp(-qv * Tv) * norm.cdf(-d1)
    p = np.where(cv, call_p, put_p)
    price[valid] = p
    return price


def implied_vol_vec(price, S, K, T, r, q=0.0, is_call=True,
                    lo=1e-4, hi=5.0, n_iter=60, tol=1e-9):
    """向量化 IV 反推（批量二分粗解 + brentq 精修）。

    先对全部行同时做 60 次二分到 ~1e-9（绝大多数行已收敛），
    再用残差阈值挑出未收敛/可疑的少数行，逐行 brentq 精修保证精度。
    这样既快（向量化处理 95%+ 行）又准（brentq 兜底）。
    """
    price = np.asarray(price, dtype=float)
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float); r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float); is_call = np.asarray(is_call, dtype=bool)
    n = len(price)
    iv = np.full(n, np.nan)
    valid = (price > 0) & (T > 0) & (S > 0) & (K > 0) & np.isfinite(price)
    if not valid.any():
        return iv
    idx = np.where(valid)[0]
    pv = price[idx]; Sv = S[idx]; Kv = K[idx]; Tv = T[idx]
    rv = r[idx]; qv = q[idx]; cv = is_call[idx]
    disc_K = Kv * np.exp(-rv * Tv)
    disc_S = Sv * np.exp(-qv * Tv)
    intrinsic = np.where(cv, np.maximum(disc_S - disc_K, 0.0),
                         np.maximum(disc_K - disc_S, 0.0))
    ok = pv >= intrinsic - 1e-8
    if not ok.any():
        return iv
    sub = idx[ok]
    pv = pv[ok]; Sv = Sv[ok]; Kv = Kv[ok]; Tv = Tv[ok]
    rv = rv[ok]; qv = qv[ok]; cv = cv[ok]

    lo_a = np.full(len(sub), lo)
    hi_a = np.full(len(sub), hi)
    mid = 0.5 * (lo_a + hi_a)
    for _ in range(n_iter):
        p_mid = bs_price_vec(Sv, Kv, Tv, rv, mid, qv, cv)
        diff = p_mid - pv
        hi_a = np.where(diff > 0, mid, hi_a)
        lo_a = np.where(diff <= 0, mid, lo_a)
        mid = 0.5 * (lo_a + hi_a)
    iv[sub] = mid

    # 精修：残差 |p(mid)-pv|/pv 较大 或 区间仍宽 的行，逐行 brentq
    p_mid = bs_price_vec(Sv, Kv, Tv, rv, mid, qv, cv)
    resid = np.abs(p_mid - pv) / np.maximum(pv, 1e-8)
    wide = (hi_a - lo_a) > 1e-6
    bad = resid > 1e-4
    refine = np.where(bad | wide)[0]
    for j in refine:
        otype = 'C' if cv[j] else 'P'
        iv[sub[j]] = implied_vol(pv[j], Sv[j], Kv[j], Tv[j], rv[j],
                                 q=qv[j], option_type=otype)
    return iv


def bs_delta_vec(S, K, T, r, sigma, q=0.0, is_call=True):
    """向量化 BS delta。"""
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float); r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float); q = np.asarray(q, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    delta = np.full_like(S, np.nan, dtype=float)
    valid = (T > 0) & (sigma > 0) & np.isfinite(sigma)
    if not valid.any():
        return delta
    Sv, Kv, Tv, rv, sv, qv = S[valid], K[valid], T[valid], r[valid], sigma[valid], q[valid]
    cv = is_call[valid]
    d1 = (np.log(Sv / Kv) + (rv - qv + 0.5 * sv ** 2) * Tv) / (sv * np.sqrt(Tv))
    d = np.where(cv, np.exp(-qv * Tv) * norm.cdf(d1),
                 -np.exp(-qv * Tv) * norm.cdf(-d1))
    delta[valid] = d
    return delta

def put_call_parity_violation(C, P, S, K, T, r, q=0.0):
    """返回 put-call parity 偏差比例：|C - P - (S e^{-qT} - K e^{-rT})| / S。

    中国 ETF/商品期权用结算价 settle 反推 IV，结算价由交易所按隐含波动率曲面
    定标、再贴现，理论上满足 parity；偏差过大通常来自数据噪声或涨跌停扭曲。
    """
    theo = S * np.exp(-q * T) - K * np.exp(-r * T)
    diff = C - P - theo
    if S <= 0:
        return np.nan
    return abs(diff) / S


# =====================================================
# 3. delta 网格 PCHIP 单调插值（构造标准化 IV 曲面）
# =====================================================

def interp_iv_by_delta(deltas, ivs, target_deltas):
    """对一组 (|delta|, IV) 用 PCHIP 单调插值，得到固定 delta 处的 IV。

    PCHIP 保留单调性、避免普通三次样条在稀疏行权价处产生的 IV 振荡，
    这对 delta 网格稀疏的中国期权尤其重要。外推返回 NaN。
    """
    deltas = np.asarray(deltas, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    target_deltas = list(target_deltas)
    mask = ~(np.isnan(deltas) | np.isnan(ivs))
    if mask.sum() < 2:
        return {d: np.nan for d in target_deltas}
    d = deltas[mask]
    v = ivs[mask]
    order = np.argsort(d)
    d, v = d[order], v[order]
    # 同 delta 取均值并去重
    df = pd.DataFrame({'d': d, 'v': v}).groupby('d', as_index=False).mean()
    d, v = df['d'].values, df['v'].values
    if len(d) < 2:
        return {td: np.nan for td in target_deltas}
    try:
        pch = PchipInterpolator(d, v, extrapolate=False)
        return {td: float(pch(td)) for td in target_deltas}
    except Exception:
        f = interp1d(d, v, kind='linear', bounds_error=False, fill_value=np.nan)
        return {td: float(f(td)) for td in target_deltas}


def interp_iv_by_maturity(t1, iv1, t2, iv2, target_t=30 / 365):
    """在两个到期之间按 (T, IV) 线性插值；单边缺失时取另一边。"""
    if np.isnan(iv1) and np.isnan(iv2):
        return np.nan
    if np.isnan(iv1):
        return iv2
    if np.isnan(iv2):
        return iv1
    if t1 == t2:
        return 0.5 * (iv1 + iv2)
    w = (target_t - t1) / (t2 - t1)
    return iv1 + w * (iv2 - iv1)


# =====================================================
# 4. Carr–Madan 风险中性方差（model-free）
# =====================================================

def carr_madan_variance(strikes, otm_prices, is_call, S, r, T, q=0.0):
    """Carr-Madan (1998) / Britten-Jones & Neuberger (2000) 离散化风险中性方差。

        VARQ = 2*exp(rT)/T * ∫ OTM(K)/K² dK      （年化方差）
        VAR_PLUS  = call 端 (K>F) 半方差
        VAR_MINUS = put  端 (K<F) 半方差
    """
    F = S * np.exp((r - q) * T)
    mask = ~np.isnan(otm_prices) & (otm_prices > 0)
    K = np.asarray(strikes)[mask]
    P = np.asarray(otm_prices)[mask]
    C = np.asarray(is_call)[mask]
    if len(K) < 2:
        return {'VARQ': np.nan, 'VAR_PLUS': np.nan, 'VAR_MINUS': np.nan}
    order = np.argsort(K)
    K, P, C = K[order], P[order], C[order]
    dK = np.empty_like(K)
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    if len(K) > 2:
        dK[1:-1] = (K[2:] - K[:-2]) / 2
    integrand = P / (K ** 2)
    integral_total = np.sum(integrand * dK)
    upper = K > F
    lower = K < F
    integral_plus = np.sum(integrand[upper] * dK[upper]) if upper.any() else 0.0
    integral_minus = np.sum(integrand[lower] * dK[lower]) if lower.any() else 0.0
    coef = 2 * np.exp(r * T) / T
    return {
        'VARQ': coef * integral_total,
        'VAR_PLUS': coef * integral_plus,
        'VAR_MINUS': coef * integral_minus,
    }


def kadan_tang_lb(VARQ, r, T):
    """Kadan-Tang (2020) / Martin (2017) 期望收益下界（前向、model-free）。

    下界 = exp(rT) * T * VARQ，量纲为持有期 T 上的"超额简单收益下界"。
    这是对称（BMS/方差互换）分量；Kadan-Tang 的非对称高阶修正依赖三阶矩
    （立方合约），在行权价稀疏的中国期权曲面上不可靠估计，故取领先阶并在
    报告中以 AVAR（方差不对称）做定性补充。与 VARQ（年化方差）在量纲上不同。
    """
    if np.isnan(VARQ) or T <= 0:
        return np.nan
    return np.exp(r * T) * T * VARQ


# =====================================================
# 5. Newey–West HAC 标准误
# =====================================================

def newey_west_se(x, y, lags=4):
    """单变量回归 y = α + βx 的 Newey-West HAC 版本。"""
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return {k: np.nan for k in ['alpha', 'beta', 'alpha_t', 'beta_t',
                                    'alpha_se', 'beta_se', 'alpha_p', 'beta_p',
                                    'r2', 'n']}
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return {
        'alpha': model.params[0], 'beta': model.params[1],
        'alpha_se': model.bse[0], 'beta_se': model.bse[1],
        'alpha_t': model.tvalues[0], 'beta_t': model.tvalues[1],
        'alpha_p': model.pvalues[0], 'beta_p': model.pvalues[1],
        'r2': model.rsquared, 'n': int(model.nobs),
    }


# =====================================================
# 6. 期限匹配无风险利率（用 SHIBOR 曲线插值到目标 T）
# =====================================================

# SHIBOR 标准期限（天）与列名
SHIBOR_TERMS = {'ON': 1, '1W': 7, '2W': 14, '1M': 30, '3M': 90,
                '6M': 180, '9M': 270, '1Y': 365}


def term_matched_rf(shibor_row: pd.Series, target_days: float) -> float:
    """给定某日 SHIBOR 横截面行，按目标持有天数线性插值得到年化无风险利率。

    shibor_row: 含 '1W','1M','3M',... 百分数列的 Series。
    返回年化利率（小数）。
    """
    pts = []
    for col, days in SHIBOR_TERMS.items():
        if col in shibor_row.index and not pd.isna(shibor_row[col]):
            pts.append((days, float(shibor_row[col])))
    if not pts:
        return np.nan
    pts.sort()
    days = np.array([p[0] for p in pts], dtype=float)
    rates = np.array([p[1] for p in pts], dtype=float) / 100.0
    td = float(target_days)
    if td <= days[0]:
        return float(rates[0])
    if td >= days[-1]:
        return float(rates[-1])
    return float(np.interp(td, days, rates))


# =====================================================
# 7. 日期与月度汇总工具
# =====================================================

def to_datetime(s):
    """tushare 'YYYYMMDD' 字符串/数字列 → pandas Timestamp。"""
    return pd.to_datetime(s.astype(str), format='%Y%m%d')


def years_between(d1, d2) -> float:
    return (pd.Timestamp(d2) - pd.Timestamp(d1)).days / 365.0


def monthly_last(df: pd.DataFrame, date_col: str = 'trade_date',
                 id_cols=('underlying',)) -> pd.DataFrame:
    """取每个 (id_cols, 月份) 的最后一个交易日记录。

    修正点：原实现只按 ym 全局分组，导致每月只保留一条记录、
    其余标的因子全部丢失。这里按 id_cols+ym 分组，保留全部截面。
    若 df 无 underlying 列则退化为按 ym 取月末。
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['ym'] = df[date_col].dt.to_period('M')
    keys = [c for c in id_cols if c in df.columns]
    if not keys:
        last_idx = df.groupby('ym')[date_col].idxmax()
    else:
        last_idx = df.groupby(keys + ['ym'])[date_col].idxmax()
    out = df.loc[last_idx].drop(columns=['ym'])
    return out.reset_index(drop=True)
