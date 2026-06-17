"""02_compute_iv.py
=========================================================
对所有标的（ETF + 商品）的期权日行情，按 BS 反推隐含波动率。

关键修正（对应评语/审计）：
- 无风险利率 r：用 SHIBOR 期限结构按合约持有天数 T_days 线性插值（不再硬填 0.025）。
- 股息率 q：
    * ETF：用 tushare fund_div 的实际分红推算年化股息率（不再硬编码 0）。
    * 商品期权（标的为期货）：令 q = r，使 BS 退化为 Black-76（期权定价的规范做法）。
- put-call parity 校验：按 (标的, 日, 到期, 行权价) 配对 call/put，剔除 parity 偏差
  超过阈值（默认 3% 标的价）的合约对，减少结算价噪声与涨跌停扭曲。
- 仍用 settle（结算价）作为期权价格——中国主流可得且由交易所曲面定标。

输入：data/raw/opt_daily_*.parquet, opt_basic_*.parquet,
       fund_daily_*.parquet, fut_daily_*.parquet, shibor.parquet,
       fund_div_*.parquet, contracts_*.parquet
输出：data/processed/iv_panel.parquet
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import UNDERLYINGS
from utils import (implied_vol, bs_delta, to_datetime, term_matched_rf,
                   put_call_parity_violation)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

PARITY_TOL = 0.03  # put-call parity 偏差占标的价格的阈值


# -----------------------------------------------------
# 加载各标的的标的价格
# -----------------------------------------------------
def load_underlying_prices() -> pd.DataFrame:
    """合并所有标的的日行情 → [underlying, trade_date, S]。"""
    rows = []
    for u in UNDERLYINGS:
        code = u["code"]
        if u["asset"] == "etf":
            f = os.path.join(RAW_DIR, f"fund_daily_{code.replace('.', '_')}.parquet")
        else:
            f = os.path.join(RAW_DIR, f"fut_daily_{u['fut_symbol'].replace('.', '_')}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df = df[["trade_date", "close"]].copy()
        df["trade_date"] = to_datetime(df["trade_date"])
        df["underlying"] = code
        df = df.rename(columns={"close": "S"})
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["underlying", "trade_date", "S"])
    return pd.concat(rows, ignore_index=True)


def load_shibor_term() -> pd.DataFrame:
    """SHIBOR 全期限，行=日期。"""
    f = os.path.join(RAW_DIR, "shibor.parquet")
    df = pd.read_parquet(f)
    df["date"] = to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_etf_div() -> pd.DataFrame:
    """ETF 分红记录 → [underlying, ex_date, div_cash]。"""
    rows = []
    for u in UNDERLYINGS:
        if u["asset"] != "etf":
            continue
        f = os.path.join(RAW_DIR, f"fund_div_{u['code'].replace('.', '_')}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        # 字段兼容：除权日取 ex_date 或 record_date
        for col in ["ex_date", "record_date"]:
            if col in d.columns:
                d["ex_date"] = pd.to_datetime(d[col], errors="coerce")
                break
        if "div_cash" not in d.columns:
            continue
        d = d[["ex_date", "div_cash"]].dropna()
        d["underlying"] = u["code"]
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["underlying", "ex_date", "div_cash"])
    return pd.concat(rows, ignore_index=True)


def build_dividend_yield(prices: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """对每个 ETF，按除权日计算过去 365 天累计分红 / 当日收盘 = 年化股息率 q。"""
    if divs.empty:
        prices = prices.copy()
        prices["q"] = 0.0
        return prices
    out = []
    for code, g in prices.groupby("underlying"):
        g = g.sort_values("trade_date").copy()
        dv = divs[divs["underlying"] == code].sort_values("ex_date")
        # 滚动累计：每个交易日过去 365 天的分红之和
        cum = []
        dv_dates = dv["ex_date"].values
        dv_cash = dv["div_cash"].values
        for td in g["trade_date"].values:
            lo = td - np.timedelta64(365, "D")
            mask = (dv_dates > lo) & (dv_dates <= td)
            cum.append(float(dv_cash[mask].sum()))
        g["div_ttm"] = cum
        g["q"] = (g["div_ttm"] / g["S"]).clip(lower=0.0).fillna(0.0)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    # 商品标的没有分红记录 → q=0（后续在主流程里商品改 q=r）
    if "q" not in res.columns:
        res["q"] = 0.0
    return res


def load_contracts() -> pd.DataFrame:
    """合并所有标的的期权合约清单，补 underlying 列。"""
    rows = []
    for u in UNDERLYINGS:
        f = os.path.join(RAW_DIR, f"contracts_{u['code'].replace('.', '_')}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df["underlying"] = u["code"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    c = pd.concat(rows, ignore_index=True)
    # maturity_date：ETF 来自 opt_basic（YYYYMMDD 字符串），商品来自 ts_code 解析（已 Timestamp）
    # 混合 dtype → 统一转 datetime，自动推断格式
    c["maturity_date"] = pd.to_datetime(c["maturity_date"], errors="coerce")
    return c


def load_opt_daily() -> pd.DataFrame:
    from config import OPT_EXCHANGES
    rows = []
    for ex in OPT_EXCHANGES:
        f = os.path.join(RAW_DIR, f"opt_daily_{ex}.parquet")
        if not os.path.exists(f):
            continue
        rows.append(pd.read_parquet(f))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["trade_date"] = to_datetime(df["trade_date"])
    return df


# -----------------------------------------------------
# 主流程
# -----------------------------------------------------
def main():
    print("[02] 计算隐含波动率 ...")
    prices = load_underlying_prices()
    shibor = load_shibor_term()
    divs = load_etf_div()
    contracts = load_contracts()
    opt_daily = load_opt_daily()
    print(f"  标的日行情: {len(prices)}  合约清单: {len(contracts)}  期权日行情: {len(opt_daily)}")
    if contracts.empty or opt_daily.empty:
        print("  ⚠️  缺合约清单或期权日行情，退出。")
        return

    prices = build_dividend_yield(prices, divs)

    # 合并：opt_daily ⨝ contracts ⨝ underlying ⨝ shibor
    keys = ["ts_code", "exercise_price", "call_put", "maturity_date", "underlying"]
    df = opt_daily.merge(contracts[keys], on="ts_code", how="inner")
    df = df.merge(prices[["underlying", "trade_date", "S", "q"]],
                  on=["underlying", "trade_date"], how="left")

    # 期限匹配 r：用 SHIBOR 行按 T_days 插值
    sh_map = shibor.set_index("date")
    def rf_for_row(td, tdays):
        # 取最近一个 ≤ td 的 SHIBOR 行
        idx = sh_map.index.searchsorted(td, side="right") - 1
        if idx < 0:
            return np.nan
        row = sh_map.iloc[idx]
        return term_matched_rf(row, tdays)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["T_days"] = (df["maturity_date"] - df["trade_date"]).dt.days
    print("  计算期限匹配无风险利率 ...")
    r_arr = np.array([rf_for_row(td, td_)
                      for td, td_ in zip(df["trade_date"].values, df["T_days"].values)])
    df["r"] = r_arr
    df["r"] = df["r"].ffill().fillna(0.025)
    df["T"] = df["T_days"] / 365.0

    # 商品期权：q = r（Black-76）
    comm_codes = {u["code"] for u in UNDERLYINGS if u["asset"] == "commodity"}
    df.loc[df["underlying"].isin(comm_codes), "q"] = df.loc[
        df["underlying"].isin(comm_codes), "r"]

    # 基础清洗
    n0 = len(df)
    df = df[df["T_days"].between(1, 365)]
    df = df[(df["settle"] > 0) & (df["S"] > 0) & (df["exercise_price"] > 0)]
    df = df[(df["vol"] > 0) | (df["oi"] > 0)]
    print(f"  基础清洗 {n0} → {len(df)}")

    # put-call parity 配对校验
    print("  put-call parity 校验 ...")
    df = df.sort_values(["underlying", "trade_date", "maturity_date",
                         "exercise_price", "call_put"]).reset_index(drop=True)
    # 聚合可能的重复（同标的/日/到期/行权价/认购认沽多合约）→ 取 settle 均值
    keys = ["underlying", "trade_date", "maturity_date", "exercise_price"]
    agg = df.groupby(keys + ["call_put"]).agg(
        settle=("settle", "mean"),
        S=("S", "first"), r=("r", "first"), q=("q", "first"), T=("T", "first")
    ).reset_index()
    piv = agg.pivot_table(index=keys, columns="call_put", values="settle").reset_index()
    meta = agg.groupby(keys)[["S", "r", "q", "T"]].first().reset_index()
    pairs = piv.merge(meta, on=keys)
    pairs = pairs.dropna(subset=["C", "P"])
    # put-call parity 偏差比例（向量化）：|C - P - (S e^{-qT} - K e^{-rT})| / S
    theo = pairs["S"] * np.exp(-pairs["q"] * pairs["T"]) - \
        pairs["exercise_price"] * np.exp(-pairs["r"] * pairs["T"])
    pairs["viol"] = (pairs["C"] - pairs["P"] - theo).abs() / pairs["S"].replace(0, np.nan)
    bad_keys = set(pairs[pairs["viol"] > PARITY_TOL][keys].apply(tuple, axis=1))
    print(f"    配对 {len(pairs)}，剔除 parity 偏差>{PARITY_TOL*100:.0f}% 的 {len(bad_keys)} 个行权价对")
    mask_bad = df[keys].apply(tuple, axis=1).isin(bad_keys)
    df = df[~mask_bad.values].reset_index(drop=True)
    print(f"  parity 清洗后 {len(df)}")

    # 反推 IV（向量化批量，数十倍加速）
    print(f"  对 {len(df)} 行反推 IV（向量化） ...")
    S = df["S"].values; K = df["exercise_price"].values
    T = df["T"].values; r = df["r"].values; q = df["q"].values
    p = df["settle"].values; cp = df["call_put"].values
    is_call = (cp == "C")
    from utils import implied_vol_vec, bs_delta_vec
    iv_arr = implied_vol_vec(p, S, K, T, r, q, is_call)
    delta_arr = bs_delta_vec(S, K, T, r, iv_arr, q, is_call)
    df["iv"] = iv_arr
    df["delta"] = delta_arr
    valid = df["iv"].notna()
    print(f"  IV 有效率: {valid.mean()*100:.2f}% ({valid.sum()}/{len(df)})")
    df = df[valid].reset_index(drop=True)

    keep = ["ts_code", "trade_date", "underlying", "call_put",
            "exercise_price", "maturity_date", "T_days", "T",
            "S", "settle", "vol", "oi", "amount", "r", "q", "iv", "delta"]
    df = df[keep]
    out = os.path.join(PROC_DIR, "iv_panel.parquet")
    df.to_parquet(out, index=False)
    print(f"  ✓ 输出 {len(df)} 行 → {out}")
    print(df.groupby("underlying").size())


if __name__ == "__main__":
    main()
