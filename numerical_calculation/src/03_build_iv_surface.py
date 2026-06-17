"""03_build_iv_surface.py
=========================================================
构造每个标的、每个交易日的"标准化 IV 曲面"。

替代原文用 OptionMetrics 提供的 surface 的步骤：
1. 每日选取距到期 ≥7 天的两个最近月份（near, next）
2. 对每个月份的合约（call/put）按 delta 排序，用 PCHIP/线性插值得到固定 delta 的 IV
3. 在两个月份间按 (T, IV) 线性插值得到固定 30 天到期的 IV
4. 类似地计算 60、180 天到期的 IV（用于期限结构因子）

输入: data/processed/iv_panel.parquet
输出: data/processed/iv_surface.parquet —— 标准化曲面

每行：
  underlying, trade_date,
  iv_30d_call_d50, iv_30d_call_d40, iv_30d_call_d30, iv_30d_call_d20,
  iv_30d_put_d50,  iv_30d_put_d40,  iv_30d_put_d30,  iv_30d_put_d20,
  iv_180d_call_d50, iv_180d_put_d50,  # 期限结构用
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from utils import interp_iv_by_delta, interp_iv_by_maturity

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TARGET_DELTAS = [0.20, 0.25, 0.30, 0.40, 0.50]  # 取绝对值后；d25 供 IVS_ATM（Cremers-Weinbaum）
TARGET_T = {'30d': 30/365, '60d': 60/365, '180d': 180/365}


def select_two_maturities(group: pd.DataFrame, target_T: float) -> tuple:
    """选取距 target_T 最近的两个到期日（一个 ≤、一个 ≥），返回两组合约。"""
    Ts = sorted(group['T'].unique())
    if len(Ts) == 0:
        return None, None, None, None

    Ts_arr = np.array(Ts)
    below = Ts_arr[Ts_arr <= target_T]
    above = Ts_arr[Ts_arr >= target_T]

    if len(below) == 0:
        # 所有到期都比 target 远 → 取最近两个
        if len(Ts) >= 2:
            t1, t2 = Ts[0], Ts[1]
        else:
            t1 = t2 = Ts[0]
    elif len(above) == 0:
        # 所有到期都比 target 近 → 取最远两个
        if len(Ts) >= 2:
            t1, t2 = Ts[-2], Ts[-1]
        else:
            t1 = t2 = Ts[-1]
    else:
        t1 = below.max()  # 最近的 ≤ target
        t2 = above.min()  # 最近的 ≥ target

    g1 = group[group['T'] == t1]
    g2 = group[group['T'] == t2]
    return g1, g2, t1, t2


def iv_at_delta_for_group(g: pd.DataFrame, target_deltas: list) -> dict:
    """对单个到期月份的合约，分别给 call/put 在固定 |delta| 的 IV。

    返回: {('C',d): iv, ('P',d): iv}
    """
    out = {}
    for cp in ['C', 'P']:
        sub = g[g['call_put'] == cp]
        if len(sub) < 2:
            for d in target_deltas:
                out[(cp, d)] = np.nan
            continue
        deltas = np.abs(sub['delta'].values)
        ivs = sub['iv'].values
        result = interp_iv_by_delta(deltas, ivs, target_deltas)
        for d in target_deltas:
            out[(cp, d)] = result[d]
    return out


def build_surface_for_day(day_df: pd.DataFrame) -> dict:
    """对单个 (标的, 交易日) 构造标准化曲面。

    delta 插值用 PCHIP（utils.interp_iv_by_delta），单调、避免稀疏行权价处振荡。
    180d 长端若需外推（最近到期都短于 180 天），结果为 NaN——下游 TSCALL/TSPUT
    相应记为缺失而非外推，避免期限结构因子失真。
    """
    res = {}
    for label, target_T in TARGET_T.items():
        g1, g2, t1, t2 = select_two_maturities(day_df, target_T)
        if g1 is None:
            continue

        iv1 = iv_at_delta_for_group(g1, TARGET_DELTAS)
        iv2 = iv_at_delta_for_group(g2, TARGET_DELTAS)

        for cp in ['C', 'P']:
            for d in TARGET_DELTAS:
                v1 = iv1[(cp, d)]
                v2 = iv2[(cp, d)]
                # 180d 缺长端（两端到期都短于 180 天）→ 标记 NaN，不外推
                if target_T == TARGET_T['180d'] and t1 < target_T and t2 < target_T:
                    res[f'iv_{label}_{("call" if cp=="C" else "put")}_d{int(d*100):02d}'] = np.nan
                    continue
                interp_v = interp_iv_by_maturity(t1, v1, t2, v2, target_T)
                key = f'iv_{label}_{("call" if cp=="C" else "put")}_d{int(d*100):02d}'
                res[key] = interp_v
    return res


def main():
    print("[03] 构造标准化 IV 曲面 ...")
    iv_panel = pd.read_parquet(os.path.join(PROC_DIR, "iv_panel.parquet"))
    print(f"  载入 IV 面板：{len(iv_panel)} 行")

    # 数据清洗：合理的 IV 与 delta
    iv_panel = iv_panel[iv_panel['iv'].between(0.01, 3.0)]
    iv_panel = iv_panel[iv_panel['delta'].abs().between(0.01, 0.99)]
    iv_panel = iv_panel[iv_panel['T_days'].between(5, 365)]
    print(f"  清洗后：{len(iv_panel)} 行")

    # 按 (underlying, trade_date) 分组
    rows = []
    grouped = iv_panel.groupby(['underlying', 'trade_date'])
    n_groups = len(grouped)
    print(f"  共 {n_groups} 个 (标的×交易日) 组")

    for i, ((u, td), g) in enumerate(grouped):
        # 至少 8 个合约，且 call+put 都有
        if len(g) < 8 or g['call_put'].nunique() < 2:
            continue
        feats = build_surface_for_day(g)
        feats['underlying'] = u
        feats['trade_date'] = td
        rows.append(feats)

        if (i + 1) % 1000 == 0:
            print(f"    进度 {i+1}/{n_groups}")

    surface = pd.DataFrame(rows)
    out = os.path.join(PROC_DIR, "iv_surface.parquet")
    surface.to_parquet(out, index=False)
    print(f"  ✓ 输出 {len(surface)} 行 → {out}")
    print(f"  字段: {[c for c in surface.columns if c.startswith('iv_')]}")


if __name__ == "__main__":
    main()
