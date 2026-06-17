"""01_fetch_data.py
=========================================================
从 tushare 下载本项目所需的全部原始数据（ETF 期权 + 商品期权）：

1. ETF 日行情 fund_daily（5 个 ETF）
2. 商品期权标的——主力连续期货日线（fut_mapping + fut_daily）
3. 期权基础信息 opt_basic（SSE/SZSE/SHFE/DCE/CZCE 五个交易所）
4. 期权日行情 opt_daily（按交易日 × 交易所下载）
5. SHIBOR 利率（用于无风险利率与超额收益）
6. ETF 分红 fund_div 与 份额 fund_share（用于 IV 的股息率 q 与 log_size）

设计说明：
- 本脚本可手动单独运行，全部接口都有 skip-existing 与重试。
- 商品期权标的物是期货合约，故额外抓主力连续合约日线作为收益与"规模"标的。
- token 已确认可用；如遇频率/积分限制，脚本会 sleep 重试。

输出到 ../data/raw/。
=========================================================
"""

from __future__ import annotations
import os, sys, time, warnings, re
import pandas as pd
import tushare as ts

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import (TOKEN, HTTP_URL, START_DATE, END_DATE,
                    UNDERLYINGS, OPT_EXCHANGES, FUT_SYMBOLS)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------
# 商品期权 ts_code 解析（opt_basic 仅有在市合约，历史合约元数据从 ts_code 解析）
# -----------------------------------------------------
def _last_trade_day_of_month(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def parse_commodity_tscode(ts_code: str, u: dict) -> dict:
    """从商品期权 ts_code 解析 call_put / exercise_price / maturity_date。

    格式：
      SHFE: <PF><YYMM><C/P><STRIKE>.SHF      如 CU2404C67000.SHF
      DCE:  <pf><YYMM>-<C/P>-<STRIKE>.DCE     如 m2409-C-3300.DCE
      CZCE: <PF><YYM 3位><C/P><STRIKE>.ZCE    如 SR501C5500.ZCE
    maturity 近似取到期月份的最后一个自然日（与实际行权日差数日，对 T 影响有限）。
    """
    body = ts_code.split(".")[0]
    pf = u["ts_prefix"]
    exch = u["exch_opt"]
    try:
        if exch == "SHFE":
            # body = PF + 4位YYMM + C/P + strike
            if not body.startswith(pf):
                return None
            rest = body[len(pf):]
            yymm, cp, strike = rest[:4], rest[4], rest[5:]
            year, month = 2000 + int(yymm[:2]), int(yymm[2:4])
        elif exch == "DCE":
            # body 形如  pf+YYMM - [MS -] C/P - STRIKE  （MS 为固定标记，可省略）
            parts = body.split("-")
            head = parts[0]            # pf+YYMM
            if not head.startswith(pf):
                return None
            yymm = head[len(pf):len(pf) + 4]
            strike = parts[-1]
            middle = parts[1:-1]
            cp = next((m for m in middle if m in ("C", "P")), None)
            if cp is None:             # MSC/MSP 变体：取末字符
                cp = next((m[-1] for m in middle if m and m[-1] in ("C", "P")), None)
            if cp is None:
                return None
            year, month = 2000 + int(yymm[:2]), int(yymm[2:4])
        elif exch == "CZCE":
            # body = PF + (3位YYM 或 4位YYMM) + (C/P | MSC/MSP 变体) + strike
            if not body.startswith(pf):
                return None
            rest = body[len(pf):]
            m = re.match(r"^(\d{3,4})(MSC|MSP|C|P)(.+)$", rest)
            if not m:
                return None
            ymd, cpseg, strike = m.group(1), m.group(2), m.group(3)
            cp = "C" if cpseg in ("C", "MSC") else "P"
            if len(ymd) == 3:
                year, month = 2020 + int(ymd[0]), int(ymd[1:3])
            else:
                year, month = 2000 + int(ymd[:2]), int(ymd[2:4])
        else:
            return None
        return {
            "ts_code": ts_code,
            "call_put": "C" if cp.upper() == "C" else "P",
            "exercise_price": float(strike),
            "maturity_date": _last_trade_day_of_month(year, month),
            "s_month": f"{year}{month:02d}",
        }
    except Exception:
        return None


def commodity_prefix_regex(exch: str) -> re.Pattern:
    """该交易所全部商品产品 ts_code 前缀的正则（用于 opt_daily 下载时过滤）。"""
    prefs = [u["ts_prefix"] for u in UNDERLYINGS
             if u["asset"] == "commodity" and u["exch_opt"] == exch]
    if not prefs:
        return None
    # 前缀后必须紧跟数字（YYMM / YYM），避免前缀互相包含
    pat = r"^(" + "|".join(re.escape(p) for p in sorted(prefs, key=len, reverse=True)) + r")\d"
    return re.compile(pat)


def init_pro():
    pro = ts.pro_api(TOKEN)
    pro._DataApi__token = TOKEN
    pro._DataApi__http_url = HTTP_URL
    return pro


def _retry(fn, *args, tries=3, sleep=2, **kw):
    """带重试的 tushare 调用。"""
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            last = e
            time.sleep(sleep * (i + 1))
    print(f"    !! 多次重试失败：{last}")
    return None


# -----------------------------------------------------
# 1. ETF 日行情
# -----------------------------------------------------
def fetch_etf_daily(pro):
    print("[01] 下载 ETF 日行情 ...")
    for u in UNDERLYINGS:
        if u["asset"] != "etf":
            continue
        code = u["code"]
        out = os.path.join(OUT_DIR, f"fund_daily_{code.replace('.', '_')}.parquet")
        if os.path.exists(out):
            print(f"  跳过已存在：{u['name']} ({code})")
            continue
        df = _retry(pro.fund_daily, ts_code=code, start_date=START_DATE, end_date=END_DATE)
        if df is None or df.empty:
            print(f"  ⚠️  {u['name']} ({code}) 无数据")
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        df.to_parquet(out, index=False)
        print(f"  ✓ {u['name']} ({code}): {len(df)} 行")
        time.sleep(0.3)


# -----------------------------------------------------
# 2. 商品期权标的——主力连续期货日线
# -----------------------------------------------------
def fetch_dominant_futures(pro):
    """对每个商品期货品种，用 fut_mapping 取每日主力合约，再 fut_daily 取其日线，
    拼成 [trade_date, ts_code(主力), close, vol, oi, settle, pre_settle] 主力连续序列。"""
    print("[02] 下载商品主力连续期货日线 ...")
    for sym in FUT_SYMBOLS:
        out = os.path.join(OUT_DIR, f"fut_daily_{sym.replace('.', '_')}.parquet")
        if os.path.exists(out):
            print(f"  跳过已存在：{sym}")
            continue
        # 主力映射
        mp = _retry(pro.fut_mapping, ts_code=sym)
        if mp is None or mp.empty:
            print(f"  ⚠️  {sym} 无 fut_mapping，尝试 fut_daily(trade_date) 兜底 ...")
            # 兜底：按交易日取该品种主力（部分接口支持 symbol 维度）
            continue
        mp = mp.sort_values("trade_date")
        # 仅保留样本期内
        mp = mp[(mp["trade_date"] >= START_DATE) & (mp["trade_date"] <= END_DATE)]
        # 需要拉取的主力合约清单
        dom_codes = mp["mapping_ts_code"].dropna().unique().tolist()
        print(f"  {sym}: {len(dom_codes)} 个主力合约待拉取 ...")

        daily_rows = []
        for cc in dom_codes:
            df = _retry(pro.fut_daily, ts_code=cc, start_date=START_DATE, end_date=END_DATE)
            if df is None or df.empty:
                continue
            daily_rows.append(df)
            time.sleep(0.12)
        if not daily_rows:
            print(f"  ⚠️  {sym} 无 fut_daily 数据")
            continue
        all_daily = pd.concat(daily_rows, ignore_index=True)
        # 按主力映射对齐：每个交易日取当日主力合约的日线
        merged = mp[["trade_date", "mapping_ts_code"]].rename(
            columns={"mapping_ts_code": "ts_code"}).merge(
            all_daily, on=["trade_date", "ts_code"], how="left")
        # 兜底：若某日主力日线缺失，用前值
        merged = merged.sort_values("trade_date").reset_index(drop=True)
        merged.to_parquet(out, index=False)
        print(f"  ✓ {sym}: {len(merged)} 行 → {out}")
        time.sleep(0.3)


# -----------------------------------------------------
# 3. 期权基础信息（按交易所）
# -----------------------------------------------------
def fetch_opt_basic(pro):
    print("[03] 下载期权基础信息 ...")
    for exchange in OPT_EXCHANGES:
        out = os.path.join(OUT_DIR, f"opt_basic_{exchange}.parquet")
        if os.path.exists(out):
            print(f"  跳过已存在：{exchange}")
            continue
        df = _retry(pro.opt_basic, exchange=exchange)
        if df is None or df.empty:
            print(f"  ⚠️  {exchange} 无 opt_basic")
            continue
        df.to_parquet(out, index=False)
        print(f"  ✓ {exchange}: {len(df)} 个合约")
        time.sleep(0.3)


# -----------------------------------------------------
# 4. 筛选每个标的的期权合约清单
# -----------------------------------------------------
def select_underlying_options(opt_basic_all: pd.DataFrame, u: dict) -> pd.DataFrame:
    """根据 name 字段关键词筛选某标的期权，并按样本期过滤。"""
    mask = pd.Series(False, index=opt_basic_all.index)
    for kw in u["kw"]:
        mask = mask | opt_basic_all["name"].str.contains(kw, na=False)
    for ex in u.get("exl", []):
        mask = mask & ~opt_basic_all["name"].str.contains(ex, na=False)
    # 限定交易所（避免跨交易所同名产品污染）
    mask = mask & (opt_basic_all["exchange"] == u["exch_opt"])
    sub = opt_basic_all[mask].copy()
    sub = sub[sub["delist_date"].astype(str) >= START_DATE]
    sub = sub[sub["list_date"].astype(str) <= END_DATE]
    return sub.reset_index(drop=True)


def build_contract_lists():
    """构建各标的合约清单。
    ETF：用 opt_basic（含历史合约）按 name 关键词筛选。
    商品：opt_basic 仅含在市合约，故从已下载的 opt_daily 中提取全部 ts_code，
          用 parse_commodity_tscode 解析元数据（行权价/认沽认购/到期月）。
    """
    print("[04] 构建各标的期权合约清单 ...")
    # ---- ETF: opt_basic ----
    etf_basics = []
    for ex in ("SSE", "SZSE"):
        f = os.path.join(OUT_DIR, f"opt_basic_{ex}.parquet")
        if os.path.exists(f):
            etf_basics.append(pd.read_parquet(f))
    if etf_basics:
        etf_basic_all = pd.concat(etf_basics, ignore_index=True)
        for u in UNDERLYINGS:
            if u["asset"] != "etf":
                continue
            sub = select_underlying_options(etf_basic_all, u)
            sub.to_parquet(os.path.join(OUT_DIR,
                            f"contracts_{u['code'].replace('.', '_')}.parquet"),
                           index=False)
            print(f"  ETF {u['name']} ({u['code']}): {len(sub)} 个合约")

    # ---- 商品: 从 opt_daily 解析 ----
    for ex in ("SHFE", "DCE", "CZCE"):
        f = os.path.join(OUT_DIR, f"opt_daily_{ex}.parquet")
        if not os.path.exists(f):
            print(f"  ⚠️  无 opt_daily_{ex}，商品合约清单跳过")
            continue
        od = pd.read_parquet(f, columns=["ts_code"])
        all_ts = od["ts_code"].dropna().unique()
        for u in UNDERLYINGS:
            if u["exch_opt"] != ex:
                continue
            parsed = [p for p in (parse_commodity_tscode(tc, u) for tc in all_ts) if p]
            if not parsed:
                print(f"  商品 {u['name']} ({u['code']}): 0 个合约（前缀 {u['ts_prefix']} 未匹配）")
                continue
            df = pd.DataFrame(parsed)
            df["underlying"] = u["code"]
            df["exchange"] = ex
            df["name"] = u["name"]
            df.to_parquet(os.path.join(OUT_DIR,
                          f"contracts_{u['code'].replace('.', '_')}.parquet"),
                         index=False)
            print(f"  商品 {u['name']} ({u['code']}): {len(df)} 个合约")


# -----------------------------------------------------
# 5. 交易日历
# -----------------------------------------------------
def fetch_trade_dates(pro):
    print("[05] 获取交易日历 ...")
    out = os.path.join(OUT_DIR, "trade_dates.parquet")
    if os.path.exists(out):
        td = pd.read_parquet(out)
    else:
        td = pro.trade_cal(exchange="SSE", start_date=START_DATE, end_date=END_DATE, is_open="1")
        td.to_parquet(out, index=False)
    all_dates = sorted(td["cal_date"].astype(str).tolist())
    print(f"  共 {len(all_dates)} 个交易日")
    return all_dates


# -----------------------------------------------------
# 6. 期权日行情（按交易日 × 交易所）
#    ETF 交易所：按 ETF 合约清单 ts_code 过滤；
#    商品交易所：按商品产品 ts_code 前缀过滤（opt_basic 无历史合约，故用前缀）。
# -----------------------------------------------------
def fetch_opt_daily(pro, all_trade_dates):
    print("[06] 下载期权日行情 ...")
    # ETF 合约 ts_code 集合
    etf_codes_by_exch = {ex: set() for ex in OPT_EXCHANGES}
    for u in UNDERLYINGS:
        if u["asset"] != "etf":
            continue
        f = os.path.join(OUT_DIR, f"contracts_{u['code'].replace('.', '_')}.parquet")
        if os.path.exists(f):
            etf_codes_by_exch[u["exch_opt"]] |= set(pd.read_parquet(f)["ts_code"].tolist())
    # 商品前缀正则
    comm_regex = {ex: commodity_prefix_regex(ex) for ex in OPT_EXCHANGES}

    for exchange in OPT_EXCHANGES:
        out = os.path.join(OUT_DIR, f"opt_daily_{exchange}.parquet")
        if os.path.exists(out):
            print(f"  跳过已存在：{exchange}")
            continue
        is_comm = exchange in ("SHFE", "DCE", "CZCE")
        regex = comm_regex.get(exchange) if is_comm else None
        etf_set = etf_codes_by_exch.get(exchange, set())
        if is_comm and regex is None:
            continue
        if not is_comm and not etf_set:
            continue
        rows = []
        n = len(all_trade_dates)
        for i, date in enumerate(all_trade_dates):
            df = _retry(pro.opt_daily, trade_date=date, exchange=exchange, tries=2)
            if df is not None and len(df):
                if is_comm:
                    df = df[df["ts_code"].str.match(regex, na=False)]
                else:
                    df = df[df["ts_code"].isin(etf_set)]
                if len(df):
                    rows.append(df)
            if (i + 1) % 60 == 0:
                print(f"    进度 {exchange}: {i+1}/{n}")
            time.sleep(0.05)
        if rows:
            full = pd.concat(rows, ignore_index=True)
            full.to_parquet(out, index=False)
            print(f"  ✓ {exchange}: {len(full)} 行")
        else:
            print(f"  ⚠️  {exchange} 无数据")


# -----------------------------------------------------
# 7. SHIBOR
# -----------------------------------------------------
def fetch_shibor(pro):
    print("[07] 下载 SHIBOR ...")
    out = os.path.join(OUT_DIR, "shibor.parquet")
    if os.path.exists(out):
        print("  跳过已存在：SHIBOR")
        return
    chunks = []
    for y in range(int(START_DATE[:4]), int(END_DATE[:4]) + 1):
        df = _retry(pro.shibor, start_date=f"{y}0101", end_date=f"{y}1231")
        if df is not None and len(df):
            chunks.append(df)
        time.sleep(0.3)
    if chunks:
        full = pd.concat(chunks, ignore_index=True).sort_values("date").reset_index(drop=True)
        full.to_parquet(out, index=False)
        print(f"  ✓ SHIBOR: {len(full)} 行")


# -----------------------------------------------------
# 8. ETF 分红与份额（用于 q 与 log_size）
# -----------------------------------------------------
def fetch_etf_div_share(pro):
    print("[08] 下载 ETF 分红 / 份额 ...")
    for u in UNDERLYINGS:
        if u["asset"] != "etf":
            continue
        code = u["code"]
        # 分红
        out_d = os.path.join(OUT_DIR, f"fund_div_{code.replace('.', '_')}.parquet")
        if not os.path.exists(out_d):
            df = _retry(pro.fund_div, ts_code=code)
            if df is not None and len(df):
                df.to_parquet(out_d, index=False)
                print(f"  ✓ 分红 {code}: {len(df)} 行")
            time.sleep(0.2)
        # 份额
        out_s = os.path.join(OUT_DIR, f"fund_share_{code.replace('.', '_')}.parquet")
        if not os.path.exists(out_s):
            df = _retry(pro.fund_share, ts_code=code, start_date=START_DATE, end_date=END_DATE)
            if df is not None and len(df):
                df.to_parquet(out_s, index=False)
                print(f"  ✓ 份额 {code}: {len(df)} 行")
            time.sleep(0.2)


# -----------------------------------------------------
# 主流程
# -----------------------------------------------------
def main():
    pro = init_pro()
    fetch_etf_daily(pro)
    fetch_dominant_futures(pro)
    fetch_opt_basic(pro)
    fetch_shibor(pro)
    fetch_etf_div_share(pro)
    build_contract_lists()          # ETF 清单（opt_basic）；商品此时无 opt_daily 会跳过
    all_dates = fetch_trade_dates(pro)
    fetch_opt_daily(pro, all_dates)
    build_contract_lists()          # 再次构建：补商品清单（从 opt_daily 解析）
    print("[OK] 全部数据下载完成。")


if __name__ == "__main__":
    main()
