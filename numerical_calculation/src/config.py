"""config.py
=========================================================
项目全局配置：标的菜单、样本区间、tushare 接入。

标的菜单覆盖两类：
  - ETF 期权（5）：上证50 / 沪深300 / 中证500 / 创业板 / 科创50
  - 商品期权（约 15）：上期所(铜/铝/黄金/锌/橡胶)、大商所(豆粕/玉米/铁矿石/棕榈油)、
    郑商所(白糖/棉花/PTA/甲醇/菜粕/玻璃)

每个标的的元数据：
  code        : tushare 标的代码（ETF 为 fund ts_code；商品为期货品种 symbol，如 'CU.SHF'）
  name        : 中文名
  asset       : 'etf' | 'commodity'
  exch_opt    : 期权所属交易所 SSE/SZSE/SHFE/DCE/CZCE
  mult        : 合约乘数（ETF=10000 份/张；商品=对应期货合约单位，用于 O/S 单位换算）
  market_proxy: beta 类因子的市场代理标的代码（同资产类分组）
  kw          : 从 opt_basic.name 字段匹配该标的期权的关键词
  exl         : 需要排除的关键词（避免名称污染，如 "50ETF" 误匹配科创50）
  fut_symbol  : 商品期权标的期货品种（tushare fut_mapping 用），ETF 留空
  opt_list_start : 该期权首次上市月份（用于界定有效样本起点）

样本期：2020-01-01 ~ 2024-12-31（商品期权 2020 年后陆续上市，面板非平衡）。
=========================================================
"""

from __future__ import annotations

# -----------------------------------------------------
# tushare 接入
# -----------------------------------------------------
TOKEN = "e3922f86bb1c82cf630f91975a75394803b295a83fe369b9d36ab4496242"
HTTP_URL = "http://lianghua.nanyangqiankun.top"

START_DATE = "20200101"
END_DATE = "20241231"


# -----------------------------------------------------
# 标的菜单
# -----------------------------------------------------
# 注：tushare 交易所代码在期权与期货间不一致——
#   期权 exchange 参数 / opt ts_code 后缀：SSE / SZSE / SHFE / DCE / CZCE
#   期货 ts_code 后缀：                       SHF  / --   / SHF  / DCE  / ZCE
#   故商品期权的 exch_opt 用 4/3 字母（SHFE/DCE/CZCE），其标的期货 fut_symbol 用 SHF/DCE/ZCE。
UNDERLYINGS = [
    # ---------- ETF 期权 ----------
    dict(code="510050.SH", name="上证50ETF",   asset="etf",       exch_opt="SSE", mult=10000,
         market_proxy="510050.SH", kw=["50ETF"], exl=["科创"], fut_symbol=None,
         ts_prefix=None, opt_list_start="2015-02"),
    dict(code="510300.SH", name="沪深300ETF",  asset="etf",       exch_opt="SSE", mult=10000,
         market_proxy="510300.SH", kw=["300ETF", "沪深300ETF"], exl=["中证"], fut_symbol=None,
         ts_prefix=None, opt_list_start="2019-12"),
    dict(code="510500.SH", name="中证500ETF",  asset="etf",       exch_opt="SSE", mult=10000,
         market_proxy="510500.SH", kw=["500ETF", "中证500ETF"], exl=[], fut_symbol=None,
         ts_prefix=None, opt_list_start="2022-09"),
    dict(code="159915.SZ", name="创业板ETF",   asset="etf",       exch_opt="SZSE", mult=10000,
         market_proxy="159915.SZ", kw=["创业板ETF"], exl=[], fut_symbol=None,
         ts_prefix=None, opt_list_start="2022-09"),
    dict(code="588000.SH", name="科创50ETF",   asset="etf",       exch_opt="SSE", mult=10000,
         market_proxy="588000.SH", kw=["科创50ETF", "上证科创板50ETF"], exl=[], fut_symbol=None,
         ts_prefix=None, opt_list_start="2023-06"),

    # ---------- 上期所 SHFE 商品期权（ts_code: <PREFIX><YYMM><C/P><STRIKE>.SHF）----------
    dict(code="CU.SHF", name="铜期权",     asset="commodity", exch_opt="SHFE", mult=5,
         market_proxy="CU.SHF", kw=["铜"], exl=["铝", "锌", "黄金", "白银"], fut_symbol="CU.SHF",
         ts_prefix="CU", opt_list_start="2018-09"),
    dict(code="AL.SHF", name="铝期权",     asset="commodity", exch_opt="SHFE", mult=5,
         market_proxy="AL.SHF", kw=["铝"], exl=["铜", "锌", "黄金"], fut_symbol="AL.SHF",
         ts_prefix="AL", opt_list_start="2020-02"),
    dict(code="AU.SHF", name="黄金期权",   asset="commodity", exch_opt="SHFE", mult=1000,
         market_proxy="AU.SHF", kw=["黄金"], exl=[], fut_symbol="AU.SHF",
         ts_prefix="AU", opt_list_start="2019-01"),
    dict(code="ZN.SHF", name="锌期权",     asset="commodity", exch_opt="SHFE", mult=5,
         market_proxy="ZN.SHF", kw=["锌"], exl=["铜", "铝", "黄金"], fut_symbol="ZN.SHF",
         ts_prefix="ZN", opt_list_start="2023-06"),
    dict(code="RU.SHF", name="橡胶期权",   asset="commodity", exch_opt="SHFE", mult=10,
         market_proxy="RU.SHF", kw=["橡胶"], exl=[], fut_symbol="RU.SHF",
         ts_prefix="RU", opt_list_start="2019-01"),

    # ---------- 大商所 DCE 商品期权（ts_code: <PREFIX><YYMM>-<C/P>-<STRIKE>.DCE）----------
    # DCE 期权 ts_code 前缀大写：M=豆粕 C=玉米 I=铁矿石 P=棕榈油
    dict(code="M.DCE",  name="豆粕期权",   asset="commodity", exch_opt="DCE", mult=10,
         market_proxy="M.DCE", kw=["豆粕"], exl=[], fut_symbol="M.DCE",
         ts_prefix="M", opt_list_start="2017-03"),
    dict(code="C.DCE",  name="玉米期权",   asset="commodity", exch_opt="DCE", mult=10,
         market_proxy="C.DCE", kw=["玉米"], exl=["淀粉"], fut_symbol="C.DCE",
         ts_prefix="C", opt_list_start="2019-01"),
    dict(code="I.DCE",  name="铁矿石期权", asset="commodity", exch_opt="DCE", mult=100,
         market_proxy="I.DCE", kw=["铁矿石"], exl=[], fut_symbol="I.DCE",
         ts_prefix="I", opt_list_start="2019-12"),
    dict(code="P.DCE",  name="棕榈油期权", asset="commodity", exch_opt="DCE", mult=10,
         market_proxy="P.DCE", kw=["棕榈油"], exl=[], fut_symbol="P.DCE",
         ts_prefix="P", opt_list_start="2020-08"),

    # ---------- 郑商所 CZCE 商品期权（ts_code: <PREFIX><YYM 3位>-<C/P><STRIKE>.ZCE）----------
    # 期货后缀为 ZCE（与期权交易所代码 CZCE 不同）
    dict(code="SR.CZC", name="白糖期权",   asset="commodity", exch_opt="CZCE", mult=10,
         market_proxy="SR.CZC", kw=["白糖"], exl=[], fut_symbol="SR.ZCE",
         ts_prefix="SR", opt_list_start="2017-04"),
    dict(code="CF.CZC", name="棉花期权",   asset="commodity", exch_opt="CZCE", mult=5,
         market_proxy="CF.CZC", kw=["棉花"], exl=["棉纱"], fut_symbol="CF.ZCE",
         ts_prefix="CF", opt_list_start="2019-01"),
    dict(code="TA.CZC", name="PTA期权",    asset="commodity", exch_opt="CZCE", mult=5,
         market_proxy="TA.CZC", kw=["PTA"], exl=[], fut_symbol="TA.ZCE",
         ts_prefix="TA", opt_list_start="2019-12"),
    dict(code="MA.CZC", name="甲醇期权",   asset="commodity", exch_opt="CZCE", mult=10,
         market_proxy="MA.CZC", kw=["甲醇"], exl=[], fut_symbol="MA.ZCE",
         ts_prefix="MA", opt_list_start="2019-12"),
    dict(code="RM.CZC", name="菜粕期权",   asset="commodity", exch_opt="CZCE", mult=10,
         market_proxy="RM.CZC", kw=["菜粕"], exl=[], fut_symbol="RM.ZCE",
         ts_prefix="RM", opt_list_start="2020-01"),
    dict(code="FG.CZC", name="玻璃期权",   asset="commodity", exch_opt="CZCE", mult=20,
         market_proxy="FG.CZC", kw=["玻璃"], exl=[], fut_symbol="FG.ZCE",
         ts_prefix="FG", opt_list_start="2021-06"),
]


# -----------------------------------------------------
# 便捷查询
# -----------------------------------------------------
def get_underlying(code: str) -> dict:
    for u in UNDERLYINGS:
        if u["code"] == code:
            return u
    raise KeyError(f"unknown underlying: {code}")


ETF_CODES = [u["code"] for u in UNDERLYINGS if u["asset"] == "etf"]
COMMODITY_CODES = [u["code"] for u in UNDERLYINGS if u["asset"] == "commodity"]
ALL_CODES = [u["code"] for u in UNDERLYINGS]

# 期权交易所清单（含 ETF 与商品）
OPT_EXCHANGES = ["SSE", "SZSE", "SHFE", "DCE", "CZCE"]

# 商品期权标的期货品种清单（去重保序）
FUT_SYMBOLS = []
for u in UNDERLYINGS:
    if u["fut_symbol"] and u["fut_symbol"] not in FUT_SYMBOLS:
        FUT_SYMBOLS.append(u["fut_symbol"])
