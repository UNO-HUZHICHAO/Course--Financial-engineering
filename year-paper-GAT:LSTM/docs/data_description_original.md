# 数据输出说明

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    双流异步门控融合网络                        │
│                                                             │
│  时序流 (LSTM)                     空间流 (GAT)               │
│  ─────────────                     ──────────                │
│  日频量价因子 × 16                 半年度基本面图               │
│  Seq_Len = 20天                    GATConv × 2层              │
│  → H_LSTM: [Batch, 64]            → H_GAT: [N, 64]          │
│                                                             │
│              门控融合 → 多跳推理 → 投资信号                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、Market 数据 (`Data/Market/`)

### 1.1 原有数据（来自 CSMAR 日个股回报率文件）

#### Daily_Returns.csv (6.8 MB)

| 属性 | 说明 |
|---|---|
| 格式 | CSV，首列为日期索引 `Trddt`，列名为 `Stkcd`（如 688001） |
| 取值 | `Dretwd`（考虑分红再投资的日收益率），缺失填 0 |
| 日期范围 | 2019-07-22 ~ 2025-12-31（含 2019 年预热期） |
| 有效回测窗口 | 2020-07-01 ~ 2025-12-31 |
| 维度 | 1566 交易日 × 602 科创板股票 |

#### Tradable_Mask.csv (1.8 MB)

| 属性 | 说明 |
|---|---|
| 格式 | CSV，与 Daily_Returns 完全对齐 |
| 取值 | `1` = 可交易，`0` = 不可交易（满足任一条件即为 0） |

不可交易判定规则（三条件满足其一即标记为 0）：
- `Trdsta ≠ 1`（停牌、ST 等非正常交易状态）
- `LimitStatus = 1 或 -1`（涨跌停限制）
- `Dnvaltrd = 0 或空`（无成交额）

此 mask 仅用于回测阶段的信号过滤，**不用于因子计算阶段剔除样本**。

### 1.2 新增数据（来自 Tushare API，通过 fetch_tushare_data.py 拉取）

全局时间窗口：**2020-07-01 ~ 2025-12-31**。

#### KCB50_Index_Daily.csv (99 KB)

| 属性 | 说明 |
|---|---|
| 来源 | `pro.index_daily(ts_code='000688.SH')` |
| 字段 | `ts_code, trade_date, close, open, high, low, pre_close, change, pct_chg, vol, amount` |
| 日期范围 | 2022-01-04 ~ 2025-12-31（Tushare 未覆盖 2020-2021 年的科创50指数数据） |
| 行数 | 969 |
| 用途 | 提供基准指数收益率，用于计算策略超额收益（Alpha）及信息比率（IR） |

#### KCB50_Index_Weight.csv (112 KB)

| 属性 | 说明 |
|---|---|
| 来源 | `pro.index_weight(index_code='000688.SH')`，按年份分批请求后合并 |
| 字段 | `index_code, con_code, trade_date, weight` |
| 日期范围 | 2020-07-31 ~ 2025-12-31 |
| 行数 | 3,300 |
| 用途 | 每期成分股权重，用于计算指数增强策略的跟踪误差；也可用于构造加权基准组合 |

#### STAR_Stock_Basic.csv (57 KB)

| 属性 | 说明 |
|---|---|
| 来源 | `pro.stock_basic(market='科创板')` |
| 字段 | `ts_code, symbol, name, area, industry, cnspell, market, list_date, act_name, act_ent_type` |
| 上市日期范围 | 2019-07-22 ~ 2026-04-24 |
| 股票数 | 611 |
| 用途 | 行业分类（industry）、上市日期（list_date）用于新股过滤（如剔除上市不满1年的股票）、FA 分析中的行业中性化 |

#### Shibor_RiskFree.csv (74 KB)

| 属性 | 说明 |
|---|---|
| 来源 | `pro.shibor()` |
| 字段 | `date, on, 1w, 2w, 1m, 3m, 6m, 9m, 1y` |
| 日期范围 | 2020-07-01 ~ 2025-12-31 |
| 行数 | 1,363 |
| 用途 | 无风险收益率代理。建议使用 `1y`（1年期 Shibor）作为年化无风险利率，按实际天数折算为日度无风险收益，用于计算夏普比率和超额收益 |

#### STAR_Daily_Valuation.csv (71 MB)

| 属性 | 说明 |
|---|---|
| 来源 | `pro.daily_basic(trade_date=date)`，逐交易日遍历获取，每次 sleep(0.3s) 防限流 |
| 字段 | `ts_code, trade_date, close, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv` |
| 日期范围 | 2020-07-01 ~ 2025-12-31 |
| 行数 | 586,824 |
| 股票数 | 602 |
| 获取方式 | 先通过 `trade_cal` 获取 SSE 交易日历（1,338天），逐日调用 `daily_basic` 并筛选 688/689 代码；失败 17 天（均为代理超时，占比 1.3%） |
| 用途 | **市值因子**（total_mv/circ_mv）用于组合中性化约束；**估值因子**（pe/pb/ps）可作为额外输入特征或筛选条件；**换手率**（turnover_rate）辅助流动性评估 |

**注意**：`STAR_Daily_Valuation.csv` 中的个股数与 Daily_Returns 一致（均为 602 只），均为 `ts_code` 格式（如 `688001.SH`），使用时需去除后缀 `.SH` 以与 Stkcd 对齐。

---

## 二、Factor 数据 (`Data/Factors/`)

### 2.1 因子列表

16 个因子，以 `.pkl` 格式存储，矩阵维度 **1566 交易日 × 602 股票**，与 Market 数据完全对齐。所有因子基于 CSMAR 日个股回报率文件的原始字段计算，滚动窗口统一为 **20 个交易日**，最小观测数 `max(W//2, 5) = 10`。

#### A 类：微观结构与异象因子（4个）

| 因子名 | 公式 | 说明 |
|---|---|---|
| `MAX_20` | max(Dretwd)<sub>20d</sub> | 20日最大日收益率，捕获彩票型偏好（Bali et al., 2011） |
| `ILLIQ_20` | mean(|Dretwd| / (Dnvaltrd + 1e-5))<sub>20d</sub> | Amihud 非流动性指标，衡量单位成交额对价格的冲击 |
| `IVOL_20` | std(Dretwd - r<sub>market,EW</sub>)<sub>20d</sub> | 特质波动率：日收益减去市场等权均值后的残差标准差 |
| `Spread_Proxy` | mean((Hiprc - Loprc) / (Clsprc + 1e-5))<sub>20d</sub> | 日内振幅比，作为买卖价差的代理变量 |

#### B 类：趋势与动量因子（4个）

| 因子名 | 公式 | 说明 |
|---|---|---|
| `Return_1d` | Dretwd | 当日收益率，不做滚动（直接作为因子输入，同时作为预测标签） |
| `Return_5d` | ∏(1 + Dretwd) - 1<sub>5d</sub> | 5日累计收益率（几何复合） |
| `Return_20d` | ∏(1 + Dretwd) - 1<sub>20d</sub> | 20日累计收益率（几何复合） |
| `Price_Position` | (Clsprc - minLoprc<sub>20d</sub>) / (maxHiprc<sub>20d</sub> - minLoprc<sub>20d</sub> + 1e-5) | 价格在20日区间内的相对位置，类似于随机指标 K 值 |

#### C 类：波动率与分布因子（4个）

| 因子名 | 公式 | 说明 |
|---|---|---|
| `Realized_Vol_5d` | std(Dretwd)<sub>5d</sub> | 短期（周度）已实现波动率 |
| `Realized_Vol_20d` | std(Dretwd)<sub>20d</sub> | 中期（月度）已实现波动率 |
| `Ret_Skew_20` | skew(Dretwd)<sub>20d</sub> | 20日收益偏度，捕获收益分布的非对称性 |
| `Ret_Kurt_20` | kurt(Dretwd)<sub>20d</sub> | 20日收益峰度，捕获收益分布的厚尾特征 |

#### D 类：量价交互因子（4个）

| 因子名 | 公式 | 说明 |
|---|---|---|
| `Volume_Change` | Dnvaltrd / mean(Dnvaltrd)<sub>20d</sub> | 当日成交额相对20日均值的偏离度 |
| `Turnover_Proxy` | Dnshrtrd / (Dsmvosd / Clsprc + 1e-5) | 换手率代理：成交量(股) / 流通股本(估) |
| `VP_Corr_20` | corr(Clsprc, Dnvaltrd)<sub>20d</sub> | 价格与成交额的20日滚动 Pearson 相关系数，量价背离指标 |
| `Amihud_Trend` | ILLIQ<sub>daily</sub> / (ILLIQ_20 + 1e-5) | 当日非流动性相对其20日均值的比例，流动性突变信号 |

### 2.2 数据处理流水线

每个因子在计算完成后统一经过以下**截面处理**（所有步骤均按交易日横截面独立进行）：

```
原始值 → Inf/NaN 替换为 NaN → 截面 1%-99% Winsorize 缩尾 → 截面 Z-score 标准化 → 残存 NaN 填 0
```

**关键细节**：
- 缩尾和标准化均**按交易日截面**（cross-sectional）进行，而非时间序列。这确保了每个交易日截面上因子值服从标准正态分布，消除了日历效应的干扰。
- 填 0 的含义：截面中性水平（Z-score=0），对深度学习模型数值稳定。
- Zero 值的两类来源：① 股票上市前的日期（结构性空值，~316K）；② 滚动窗口不足的前几个交易日（~5K）。
- `Return_1d` 因子既是 LSTM 的输入特征，也作为 T+1 日的收益率预测标签。

### 2.3 文件格式

```python
import pandas as pd
df = pd.read_pickle("Data/Factors/MAX_20.pkl")
# df.shape = (1566, 602)
# df.index = DatetimeIndex 2019-07-22 ~ 2025-12-31
# df.columns = ['688001', '688002', ..., '689009']
# df.values = float32, 全部由 Z-score 或 0 组成
```

---

## 三、GAT 图快照数据 (`snapshots/`)

### 3.1 截面时间点

GAT 在回测窗口（2020-07-01 ~ 2025-12-31）内以**半年度频率**生成图快照，共 **10 个截面**。每个截面代表截至该日期的企业基本面关联图谱。

| 截面日 | 节点数 | 边总数 | CSMAR边 | LLM边 | 备注 |
|---|---|---|---|---|---|
| 2021-08-31 | 165 | 1 | 1 | 0 | 最早可用的截面（早期数据不足以支撑 2020-08-31 和 2021-04-30） |
| 2022-04-30 | 267 | 1 | 1 | 0 | |
| 2022-08-31 | 329 | 6 | 2 | 4 | LLM 供应链边开始出现 |
| 2023-04-30 | 416 | 4 | 4 | 0 | |
| 2023-08-31 | 455 | 5 | 5 | 0 | |
| 2024-04-30 | 513 | 13 | 9 | 4 | |
| 2024-08-31 | 556 | 38 | 10 | 28 | LLM 边缘大幅增长，出现 Joint_RD、Tech_Licensing |
| 2025-04-30 | 565 | 44 | 10 | 34 | |
| 2025-08-31 | 564 | 31 | 11 | 20 | 出现 Equity_Investment 边 |
| 2025-12-31 | 574 | 54 | 14 | 40 | 最终截面，覆盖 574 只科创板股票 |

**注意**：早期代码预设了 2020-08-31 和 2021-04-30 两个截面，但因 CSMAR 原始数据在该时间点尚不足以构建有效的边（仅含极少量股权参控记录），实际未生成。

### 3.2 股票池构建（`build_universe`）

每个截面的股票池需同时满足三个条件：
1. **交易状态正常**：截面日当天（或最近交易日）`Trdsta == 1`
2. **上市满 1 年**：`snap_date - Listdt >= 365 days`，剔除次新股
3. **非 ST / *ST**：剔除 `Trdsta ∈ {2, 3}` 的股票

数据来源：日个股股票状态表（2020.7.1-2025.6.30 + 2025.7.1-2025.12.31 两段拼接），并通过日行情数据补全了 basic_info 中缺失的早期股票信息。

### 3.3 节点特征（4维）

每只股票在每个截面上有 4 个特征，存储在 `{YYYYMMDD}_nodes.csv` 中。

| 特征 | 含义 | 原始来源 | 处理 |
|---|---|---|---|
| `rd_intensity` | 研发强度 | FS_Comins 表：`B001216000（研发支出）/ B001101000（营业收入）` | 取截面日前最近一期合并报表（年报12-31 或半年报06-30）；先 Winsorize 1%-99% 再 Z-score |
| `tech_moth` | 技术动量 | PT_LCDOMFORAPPLY 表：截面日前 365 天内新增「已授权」发明专利数 | `log(1 + count)` 变换后，Winsorize + Z-score |
| `is_HighTech` | 高新技术企业认定 | QUA_QUALIFINFO 表：`Protype` 含「高新技术企业」且认定日期 ≤ 截面日 | 0/1 二值，不做标准化 |
| `is_LittleGiant` | 专精特新/小巨人 | 同上表，`Protype` 含「专精特新」或「小巨人」 | 0/1 二值；当前数据源中暂未匹配到该类型，默认全 0 |

**缺失值处理**：研发强度和专利数据缺失时填 0（表示无投入/无产出），二值特征缺失填 0。

### 3.4 边构建（3类 CSMAR 边 + 5类 LLM 边）

边存储在 `{YYYYMMDD}_edges.csv` 中，每行格式为 `Source, Target, Edge_Type, Edge_Weight`。

#### CSMAR 边（从结构化数据库提取）

##### (1) 资本边 `Capital_Edge`

| 属性 | 说明 |
|---|---|
| 数据源 | CA_EquityParticipation（股权参控明细表） |
| 逻辑 | Source = 上市公司代码；Target = 被参控公司名称 → 通过公司简称→代码映射和模糊匹配对齐到股票代码 |
| 权重 | `(directholdingratio + indirectholdingratio) / 100`，范围 (0, 1] |
| 策略 | 每对 (symbol, cntrlldcompname) 取最近一期记录；Source ≠ Target 且权重 > 0；去重时取最大权重 |

##### (2) 客户边 `Customer_Edge`

| 属性 | 说明 |
|---|---|
| 数据源 | SC_TopFiveSaleInfo（前五大客户销售信息表） |
| 逻辑 | 仅取合并报表（StateTypeCode=1）、排名 1~5 的记录；每只股票取最近一期 |
| 对齐 | 优先用 BusinessSymbol 直接对齐；否则对 InstitutionName 模糊匹配到股票简称 |
| 过滤 | 掩蔽名称（客户一、公司A 等）丢弃；Source 和 Target 均在当期股票池内 |
| 权重 | `ProportionOfTotalValue / 100`（销售占比） |

##### (3) 供应商边 `Supplier_Edge`

| 属性 | 说明 |
|---|---|
| 数据源 | SC_TopFivePurchaseInfo（前五大供应商采购信息表） |
| 逻辑 | 与客户边完全一致，仅数据源不同 |
| 权重 | `ProportionOfTotalValue / 100`（采购占比） |

#### LLM 边（从业绩说明会文本中通过 DeepSeek API 抽取）

##### 抽取流程

1. **数据清洗**：从 PES_Interaction（公司互动情况表）中筛选：
   - 时间范围：2020-07-01 ~ 2025-12-31（按 EndDate）
   - 问题类型：保留业务经营相关类型（创新信息、经营信息、合作信息、竞争信息、投资信息、融资信息、经济环境信息），丢弃分红/股价/股东人数等非业务提问
   - 有效文本：拼接 QuestionContent + AnswerContent，过滤长度 ≤ 50 字符的
   - 共 **4,498 条有效记录**
2. **API 调用**：DeepSeek Chat API，temperature=0.1，max_tokens=1024，max_concurrency=20
3. **时间对齐**：按 ReportDate 映射到最近的下一个 GAT 截面日，确保信息可用性（不引入未来信息）
4. **实体对齐**：`source_entity` 和 `target_entity` 通过公司简称→代码映射转换为股票代码；无法解析的丢弃

##### 5 类 LLM 关系

| 边类型 | 含义 | 权重（depth_score/5） |
|---|---|---|
| `LLM_Supply_Chain` | 供应链关系（客户/供应商/采购/销售） | 1-5 → 0.2-1.0 |
| `LLM_Strategic_Partnership` | 战略合作、框架协议、生态合作 | 同上 |
| `LLM_Joint_RD` | 联合研发、合作开发、技术共建 | 同上 |
| `LLM_Tech_Licensing` | 技术授权、专利许可、技术转让 | 同上 |
| `LLM_Equity_Investment` | 股权投资、并购、参股 | 同上 |

depth_score 含义：1=提及/传闻，3=有具体合作内容，5=已落地/有合同。权重 = depth_score / 5.0，同一对 (Source, Target, Edge_Type) 取最大权重去重。

##### LLM 边在各截面的分布

| 截面 | LLM_Supply_Chain | LLM_Strategic_Partnership | LLM_Joint_RD | LLM_Tech_Licensing | LLM_Equity_Investment | 合计 |
|---|---|---|---|---|---|---|
| 2022-08-31 | 3 | 1 | 0 | 0 | 0 | 4 |
| 2024-04-30 | 4 | 0 | 0 | 0 | 0 | 4 |
| 2024-08-31 | 23 | 2 | 2 | 1 | 0 | 28 |
| 2025-04-30 | 20 | 14 | 0 | 0 | 0 | 34 |
| 2025-08-31 | 8 | 7 | 3 | 0 | 2 | 20 |
| 2025-12-31 | 25 | 11 | 2 | 1 | 1 | 40 |

LLM 边已直接合并写入各截面的 `_edges.csv` 文件，与 CSMAR 边以统一的 `Edge_Type` 字段区分。

### 3.5 GAT 模型：数据流与维度

```
snapshots/{YYYYMMDD}_nodes.csv         snapshots/{YYYYMMDD}_edges.csv
  [N, 4] 节点特征                        [E, 4] Source, Target, Edge_Type, Edge_Weight
      │                                       │
      ▼                                       ▼
  SnapshotGraphDataset                   Stkcd → 连续索引映射
  过滤无法对齐的 Source/Target
      │                                       │
      ▼                                       ▼
  Data(x=[N, 4], edge_index=[2, E'], edge_attr=[E', 1])
      │
      ▼
  SpatialGAT:
    GATConv(in=4 → hidden=64, heads=4, edge_dim=1) + ELU
    GATConv(in=256 → out=64, heads=1, edge_dim=1)
      │
      ▼
  H_GAT = [N, 64]   每只股票一个 64 维空间表征向量
```

**边权重注入机制**：`GATConv(edge_dim=1)` 将边权重作为附加输入拼接到源节点和目标节点的特征上，共同参与注意力系数的计算。这使得高权重边（如持股比例大的资本边）对消息传递的贡献更大。

---

## 四、回测对接说明

### 4.1 时序流 (LSTM)

```
Data/Factors/*.pkl (16个, 1566×602)
    ↓ build_lstm_tensor(seq_len=20)
X_lstm: [Samples, 20, 16]  滑动窗口
Mask:   [Samples]           T+1日可交易=1
    ↓
LSTM → H_LSTM: [Samples, 64]
```

- 每个样本 = 某只股票在过去 20 个交易日的 16 维因子序列
- T+1 日不可交易的样本已被过滤（604,612 个可交易样本）
- `Return_1d` 对应 T+1 日的真实收益率，可作为回归/排序标签

### 4.2 空间流 (GAT)

```
snapshots/{date}_nodes.csv + edges.csv
    ↓ SnapshotGraphDataset
Data(x=[N,4], edge_index=[2,E], edge_attr=[E,1])
    ↓ SpatialGAT.forward()
H_GAT: [N, 64]
```

- 每个半年度截面日期产生一次 GAT 前向传播
- N = 截面上符合条件的科创板股票数（165~574）
- GAT 表征在相邻截面之间保持不变（piecewise constant），即同一截面区间内的所有交易日使用相同的 GAT 嵌入

### 4.3 门控融合

```
H_fused = gate ⊙ H_LSTM + (1 - gate) ⊙ H_GAT

gate = σ( W_g · [H_LSTM || H_GAT] + b_g )
```

- LSTM 是日频的，GAT 是半年度截面的——需要按日期将 H_GAT 广播对齐
- 不在当期 GAT 截面内的股票，使用零向量填充或跳过

### 4.4 回测评价

```
对每日截面:
  1. 取模型输出的股票得分排名
  2. 过滤 Tradable_Mask=0 的股票
  3. 构建多空组合（top-K 做多 / bottom-K 做空）
  4. 用 Daily_Returns 计算次日组合收益
  5. 使用 Shibor_1Y 折算无风险日收益
  6. 累积收益曲线 + Sharpe / MaxDD / IC / IR 等指标
  7. 以 KCB50_Index_Daily 为基准计算超额收益
```

---

## 五、文件清单速查

| 路径 | 格式 | 维度 | 内容 |
|---|---|---|---|
| **Market 数据** | | | |
| `Data/Market/Daily_Returns.csv` | CSV | 1566 × 602 | 日收益率矩阵 |
| `Data/Market/Tradable_Mask.csv` | CSV | 1566 × 602 | 可交易掩码 (0/1) |
| `Data/Market/KCB50_Index_Daily.csv` | CSV | 969 × 11 | 科创50指数日行情 |
| `Data/Market/KCB50_Index_Weight.csv` | CSV | 3300 × 4 | 科创50成分股历史权重 |
| `Data/Market/STAR_Stock_Basic.csv` | CSV | 611 × 10 | 科创板股票基础信息 |
| `Data/Market/Shibor_RiskFree.csv` | CSV | 1363 × 9 | 无风险利率代理（Shibor各期限） |
| `Data/Market/STAR_Daily_Valuation.csv` | CSV | 586824 × 17 | 科创板每日市值、估值、换手率 |
| **Factor 数据** | | | |
| `Data/Factors/MAX_20.pkl` 等 16 个 | PKL | 1566 × 602 | 滚动因子矩阵（Z-score 后） |
| **GAT 快照** | | | |
| `snapshots/{date}_nodes.csv` × 10 | CSV | N × 5 | 半年度 GAT 节点特征 |
| `snapshots/{date}_edges.csv` × 10 | CSV | E × 4 | 半年度 GAT 边（含 CSMAR + LLM） |
| **代码** | | | |
| `process_star_market.py` | Python | — | LSTM 因子工程全管线 |
| `fetch_tushare_data.py` | Python | — | Tushare 数据拉取脚本 |
| `gat_data_processor.py` | Python | — | GAT 快照生成管线 |
| `llm_relation_extractor.py` | Python | — | LLM 隐性关系抽取（DeepSeek API） |
| `gat_model.py` | Python | — | SpatialGAT 网络定义 |
| `pyg_dataset.py` | Python | — | PyG 图数据集适配器 |
| `test_gat_forward.py` | Python | — | GAT 前向传播可用性测试 |
