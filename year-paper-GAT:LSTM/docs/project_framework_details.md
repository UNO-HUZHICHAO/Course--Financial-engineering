# 科创50指数增强策略 — 项目框架详解

> 面向通用大模型排查问题的完整技术文档，覆盖从数据获取到回测评价的每一个细节。

---

## 一、项目概览

本项目实现一套**"基于双流门控融合网络的科创50指数增强策略"**。核心思路是：将日频量价因子（**时序流**，LSTM 编码）与半年度更新的产业链知识图谱（**空间流**，GAT 编码）通过一个可学习的门控机制进行凸组合融合，端到端输出个股预期超额收益打分，再通过带约束的二次规划将打分转化为投资组合权重。

**代码运行方式**：`cd src && python run_backtest.py --quick`（快速测试）/ 不加 `--quick` 跑完整回测。

**文件结构（当前整理后）**：

```
学年论文/
├── src/                          ← 核心管线代码
│   ├── run_backtest.py           ← 主入口
│   ├── data/data_module.py       ← 模块一：异构数据对齐与 Dataset
│   ├── model/model.py            ← 模块二：双流门控融合网络
│   ├── backtest/backtest_engine.py   ← 模块三：块状滚动回测引擎
│   ├── backtest/portfolio_optimizer.py ← 模块四：cvxpy 组合优化
│   └── backtest/reporter.py      ← 模块五：可视化和回测报告
├── data/                         ← 数据仓库
│   ├── raw/csmar/                ← 原始 CSMAR 数据（8张表）
│   ├── raw/lstm/                 ← 原始日个股回报率 CSV
│   ├── processed/factors/        ← 16个量价因子 .pkl
│   ├── processed/market/         ← 行情/收益率/无风险利率
│   ├── processed/snapshots/      ← 10期图谱快照 (nodes + edges CSV)
│   ├── process_star_market.py    ← LSTM 因子工程脚本
│   ├── fetch_tushare_data.py     ← Tushare API 数据拉取
│   ├── gat_data_processor.py     ← GAT 图谱构建脚本
│   └── llm_relation_extractor.py ← LLM 隐性关系抽取
├── result/                       ← 回测输出
└── docs/                         ← 文档
```

---

## 二、数据来源与预处理

### 2.1 数据来源一览

| 数据 | 来源 | 覆盖范围 | 说明 |
|---|---|---|---|
| 日个股回报率 | **CSMAR** (TRD_Dalyr) | 2019-07-22 ~ 2025-12-31 | `Dretwd` 日收益率 (考虑分红再投资)，`Hiprc/Loprc/Clsprc` 价格，`Dnvaltrd` 成交额，`Dnshrtrd` 成交量，`Dsmvosd` 流通市值 |
| 日个股交易状态 | **CSMAR** | 同上 | `Trdsta` (1=正常), `LimitStatus` (涨跌停), `Dnvaltrd=0` |
| 科创50指数日行情 | **Tushare** `pro.index_daily(ts_code='000688.SH')` | 2022-01-04 ~ 2025-12-31 | `pct_chg` 基准收益率 |
| 科创50成分股权重 | **Tushare** `pro.index_weight` | 2020-07-31 ~ 2025-12-31 | 每月成分股及其权重 |
| 股票基本信息 | **Tushare** `pro.stock_basic` | 611只科创板股票 | `industry`(申万一级行业), `list_date`(上市日期) |
| 无风险利率 | **Tushare** `pro.shibor` | 2020-07-01 ~ 2025-12-31 | 使用 `1y` (1年期 Shibor) 折算日度无风险利率 |
| 公司财务(研发) | **CSMAR** `FS_Comins` | 各期合并报表 | `B001216000` 研发支出 / `B001101000` 营业收入 |
| 专利申请 | **CSMAR** `PT_LCDOMFORAPPLY` | 截至各截面日 | 已授权发明专利数量 |
| 高新/专精特新认定 | **CSMAR** `QUA_QUALIFINFO` | 截至各截面日 | `Protype` 字段匹配 |
| 股权参控 | **CSMAR** `CA_EquityParticipation` | 各期 | Source=上市公司, Target=被参控公司 |
| 前五大客户/供应商 | **CSMAR** `SC_TopFiveSaleInfo/PurchaseInfo` | 各期合并报表 | 销售/采购占比 |
| 公司互动情况(业绩说明会) | **CSMAR** `PES_Interaction` | 2020-07-01 ~ 2025-12-31 | `QuestionContent + AnswerContent` 文本 |

### 2.2 时序量价特征工程 (process_star_market.py → data/processed/factors/)

**处理流程**：
1. 从 `raw/lstm/` 读取 CSMAR 日个股回报率 CSV（7个分片），筛选 688/689 开头的科创板股票
2. 统一 stkcd 编码为6位数字字符串，日期格式化为 `YYYY-MM-DD`
3. 构建日收益率矩阵、可交易掩码矩阵（停牌/涨跌停/无成交额 → mask=0）、价格/成交额/流通市值等矩阵
4. 基于上述矩阵计算 16 个滚动因子（窗口统一为 20 个交易日）
5. 每个因子独立存储为 `.pkl`，矩阵维度 `1566交易日 × 602股票`，float32

**16个因子分类**：

| 类别 | 因子名 | 公式 | 经济学含义 |
|---|---|---|---|
| **微观结构** | `MAX_20` | max(Dretwd)_20d | 20日最大日收益，捕获彩票型偏好 |
| | `ILLIQ_20` | mean(\|Dretwd\| / Dnvaltrd)_20d | Amihud 非流动性，单位成交额的价格冲击 |
| | `IVOL_20` | std(Dretwd - r_market,EW)_20d | 特质波动率（去市场等权均值后的残差） |
| | `Spread_Proxy` | mean((Hiprc-Loprc)/Clsprc)_20d | 日内振幅比，买卖价差代理 |
| **趋势动量** | `Return_1d` | Dretwd | 当日收益率（**滞后1天**，见下文） |
| | `Return_5d` | Π(1+Dretwd)-1 _5d | 5日累计几何收益（**滞后1天**） |
| | `Return_20d` | Π(1+Dretwd)-1 _20d | 20日累计几何收益（**滞后1天**） |
| | `Price_Position` | (Clsprc-minLoprc)/(maxHiprc-minLoprc)_20d | 价格在20日区间相对位置 (随机指标K) |
| **波动分布** | `Realized_Vol_5d` | std(Dretwd)_5d | 周度已实现波动率 |
| | `Realized_Vol_20d` | std(Dretwd)_20d | 月度已实现波动率 |
| | `Ret_Skew_20` | skew(Dretwd)_20d | 收益偏度 |
| | `Ret_Kurt_20` | kurt(Dretwd)_20d | 收益峰度 |
| **量价交互** | `Volume_Change` | Dnvaltrd / mean(Dnvaltrd)_20d | 当日成交额偏离度 |
| | `Turnover_Proxy` | Dnshrtrd / 流通股本_估算 | 换手率代理 |
| | `VP_Corr_20` | corr(Clsprc, Dnvaltrd)_20d | 量价相关系数 |
| | `Amihud_Trend` | ILLIQ_daily / ILLIQ_20 | 流动性突变信号 |

**关键处理细节**：

1. **截面处理流程**（每个交易日独立进行）：
   ```
   原始值 → Inf/NaN 替换为 NaN → 截面 1%-99% Winsorize → 截面 Z-score → 残留 NaN 填 0
   ```

2. **收益率滞后处理**（在 data_module.py 加载时执行）：
   `Return_1d`, `Return_5d`, `Return_20d` 这三个因子统一向后平移 1 天。原因是日收益率存在显著的均值回复效应（IC ≈ -0.025, t ≈ -5.60），即 `Return_1d[t]` 对 `ret[t+1]` 有负向预测力。滞后后，因子窗口 [t-19, t] 中的收益因子对应实际收益 [t-20, t-1]，避免模型学到反向排序。

3. **截面 Rank 百分位转换**（在 data_module.py 加载时执行）：
   对每日每个因子做截面 Rank → Percentile → [-0.5, 0.5]。目的是：
   - 消除极端值（科创板单日涨跌 ±20% 产生巨大噪声）
   - 将非线性因子值映射为统一量纲的单调排序信号
   - 与 Rank IC / Spearman 相关系数损失函数对齐

### 2.3 GAT 图谱构建 (gat_data_processor.py → data/processed/snapshots/)

**截面时间点**（半年度，共10期）：
`20210831, 20220430, 20220831, 20230430, 20230831, 20240430, 20240831, 20250430, 20250831, 20251231`

每个截面生成一个 `{YYYYMMDD}_nodes.csv` 和一个 `{YYYYMMDD}_edges.csv`。

**股票池筛选**（每个截面独立）：
1. 截面日当天（或最近交易日） `Trdsta == 1`（正常交易）
2. 上市天数 `snap_date - Listdt >= 365`（剔除次新股）
3. `Trdsta ∉ {2, 3}`（非 ST / *ST）

**4维节点特征**：

| 特征 | 计算方式 | 标准化 |
|---|---|---|
| `rd_intensity` | 研发支出 / 营业收入（取截面日前最近一期合并报表） | Winsorize 1%-99% + Z-score |
| `tech_moth` | log(1 + 截面日前365天内授权发明专利数) | Winsorize + Z-score |
| `is_HighTech` | Protype 含"高新技术企业"且认定日 ≤ 截面日 | 0/1 二值，不做标准化 |
| `is_LittleGiant` | Protype 含"专精特新"或"小巨人" | 0/1 二值，当前数据源暂缺，默认全0 |

**边构建（CSMAR + LLM）**：

**CSMAR 边（3类）：**
- `Capital_Edge`：从股权参控表提取。权重 = (直接持股+间接持股) / 100。取最近一期记录，去重取最大权重。
- `Customer_Edge`：从前五大客户表提取。权重 = 销售占比 / 100。仅合并报表，模糊匹配公司名→股票代码。
- `Supplier_Edge`：从前五大供应商表提取。权重 = 采购占比 / 100。逻辑同上。

**LLM 边（5类，通过 DeepSeek API 从业绩说明会文本抽取，4,498条记录）：**
- `LLM_Supply_Chain`：供应链关系
- `LLM_Strategic_Partnership`：战略合作
- `LLM_Joint_RD`：联合研发
- `LLM_Tech_Licensing`：技术授权/许可
- `LLM_Equity_Investment`：股权投资/并购

LLM 边权重 = depth_score / 5.0（1=提及, 3=具体合作, 5=已落地），去重时取最大权重。LLM 边按 `ReportDate` 对齐到最近的下一个 GAT 截面日，确保不引入未来信息。

**各截面数据规模**：
| 截面日 | 节点数 | CSMAR边 | LLM边 | 总边数 |
|---|---|---|---|---|
| 20210831 | 165 | 1 | 0 | 1 |
| 20220430 | 267 | 1 | 0 | 1 |
| 20220831 | 329 | 2 | 4 | 6 |
| 20230430 | 416 | 4 | 0 | 4 |
| 20230831 | 455 | 5 | 0 | 5 |
| 20240430 | 513 | 9 | 4 | 13 |
| 20240831 | 556 | 10 | 28 | 38 |
| 20250430 | 565 | 10 | 34 | 44 |
| 20250831 | 564 | 11 | 20 | 31 |
| 20251231 | 574 | 14 | 40 | 54 |

---

## 三、模块一：异构数据对齐与 DataLoader (`src/data/data_module.py`)

### 3.1 核心类

**`SnapshotIndex`**：GAT 快照索引管理器
- 预加载全部 10 期 PyG `Data` 对象（node features + edge_index + edge_attr）
- 维护每期快照的 `stkcd → node_idx` 映射
- 提供 `get_snapshot(trade_date)`：按交易日向过去寻址最近快照

**`DualStreamDataset`**：PyTorch Dataset
- 加载因子立方体 `[T, N, F]` float32
- 加载标签（T+1 日收益率，截面 Z-score 标准化）
- 加载交易掩码（`Tradable_Mask`）、上市天数（用于过滤上市不满1年的股票）
- 构建有效样本索引（过滤停牌、次新股、NaN 标签）
- **单样本返回**：`(X_lstm [20,16], snap_key, node_idx, label, mask, trade_date, stkcd)`
- **`get_daily_samples(date)`**：批量返回当日所有股票的因子序列（OOS 推断提速）
- **`get_indices_in_range(start, end)`**：按日期范围切片样本索引

### 3.2 数据处理细节（加载时执行）

**收益率滞后**：`Return_1d/Return_5d/Return_20d` 因子向后平移1天，首日填0。

**截面 Rank 转换**：每日每个因子独立做 `argsort → percentile → [-0.5, 0.5]`。

**标签 Z-score**：对每日 T+1 收益率截面做 `(x - mean) / (std + 1e-8)`，仅用可交易股票计算均值和标准差。这使模型以"相对排序"而非"绝对数值"为学习目标。

**过滤条件**（构建 sample index 时）：
1. `mask[t, s] == 1`（T+1 日可交易）
2. `listed_days[t, s] >= 365`（上市满1年）
3. `not np.isnan(labels[t, s])`（标签有效）

### 3.3 collate 函数

`dual_stream_collate`：纯 CPU 输出。图谱 Data 对象不在 worker 进程中采集（避免 SnapshotIndex 多进程 pickle 问题），由主进程训练循环根据 `snap_keys` 自行获取。

### 3.4 对齐策略

- **同截面股票**：时序流和空间流的节点按 `stkcd` 对齐
- **不在图谱的股票**：`node_idx = -1` → GAT 编码为零向量（等价于纯 LSTM 预测）
- **不在股票池但在图谱中**：忽略，不影响
- **设备迁移**：图谱 Data 在训练循环中 `.to(device)` 与模型参数对齐

---

## 四、模块二：双流门控融合网络 (`src/model/model.py`)

### 4.1 整体架构

```
X_lstm [B, 20, 16] ──→ LSTM(单层, hidden=64) ──→ H_LSTM [B, 64] ──┐
                                                                    ├──→ Gate → H_fused [B,64] → Linear → Score [B,1]
Graph Data ──→ SpatialGAT(2层, edge_dim=1) ──→ H_GAT [B, 64] ─────┘
```

### 4.2 子模块详解

**TemporalLSTM**：
- 单层单向 LSTM：`input_size=16, hidden_size=64, batch_first=True`
- 取最后时间步隐藏状态 `h_n[-1]` 作为时序表征
- 输出前经过 `Dropout(0.1)`

**SpatialGAT**：
- **第一层** `GATConv(in=4, hidden=64, heads=4, edge_dim=1)` + ELU + Dropout(0.1)
  - 输出维度：`64 × 4 = 256` (concat heads)
- **第二层** `GATConv(in=256, out=64, heads=1, edge_dim=1)`
  - 输出维度：`64` (单头，平均)
- **edge_dim=1**：边权重 `[E, 1]` 作为附加输入嵌入注意力系数计算
- **稀疏图谱回退**：当某快照边数 < `min_edges_for_gat=5` 时，该快照不执行 GAT 前向，所有节点嵌入保持零向量（避免稀疏图 GATConv 产生噪声嵌入）

**GatedFusion**：
```
g = σ(W_g · [H_LSTM || H_GAT] + b_g)     # W_g: [128, 1]
H_fused = g ⊙ H_LSTM + (1 - g) ⊙ H_GAT   # 凸组合
```
- 门控值 g ∈ (0, 1)，用于可解释性：g 接近 1 = 时序占主导，g 接近 0 = 空间占主导

**Prediction Head**：
- `Linear(64, 1)` → 一维 Score

### 4.3 前向传播流程

`DualStreamNet.forward(X_lstm, snap_keys, node_indices, snapshot_graphs)`：

1. LSTM 编码：`H_lstm = self.lstm(X_lstm)` → [B, 64]
2. GAT 编码（按图谱分组 + 节点采样）：
   - 遍历 batch 涉及的所有唯一 snap_key
   - 对每个图谱：`gat_out = self.gat(g.x, g.edge_index, g.edge_attr)` → [N, 64]
   - 按 `node_indices` 从 `gat_out` 中采样，`node_idx=-1` 保持零向量
   - 结果：`H_gat` → [B, 64]
3. 门控融合：`H_fused, gates = self.gated_fusion(H_lstm, H_gat)` → [B, 64], [B, 1]
4. 预测打分：`scores = self.pred_head(H_fused)` → [B, 1]

**训练加速**：提供 `forward_with_precomputed_gat(X_lstm, H_gat)` 接口。每个 epoch 预计算所有图谱的节点嵌入（单次 GAT 前向），batch 内直接查表采样，避免重复 GATConv 前向（加速 3~5x）。

**OOS 推断加速**：`predict_snapshot_embeddings()` 无梯度预计算图谱嵌入，同一 OOS 区间内逐日复用。

### 4.4 损失函数

**Pearson 相关系数损失**（主要使用）：
```python
loss = 1 - corr(scores, labels)  # 最大化排序一致性
```
- 相比 MSE，更关注相对排序而非绝对数值精度
- 适合量化选股"选好股 > 猜准收益"的场景

**组合损失**（备选，在 cloud_upload 版本中使用）：
```python
loss = 0.7 × rank_loss + 0.3 × MSE_loss
```

### 4.5 模型规模

总参数量约 **50,000+**（具体取决于模块维度配置），全部可训练。

---

## 五、模块三：块状滚动回测引擎 (`src/backtest/backtest_engine.py`)

### 5.1 核心逻辑：Expanding Window + In-Band Validation

**锚点**：每年 **4月30日** 和 **8月31日**（与 GAT 快照更新同步）

**以2024年4月30日为例**：
```
训练集: 初始日期(2020-07-01) ~ 2024-02-28          (Expanding, 近4年数据)
验证集: 2024-03-01 ~ 2024-04-30                    (In-Band, 2个月, Early Stopping patience=5)
推断集: 2024-05-01 ~ 2024-08-31                    (OOS, 锁死权重, 逐日滚动预测)
```

### 5.2 训练流程（每个锚点）

1. **构建 DataLoader**：按日期范围切片训练/验证子集，创建 DataLoader
2. **创建模型**：每个锚点从零初始化（不复用历史锚点权重，保持 OOS 纯粹性）
3. **训练循环**：
   - 每 epoch 开始：预计算所有训练集图谱的 GAT 嵌入
   - Batch 内：LSTM 前向 + 查表获取 H_GAT + 门控融合 + 预测 → 计算 Pearson loss
   - 优化器：Adam (lr=1e-3)，梯度裁剪 max_norm=1.0
   - 验证：每 epoch 在验证集上计算 Pearson loss
   - Early Stopping：patience=5，保存最优验证 loss 的模型权重
4. **OOS 推断**：加载最优权重，在该区间的每一日：
   - 用 `get_daily_samples()` 获取当日所有有效股票的因子序列
   - 预计算图谱嵌入（同区间内快照固定不变，复用）
   - 逐日批量前向 → 输出 scores 和 gate_values
5. **清理**：显式删除模型/DataLoader，释放 MPS/CUDA 缓存，确保下一锚点 worker 可启动

### 5.3 输出

`pd.DataFrame` 列：`[trade_date, stkcd, score, gate_value, anchor_date]`

覆盖全回测区间所有 OOS 交易日，每个 (交易日, 股票) 对一条记录。

---

## 六、模块四：组合二次规划优化器 (`src/backtest/portfolio_optimizer.py`)

### 6.1 问题形式化

对每个交易日，求解：
```
max  Σ(w_i × Score_i)
s.t. w_i ≥ 0,  Σw_i = 1
     |w_i - w_bench,i| ≤ δ              (仅成分股，δ=0.02)
     |Σ_{ind_k} w - Σ_{ind_k} w_bench| ≤ η   (η=0.05)
     Σ_{非成分股} w ≤ 0.20
```

### 6.2 三级动态放宽金字塔

当最优问题不可行时，依次放宽约束：

| Level | 个股偏离 | 行业偏离 | 非成分股上限 | 含义 |
|---|---|---|---|---|
| 1 | ±2% | ±5% | ≤20% | 严格指数增强 |
| 2 | ±5% | ±10% | ≤30% | 标准放宽 |
| 3 | ±8% | ±15% | 无上限 | 极度宽松 |
| 4 | ±10% | 无约束 | 无上限 | 最小约束 |

全部失败则回退到基准指数权重。

### 6.3 缺失分数填充

对当天 Active_Mask==1 但缺少预测得分的成分股，填充为全市场有效分数的**中位数**。这样它们以"中庸"姿态参与竞争，避免因极低分数（-1e9）占满约束下界导致不可行。

### 6.4 求解器

优先级：`CLARABEL → ECOS → OSQP → SCS`

### 6.5 输入输出

- 输入：`predictions` DataFrame [trade_date, stkcd, score]
- 输出：`weights` DataFrame [trade_date, stkcd, weight, is_constituent]

---

## 七、模块五：回测报告 (`src/backtest/reporter.py`)

### 7.1 净值计算

核心公式（**注意时序对齐**）：
```
r_p,t = Σ(w_{i,t-1} × r_{i,t})        ← 昨日权重 × 今日收益
cost_t = 0.2% × Σ|w_{i,t} - w_{i,t-1}| ← 双边调仓摩擦成本
r_net,t = r_p,t - cost_t
NAV_t = NAV_{t-1} × (1 + r_net,t)
```

**t=0 特殊处理**：`w[0]` 由 `score[0]` 优化得出，`score[0]` 预测的是 `ret[1]`。因此 t=0 时使用基准收益替代，且不计换手成本（初始建仓不计成本）。`w[0]` 从 t=1 开始才产生策略收益。

### 7.2 绩效指标

| 指标 | 说明 |
|---|---|
| 年化收益率 | `(NAV_end/NAV_start)^{252/T} - 1` |
| 年化波动率 | `std(daily_ret) × sqrt(252)` |
| 夏普比率 | `(年化收益率 - 年化无风险) / 年化波动率` |
| 最大回撤 (MaxDD) | `max(1 - NAV_t / max(NAV_{0..t}))` |
| 超额收益 | 策略年化收益 - 基准年化收益 |
| 信息比率 (IR) | `mean(超额日收益) / std(超额日收益) × sqrt(252)` |
| 胜率 | 超额日收益 > 0 的交易日占比 |
| 换手率 | 日均双边换手 |

### 7.3 输出

- 三张图表：净值对比图、超额收益回撤图、门控可解释性归因图
- 一份 Markdown 报告

---

## 八、已知问题 & 排查方向

### 8.1 当前现象

回测结果"极为稀烂"（用户反馈），需要系统性排查。

### 8.2 可能的问题方向

**数据层面：**
- [ ] 因子截面 Rank 转换是否正确（确保每个交易日独立、无前视偏差）
- [ ] 收益率滞后处理是否正确（Return_1d/5d/20d 滞后1天）
- [ ] 标签 Z-score 标准化是否正确（每日截面独立）
- [ ] Tradable_Mask 与 T+1 对齐是否正确
- [ ] GAT 快照寻址是否正确（向前寻址，不引入未来信息）

**模型层面：**
- [ ] GAT 稀疏图问题：早期截图边数极少（1条边），GATConv 在 1 条边 + 165 节点时是否产生有意义的嵌入？见 `min_edges_for_gat=5` 的回退机制是否生效
- [ ] 门控是否退化：Pearson loss 下门控 g 是否坍塌到常数（如全 0.5 或全 1.0）？检查 gate_value 分布
- [ ] LSTM 是否学到有效信号：Pearson 相关系数损失下，IC 是否为正且显著？
- [ ] 因子特征质量：16个因子对科创50的 Rank IC 分别为多少？是否有因子 IC 接近于 0 或负？

**回测层面：**
- [ ] 锚点训练窗口是否过短（早期锚点训练集仅 1 年+）
- [ ] 每锚点从零初始化是否合理（vs. 持续训练/微调）
- [ ] Early Stopping patience=5 是否过小
- [ ] t=0 的基准收益填充逻辑是否正确

**优化层面：**
- [ ] 组合优化是否过度频繁回退到基准权重？检查 `fallback_benchmark` 占比
- [ ] 0.2% 双边手续费是否对日频调仓过高？
- [ ] 约束（±2% 个股偏离）是否过紧？

### 8.3 快速诊断命令

```bash
# 检查因子 IC
cd src && python -c "
from data.data_module import DualStreamDataset, SnapshotIndex
import numpy as np
snap = SnapshotIndex()
ds = DualStreamDataset(snapshot_index=snap)
# 计算每个因子与 T+1 收益的 Rank IC
"

# 检查 gate 分布
cd src && python -c "
import pandas as pd
preds = pd.read_csv('../result/predictions.csv')
print(preds['gate_value'].describe())
print('gate=0.5占比:', (preds['gate_value'].between(0.49,0.51)).mean())
"

# 检查回退率
cd src && python -c "
import pandas as pd
w = pd.read_csv('../result/optimal_weights.csv')
# 查看是否有 fallback 标记
"
```
