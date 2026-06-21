# 科创50指数增强策略 — 完整建模算法说明

> 版本日期: 2026-05-27  
> 描述: 基于大语言模型知识图谱与双流异步门控网络的科创50指数增强策略，  
> 包含异构数据对齐、双流融合模型、块状滚动回测、分层等权增强等完整管线。

---

## 1. 整体架构概览

```
原始金融数据
    │
    ├── 日频量价因子（16维）──────► LSTM 时序编码 ──────┐
    │   (Return_1d, IVOL_20, ...)                       │
    │                                                   ├──► GatedFusion ──► Score
    ├── 半年度产业链图谱 ──────────► GAT 空间编码 ──────┘       (门控融合)
    │   (LLM提取边权重 + 节点属性)
    │
    ▼
块状滚动回测 (Expanding Window + 10锚点)
    │
    ▼
双策略组合优化
    ├── layered_equal_weight: 基准底仓 + Top60分层等权 (主结果)
    └── qp_conservative: 稳健型二次规划 (对比基线)
    │
    ▼
绩效评价与可视化报告
```

**核心创新点：**
- 双流异步融合：LSTM（技术面时序）与 GAT（基本面图谱）通过自适应门控动态融合
- Gate Floor + Residual 机制：防止门控塌缩到单一模态
- 分层等权增强：用低换手、低 active-share 的组合构建保护弱信号
- 块状滚动回测：每半年重新训练，OOS 区间锁死权重

---

## 2. 数据模块 (`src/data/data_module.py`)

### 2.1 数据来源

| 数据类别 | 内容 | 维度 |
|----------|------|------|
| 日频量价因子 | 16个技术因子 (Return_1d/5d/20d, IVOL_20, MAX_20, ILLIQ_20, ...) | [1557天, 599股, 16] |
| 日收益率矩阵 | Daily_Returns.csv | [1557天, 599股] |
| 可交易掩码 | Tradable_Mask.csv (停牌/ST/新股过滤) | [1557天, 599股] |
| 指数权重 | KCB50_Index_Weight.csv (科创50成分股权重) | 月末 |
| 指数行情 | KCB50_Index_Daily.csv (基准收益率) | 日频 |
| GAT 图谱快照 | 10期半年度快照 (2021Q2 ~ 2025Q4)，节点+边CSV | 每期 ~100-600节点 |
| 无风险利率 | Shibor 1年期日化 | 日频 |

### 2.2 SnapshotIndex — GAT 快照索引

**职责：** 加载全部 10 期 PyG Data 对象，维护 stkcd → node_idx 映射，提供按交易日向过去寻址最近快照的接口。

**关键方法：**
- `get_snapshot(trade_date)` → 返回不晚于 trade_date 的最近快照键及其 PyG Data
- `get_node_idx(snap_key, stkcd)` → 返回某股票在某快照中的节点索引
- `get_graph(snap_key)` → 返回指定快照的 PyG Data 对象

### 2.3 DualStreamDataset — 双流数据集

**设计：** 继承 PyTorch Dataset，每个样本对应一个 `(交易日, 股票)` 对。

**单样本返回：**
```
(X_lstm: [20,16], snap_key: str, node_idx: int, label: float, mask: int8, trade_date: str, stkcd: str)
```

**数据预处理管线：**
1. `_load_factor_cube()` — 16个 .pkl 因子文件堆叠为 [T, N, F] float32 立方体
2. `_load_labels_and_mask()` — 标签 = T+1 日收益率截面 Z-score；掩码 = 可交易标记
3. `_load_stock_info()` — 股票上市日期，过滤上市不满 365 天的新股
4. `_normalize_factors()` — 截面 Z-score 标准化（仅用 mask==1 且 listed_days>=365 的股票计算均值/标准差）
5. `_build_sample_index()` — 构建有效样本索引，过滤条件：日期范围 + seq_len 充足 + mask==1 + 上市满1年 + 标签非 NaN

**有效样本数：** 479,286 (2020-07-01 ~ 2025-12-31)

### 2.4 序列长度与回测区间

| 参数 | 值 | 说明 |
|------|-----|------|
| `SEQ_LEN` | 20 天 | 过去 20 个交易日的因子序列 |
| `BACKTEST_START` | 2020-07-01 | 回测起始日 |
| `BACKTEST_END` | 2025-12-31 | 回测结束日 |

---

## 3. 模型架构 (`src/model/model.py`)

### 3.1 TemporalLSTM — 日频量价序列编码器

```
输入: X_lstm [B, 20, 16]
架构: 单层 LSTM (hidden=64, batch_first=True, dropout=0.1)
输出: H_LSTM [B, 64] (最后时间步隐藏状态)
```

**设计理由：** 16 维因子 × 20 日时序 → LSTM 捕捉因子间的短期动量与交互效应。

### 3.2 SpatialGAT — 产业链图谱空间编码器

```
输入: 节点特征 [N, 4], 边索引 [2, E], 边权重 [E, 1]
架构:
  Conv1: GATConv(4 → 64, heads=4, edge_dim=1) → ELU → Dropout
  Conv2: GATConv(256 → 64, heads=1, edge_dim=1)
输出: H_GAT [N, 64] (节点表征矩阵)
```

**设计理由：** 双层图注意力网络，将 LLM 提取的产业链边权重注入注意力计算。第一层使用 4 头注意力捕获多种关系模式，第二层压缩为统一表征。

**防御逻辑：** 若某快照边数 < min_edges_for_gat (5)，跳过 GAT 前向，该快照对应样本的 H_GAT 保持零向量（退化到纯 LSTM 预测）。

### 3.3 GatedFusion — 自适应门控融合 (关键升级)

这是本策略最核心的改进模块。

```
输入: H_LSTM [B, 64], H_GAT [B, 64]

步骤:
  1. 投影层:
     h_lstm_proj = Linear_64(LSTM表征)
     h_gat_proj  = Linear_64(GAT表征)

  2. 门控计算 (双层 MLP):
     concat = [h_lstm_proj || h_gat_proj]  → [B, 128]
     raw_gate = σ(Linear_64(GELU(Linear_128(concat))))  → [B, 1]

  3. Gate Floor 防塌缩:
     gate = gate_floor + (1 - 2×gate_floor) × raw_gate
     其中 gate_floor = 0.20
     → gate ∈ [0.20, 0.80]，永不完全偏向单流

  4. GAT 有效性检测:
     若 ||h_gat|| ≈ 0 (该样本无有效图谱) → gate = 1.0 (纯 LSTM)

  5. GAT Residual 残差注入:
     residual_scale = σ(trainable_logit)  ← 可学习参数
     h_fused = LayerNorm(
         gate × h_lstm_proj
       + (1-gate) × h_gat_proj
       + residual_scale × h_gat_proj    ← 残差注入
     )

输出: H_fused [B, 64], gate [B, 1]
```

**升级要点：**

| 改动 | 旧版 | 新版 |
|------|------|------|
| 门控网络 | 单层 Linear(128→1) | 双层 MLP(128→64→1) + GELU |
| LSTM/GAT 投影 | 无 | 各加一个 Linear 投影层 |
| Gate Floor | 无 | gate ∈ [0.20, 0.80] |
| GAT Residual | 无 | 可学习残差注入 |
| 图层归一化 | 无 | LayerNorm 稳定训练 |
| 无图谱回退 | 零向量混入融合 | 自动退化纯 LSTM |

### 3.4 DualStreamNet — 顶层容器

```
class DualStreamNet:
  - lstm: TemporalLSTM
  - gat: SpatialGAT
  - gated_fusion: GatedFusion
  - pred_head: Linear(64→1)

forward():
  1. H_lstm  = lstm(X_lstm)
  2. H_gat   = _gat_gather(Graph, node_indices)  ← 按节点采样
  3. H_fused, gates = gated_fusion(H_lstm, H_gat)
  4. scores  = pred_head(H_fused)

OOS 推断优化:
  predict_snapshot_embeddings() → 预计算所有图谱节点嵌入
  forward_with_precomputed_gat() → 跳过 GATConv，直接查表
```

### 3.5 损失函数

**combined_loss(scores, labels, masks, gates, rank_weight=0.7, gate_reg_weight=0.02):**

```
loss = rank_weight × (1 - PearsonCorr(scores, labels))
     + (1 - rank_weight) × MSE(scores, labels)
     + gate_reg_weight × [
         (gate_mean - 0.60)²                          ← 鼓励 gate 均值靠近 0.60
       + 0.5 × ReLU(0.08 - gate_std)²                 ← 惩罚 gate 方差过低 (塌缩)
       ]
```

**设计理由：**
- Pearson 相关系数损失直接优化排序一致性（Rank IC 导向）
- MSE 提供梯度稳定性
- Gate 正则防止训练早期门控直接塌缩为"几乎永远只看 LSTM"或"完全平均"

---

## 4. 块状滚动回测引擎 (`src/backtest/backtest_engine.py`)

### 4.1 锚点设计

每年两个锚点（4/30 和 8/31），与 GAT 快照更新同步：

```
锚点日        训练截止      验证窗口             OOS 推断窗口
2021-04-30    2021-02-26    2021-03-01~04-30    2021-05-06~08-30
2021-08-31    2021-06-30    2021-07-01~08-31    2021-09-01~2022-04-29
2022-05-05    2022-02-28    2022-03-01~05-05    2022-05-06~08-30
...
2025-09-01    2025-06-30    2025-07-01~09-01    2025-09-02~12-30
```

共 10 个锚点。

### 4.2 Expanding Window 训练

```
Anchor 1: Train=[2020-07, 2021-02-26]          7K 样本
Anchor 2: Train=[2020-07, 2021-06-30]         15K 样本
...
Anchor 10: Train=[2020-07, 2025-06-30]       408K 样本
```

**关键设计：** 训练集始终从回测起始日开始，随着锚点推进不断扩张（Expanding Window），确保不丢失早期数据中的模式。

### 4.3 训练流程

```
每个锚点:
  1. 构建训练/验证 DataLoader (按日期切片)
  2. 创建新模型 (随机初始化)
  3. 预收集图谱数据 (避免 PyG 多进程 pickle 问题)
  4. 训练循环:
     - Adam optimizer (lr=1e-3)
     - combined_loss (70% Rank + 30% MSE + gate_reg)
     - clip_grad_norm (max_norm=1.0)
     - Early Stopping (patience=5)
  5. 加载最优模型 → OOS 推断 (锁死权重，逐日滚动)
  6. 清理 GPU 缓存，进入下一锚点
```

### 4.4 OOS 推断优化

- 同一 OOS 区间内 GAT 快照固定不变 → **预计算所有图谱节点嵌入**
- 逐日批量前向: `get_daily_samples()` 直接从 numpy cube 提取，避免逐条 `__getitem__`
- LSTM + Fusion + PredHead 一步完成，GATConv 只在每锚点开始时执行一次

---

## 5. 组合优化 (`src/backtest/portfolio_optimizer.py`)

### 5.1 策略一：layered_equal_weight (主结果，推荐)

```
最终权重 = (1 - active_share) × w_benchmark + active_share × w_layered

其中:
  active_share = 0.10 (主动偏离10%)
  w_layered = Top60 股票按三层分配:
    Tier1 (Top20): 40% × 0.10 / 20 = 0.20% per stock
    Tier2 (21-40): 35% × 0.10 / 20 = 0.175% per stock
    Tier3 (41-60): 25% × 0.10 / 20 = 0.125% per stock
```

**设计逻辑：**
- 90% 基准底仓保证跟踪指数
- 10% 主动仓位按模型打分分层等权 → 避免 QP 的极端集中
- 双周调仓 (T=10) → 低换手
- 10 日分数平滑 → 过滤日间噪声

**参数配置：**

| 参数 | 值 | 说明 |
|------|-----|------|
| tier_sizes | (20, 20, 20) | 三层各20只 |
| tier_allocations | (0.4, 0.35, 0.25) | 自上而下递减 |
| active_share | 0.10 | 主动偏离10% |
| smoothing_window | 10 | 10日分数平滑 |
| rebalance_freq | 10 | 双周调仓 |
| constituent_only | True | 仅成分股 |

### 5.2 策略二：qp_conservative (对比基线)

稳健型二次规划，逐日求解：

```
目标函数:
  max Σ(w_i × score_i) - λ_track × tracking_error - λ_turn × turnover

四级约束金字塔:
  Level1_严格增强: 个股±2%, 行业±5%, 非成分股≤5%
  Level2_标准增强: 个股±3%, 行业±8%, 非成分股≤8%
  Level3_温和放宽: 个股±5%, 行业±10%, 非成分股≤10%
  Level4_保底放宽: 个股±6%, 行业±12%, 非成分股≤12%
  
  Fallback: 基准指数权重

求解器优先级: CLARABEL → ECOS → OSQP → SCS
```

**配置：** 周频调仓 (T=5)，3 日分数平滑。

---

## 6. 绩效报告 (`src/backtest/reporter.py`)

### 6.1 净值计算

```
r_portfolio[t] = Σ(w_i[t-1] × r_i[t])
cost[t] = 0.2% × Σ|w_i[t] - drifted_w_i[t]|    (仅调仓日)
r_net[t] = r_portfolio[t] - cost[t]
NAV[t] = NAV[t-1] × (1 + r_net[t])
```

**时序对齐：** w[t] 由 score[t] 优化得出，score[t] 预测 ret[t+1]。w[t] 在 t+1 日产生收益。t=0 日使用基准收益初始化。

### 6.2 核心指标

| 指标 | 公式 |
|------|------|
| 年化收益率 | (1+Π(1+r))^(252/n) - 1 |
| 年化超额 (Alpha) | r_strategy - r_benchmark |
| 超额最大回撤 | min((cumexcess - running_max)/running_max) |
| 跟踪误差 | std(excess_ret) × √252 |
| 信息比率 | mean(excess_ret)/std(excess_ret) × √252 |
| 夏普比率 | mean(r_strategy - r_rf)/std(r_strategy - r_rf) × √252 |
| 年化换手率 | Σturnover / years |

### 6.3 三张图表

1. **净值走势对比** — 策略 vs 科创50基准，含回撤阴影
2. **超额收益与回撤** — 对数超额累计 + 超额最大回撤标注
3. **门控权重归因** — 全市场 gate 均值 MA20 时序，标注量价/基本面主导期

---

## 7. 端到端运行流程

```bash
# 1. 上传代码到云端 (已集成在 deploy_to_autodl.py/run_on_autodl.py)
# 2. 环境配置
bash setup_cloud.sh

# 3. 运行完整回测
python3 src/run_backtest.py --epochs 100

# 4. 输出产物
result/
  ├── predictions.csv              # 模型预测分数
  ├── optimal_weights.csv          # 主结果权重
  ├── backtest_report.md           # 主结果报告
  ├── strategy_comparison.csv      # 双策略对比
  ├── chart1_nav_comparison.png
  ├── chart2_excess_return.png
  ├── chart3_gate_attribution.png
  ├── layered_equal_weight/        # 分层等权子结果
  └── qp_conservative/             # 稳健QP子结果

# 5. 仅重建组合结果 (已有 predictions.csv 时无需重训)
python3 src/analysis/rebuild_strategy_results.py
```

---

## 8. 扩展方向

1. **GAT 节点基本面特征**：目前 GAT 节点仅 4 维 LLM 提取属性。可将 ROE、毛利率、研发费用率、营收增速等财报数据作为节点特征，使 GAT 分支具备真实的基本面信息。

2. **更多因子**：当前 16 个因子均为量价技术因子。可加入资金流向（北向资金、主力净流入）、波动率曲面、期权隐含信息等。

3. **损失函数调优**：可尝试 ListNet/LambdaRank 等 learning-to-rank 损失，或引入 IC 加权让高 IC 日权重更大。

4. **调仓频率自适应**：根据市场波动率动态调整调仓频率，高波动时降频以减少交易成本。

5. **多周期预测融合**：同时预测 T+1、T+5、T+20 收益，不同预测周期用不同组合约束。
