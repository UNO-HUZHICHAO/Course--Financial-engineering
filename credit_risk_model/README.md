# 信用风险预测模型

基于 Kaggle [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) 竞赛的信用违约预测方案。

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.x-orange.svg)](https://catboost.ai/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458.svg)](https://pandas.pydata.org/)

---

## 📋 目录

- [项目背景与问题](#-项目背景与问题)
- [快速开始](#-快速开始)
- [架构概览](#-架构概览)
- [数据流程](#-数据流程)
- [模型设计](#-模型设计)
- [运行结果](#-运行结果)
- [文件说明](#-文件说明)
- [配置参数](#-配置参数)

---

## 🎯 项目背景与问题

缺乏信用记录的人（如年轻人、偏好现金交易者）在传统金融体系中往往无法获得贷款。消费金融公司需要借助数据科学准确判断哪些客户能还款、哪些不能，从而让贷款惠及最需要的人群。Home Credit 是一家成立于 1997 年的国际消费金融公司，专注于为缺乏信用记录的人群提供负责任的贷款服务。

### 核心挑战：模型的时间稳定性

实际操作中，客户的还款行为会随时间变化，评分卡（scorecard）需要定期重新开发、验证和部署，这个过程耗时很长。一个模型如果在部署后预测性能突然下降，意味着平均而言贷款会发放给更差的客户——但等到首批贷款的还款日才发现问题，已经太晚了。

因此竞赛的核心目标不仅是**预测违约准确率**（AUC），更关键的是**模型在不同时间窗口的稳定性**——在性能与稳定之间找到平衡。

### 本项目方案

使用 LightGBM + CatBoost 双模型、3 折交叉验证、投票融合，基于基础表 + 征信局静态表 + 其他静态表三张核心表，构建信用违约预测模型。

---

## 🚀 快速开始

### 环境要求

```bash
# Python 3.12+
pip install pandas numpy scikit-learn lightgbm catboost matplotlib
```

### 一键运行

```bash
cd /Users/huzhichao/Desktop/credit_risk_model
python simple_version/main_simple.py
```

程序会执行完整流程：**加载数据 → 预处理 → 训练模型 → 预测 → 输出结果**。

---

## 🏗 架构概览

```
simple_version/
├── main_simple.py          ← 主入口，编排完整流程
├── config_simple.py        ← 所有参数集中配置
├── preprocessing_simple.py ← 数据加载、清洗、特征工程
├── model_simple.py         ← 模型训练、融合、评估
├── visualization.py        ← 可视化图表生成
└── results/                ← 输出结果（自动生成）
    ├── submission.csv          ← 提交文件
    ├── feature_importance.csv  ← 特征重要性
    ├── results_summary.txt     ← 结果摘要
    ├── model_comparison.png    ← 模型对比图
    ├── feature_importance.png  ← 特征重要性图
    ├── prediction_distribution.png ← 预测分布图
    └── target_distribution.png     ← 目标分布图
```

---

## 📊 数据流程

### 数据表合并

程序加载 **3 张静态表**（depth=0），通过 `case_id` 左连接到基础表：

```
train_base.csv (含 target)
    ├── LEFT JOIN on case_id ── train_static_cb_0.csv  (征信局静态数据)
    └── LEFT JOIN on case_id ── train_static_0_0.csv   (其他静态数据)
```

### 预处理管线

原始数据中存在大量噪声和缺失，需要经过 5 个步骤才能送入模型：

```
原始数据 (500+ 列, 大量缺失)
  │
  ├─[1] 过滤高缺失列 (缺失率 > 95%)
  │      如果一列中 95% 的值都是空的，存留的信息不足以支撑预测
  │
  ├─[2] 过滤低效类别列 (唯一值 > 100 或 = 1)
  │      唯一值过多 → One-Hot 后特征爆炸
  │      唯一值为 1 → 对区分客户毫无帮助
  │
  ├─[3] 日期特征转换 (绝对日期 → 相对天数)
  │      "2024-03-01" 换为 "距决策日 -45 天"
  │      避免模型记住特定日期，而非学习客户行为模式
  │
  ├─[4] 缺失值填充
  │      数值列: 中位数（比均值更抗极端值）
  │      类别列: "missing"（明确标记为缺失状态）
  │
  └─[5] 类别类型转换 (object → category)
         让 LightGBM/CatBoost 直接解析类别语义

处理后 (约 158 列特征)
```

### 特征后缀说明

Home Credit 使用特征后缀约定区分类型：

| 后缀 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `P` | 数值 | 个人属性 | `main_income_P` |
| `A` | 数值 | 申请相关 | `annuity_780A` |
| `D` | 日期 | 日期字段 | `birth_259D` |
| `M` | 类别 | 类别特征 | `employment_type_M` |
| `T` | 时间 | 时间特征 | `approvaldate_319D` |
| `L` | 其他 | 其他特征 | `days360_512L` |

---

## 🧠 模型设计

### 整体方案：双模型 + 交叉验证 + 投票融合

单模型可能学偏、单折可能运气好。通过组合多个不同训练来源的模型，降低单个模型的随机性：

```
训练数据 (1,526,659 条)
  │
  ├─ Fold 1 ──┬─ LightGBM ──┐
  │           └─ CatBoost ──┤
  ├─ Fold 2 ──┬─ LightGBM ──┤
  │           └─ CatBoost ──┼── VotingModel (6 个模型)
  └─ Fold 3 ──┬─ LightGBM ──┤       │
              └─ CatBoost ──┘       │
                                    ▼
                              预测概率取平均
                                  │
                                  ▼
                             submission.csv
```

### 为什么选 LightGBM + CatBoost

| 特性 | LightGBM | CatBoost |
|------|----------|----------|
| 树生长策略 | Leaf-wise（优先分裂增益大的叶子） | Symmetric（对称树） |
| 类别特征 | 直接支持 category 类型 | Ordered Target Statistics（排序编码） |
| 过拟合控制 | GOSS 采样（保留大梯度样本） | Ordered Boosting |
| 速度 | ⚡ 快 | 🐢 较慢 |
| 默认效果 | 需调参 | 开箱即用 |

两种模型内部机制不同，犯的错误也不同。将它们的预测取平均后，各自的偏差会互相抵消一部分，这正是集成学习降低方差的核心思想。

### 交叉验证策略 · StratifiedKFold

- **3 折**分层划分，每折严格保持违约/非违约比例一致（~3.14%）
- 如果违约客户太少（仅 3%），随机划分可能导致某折没有违约样本
- 每折训练 1 个 LightGBM + 1 个 CatBoost，共 **6 个基模型**
- 使用 `early_stopping=50`：验证集 AUC 连续 50 轮不涨就停训

---

## 📈 运行结果

> 运行日期：2026-05-12 | 训练样本：1,526,659 条 | 特征数：158 | 违约率：3.14%

### 模型性能

| 指标 | LightGBM | CatBoost |
|------|----------|----------|
| Fold 1 AUC | 0.7668 | 0.7582 |
| Fold 2 AUC | 0.7645 | 0.7563 |
| Fold 3 AUC | 0.7664 | 0.7586 |
| **平均 AUC** | **0.7659** | **0.7577** |
| 标准差 | ±0.0010 | ±0.0010 |

关键发现：
- LightGBM 略优于 CatBoost（高约 0.008），且两者趋势一致
- **三折 AUC 标准差仅 ±0.001**，说明模型在不同数据切分上表现一致，具有一定的稳定性
- 融合模型预期 AUC 约在 0.7618（两者平均），虽略低于单一最强模型，但**更稳定、更抗漂移**

### 预测统计

| 指标 | 值 |
|------|-----|
| 测试样本数 | 10 |
| 预测概率范围 | [0.0086, 0.2860] |
| 预测概率均值 | 0.0863 |
| 预测概率中位数 | 0.0684 |

预测分布偏右（均值 > 中位数），说明存在少量高风险客户拉高了平均分，这在风控场景中有参考价值。

### 特征重要性 Top 10

| 排名 | 特征 | 业务含义 | 重要性 |
|------|------|----------|--------|
| 1 | `annuity_780A` | 月供还款金额 | 170.34 |
| 2 | `days360_512L` | 360天相关派生特征 | 165.13 |
| 3 | `price_1097A` | 商品/服务价格 | 162.69 |
| 4 | `pmtssum_45A` | 历史还款总额 | 155.42 |
| 5 | `pmtnum_254L` | 历史还款期数 | 153.82 |
| 6 | `lastcancelreason_561M` | 上次申请取消原因 | 133.37 |
| 7 | `credamount_770A` | 授信金额 | 127.32 |
| 8 | `totalsettled_863A` | 已结算总金额 | 123.10 |
| 9 | `thirdquarter_1082L` | 第三季度相关指标 | 118.83 |
| 10 | `lastapprcommoditycat_1041M` | 上次审批商品类别 | 116.52 |

> 🔍 最关键的三个信号：**月供金额**（还款压力）、**还款历史**（行为惯性）、**上次取消原因**（意愿信号）——这与风控业务直觉高度一致。

### 可视化图表

程序运行后自动在 `results/` 目录生成：

| 图表 | 文件 | 内容 |
|------|------|------|
| 模型对比 | `model_comparison.png` | 各折 AUC 柱状对比 + 平均 AUC（含标准差） |
| 特征重要性 | `feature_importance.png` | Top 20 重要特征水平柱状图 |
| 预测分布 | `prediction_distribution.png` | 预测概率直方图 + 箱线图 |
| 目标分布 | `target_distribution.png` | 训练集违约/非违约分布饼图（可见严重不平衡） |

---

## 📁 文件说明

| 文件 | 职责 |
|------|------|
| `main_simple.py` | 主程序入口，串联 12 步完整流程 |
| `config_simple.py` | 配置中心（路径、模型参数、阈值全部集中） |
| `preprocessing_simple.py` | 数据加载合并 + 5 步预处理管线 |
| `model_simple.py` | VotingModel 融合类 + 交叉验证训练 + 预测 + 评估 |
| `visualization.py` | matplotlib 图表生成（自动适配 macOS 中文字体） |

### 输入数据

```bash
csv_files/
├── train/
│   ├── train_base.csv          # 基础表（含目标变量 target）
│   ├── train_static_cb_0.csv   # 征信局静态表（信用评分等）
│   └── train_static_0_0.csv    # 其他静态表（申请信息等）
└── test/
    ├── test_base.csv
    ├── test_static_cb_0.csv
    └── test_static_0_0.csv
```

---

## ⚙️ 配置参数

### 模型超参数 (config_simple.py)

```python
# LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# CatBoost
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
    "random_seed": 42,
}
```

### 数据处理参数

| 参数 | 值 | 作用 |
|------|-----|------|
| `NULL_THRESHOLD` | 0.95 | 缺失率 > 95% 的列直接删除 |
| `CAT_UNIQUE_THRESHOLD` | 100 | 唯一值 > 100 的类别列删除 |
| `N_SPLITS` | 3 | 交叉验证折数 |
| `EARLY_STOPPING_ROUNDS` | 50 | 验证集 50 轮不涨即停训 |

---

## 📝 参考资料

- [Kaggle Competition - Home Credit Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
- [LightGBM 官方文档](https://lightgbm.readthedocs.io/)
- [CatBoost 官方文档](https://catboost.ai/en/docs/)
- [Scikit-learn StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [World Bank - Global Findex Database (无银行账户人口统计)](https://globalfindex.worldbank.org/)
