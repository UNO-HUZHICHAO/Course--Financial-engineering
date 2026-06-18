# Jane Street 实时市场预测 —— 项目交付包（复刻用）

本包为课程报告《基于线性与非线性模型对比的金融市场实时预测系统设计与实现》的配套代码与数据，
供复刻报告中的实验结果使用。**本包不含报告正文**，报告另行提交。

## 一、目录结构

```
js_forecast_submit/
├── data/
│   ├── raw/                    # 原始比赛数据（约11GB，Jane Street 官方发布）
│   │   ├── train.parquet/      # 训练数据，按 partition_id=0..9 分区（约4712万行）
│   │   ├── lags.parquet        # 官方滞后（仅 date_id=0；完整滞后由 data_prep 重建）
│   │   ├── test.parquet
│   │   ├── features.csv
│   │   ├── responders.csv
│   │   └── sample_submission.csv
│   └── processed/              # 预处理产物
│       └── validation.parquet  # 含滞后特征的验证集（date_id>1577，约449万行，1.2GB）
│                                #   注：training.parquet 未随包分发，由 data_prep.py 从 raw 重建
├── src/                        # 全部源代码
│   ├── config.py               # 路径/特征/超参/类别映射（单一真相源）
│   ├── utils.py                # 随机种子、加权R²、标准化、类别编码、设备判定
│   ├── data_prep.py            # 原始数据→含滞后特征的训练/验证集 + 标准化统计量
│   ├── ridge_train.py          # Ridge 回归训练（无泄露）
│   ├── xgb_train.py            # XGBoost 训练（4种子，HONEST_TRAIN 无泄露模式）
│   ├── nn_train.py             # MLP 训练（5折日期交叉验证，PyTorch Lightning）
│   ├── tabm_train.py           # TabM 训练（本环境数值不稳定，未纳入最终方案，保留供参考）
│   ├── tabm_reference.py       # TabM 模型架构参考实现
│   ├── evaluate.py             # 单模型对比评估（Ridge/XGBoost/NN 加权R² + 预测 + 全部图表）
│   ├── backtest.py             # 单模型多空策略回测（诚实口径，含 per-bar Sharpe）
│   ├── ensemble.py             # 评估依赖的预测函数库（predict_ridge/xgb/nn/tabm）
│   └── run_all.py              # 一键编排：数据预处理→训练→评估→回测
├── result/
│   ├── metrics/                # 指标 CSV（overall/per_symbol/corr/daily_r2/backtest）
│   └── figures/                # 可视化 PNG（9张，与报告图1-图9对应）
└── README.md
```

> **注**：`result/models/`（训练产出的模型权重）与 `result/predictions/`（逐样本预测 parquet）
> 为中间产物，未随包分发。老师可运行训练脚本重新生成，或仅基于 `result/metrics` 与
> `result/figures` 查阅报告中的数字与图表。

## 二、运行环境

- **硬件**：GPU（单卡，显存≥16GB）+ 64GB 内存 + 50GB 数据盘。本项目实际运行于
  AutoDL（RTX PRO 6000 96GB + 110GB 内存 + Ubuntu 22.04）。
- **软件**：Python 3.12，PyTorch 2.x（CUDA），PyTorch Lightning 2.x，XGBoost，Polars，scikit-learn。
  完整依赖见 `src/` 各脚本 import；TabM 相关依赖（`rtdl_num_embeddings`、`delu`）仅 `tabm_train.py` 需要。

```bash
pip install torch pytorch-lightning xgboost polars pandas scikit-learn matplotlib seaborn scipy dill joblib tqdm
```

## 三、复刻步骤

### 方式 A：完整复刻（从原始数据重训全部模型，约 2.5 小时）

```bash
cd src
python run_all.py
```

`run_all.py` 依次执行：data_prep（重建含滞后特征的训练/验证集）→ xgb_train（4种子，无泄露）
→ nn_train（5折，无泄露，默认 max_epochs=15）→ evaluate（单模型R²+预测+图表）→ backtest（回测）。

> NN 训练最耗时（5折约 1.5-2h）。若仅验证流程可跑通，可设环境变量缩减：
> `NN_MAX_EPOCHS=3 python run_all.py`（精度会下降，仅用于跑通验证）。

### 方式 B：仅复刻评估与回测（跳过训练，使用已有 metrics/figures 验证）

本包已含 `data/processed/validation.parquet`，可直接运行评估与回测复现报告数字：

```bash
cd src
python evaluate.py     # 需先有训练好的模型权重（见下方说明）
python backtest.py
```

> 注意：`evaluate.py` 需加载 `result/models/` 下的模型权重（Ridge.dill、result_seed*.pkl、nn_*.ckpt），
> 这些中间产物未随包分发。如需复跑 evaluate，请先按方式 A 训练生成模型，或仅查阅
> `result/metrics/`、`result/figures/` 中已产出的结果。

### 方式 C：分步执行

```bash
cd src
python data_prep.py        # 1) 建滞后特征 + 标准化统计量（从 raw 重建 processed）
python ridge_train.py      # 2) Ridge（无泄露）
python xgb_train.py        # 3) XGBoost 4种子（HONEST_TRAIN=1 无泄露）
python nn_train.py         # 4) MLP 5折（HONEST_TRAIN=1 无泄露，最耗时）
python evaluate.py         # 5) 单模型R² + 预测 + 图表
python backtest.py         # 6) 多空策略回测 + 回测图表
```

无泄露训练通过环境变量 `HONEST_TRAIN=1` 控制（`run_all.py` 已自动注入）：
XGBoost/NN 仅用 `date_id ≤ 1577` 训练，评估与回测仅在 `date_id > 1577` 上进行。

## 四、关键实验设定（务必阅读）

1. **无数据泄露**：所有模型仅在 `date_id ≤ 1577` 上训练，在 `date_id > 1577`（121个交易日，约449万条）
   上评估与回测。训练与评估在时间上严格分离，结果为模型真实泛化能力的无偏估计。
   主动放弃了原竞赛的 "LB boosting trick"（训练集+验证集合并训练），因其会扭曲回测结论且对本任务贡献极小。

2. **模型体系**：Ridge（线性基线，R²=0.0032）+ XGBoost（主力非线性，R²=0.8596）为主对比；
   MLP 神经网络（R²=0.8541）为扩展实验，仅报告 R²，不参与收益回测。TabM 因本环境训练数值不稳定未纳入。

3. **回测口径（重要）**：`result/metrics/backtest_metrics_honest.csv` 中的"年化收益""Sharpe""最大回撤"
   基于 Jane Street 脱敏目标变量 `responder_6`（量级±5，非真实百分比收益）计算，
   **与实盘金融的 Sharpe/年化不是同一个东西**，仅用于横向比较模型间相对优劣。
   XGBoost 的 Sharpe≈113、胜率100%、回撤0 是"高R²(0.86)→横截面排序近乎完美→每bar多空价差99.93%为正
   →968 bar/天累加放大→逐日全正"的确定结果，是预测精度的体现而非数据泄露。
   详见报告 5.4 节。

## 五、产出说明

- `result/metrics/overall_r2.csv`：各单模型加权 R²
- `result/metrics/per_symbol_r2.csv`：分 symbol 的 R²
- `result/metrics/corr_pearson.csv` / `corr_spearman.csv`：模型预测相关性
- `result/metrics/daily_r2.csv`：逐日 R²
- `result/metrics/backtest_metrics_honest.csv`：单模型多空策略回测指标
- `result/metrics/backtest_daily_honest.csv`：逐日回测收益
- `result/figures/*.png`：9张图，与报告图1-图9一一对应
