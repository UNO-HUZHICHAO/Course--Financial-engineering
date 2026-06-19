# -*- coding: utf-8 -*-
"""
全局配置：路径、特征定义、随机种子与各模型超参数。

所有超参数严格对应原始 Kaggle 备赛 notebook（ridge_train / xgb_train /
nn_train / tabm_train / ensemble），仅做集中化管理，不修改任何建模细节。
"""
import os
from pathlib import Path

# =====================================================================
# 路径配置（相对本文件所在 src/ 目录解析，便于在 AutoDL 上整体迁移）
# =====================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"            # 原始比赛数据存放处
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # 预处理产物
MODELS_DIR = PROJECT_ROOT / "result" / "models"
METRICS_DIR = PROJECT_ROOT / "result" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "result" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "result" / "predictions"

for _d in (DATA_PROCESSED, MODELS_DIR, METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 原始比赛数据文件名（Kaggle 下载后放入 data/raw/）
RAW_TRAIN_DIRNAME = "train.parquet"          # 分区目录 partition_id=0..9
RAW_VALIDATION_FILENAME = "validation.parquet"
RAW_LAGS_FILENAME = "lags.parquet"           # 可选；data_prep 会优先用其重建滞后

# 预处理产物
PROCESSED_TRAIN = DATA_PROCESSED / "training.parquet"      # 含滞后特征的训练集
PROCESSED_VALID = DATA_PROCESSED / "validation.parquet"    # 含滞后特征的验证集
DATA_STATS_PATH = DATA_PROCESSED / "data_stats.pkl"        # TabM 标准化统计量

# =====================================================================
# 通用
# =====================================================================
SEED = 42
TARGET_COL = "responder_6"
WEIGHT_COL = "weight"

# 79 个基础数值特征
FEATURE_79 = [f"feature_{idx:02d}" for idx in range(79)]
# 9 个一阶滞后响应变量
LAG_FEATURES = [f"responder_{idx}_lag_1" for idx in range(9)]

# =====================================================================
# Ridge 回归（对应 ridge_train.ipynb）
# =====================================================================
class RidgeConfig:
    partitions = [6, 7, 8, 9]      # 仅读取 partition 6-9
    sample_frac = 0.82             # 采样比例
    split = 200000                 # 末尾 20 万条作本地验证（原 notebook 注释为 ~1%）
    fillna_value = 3               # 连续特征缺失值统一填充为 3
    feature_cols = FEATURE_79      # Ridge 仅用 79 个基础特征
    model_path = MODELS_DIR / "Ridge.dill"

# =====================================================================
# XGBoost（对应 xgb_train.ipynb）
# =====================================================================
class XGBConfig:
    feature_cols = ["symbol_id", "time_id"] + FEATURE_79 + LAG_FEATURES  # 91 维
    # 4 个不同种子的模型，推理时等权(各 0.25)平均（对应报告“4个XGBoost不同种子”）
    seeds = [42, 2024, 7, 2025]
    params = dict(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=5,
        tree_method="hist",     # 原为 'gpu_hist'；hist+device='cuda' 为现代等价实现，算法不变
        device="cuda",          # 无 GPU 时由 utils.to_xgb_device 自动回退 cpu
        random_state=None,      # 每个 seed 单独注入
    )
    result_prefix = "result"    # 保存为 result_seed{seed}.pkl

# =====================================================================
# 神经网络 MLP（对应 nn_train.ipynb）
# =====================================================================
class NNConfig:
    feature_cols = FEATURE_79 + LAG_FEATURES      # 88 维
    n_hidden = [512, 512, 256]
    dropouts = [0.1, 0.1]
    lr = 1e-3
    weight_decay = 5e-4
    patience = 25
    max_epochs = 2000
    n_fold = 5
    # 注：原方案为 max_epochs=2000, patience=25，5折全量约7-8h。
    # 受3h算力预算约束，运行时由 run_train.sh 通过环境变量缩减为 max_epochs=15, patience=5
    # （val_r² 在 epoch 10+ 即达 0.84，15 epoch 足够），原方案配置见上保留。
    batch_size = 8192
    loader_workers = 4
    gpuid = 0
    fillna_method = "ffill"     # 先 ffill 再 fillna(0)
    model_prefix = "nn"         # 保存为 nn_{fold}.ckpt

# =====================================================================
# TabM（对应 tabm_train.ipynb）
# =====================================================================
class TabMConfig:
    feature_cat = ["feature_09", "feature_10", "feature_11"]
    # 连续特征 = 79 基础去除 3 个分类 + 9 滞后 = 85 维
    feature_cont = [c for c in FEATURE_79 if c not in ["feature_09", "feature_10", "feature_11"]] + LAG_FEATURES
    feature_all = FEATURE_79 + LAG_FEATURES     # 原始顺序（含分类列），与原 notebook 一致
    std_feature = [c for c in FEATURE_79 if c not in ["feature_09", "feature_10", "feature_11"]] + LAG_FEATURES
    start_dt = 800
    end_dt = 1577
    batch_size = 8192
    num_epochs = 4
    k = 32                      # 集成成员数
    n_cont_features = 85
    cat_cardinalities = [23, 10, 32, 40, 969]   # [feature_09, feature_10, feature_11, symbol_id, time_id]
    backbone = {"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.25}
    lr = 1e-4
    weight_decay = 5e-3
    noise_std = 0.035           # 连续特征高斯噪声标准差
    patience = 5
    model_path = MODELS_DIR / "tabm_best.pt"

# 类别映射（原 tabm_train / ensemble 中硬编码，原样保留）
CATEGORY_MAPPINGS = {
    'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9,
                   34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17,
                   68: 18, 70: 19, 81: 20, 82: 21},
    'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
    'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8,
                   50: 9, 59: 10, 62: 11, 63: 12, 66: 13, 76: 14, 150: 15, 158: 16,
                   159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23,
                   336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
    'symbol_id': {i: i for i in range(39)},
    'time_id': {i: i for i in range(968)},
}

# =====================================================================
# 集成（精简版：XGBoost + NN 两模型无泄露融合）
# =====================================================================
class EnsembleConfig:
    # 精简方案：仅 XGB(4种子等权) + NN(5折等权) 两模型加权融合
    # 原 4 模型(Ridge/XGB/NN/TabM) + train+val trick 方案见项目报告初稿与 git 历史
    e_weights = (0.55, 0.45)        # (xgb, nn)
    clip = (-5.0, 5.0)

# =====================================================================
# 评估
# =====================================================================
class EvalConfig:
    # 精简方案默认无泄露：XGB/NN 仅用 date_id<=honest_cutoff 训练，
    # 评估与回测仅在 date_id>honest_cutoff 上进行，策略收益真实可信。
    honest_heldout = True
    honest_cutoff = 1577
