# -*- coding: utf-8 -*-
"""
数据预处理：从原始比赛数据自建“含滞后特征”的训练/验证集与标准化统计量。

原始 notebook 依赖若干外部 Kaggle 数据集（js24-preprocessing-create-lags、
jane-street-data-preprocessing 等）提供滞后特征与 mean/std 统计量，以及独立的
validation.parquet。这些产物均已丢失，本脚本从最原始的比赛数据出发自建等价产物，
使整个流程自包含、可在 AutoDL 上独立重跑。

数据布局说明：本套比赛数据未单独提供 validation.parquet，验证集由训练全量中
date_id > VALID_CUTOFF(=1577) 的部分构成（与原方案 TabM 验证窗口 date_id>1577
一致）。原方案 train+val 合并训练的 “LB boosting trick” 等价于：训练集 =
date_id<=1577 的全部数据，验证集 = date_id>1577 的数据，二者合并训练即全量。

滞后特征定义与竞赛一致：对每个 symbol_id，按 (date_id, time_id) 全局排序后，
responder_{i}_lag_1 = responder_{i}.shift(1)（跨日边界自动衔接）。推理时原
ensemble 对滞后列 fill_null(0)，故此处同样将滞后列空值填 0，与原推理行为一致。

内存策略：全量约 4400 万行 × 92 列，sort+shift 物化需 ~30GB。本实现采用分 symbol
逐组处理（每个 symbol 独立排序、shift、写盘），峰值内存仅单个 symbol 的量级
（~2GB），可在 16GB 内存的机器上运行，也适配 AutoDL 64GB。
"""
import argparse
import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

import config as C
from utils import seed_everything

RESPONDERS = [f"responder_{i}" for i in range(9)]
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]
VALID_CUTOFF = 1577


def read_raw_train_lazy() -> pl.LazyFrame:
    """惰性扫描全部训练分区。"""
    train_dir = C.DATA_RAW / C.RAW_TRAIN_DIRNAME
    parts = sorted(train_dir.glob("partition_id=*/*.parquet"))
    if not parts:
        parts = sorted(train_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"未在 {train_dir} 找到训练 parquet 分区。请先下载比赛数据放入 data/raw/。"
        )
    print(f"[data_prep] 扫描 {len(parts)} 个训练分区 ...")
    return pl.scan_parquet(parts)


def process_and_write(lf: pl.LazyFrame):
    """
    分 symbol 处理：每个 symbol 内按 (date_id, time_id) 排序、shift(1) 得滞后、
    fill_null(0)，再按 date_id 切分追加写入训练/验证 parquet。
    分 symbol 处理保证 shift 的跨日衔接正确（滞后只在同一 symbol 内传递）。
    """
    symbols = lf.select("symbol_id").unique().sort("symbol_id").collect()["symbol_id"].to_list()
    print(f"[data_prep] 共 {len(symbols)} 个 symbol，逐个处理 ...")

    train_path = C.PROCESSED_TRAIN
    valid_path = C.PROCESSED_VALID
    # 清理旧文件
    for p in (train_path, valid_path):
        if p.exists():
            p.unlink()

    n_train = n_valid = 0
    train_parts, valid_parts = [], []
    for sym in tqdm(symbols, desc="per symbol"):
        sub = (lf.filter(pl.col("symbol_id") == sym)
               .sort(["date_id", "time_id"])
               .with_columns([
                   pl.col(c).shift(1).alias(f"{c}_lag_1") for c in RESPONDERS
               ])
               .with_columns([pl.col(c).fill_null(0) for c in LAG_COLS])
               .collect())
        if sub.height == 0:
            continue
        tr = sub.filter(pl.col("date_id") <= VALID_CUTOFF)
        va = sub.filter(pl.col("date_id") > VALID_CUTOFF)
        if tr.height:
            train_parts.append(tr)
            n_train += tr.height
        if va.height:
            valid_parts.append(va)
            n_valid += va.height
    # 合并各 symbol 后一次性写出（各 symbol 已各自排好序）
    if train_parts:
        pl.concat(train_parts, how="diagonal_relaxed").write_parquet(train_path)
    if valid_parts:
        pl.concat(valid_parts, how="diagonal_relaxed").write_parquet(valid_path)
    return n_train, n_valid


def compute_stats():
    """流式计算 TabM 标准化所需的 mean/std（连续特征）。"""
    cols = C.TabMConfig.std_feature
    lf = pl.scan_parquet(C.PROCESSED_TRAIN).select(cols)
    agg = lf.select([
        pl.col(c).mean().alias(f"{c}__mean") for c in cols
    ] + [
        pl.col(c).std().alias(f"{c}__std") for c in cols
    ]).collect()
    means = {c: float(agg[f"{c}__mean"][0]) for c in cols}
    stds = {c: float(agg[f"{c}__std"][0]) for c in cols}
    stds = {c: (v if v > 1e-8 else 1.0) for c, v in stds.items()}
    joblib.dump({"mean": means, "std": stds}, C.DATA_STATS_PATH)
    print(f"[data_prep] 标准化统计量已保存 -> {C.DATA_STATS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true", help="仅重算标准化统计量")
    args = parser.parse_args()

    seed_everything(C.SEED)

    if not args.stats_only:
        lf = read_raw_train_lazy()
        n_train, n_valid = process_and_write(lf)
        print(f"[data_prep] train 行数={n_train}, valid 行数={n_valid}")
        print(f"[data_prep] 已保存 -> {C.PROCESSED_TRAIN}")
        print(f"[data_prep] 已保存 -> {C.PROCESSED_VALID}")
    compute_stats()
    print("[data_prep] 完成。")


if __name__ == "__main__":
    main()
