# -*- coding: utf-8 -*-
"""
公共工具：随机种子、加权 R² 指标、标准化、类别编码、设备判定等。
算法细节均与原始 notebook 一致。
"""
import os
import random
import numpy as np
import pandas as pd
import polars as pl
import torch


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_xgb_device(params: dict) -> dict:
    """根据是否有 GPU 自动调整 XGBoost 的 device / tree_method，算法不变。"""
    params = dict(params)
    if not torch.cuda.is_available():
        params["device"] = "cpu"
        params["tree_method"] = "hist"
    return params


# ---------------------------------------------------------------------
# 评价指标：加权 R²（与竞赛一致）
# ---------------------------------------------------------------------
def custom_metric(y_true, y_pred, weight):
    """ridge_train.ipynb 中的实现。"""
    weighted_r2 = 1 - (np.sum(weight * (y_true - y_pred) ** 2) / np.sum(weight * y_true ** 2))
    return weighted_r2


def r2_val(y_true, y_pred, sample_weight):
    """nn_train / ensemble 中的加权 R² 实现。"""
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / \
        (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2


def r2_val_sum(y_true, y_pred, sample_weight):
    """tabm_train 中的加权 R² 实现（基于求和）。"""
    residuals = sample_weight * (y_true - y_pred) ** 2
    weighted_residual_sum = np.sum(residuals)
    weighted_true_sum = np.sum(sample_weight * (y_true) ** 2)
    return 1 - weighted_residual_sum / weighted_true_sum


# ---------------------------------------------------------------------
# TabM 预处理：标准化与类别编码（原样保留）
# ---------------------------------------------------------------------
def standardize(df: pl.DataFrame, feature_cols, means, stds) -> pl.DataFrame:
    return df.with_columns([
        ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
    ])


def encode_column(df: pl.DataFrame, column: str, mapping: dict) -> pl.DataFrame:
    """对应 ensemble.ipynb 中的 encode_column：未见类别映射为 max(mapping)+1。"""
    max_value = max(mapping.values())

    def encode_category(category):
        return mapping.get(category, max_value + 1)

    return df.with_columns(
        pl.col(column).map_elements(encode_category).alias(column)
    )


def encode_column_train(df: pl.DataFrame, column: str, mapping: dict) -> pl.DataFrame:
    """对应 tabm_train.ipynb 中的 encode_column：未见类别映射为 -1。"""
    def encode_category(category):
        return mapping.get(category, -1)
    return df.with_columns(
        pl.col(column).map_elements(encode_category, return_dtype=pl.Int16).alias(column)
    )


def get_device(gpuid: int = 0):
    return torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")
