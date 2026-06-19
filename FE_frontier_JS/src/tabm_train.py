# -*- coding: utf-8 -*-
"""
TabM 训练（对应 tabm_train.ipynb）。

建模细节原样保留：
- 连续特征 85 维（76 基础 + 9 滞后，剔除 feature_09/10/11）
- 分类特征 5 个：feature_09/10/11/symbol_id/time_id，基数 [23,10,32,40,969]
- k=32 集成成员，骨干 MLP 3×512 dropout=0.25
- 损失：R² Loss（与竞赛指标对齐）
- AdamW(lr=1e-4, wd=5e-3)，make_parameter_groups 分组权重衰减
- 连续特征注入高斯噪声 std=0.035（仅训练）
- 划分：train 用 date_id∈[800,1577] + valid≤1577；valid 用 date_id>1577
- early stopping patience=5（mode=max），num_epochs=4

一致性修正：原 notebook 定义了 standardize 但训练时未调用，而 ensemble 推理却调用，
导致训练/推理不一致。本实现按报告所述在训练与推理中统一应用 Z-score 标准化。
"""
import argparse
import gc
import math
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import delu
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader

import config as C
from utils import seed_everything, standardize, encode_column_train, r2_val_sum, get_device
from tabm_reference import Model, make_parameter_groups


class R2Loss(nn.Module):
    """原 tabm_train 中的 R² 损失。"""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.sum((y_pred - y_true) ** 2)
        var_y = torch.sum(y_true ** 2)
        return mse_loss / (var_y + 1e-38)


def build_data():
    cfg = C.TabMConfig
    train_original = pl.scan_parquet(C.PROCESSED_TRAIN)
    valid_original = pl.scan_parquet(C.PROCESSED_VALID)

    # 类别编码（OOV -> -1，对应 tabm_train 的 encode_column）
    for col in cfg.feature_cat + ["symbol_id", "time_id"]:
        train_original = encode_column_train(train_original, col, C.CATEGORY_MAPPINGS[col])
        valid_original = encode_column_train(valid_original, col, C.CATEGORY_MAPPINGS[col])

    select_cols = cfg.feature_all + [C.TARGET_COL, C.WEIGHT_COL, "symbol_id", "time_id"]
    # 全部 cast 为 Float32，保证 to_numpy() 返回连续 float 数组（避免混合 dtype 问题）
    cast_expr = [pl.col(c).cast(pl.Float32) for c in select_cols]

    # 划分
    train_data1 = train_original.filter(
        (pl.col("date_id") >= cfg.start_dt) & (pl.col("date_id") <= cfg.end_dt)
    ).select(cast_expr)

    train_data2 = valid_original.filter(
        pl.col("date_id") <= cfg.end_dt
    ).select(cast_expr)

    train_data = pl.concat([train_data1, train_data2])
    valid_data = valid_original.filter(
        pl.col("date_id") > cfg.end_dt
    ).sort(["date_id", "time_id"]).select(cast_expr)

    # 缺失值填充：原始特征存在大量 nan（金融数据常态），TabM 不像树模型能天然处理 nan，
    # 必须在标准化前对连续特征 fill_null(0)（与 NN 的 fillna(0) 一致），否则前向传播产生 nan。
    fill_expr = [pl.col(c).fill_null(0.0) for c in cfg.feature_all]
    train_data = train_data.with_columns(fill_expr)
    valid_data = valid_data.with_columns(fill_expr)

    # 标准化（连续特征），与推理一致
    stats = joblib.load(C.DATA_STATS_PATH)
    means, stds = stats["mean"], stats["std"]
    train_data = standardize(train_data, cfg.std_feature, means, stds)
    valid_data = standardize(valid_data, cfg.std_feature, means, stds)

    train_data = train_data.collect()
    valid_data = valid_data.collect()
    print(f"[TabM] train_data={train_data.shape}, valid_data={valid_data.shape}")
    return train_data, valid_data


def make_loaders(train_data, valid_data):
    cfg = C.TabMConfig
    train_tensor = torch.tensor(train_data.to_numpy(), dtype=torch.float32)
    valid_tensor = torch.tensor(valid_data.to_numpy(), dtype=torch.float32)
    train_ds = TensorDataset(train_tensor)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=4,
                          pin_memory=True, shuffle=True)
    valid_ds = TensorDataset(valid_tensor)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, num_workers=4,
                          pin_memory=True, shuffle=False)
    return train_dl, valid_dl


def build_model(device):
    cfg = C.TabMConfig
    model = Model(
        n_num_features=cfg.n_cont_features,
        cat_cardinalities=cfg.cat_cardinalities,
        n_classes=None,
        backbone=cfg.backbone,
        bins=None,
        num_embeddings=None,
        arch_type="tabm",
        k=cfg.k,
    ).to(device)
    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    loss_fn = R2Loss()
    return model, optimizer, loss_fn


def split_inputs(tensor_row, device):
    """与原 tabm_train 一致：最后 4 列为 [y, weight, symbol_id, time_id]。"""
    X_input = tensor_row[:, :-4].to(device)
    y_input = tensor_row[:, -4].to(device)
    w_input = tensor_row[:, -3].to(device)
    symbol_input = tensor_row[:, -2].to(device)
    time_input = tensor_row[:, -1].to(device)

    x_cont = X_input[:, [c for c in range(X_input.shape[1]) if c not in [9, 10, 11]]]
    x_cat = X_input[:, [9, 10, 11]]
    x_cat = torch.concat([x_cat, symbol_input.unsqueeze(-1), time_input.unsqueeze(-1)],
                         axis=1).to(torch.int64)
    return x_cont, x_cat, y_input, w_input


def main():
    seed_everything(C.SEED)
    device = get_device(C.NNConfig.gpuid)
    cfg = C.TabMConfig

    train_data, valid_data = build_data()
    train_dl, valid_dl = make_loaders(train_data, valid_data)
    del train_data, valid_data
    gc.collect()

    model, optimizer, loss_fn = build_model(device)

    timer = delu.tools.Timer()
    patience = cfg.patience
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best = {"val": -math.inf, "epoch": -1}
    timer.run()

    for epoch in range(cfg.num_epochs):
        model.train()
        train_pred_list = []
        with tqdm(train_dl, total=len(train_dl), leave=True) as pbar:
            for train_tensor in pbar:
                optimizer.zero_grad()
                x_cont, x_cat, y_input, w_input = split_inputs(train_tensor[0], device)
                # 训练时连续特征注入噪声
                x_cont = x_cont + torch.randn_like(x_cont) * cfg.noise_std
                output = model(x_cont, x_cat).squeeze(-1)
                loss = loss_fn(output.flatten(0, 1), y_input.repeat_interleave(cfg.k))
                train_pred_list.append((output.mean(1), y_input, w_input))
                loss.backward()
                optimizer.step()
                pbar.set_postfix(OrderedDict(
                    epoch=f"{epoch+1}/{cfg.num_epochs}",
                    loss=f"{loss.item():.6f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.3e}",
                ))
                pbar.update(1)

        weights_train = torch.cat([x[2] for x in train_pred_list]).cpu().numpy()
        y_train = torch.cat([x[1] for x in train_pred_list]).cpu().numpy()
        prob_train = torch.cat([x[0] for x in train_pred_list]).detach().cpu().numpy()
        train_r2 = r2_val_sum(y_train, prob_train, weights_train)

        model.eval()
        valid_pred_list = []
        for valid_tensor in tqdm(valid_dl, desc=f"val epoch {epoch+1}"):
            x_cont, x_cat, y_valid, w_valid = split_inputs(valid_tensor[0], device)
            # 推理不加噪声（与 ensemble 推理一致）
            with torch.no_grad():
                y_pred = model(x_cont, x_cat).squeeze(-1)
            valid_pred_list.append((y_pred.mean(1), y_valid, w_valid))

        weights_eval = torch.cat([x[2] for x in valid_pred_list]).cpu().numpy()
        y_eval = torch.cat([x[1] for x in valid_pred_list]).cpu().numpy()
        prob_eval = torch.cat([x[0] for x in valid_pred_list]).cpu().numpy()
        val_r2 = r2_val_sum(y_eval, prob_eval, weights_eval)

        print(f"Epoch {epoch+1}: train_r2={train_r2:.6f}, val_r2={val_r2:.6f}, [time] {timer}")

        if val_r2 > best["val"]:
            print("New best epoch!")
            best = {"val": val_r2, "epoch": epoch}
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "r2": val_r2,
            }
            torch.save(checkpoint, cfg.model_path)
            print(f"[TabM] 最佳模型已保存 -> {cfg.model_path}")

        early_stopping.update(val_r2)
        if early_stopping.should_stop():
            print("Early stop")
            break

    print(f"[TabM] 训练结束，best val_r2={best['val']:.6f} @ epoch {best['epoch']}")


if __name__ == "__main__":
    main()
