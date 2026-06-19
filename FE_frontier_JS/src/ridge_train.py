# -*- coding: utf-8 -*-
"""
Ridge 回归训练（对应 ridge_train.ipynb）。

建模细节原样保留：
- 仅读取 partition 6-9，sample(frac=0.82, random_state=2025)
- 79 个基础特征，缺失值统一 fillna(3)
- 末尾 20 万条作本地验证集（时间上位于训练集之后）
- sklearn Ridge() 默认正则化参数
- 评价指标：加权 R²
"""
import argparse
import gc
import dill
import numpy as np
import polars as pl
import pandas as pd
from sklearn.linear_model import Ridge

import config as C
from utils import seed_everything, custom_metric


def load_ridge_data():
    """读取 partition 6-9，采样 0.82，返回 pandas DataFrame 与 weight 列表。"""
    train_dir = C.DATA_RAW / C.RAW_TRAIN_DIRNAME
    datas, weights = [], []
    for i in C.RidgeConfig.partitions:
        path = train_dir / f"partition_id={i}" / "part-0.parquet"
        train = pl.read_parquet(path).to_pandas().sample(frac=C.RidgeConfig.sample_frac,
                                                          random_state=2025)
        weights += list(train["weight"].values)
        train.drop(["weight"], axis=1, inplace=True)
        datas.append(train)
    train = pd.concat(datas)
    del datas
    gc.collect()
    return train, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    # 原 notebook 使用 seed=2025
    seed_everything(2025)

    print("< read parquet >")
    train, weights = load_ridge_data()
    print(f"train.shape:{train.shape}")

    print("< get X,y >")
    cols = C.RidgeConfig.feature_cols
    X = train[cols].fillna(C.RidgeConfig.fillna_value).values
    y = train["responder_6"].values
    del train
    gc.collect()

    print("< train test split >")
    split = C.RidgeConfig.split
    train_X, train_y = X[:-split], y[:-split]
    test_X, test_y = X[-split:], y[-split:]
    train_weight, test_weight = weights[:-split], weights[-split:]
    print(f"train_X.shape:{train_X.shape}, test_X.shape:{test_X.shape}")

    print("< fit and predict >")
    model = Ridge()
    model.fit(train_X, train_y)
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    print(f"train weighted_r2:{custom_metric(train_y, train_pred, weight=train_weight)}")
    print(f"test weighted_r2:{custom_metric(test_y, test_pred, weight=test_weight)}")

    if not args.no_save:
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.RidgeConfig.model_path, "wb") as fp:
            dill.dump(model, fp)
        print(f"模型已保存 -> {C.RidgeConfig.model_path}")


if __name__ == "__main__":
    main()
