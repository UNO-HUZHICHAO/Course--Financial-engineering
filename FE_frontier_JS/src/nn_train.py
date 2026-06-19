# -*- coding: utf-8 -*-
"""
MLP 神经网络训练（对应 nn_train.ipynb，基于 PyTorch Lightning）。

建模细节原样保留：
- 输入 88 维（79 基础特征 + 9 滞后），缺失值 ffill+0
- 架构：BN+Dropout(0.1)+Linear(512) -> SiLU+BN+Dropout(0.1)+Linear(512)
        -> SiLU+BN+Dropout(0.1)+Linear(256) -> Linear+Tanh，输出×5 截断到 [-5,5]
- 损失：加权 MSE
- Adam(lr=1e-3, wd=5e-4) + ReduceLROnPlateau(factor=0.5, patience=5)
- EarlyStopping(patience=25, max_epochs=2000)
- 5 折日期交叉验证，5 个模型推理时简单平均
- train+valid 合并训练（LB trick）
"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl_p
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from torch.utils.data import Dataset, DataLoader

import config as C
from utils import r2_val, get_device

feature_names = C.NNConfig.feature_cols
label_name = C.TARGET_COL
weight_name = C.WEIGHT_COL


class CustomDataset(Dataset):
    def __init__(self, df, accelerator):
        self.features = torch.FloatTensor(df[feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(df[label_name].values).to(accelerator)
        self.weights = torch.FloatTensor(df[weight_name].values).to(accelerator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


class DataModule(LightningDataModule):
    def __init__(self, train_df, batch_size, valid_df=None, accelerator="cpu"):
        super().__init__()
        self.df = train_df
        self.batch_size = batch_size
        self.dates = self.df["date_id"].unique()
        self.accelerator = accelerator
        self.train_dataset = None
        self.valid_df = valid_df
        self.val_dataset = None

    def setup(self, fold=0, N_fold=5, stage=None):
        selected_dates = [d for ii, d in enumerate(self.dates) if ii % N_fold != fold]
        df_train = self.df.loc[self.df["date_id"].isin(selected_dates)]
        self.train_dataset = CustomDataset(df_train, self.accelerator)
        if self.valid_df is not None:
            self.val_dataset = CustomDataset(self.valid_df, self.accelerator)

    def train_dataloader(self, n_workers=0):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=n_workers)

    def val_dataloader(self, n_workers=0):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=n_workers)


class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return 5 * self.model(x).squeeze(-1)

    def training_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction="none") * w
        loss = loss.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction="none") * w
        loss = loss.mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in self.trainer.logged_metrics.items()}
        formatted = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted}")


class Args:
    usegpu = True
    gpuid = C.NNConfig.gpuid
    seed = C.SEED
    use_wandb = False
    loader_workers = C.NNConfig.loader_workers
    bs = C.NNConfig.batch_size
    lr = C.NNConfig.lr
    weight_decay = C.NNConfig.weight_decay
    dropouts = C.NNConfig.dropouts
    n_hidden = C.NNConfig.n_hidden
    patience = C.NNConfig.patience
    max_epochs = C.NNConfig.max_epochs
    N_fold = C.NNConfig.n_fold


def main():
    args = Args()
    # 支持环境变量覆盖，便于在算力预算约束下缩减训练规模（原方案默认值见 config.py）
    import os as _os
    if _os.environ.get("NN_MAX_EPOCHS"):
        args.max_epochs = int(_os.environ["NN_MAX_EPOCHS"])
    if _os.environ.get("NN_PATIENCE"):
        args.patience = int(_os.environ["NN_PATIENCE"])
    print(f"[NN] max_epochs={args.max_epochs}, patience={args.patience}, n_fold={args.N_fold}")
    pl_p.seed_everything(args.seed)

    # 读取处理后的训练/验证集（含滞后特征）
    df = pl.read_parquet(C.PROCESSED_TRAIN).to_pandas()
    valid = pl.read_parquet(C.PROCESSED_VALID).to_pandas()
    import os as _os2
    if not _os2.environ.get("HONEST_TRAIN"):
        df = pd.concat([df, valid]).reset_index(drop=True)  # LB trick
    else:
        print("[NN] HONEST_TRAIN 模式：不合并 validation（无数据泄露）")

    # 缺失值 ffill + 0
    df[feature_names] = df[feature_names].ffill().fillna(0)
    valid[feature_names] = valid[feature_names].ffill().fillna(0)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() and args.usegpu else "cpu")
    use_gpu = torch.cuda.is_available() and args.usegpu
    accelerator = "gpu" if use_gpu else "cpu"
    # CPU 模式下 devices 必须是 int；GPU 模式下指定单卡
    trainer_devices = [args.gpuid] if use_gpu else 1
    loader_device = "cpu"

    data_module = DataModule(df, batch_size=args.bs, valid_df=valid, accelerator=loader_device)
    del df
    gc.collect()

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for fold in range(args.N_fold):
        print(f"\n[NN] ===== fold {fold} =====")
        data_module.setup(fold, args.N_fold)
        input_dim = data_module.train_dataset.features.shape[1]
        model = NN(input_dim=input_dim, hidden_dims=args.n_hidden, dropouts=args.dropouts,
                   lr=args.lr, weight_decay=args.weight_decay)

        early_stopping = EarlyStopping("val_loss", patience=args.patience, mode="min", verbose=False)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(C.MODELS_DIR), filename=f"{C.NNConfig.model_prefix}_{fold}",
            monitor="val_loss", mode="min", save_top_k=1, verbose=False)
        timer = Timer()

        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            devices=trainer_devices,
            logger=None,
            callbacks=[early_stopping, checkpoint_callback, timer],
            enable_progress_bar=True,
        )
        trainer.fit(model, data_module.train_dataloader(args.loader_workers),
                    data_module.val_dataloader(args.loader_workers))
        print(f"[NN] fold-{fold} 训练完成，用时 {timer.time_elapsed('train'):.2f}s")
        print(f"[NN] 最佳模型: {checkpoint_callback.best_model_path}")
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
