# -*- coding: utf-8 -*-
"""
一键编排：数据预处理 -> 四模型训练 -> 集成评估与可视化。

用法：
    cd src
    python run_all.py                 # 全流程（NN 默认减 epoch，约2.5h）
    python run_all.py --full-nn       # NN 用原方案 max_epochs=2000，约7-8h
    python run_all.py --skip data     # 跳过数据预处理
    python run_all.py --only ensemble # 仅运行集成评估

NN 训练规模说明：
- 默认 NN_MAX_EPOCHS=15 NN_PATIENCE=5（受算力预算约束的实际运行配置，约2.5h完成5折）
- --full-nn 使用原方案 max_epochs=2000 patience=25（约7-8h，需充足算力）
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, label, env_extra=None):
    print(f"\n{'='*70}\n>>> [{label}] python {script}\n{'='*70}")
    t0 = time.time()
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    ret = subprocess.call([sys.executable, str(HERE / script)], env=env)
    dt = time.time() - t0
    if ret != 0:
        print(f"[run_all] {script} 失败 (exit={ret})，用时 {dt:.1f}s")
        sys.exit(ret)
    print(f"[run_all] {script} 完成，用时 {dt:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["data", "ridge", "xgb", "nn", "tabm", "ensemble"],
                        help="跳过的阶段")
    parser.add_argument("--only", choices=["data", "ridge", "xgb", "nn", "tabm", "ensemble"],
                        help="仅运行某一阶段")
    parser.add_argument("--full-nn", action="store_true",
                        help="NN 使用原方案 max_epochs=2000 patience=25（约7-8h）")
    args = parser.parse_args()

    # NN 环境变量：默认减 epoch（与报告实际运行配置一致），--full-nn 用原方案
    nn_env = None if args.full_nn else {"NN_MAX_EPOCHS": "15", "NN_PATIENCE": "5"}
    # 精简方案默认无泄露训练（HONEST_TRAIN=1）：XGB/NN 仅用 date_id<=1577 训练
    honest = {"HONEST_TRAIN": "1"}
    xgb_env = dict(honest)
    nn_env_full = dict(honest)
    if nn_env:
        nn_env_full.update(nn_env)

    stages = ["data", "xgb", "nn", "ensemble", "backtest"]
    if args.only:
        stages = [args.only]
    else:
        stages = [s for s in stages if s not in args.skip]

    mapping = {
        "data": ("data_prep.py", "数据预处理", None),
        "xgb": ("xgb_train.py", "XGBoost 训练(无泄露)", xgb_env),
        "nn": ("nn_train.py", "NN 训练(无泄露)", nn_env_full),
        "ensemble": ("ensemble.py", "集成评估与可视化", None),
        "backtest": ("backtest.py", "策略回测", None),
    }
    for s in stages:
        script, label, env_extra = mapping[s]
        run(script, label, env_extra)
    print("\n[run_all] 全部阶段完成。结果见 result/ 目录。")


if __name__ == "__main__":
    main()
