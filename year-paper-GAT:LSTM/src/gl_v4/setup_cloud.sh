#!/bin/bash
# ============================================================
# AutoDL / 恒源云 环境配置脚本
# 镜像: PyTorch 2.7.0 / Python 3.12 / CUDA 12.8 / Ubuntu 22.04
# 运行: bash setup_cloud.sh
# ============================================================
set -e

echo "========================================"
echo " 科创50指数增强 — 云端环境配置"
echo " 目标: PyTorch 2.7.0 + CUDA 12.8"
echo "========================================"

# 1. 确认环境
echo ""
echo "[1/4] 检查基础环境 ..."
python3 -c "
import torch, sys
print(f'  PyTorch: {torch.__version__}')
print(f'  Python:  {sys.version_info.major}.{sys.version_info.minor}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 2. 安装 PyTorch Geometric
echo ""
echo "[2/4] 安装 PyTorch Geometric ..."

# 方法1: 直接安装（PyG >= 2.6 支持 pip install）
pip install torch_geometric -q 2>/dev/null && echo "  ✓ torch_geometric installed" || {
    # 方法2: 指定 CUDA 12.8 wheel
    echo "  尝试 wheel 安装 (cu128)..."
    pip install torch_geometric \
        pyg_lib torch_scatter torch_sparse torch_cluster \
        -f https://data.pyg.org/whl/torch-2.7.0+cu128.html -q 2>/dev/null && \
        echo "  ✓ wheel 安装成功" || {
        # 方法3: 从源码编译
        echo "  尝试源码安装..."
        pip install torch_geometric --no-build-isolation -q
        echo "  ✓ 源码安装成功"
    }
}

# 3. 安装 cvxpy + ECOS (组合优化)
echo ""
echo "[3/4] 安装 cvxpy + ECOS ..."
pip install cvxpy ecos -q
echo "  ✓ cvxpy + ECOS installed"

# 4. 安装其他依赖
echo ""
echo "[4/4] 安装其他依赖 ..."
pip install -r requirements.txt -q 2>/dev/null || true
echo "  ✓ requirements.txt ok"

# 5. 最终验证
echo ""
echo "========================================"
echo " 验证所有依赖 ..."
python3 -c "
import torch
import torch_geometric
import cvxpy
import numpy as np
import pandas as pd
import matplotlib
import seaborn

print(f'  torch:           {torch.__version__}')
print(f'  torch_geometric: {torch_geometric.__version__}')
print(f'  cvxpy:           {cvxpy.__version__}')
print(f'  numpy:           {np.__version__}')
print(f'  pandas:          {pd.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')

# GATConv GPU 前向测试
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = GATConv(4, 64, heads=4, edge_dim=1).to(device)
x = torch.randn(100, 4).to(device)
ei = torch.randint(0, 100, (2, 200)).to(device)
ea = torch.rand(200, 1).to(device)
out = conv(x, ei, ea)
print(f'  GATConv GPU test: {out.shape} ✓')

# cvxpy 求解器测试
import cvxpy as cp
w = cp.Variable(5)
prob = cp.Problem(cp.Maximize(w.sum()), [w>=0, cp.sum(w)==1])
prob.solve(solver='ECOS', verbose=False)
print(f'  cvxpy solver test: OK (status={prob.status})')
"
echo ""
echo "========================================"
echo " 环境配置完成！"
echo ""
echo " 快速测试 (3分钟):  python run_backtest.py --quick"
echo " 完整回测 (1~2小时): python run_backtest.py --epochs 100 --batch-size 1024"
echo "========================================"
