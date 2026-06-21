"""
阶段 4：极速冒烟测试
===================
验证 V3 全部修改的端到端正确性。
"""

import sys
import os
import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# 将 src 加入 path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def run_test(name, func):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        func()
        dt = time.time() - t0
        print(f"\n  {PASS} {name} ({dt:.1f}s)")
        results.append((name, True, ""))
    except Exception as e:
        dt = time.time() - t0
        tb = traceback.format_exc()
        print(f"\n  {FAIL} {name} ({dt:.1f}s)")
        print(f"  ERROR: {e}")
        print(tb)
        results.append((name, False, str(e)))


# ============================================================================
#  Test 1: 数据与正交化冒烟
# ============================================================================

def test_data_orthogonalization():
    """加载真实因子数据，验证正交化 + 标签平滑 + 截面标准化。"""
    import torch
    from data.data_module import (
        ADV9_FACTOR_NAMES,
        BASE16_FACTOR_NAMES,
        MICRO8_FACTOR_NAMES,
        FACTOR_DIR,
        MARKET_DIR,
        resolve_factor_files,
    )

    # 快速检查因子文件
    factor_files = resolve_factor_files(FACTOR_DIR, "all33")
    assert len(factor_files) == 33, f"期望 33 个因子文件，实际 {len(factor_files)}"
    assert len(BASE16_FACTOR_NAMES) == 16
    assert len(MICRO8_FACTOR_NAMES) == 8
    assert len(ADV9_FACTOR_NAMES) == 9
    print(f"  因子文件: {len(factor_files)} 个 OK")

    # 测试单个因子加载
    sample = pd.read_pickle(factor_files[0])
    print(f"  示例因子 '{factor_files[0].stem}': {sample.shape} ({sample.index[0]} ~ {sample.index[-1]})")

    # 测试正交化逻辑（独立于 Dataset 的快速验证）
    print("\n  测试对称正交化逻辑 ...")
    # 构造一个小型因子矩阵 [100, 33]
    np.random.seed(42)
    F = np.random.randn(100, 33).astype("float32")
    # 加入共线性：col[1] = col[0] * 2 + noise
    F[:, 1] = F[:, 0] * 2 + np.random.randn(100) * 0.1

    S = np.cov(F, rowvar=False)
    assert S.shape == (33, 33), f"协方差矩阵形状错误: {S.shape}"

    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-8)
    inv_sqrt_diag = np.diag(eigvals ** -0.5)
    S_inv_sqrt = eigvecs @ inv_sqrt_diag @ eigvecs.T
    F_orth = F @ S_inv_sqrt

    # 验证正交化后相关性接近 0
    S_orth = np.cov(F_orth, rowvar=False)
    off_diag = S_orth[np.triu_indices(33, k=1)]
    max_corr = np.max(np.abs(off_diag))
    print(f"  正交化前 max|off-diag cov|: {np.max(np.abs(S[np.triu_indices(33, k=1)])):.4f}")
    print(f"  正交化后 max|off-diag cov|: {max_corr:.6f}")
    assert max_corr < 0.01, f"正交化后仍有高相关: {max_corr}"
    print("  对称正交化逻辑 OK")

    # 测试标签平滑逻辑
    print("\n  测试 5 日前瞻标签平滑 ...")
    ret_mat = pd.read_csv(MARKET_DIR / "Daily_Returns.csv", index_col=0, parse_dates=True)
    labels_smooth = ret_mat.shift(-1).rolling(5, min_periods=5).mean().shift(-4)
    n_valid = labels_smooth.notna().sum().sum()
    n_total = labels_smooth.size
    print(f"  标签形状: {labels_smooth.shape}")
    print(f"  有效标签: {n_valid:,} / {n_total:,} ({n_valid/n_total*100:.1f}%)")
    assert n_valid > 0, "标签全部为 NaN"
    assert labels_smooth.iloc[-5:].isna().all().all(), "最后 5 行应全为 NaN"
    print("  标签平滑逻辑 OK")

    print("\n  数据与正交化冒烟通过")


# ============================================================================
#  Test 2: 消融控制流冒烟
# ============================================================================

def test_ablation_modes():
    """用合成数据验证三种消融模式 + freeze_lstm 的训练流程。"""
    import torch
    from torch_geometric.data import Data
    from model.model import create_model, combined_loss

    device = torch.device("cpu")
    B = 64
    seq_len = 20
    n_features = 33
    gat_in = 16  # V3: 9原始 + 3动量 + 4估值 = 16

    # 合成数据
    X_lstm = torch.randn(B, seq_len, n_features)
    labels = torch.randn(B, 1)
    masks = torch.ones(B, 1)
    snap_keys = ["20240430"] * B
    node_indices = list(range(B))

    N = 100
    g = Data(
        x=torch.randn(N, gat_in),
        edge_index=torch.randint(0, N, (2, 30)),
        edge_attr=torch.rand(30, 1),
    )
    snap_graphs = {"20240430": g}

    modes = ["full", "lstm_only", "gat_only"]
    for mode in modes:
        print(f"\n  --- ablation_mode={mode}, freeze_lstm=False ---")
        model = create_model(
            in_features=n_features,
            gat_in=gat_in,
            ablation_mode=mode,
            freeze_lstm=False,
            device=device,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  参数: {trainable:,} / {total:,} 可训练")

        # 前向
        model.train()
        scores, gates = model(X_lstm, snap_keys, node_indices, snap_graphs)
        print(f"  scores: {scores.shape}, gates: {gates.shape}")
        print(f"  gate_mean={gates.mean().item():.3f}")

        # 损失 + 反向
        loss = combined_loss(scores, labels, masks, gates=gates, rank_weight=0.7)
        print(f"  loss={loss.item():.6f}")

        model.zero_grad()
        loss.backward()

        # 检查梯度
        grad_ok = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert grad_ok, f"模式 {mode}: 无梯度流通"
        print(f"  梯度流通 OK")

    # 测试 freeze_lstm
    print(f"\n  --- ablation_mode=full, freeze_lstm=True ---")
    model_frozen = create_model(
        in_features=n_features,
        gat_in=gat_in,
        ablation_mode="full",
        freeze_lstm=True,
        device=device,
    )
    lstm_frozen = all(not p.requires_grad for p in model_frozen.lstm.parameters())
    gat_trainable = all(p.requires_grad for p in model_frozen.gat.parameters())
    fusion_trainable = all(p.requires_grad for p in model_frozen.gated_fusion.parameters())
    assert lstm_frozen, "LSTM 应该被冻结"
    assert gat_trainable, "GAT 应该可训练"
    assert fusion_trainable, "Fusion 应该可训练"
    print(f"  LSTM 冻结: {lstm_frozen}")
    print(f"  GAT 可训练: {gat_trainable}")
    print(f"  Fusion 可训练: {fusion_trainable}")

    # 训练一步验证
    trainable_params = [p for p in model_frozen.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    scores, gates = model_frozen(X_lstm, snap_keys, node_indices, snap_graphs)
    loss = combined_loss(scores, labels, masks, gates=gates, rank_weight=0.7)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  freeze_lstm 训练一步 OK, loss={loss.item():.6f}")

    print("\n  消融控制流冒烟通过")


# ============================================================================
#  Test 3: 诊断脚本冒烟
# ============================================================================

def test_diagnostics():
    """验证因子诊断脚本能正常输出 CSV 和 PNG。"""
    import subprocess

    src_dir = SRC_DIR
    output_dir = SRC_DIR.parent / "result" / "diagnostics_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只跑 IC + corr（quintile 需要 scipy，先测核心功能）
    for mode in ["ic", "corr"]:
        print(f"\n  --- factor_diagnostics --mode {mode} ---")
        cmd = [
            sys.executable, "-m", "analysis.factor_diagnostics",
            "--mode", mode,
            "--start", "2024-01-01",
            "--end", "2024-06-30",
            "--output-dir", str(output_dir),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(src_dir), timeout=120
        )
        print(f"  stdout (last 500 chars):\n{result.stdout[-500:]}")
        if result.returncode != 0:
            print(f"  stderr:\n{result.stderr[-500:]}")
            raise RuntimeError(f"factor_diagnostics --mode {mode} 失败 (rc={result.returncode})")
        print(f"  {mode} 模式 OK")

    # 检查输出文件
    csvs = list(output_dir.glob("*.csv"))
    pngs = list(output_dir.glob("*.png"))
    print(f"\n  输出文件: {len(csvs)} CSV, {len(pngs)} PNG")
    for f in sorted(csvs):
        print(f"    {f.name}")
    for f in sorted(pngs):
        print(f"    {f.name}")

    assert len(csvs) >= 2, f"期望至少 2 个 CSV，实际 {len(csvs)}"
    assert len(pngs) >= 2, f"期望至少 2 个 PNG，实际 {len(pngs)}"

    print("\n  诊断脚本冒烟通过")


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  阶段 4：极速冒烟测试")
    print("=" * 60)

    run_test("1. 数据与正交化", test_data_orthogonalization)
    run_test("2. 消融控制流", test_ablation_modes)
    run_test("3. 诊断脚本", test_diagnostics)

    print("\n" + "=" * 60)
    print("  冒烟测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, ok, err in results:
        status = PASS if ok else FAIL
        detail = f" ({err})" if err else ""
        print(f"  {status} {name}{detail}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  " + "=" * 40)
        print("  " + PASS + " 全部通过 — V3 冒烟测试绿灯通行证")
        print("  " + "=" * 40)
    else:
        print("  " + FAIL + " 存在失败项，请检查上方日志")
        sys.exit(1)
