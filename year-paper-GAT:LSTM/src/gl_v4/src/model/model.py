"""
模块二：双流异步门控融合网络
=============================
端到端融合日频量价时序特征（LSTM）与半年度产业链图谱特征（GAT），
通过自适应门控机制动态分配权重，输出个股预期超额收益打分。

架构:
    X_lstm [B,20,16] ──→ LSTM ──→ H_LSTM [B,64] ──┐
                                                      ├──→ Gate → H_fused [B,64] → Score [B,1]
    Graph Data ──→ SpatialGAT ──→ H_GAT [B,64] ─────┘

门控公式:
    g = σ(W_g [H_LSTM || H_GAT] + b_g)
    H_fused = g ⊙ H_LSTM + (1 - g) ⊙ H_GAT
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# ============================================================================
#  TemporalLSTM — 日频量价序列编码器
# ============================================================================

class TemporalLSTM(nn.Module):
    """
    单层 LSTM 时序编码器，将过去20日 16 维因子序列压缩为 64 维表征。

    Parameters
    ----------
    in_features : int
        输入因子维度（默认 16）。
    hidden_size : int
        LSTM 隐藏层维度（默认 64）。
    num_layers : int
        LSTM 层数（默认 1）。
    dropout : float
        Dropout 比率（默认 0.1）。
    """

    def __init__(
        self,
        in_features: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 20, 16] 时序因子序列

        Returns
        -------
        H_LSTM : [B, 64] 最后时间步隐藏状态
        """
        # lstm_out: [B, 20, hidden_size]
        # h_n:      [1, B, hidden_size]
        lstm_out, (h_n, _) = self.lstm(x)
        # 取最后时间步的隐藏状态
        out = h_n[-1]  # [B, hidden_size]
        return self.dropout(out)


# ============================================================================
#  SpatialGAT — 产业链图谱空间编码器（复用已有实现并做小幅增强）
# ============================================================================

class SpatialGAT(nn.Module):
    """
    双层图注意力网络，将边权重注入注意力系数计算。

    第一层: in_channels → hidden_channels, heads=4, concat=True  → 输出 4×hidden
    第二层: 4×hidden → out_channels, heads=1, concat=False       → 输出 out_channels

    Parameters
    ----------
    in_channels : int
        输入节点特征维度（默认 4）。
    hidden_channels : int
        隐藏层维度（默认 64）。
    out_channels : int
        输出表征维度（默认 64）。
    heads : int
        第一层注意力头数（默认 4）。
    dropout : float
        Dropout 比率（默认 0.1）。
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            edge_dim=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, in_channels] 节点特征矩阵
        edge_index : [2, E] 边索引
        edge_attr : [E, 1] 边权重

        Returns
        -------
        H_GAT : [N, out_channels] 空间表征矩阵
        """
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x  # [N, out_channels]


# ============================================================================
#  GatedFusion / FiLMFusion — 融合模块
# ============================================================================

class GatedFusion(nn.Module):
    """
    学习一个标量门控值 g ∈ (0,1)，对 LSTM 和 GAT 表征做凸组合。

    公式:
        g = σ(W_g [H_LSTM || H_GAT] + b_g)
        H_fused = g ⊙ H_LSTM + (1 - g) ⊙ H_GAT

    消融模式 (ablation_mode):
        "full"     : 正常双流门控（默认）
        "lstm_only": 强制 gate = 1.0，禁用 GAT（纯 LSTM 基线）
        "gat_only" : 强制 gate = 0.0，禁用 LSTM（纯 GAT 基线）
    """

    def __init__(
        self,
        hidden_size: int = 64,
        gate_floor: float = 0.20,
        ablation_mode: str = "full",
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.gat_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.gate_floor = gate_floor
        self.gat_residual_logit = nn.Parameter(torch.tensor(-0.7))

    def forward(
        self, h_lstm: torch.Tensor, h_gat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_lstm : [B, 64] LSTM 时序表征
        h_gat  : [B, 64] GAT 空间表征

        Returns
        -------
        H_fused : [B, 64] 融合表征
        gate    : [B, 1]  门控标量（用于可解释性分析）
        """
        h_lstm_proj = self.lstm_proj(h_lstm)
        h_gat_proj = self.gat_proj(h_gat)

        # ── 消融模式：强制门控值 ──────────────────────────────────────────
        if self.ablation_mode == "lstm_only":
            # 强制 gate = 1.0，完全禁用 GAT
            gate = torch.ones(h_lstm.shape[0], 1, device=h_lstm.device)
            h_fused = self.layer_norm(h_lstm_proj)
            return h_fused, gate
        elif self.ablation_mode == "gat_only":
            # 强制 gate = 0.0，完全禁用 LSTM
            gate = torch.zeros(h_lstm.shape[0], 1, device=h_lstm.device)
            h_fused = self.layer_norm(h_gat_proj)
            return h_fused, gate

        # ── 正常双流门控 ──────────────────────────────────────────────────
        concat = torch.cat([h_lstm_proj, h_gat_proj], dim=-1)  # [B, 128]
        raw_gate = torch.sigmoid(self.gate_layer(concat))  # [B, 1]

        # 限制门控塌缩到 0/1 两端；若该样本无有效 GAT，则直接退化为纯 LSTM。
        gate = self.gate_floor + (1.0 - 2.0 * self.gate_floor) * raw_gate
        gat_available = (h_gat.abs().sum(dim=-1, keepdim=True) > 1e-8).float()
        gate = gat_available * gate + (1.0 - gat_available) * torch.ones_like(gate)

        residual_scale = torch.sigmoid(self.gat_residual_logit) * gat_available
        h_fused = self.layer_norm(
            gate * h_lstm_proj + (1.0 - gate) * h_gat_proj + residual_scale * h_gat_proj
        )
        return h_fused, gate


class FiLMFusion(nn.Module):
    """
    使用 GAT 生成逐维调制参数，对 LSTM 表征做 FiLM 调制。

    公式:
        gamma, beta = MLP(H_GAT)
        H_new = gamma ⊙ H_LSTM + beta

    仍返回一个 [B,1] 标量作为 diagnostics_value，表示平均调制强度，
    便于复用现有可视化与结果表结构。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        ablation_mode: str = "full",
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.gat_proj = nn.Linear(hidden_size, hidden_size)
        self.gamma_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, h_lstm: torch.Tensor, h_gat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_lstm_proj = self.lstm_proj(h_lstm)
        h_gat_proj = self.gat_proj(h_gat)

        if self.ablation_mode == "lstm_only":
            diagnostics_value = torch.ones(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_lstm_proj), diagnostics_value
        elif self.ablation_mode == "gat_only":
            diagnostics_value = torch.zeros(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_gat_proj), diagnostics_value

        gat_available = (h_gat.abs().sum(dim=-1, keepdim=True) > 1e-8).float()
        raw_gamma = self.gamma_layer(h_gat_proj)
        gamma = 1.0 + 0.5 * torch.tanh(raw_gamma)
        beta = self.beta_layer(h_gat_proj)

        h_film = gamma * h_lstm_proj + beta
        h_fused = gat_available * h_film + (1.0 - gat_available) * h_lstm_proj
        h_fused = self.layer_norm(h_fused)

        diagnostics_value = gamma.mean(dim=-1, keepdim=True)
        diagnostics_value = gat_available * diagnostics_value + (1.0 - gat_available) * torch.ones_like(diagnostics_value)
        return h_fused, diagnostics_value


class StableFiLMFusion(nn.Module):
    """
    稳健版 FiLM。

    不再依赖规则型开关，而是直接限制 GAT 对 LSTM 的调制幅度：
    1. 收窄 gamma 的缩放范围
    2. 将 beta 约束到与 LSTM 范数同量级的小范围
    3. 对整体 FiLM 增量做范数裁剪，避免弱窗口里一次性调制过冲
    """

    def __init__(
        self,
        hidden_size: int = 64,
        ablation_mode: str = "full",
        gamma_limit: float = 0.25,
        beta_limit: float = 0.25,
        max_delta_ratio: float = 0.35,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.gamma_limit = gamma_limit
        self.beta_limit = beta_limit
        self.max_delta_ratio = max_delta_ratio

        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.gat_proj = nn.Linear(hidden_size, hidden_size)
        self.gamma_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, h_lstm: torch.Tensor, h_gat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_lstm_proj = self.lstm_proj(h_lstm)
        h_gat_proj = self.gat_proj(h_gat)

        if self.ablation_mode == "lstm_only":
            diagnostics_value = torch.ones(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_lstm_proj), diagnostics_value
        elif self.ablation_mode == "gat_only":
            diagnostics_value = torch.zeros(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_gat_proj), diagnostics_value

        gat_available = (h_gat.abs().sum(dim=-1, keepdim=True) > 1e-8).float()

        raw_gamma = self.gamma_layer(h_gat_proj)
        gamma = 1.0 + self.gamma_limit * torch.tanh(raw_gamma)

        lstm_norm = h_lstm_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        raw_beta = self.beta_layer(h_gat_proj)
        beta = self.beta_limit * lstm_norm * torch.tanh(raw_beta)

        film_delta = (gamma - 1.0) * h_lstm_proj + beta
        delta_norm = film_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_scale = (self.max_delta_ratio * lstm_norm / delta_norm).clamp(max=1.0)

        h_fused = h_lstm_proj + gat_available * delta_scale * film_delta
        h_fused = self.layer_norm(h_fused)

        diagnostics_value = 1.0 - 0.45 * gat_available * delta_scale
        diagnostics_value = gat_available * diagnostics_value + (1.0 - gat_available) * torch.ones_like(diagnostics_value)
        return h_fused, diagnostics_value


class AdaptiveFiLMFusion(nn.Module):
    """
    自适应限幅版 FiLM。

    与固定阈值裁剪不同，不预先规定统一的调制强度，而是对每个样本根据：
    1. LSTM/GAT 方向一致性
    2. GAT/LSTM 强度比例
    3. FiLM 增量相对 LSTM 的幅度
    学习一个连续 trust_score，再用该分数缩放 FiLM 增量。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        ablation_mode: str = "full",
        gamma_limit: float = 0.50,
        delta_soft_cap: float = 1.20,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.gamma_limit = gamma_limit
        self.delta_soft_cap = delta_soft_cap

        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.gat_proj = nn.Linear(hidden_size, hidden_size)
        self.gamma_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, h_lstm: torch.Tensor, h_gat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_lstm_proj = self.lstm_proj(h_lstm)
        h_gat_proj = self.gat_proj(h_gat)

        if self.ablation_mode == "lstm_only":
            diagnostics_value = torch.ones(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_lstm_proj), diagnostics_value
        elif self.ablation_mode == "gat_only":
            diagnostics_value = torch.zeros(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_gat_proj), diagnostics_value

        gat_available = (h_gat.abs().sum(dim=-1, keepdim=True) > 1e-8).float()

        raw_gamma = self.gamma_layer(h_gat_proj)
        gamma = 1.0 + self.gamma_limit * torch.tanh(raw_gamma)
        beta = self.beta_layer(h_gat_proj)
        film_delta = (gamma - 1.0) * h_lstm_proj + beta

        lstm_norm = h_lstm_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gat_norm = h_gat_proj.norm(dim=-1, keepdim=True)
        delta_norm = film_delta.norm(dim=-1, keepdim=True)

        cosine = F.cosine_similarity(h_lstm_proj, h_gat_proj, dim=-1, eps=1e-8).unsqueeze(-1)
        norm_ratio = gat_norm / lstm_norm
        delta_ratio = delta_norm / lstm_norm

        cosine_score = ((cosine + 0.20) / 1.20).clamp(0.0, 1.0)
        ratio_score = (1.0 - (norm_ratio - 1.0).abs() / 1.20).clamp(0.0, 1.0)
        delta_score = (1.0 - (delta_ratio / self.delta_soft_cap - 1.0).clamp_min(0.0)).clamp(0.0, 1.0)
        prior_score = gat_available * cosine_score * ratio_score * delta_score

        scale_features = torch.cat([
            cosine.clamp(-1.0, 1.0),
            norm_ratio.clamp(0.0, 3.0),
            delta_ratio.clamp(0.0, 3.0),
            gat_available,
        ], dim=-1)
        adaptive_score = torch.sigmoid(self.scale_mlp(scale_features))
        trust_score = prior_score * (0.5 + 0.5 * adaptive_score)

        h_fused = h_lstm_proj + trust_score * film_delta
        h_fused = self.layer_norm(h_fused)

        diagnostics_value = 1.0 - 0.45 * trust_score
        diagnostics_value = gat_available * diagnostics_value + (1.0 - gat_available) * torch.ones_like(diagnostics_value)
        return h_fused, diagnostics_value


class GuardedFiLMFusion(nn.Module):
    """
    规则型 FiLM 保护开关。

    仅当 GAT 与 LSTM 的方向不明显冲突、且 GAT 强度与 LSTM 处于可接受比例时，
    才允许 FiLM 调制生效；否则回退到纯 LSTM 表征。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        ablation_mode: str = "full",
        min_cosine: float = -0.05,
        min_norm_ratio: float = 0.60,
        max_norm_ratio: float = 1.80,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.min_cosine = min_cosine
        self.min_norm_ratio = min_norm_ratio
        self.max_norm_ratio = max_norm_ratio

        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.gat_proj = nn.Linear(hidden_size, hidden_size)
        self.gamma_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, h_lstm: torch.Tensor, h_gat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_lstm_proj = self.lstm_proj(h_lstm)
        h_gat_proj = self.gat_proj(h_gat)

        if self.ablation_mode == "lstm_only":
            diagnostics_value = torch.ones(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_lstm_proj), diagnostics_value
        elif self.ablation_mode == "gat_only":
            diagnostics_value = torch.zeros(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_gat_proj), diagnostics_value

        gat_available = (h_gat.abs().sum(dim=-1, keepdim=True) > 1e-8).float()

        raw_gamma = self.gamma_layer(h_gat_proj)
        gamma = 1.0 + 0.5 * torch.tanh(raw_gamma)
        beta = self.beta_layer(h_gat_proj)
        h_film = gamma * h_lstm_proj + beta

        cosine = F.cosine_similarity(h_lstm_proj, h_gat_proj, dim=-1, eps=1e-8).unsqueeze(-1)
        lstm_norm = h_lstm_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gat_norm = h_gat_proj.norm(dim=-1, keepdim=True)
        norm_ratio = gat_norm / lstm_norm

        cosine_ok = (cosine >= self.min_cosine).float()
        norm_ok = (
            (norm_ratio >= self.min_norm_ratio) &
            (norm_ratio <= self.max_norm_ratio)
        ).float()
        trust_mask = gat_available * cosine_ok * norm_ok

        h_fused = trust_mask * h_film + (1.0 - trust_mask) * h_lstm_proj
        h_fused = self.layer_norm(h_fused)

        # 诊断值沿用“越高越偏 LSTM”的口径：
        # 被保护开关拦下时为 1，允许使用图调制时回到更中性的 0.55。
        diagnostics_value = trust_mask * 0.55 + (1.0 - trust_mask) * 1.0
        return h_fused, diagnostics_value


class SoftGuardedFiLMFusion(nn.Module):
    """
    软保护版 FiLM。

    与硬开关不同，不是简单地“通过/回退”，而是根据
    1. LSTM/GAT 方向一致性
    2. GAT/LSTM 强度比例
    生成 [0,1] 的 trust_score，对 FiLM 增量做连续缩放。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        ablation_mode: str = "full",
        min_cosine: float = -0.05,
        target_norm_ratio: float = 1.00,
        ratio_tolerance: float = 0.60,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.min_cosine = min_cosine
        self.target_norm_ratio = target_norm_ratio
        self.ratio_tolerance = ratio_tolerance

        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.gat_proj = nn.Linear(hidden_size, hidden_size)
        self.gamma_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, h_lstm: torch.Tensor, h_gat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_lstm_proj = self.lstm_proj(h_lstm)
        h_gat_proj = self.gat_proj(h_gat)

        if self.ablation_mode == "lstm_only":
            diagnostics_value = torch.ones(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_lstm_proj), diagnostics_value
        elif self.ablation_mode == "gat_only":
            diagnostics_value = torch.zeros(h_lstm.shape[0], 1, device=h_lstm.device)
            return self.layer_norm(h_gat_proj), diagnostics_value

        gat_available = (h_gat.abs().sum(dim=-1, keepdim=True) > 1e-8).float()

        raw_gamma = self.gamma_layer(h_gat_proj)
        gamma = 1.0 + 0.5 * torch.tanh(raw_gamma)
        beta = self.beta_layer(h_gat_proj)
        film_delta = (gamma - 1.0) * h_lstm_proj + beta

        cosine = F.cosine_similarity(h_lstm_proj, h_gat_proj, dim=-1, eps=1e-8).unsqueeze(-1)
        cosine_score = ((cosine - self.min_cosine) / (1.0 - self.min_cosine)).clamp(0.0, 1.0)

        lstm_norm = h_lstm_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gat_norm = h_gat_proj.norm(dim=-1, keepdim=True)
        norm_ratio = gat_norm / lstm_norm
        ratio_error = (norm_ratio - self.target_norm_ratio).abs()
        ratio_score = (1.0 - ratio_error / self.ratio_tolerance).clamp(0.0, 1.0)

        trust_score = gat_available * cosine_score * ratio_score
        h_fused = h_lstm_proj + trust_score * film_delta
        h_fused = self.layer_norm(h_fused)

        diagnostics_value = 1.0 - 0.45 * trust_score
        diagnostics_value = gat_available * diagnostics_value + (1.0 - gat_available) * torch.ones_like(diagnostics_value)
        return h_fused, diagnostics_value


# ============================================================================
#  DualStreamNet — 顶层容器
# ============================================================================

class DualStreamNet(nn.Module):
    """
    双流异步门控融合网络，端到端输出个股预期超额收益打分。

    所有子模块（LSTM / GAT / Fusion / PredHead）统一驻留在同一设备上。
    PyG Data 对象在 forward 前需 .to(device) 以对齐设备。

    使用方式:
        scores, gates = model(X_lstm, snap_keys, node_indices, snapshot_graphs)
    """

    def __init__(
        self,
        in_features: int,
        lstm_hidden: int = 64,
        gat_in: int = 4,
        gat_hidden: int = 64,
        gat_out: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
        min_edges_for_gat: int = 5,
        ablation_mode: str = "full",
        fusion_mode: str = "gate",
        freeze_lstm: bool = False,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.fusion_mode = fusion_mode
        self.lstm = TemporalLSTM(
            in_features=in_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            dropout=dropout,
        )
        self.gat = SpatialGAT(
            in_channels=gat_in,
            hidden_channels=gat_hidden,
            out_channels=gat_out,
            heads=gat_heads,
            dropout=dropout,
        )
        if fusion_mode == "gate":
            self.fusion = GatedFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
            )
        elif fusion_mode == "film":
            self.fusion = FiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
            )
        elif fusion_mode == "film_stable":
            self.fusion = StableFiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
            )
        elif fusion_mode == "film_stable_loose":
            self.fusion = StableFiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
                gamma_limit=0.35,
                beta_limit=0.40,
                max_delta_ratio=0.60,
            )
        elif fusion_mode == "adaptive_film":
            self.fusion = AdaptiveFiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
            )
        elif fusion_mode == "film_guard":
            self.fusion = GuardedFiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
            )
        elif fusion_mode == "soft_film_guard":
            self.fusion = SoftGuardedFiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
            )
        elif fusion_mode == "soft_film_guard_loose":
            self.fusion = SoftGuardedFiLMFusion(
                hidden_size=gat_out,
                ablation_mode=ablation_mode,
                min_cosine=-0.15,
                target_norm_ratio=1.00,
                ratio_tolerance=1.20,
            )
        else:
            raise ValueError(f"未知 fusion_mode={fusion_mode}，可选 ['gate', 'film', 'film_stable', 'film_stable_loose', 'adaptive_film', 'film_guard', 'soft_film_guard', 'soft_film_guard_loose']")
        self.pred_head = nn.Linear(gat_out, 1)

        self._gat_out_dim = gat_out
        self.min_edges_for_gat = min_edges_for_gat

        # 冻结 LSTM 权重（仅更新 GAT + Fusion + PredHead）
        if freeze_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

    def forward(
        self,
        X_lstm: torch.Tensor,
        snap_keys: List[str],
        node_indices: List[int],
        snapshot_graphs: Dict[str, Data],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X_lstm : [B, 20, F]
            时序因子序列，F 由模型构造时显式指定。
        snap_keys : List[str] 长度 B
            每个样本对应的 GAT 快照键（空前缀 "" 表示无图谱）。
        node_indices : List[int] 长度 B
            每个样本在对应快照中的节点索引（-1 表示零向量补齐）。
        snapshot_graphs : Dict[str, Data]
            本 batch 涉及的所有图谱 Data 对象。

        Returns
        -------
        scores : [B, 1] 预期超额收益打分
        gates  : [B, 1] 门控标量（用于可解释性归因）
        """
        B = X_lstm.shape[0]
        device = X_lstm.device

        # 1. LSTM 时序编码
        H_lstm = self.lstm(X_lstm)  # [B, 64]

        # 2. GAT 空间编码：逐图谱前向 + 按节点索引采样
        H_gat = self._gat_gather(snap_keys, node_indices, snapshot_graphs, device)

        # 3. 门控融合
        H_fused, gates = self.fuse_representations(H_lstm, H_gat)  # [B, 64], [B, 1]

        # 4. 预测打分
        scores = self.pred_head(H_fused)  # [B, 1]

        return scores, gates

    def _gat_gather(
        self,
        snap_keys: List[str],
        node_indices: List[int],
        snapshot_graphs: Dict[str, Data],
        device: torch.device,
    ) -> torch.Tensor:
        """
        对每个需要的图谱执行 GAT 前向传播，再按 node_indices 采样节点嵌入。

        图谱 Data 对象在此处被 .to(device) 迁移至模型所在设备，
        确保与 GAT 层参数设备一致。

        若某快照的边数 < min_edges_for_gat，跳过 GAT 前向，
        该快照对应样本的 H_GAT 保持零向量（等价于纯 LSTM 预测）。
        """
        B = len(snap_keys)
        H_gat = torch.zeros(B, self._gat_out_dim, device=device)

        if not snapshot_graphs:
            return H_gat

        from collections import defaultdict
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, sk in enumerate(snap_keys):
            if sk != "" and sk in snapshot_graphs:
                groups[sk].append(i)

        for snap_key, batch_indices in groups.items():
            g = snapshot_graphs[snap_key]
            # 稀疏图谱回退：边数不足时 GAT 提供噪声嵌入，直接跳过
            if g.edge_index.shape[1] < self.min_edges_for_gat:
                continue

            g = g.to(device)

            gat_out = self.gat(g.x, g.edge_index, g.edge_attr)  # [N, out_channels]

            for i in batch_indices:
                ni = node_indices[i]
                if 0 <= ni < gat_out.shape[0]:
                    H_gat[i] = gat_out[ni]

        return H_gat

    def forward_with_precomputed_gat(
        self,
        X_lstm: torch.Tensor,
        H_gat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用预计算的 GAT 节点嵌入执行前向传播（训练加速版）。

        跳过 _gat_gather 的逐 batch GATConv 前向，直接从预计算嵌入矩阵中读取。
        每个 epoch 只需预计算一次图谱嵌入，batch 内仅执行 LSTM + Fusion + PredHead。

        Parameters
        ----------
        X_lstm : [B, 20, F]
        H_gat  : [B, 64] 预计算的 GAT 节点嵌入（已在目标设备上）

        Returns
        -------
        scores : [B, 1], gates : [B, 1]
        """
        H_lstm = self.lstm(X_lstm)
        H_fused, gates = self.fuse_representations(H_lstm, H_gat)
        scores = self.pred_head(H_fused)
        return scores, gates

    def fuse_representations(
        self,
        h_lstm: torch.Tensor,
        h_gat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fusion(h_lstm, h_gat)

    @torch.no_grad()
    def predict_snapshot_embeddings(
        self, snapshot_graphs: Dict[str, Data], device: torch.device | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        无梯度预计算所有图谱的节点嵌入（OOS 推断加速）。

        图谱 Data 在此处完整迁移至目标设备，嵌入返回时保持在目标设备上。
        """
        self.eval()
        embeddings: Dict[str, torch.Tensor] = {}
        for snap_key, g in snapshot_graphs.items():
            g = g.to(device) if device is not None else g
            emb = self.gat(g.x, g.edge_index, g.edge_attr)
            embeddings[snap_key] = emb
        return embeddings


# ============================================================================
#  损失函数 — 排序导向回归
# ============================================================================

def rank_correlation_loss(
    scores: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor | None = None
) -> torch.Tensor:
    """
    负皮尔逊相关系数损失，直接优化预测得分与标签的线性排序一致性。

    相比 MSE，该损失更关注相对排序而非绝对数值精度，适合量化选股的
    "选好股 > 猜准收益" 场景。

    公式:
        corr = Cov(score, label) / (σ_score * σ_label)
        loss = 1 - corr   (→ 最大化相关系数)

    Parameters
    ----------
    scores : [B, 1] 或 [B]
    labels : [B, 1] 或 [B]
    masks  : [B, 1] 可选，mask=0 的样本不参与计算
    """
    s = scores.view(-1)
    l = labels.view(-1)

    if masks is not None:
        m = masks.view(-1).bool()
        s = s[m]
        l = l[m]

    if len(s) < 3:
        return torch.tensor(0.0, device=s.device, requires_grad=True)

    s_mean = s - s.mean()
    l_mean = l - l.mean()
    cov = (s_mean * l_mean).sum()
    std = s_mean.norm() * l_mean.norm()
    corr = cov / (std + 1e-8)
    return 1.0 - corr


def combined_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None = None,
    gates: torch.Tensor | None = None,
    rank_weight: float = 0.7,
    gate_reg_weight: float = 0.02,
    gate_target: float = 0.60,
    gate_std_floor: float = 0.08,
) -> torch.Tensor:
    """
    MSE + Rank Correlation 的组合损失。

    rank_weight 控制排序损失的比重：
        1.0  = 纯排序损失
        0.0  = 纯 MSE
        0.7  = 70% 排序 + 30% MSE（推荐默认）

    组合两个损失能同时利用 MSE 的梯度稳定性和排序损失的方向正确性。
    """
    mse = torch.nn.functional.mse_loss(scores.view(-1), labels.view(-1))
    rank = rank_correlation_loss(scores, labels, masks)
    loss = rank_weight * rank + (1.0 - rank_weight) * mse

    if gates is not None and gate_reg_weight > 0:
        g = gates.view(-1)
        mean_penalty = (g.mean() - gate_target) ** 2
        std_penalty = F.relu(gate_std_floor - g.std(unbiased=False)) ** 2
        loss = loss + gate_reg_weight * (mean_penalty + 0.5 * std_penalty)

    return loss


# ============================================================================
#  便捷工厂函数
# ============================================================================

def create_model(
    in_features: int | None = None,
    lstm_hidden: int = 64,
    gat_in: int | None = None,
    gat_hidden: int = 64,
    gat_out: int = 64,
    gat_heads: int = 4,
    dropout: float = 0.1,
    min_edges_for_gat: int = 5,
    ablation_mode: str = "full",
    fusion_mode: str = "gate",
    freeze_lstm: bool = False,
    device: torch.device | None = None,
) -> DualStreamNet:
    """创建双流融合网络并移至目标设备（GAT 自动处理 MPS 兼容）。"""
    if in_features is None:
        raise ValueError("create_model() 需要显式传入 in_features，不能再默认假定 16 维 LSTM 输入。")
    if gat_in is None:
        raise ValueError("create_model() 需要显式传入 gat_in，不能再默认假定 4 维节点特征。")
    model = DualStreamNet(
        in_features=in_features,
        lstm_hidden=lstm_hidden,
        gat_in=gat_in,
        gat_hidden=gat_hidden,
        gat_out=gat_out,
        gat_heads=gat_heads,
        dropout=dropout,
        min_edges_for_gat=min_edges_for_gat,
        ablation_mode=ablation_mode,
        fusion_mode=fusion_mode,
        freeze_lstm=freeze_lstm,
    )
    if device is not None:
        model = model.to(device)
    return model


# ============================================================================
#  测试入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("模块二测试：双流异步门控融合网络")
    print("=" * 60)

    device = torch.device("cpu")

    # 1. 构造虚拟数据
    print("\n[1] 构造虚拟输入 ...")
    B = 32
    X_lstm = torch.randn(B, 20, 33)
    snap_keys = ["20240430"] * 16 + ["20240831"] * 16
    node_indices = list(range(16)) + list(range(16))

    # 构造两个虚拟图谱
    N1, N2 = 200, 250
    gat_in = 11
    g1 = Data(
        x=torch.randn(N1, gat_in),
        edge_index=torch.randint(0, N1, (2, 50)),
        edge_attr=torch.rand(50, 1),
    )
    g2 = Data(
        x=torch.randn(N2, gat_in),
        edge_index=torch.randint(0, N2, (2, 60)),
        edge_attr=torch.rand(60, 1),
    )
    snapshot_graphs = {"20240430": g1, "20240831": g2}

    # 2. 创建模型
    print("\n[2] 创建 DualStreamNet ...")
    model = create_model(in_features=33, gat_in=gat_in, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:    {total_params:,}")
    print(f"  可训练参数:  {trainable_params:,}")

    # 3. 前向传播
    print("\n[3] 前向传播测试 ...")
    model.train()
    scores, gates = model(X_lstm, snap_keys, node_indices, snapshot_graphs)
    print(f"  scores: {scores.shape}, range=[{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"  gates:  {gates.shape}, mean={gates.mean().item():.4f}, std={gates.std().item():.4f}")

    # 4. 梯度流通测试
    print("\n[4] 梯度流通测试 ...")
    loss = scores.mean()
    loss.backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    print(f"  LSTM 梯度范数:   {grad_norms.get('lstm.lstm.weight_ih_l0', 0):.6f}")
    print(f"  GAT 梯度范数:    {grad_norms.get('gat.conv1.lin_src.weight', 0):.6f}")
    print(f"  Fusion 梯度范数: {grad_norms.get('fusion.lstm_proj.weight', 0):.6f}")
    print(f"  PredHead 梯度:   {grad_norms.get('pred_head.weight', 0):.6f}")

    # 5. 零向量补齐测试
    print("\n[5] 零向量补齐测试（部分 node_idx=-1）...")
    node_indices_mixed = list(range(12)) + [-1] * 4 + list(range(12)) + [-1] * 4
    scores2, gates2 = model(X_lstm, snap_keys, node_indices_mixed, snapshot_graphs)
    print(f"  scores: {scores2.shape}")

    # 6. 无图谱场景
    print("\n[6] 无图谱场景（空快照）...")
    snap_keys_empty = [""] * B
    node_indices_empty = [-1] * B
    scores3, gates3 = model(X_lstm, snap_keys_empty, node_indices_empty, {})
    print(f"  scores: {scores3.shape}")
    # 此时 H_GAT 全为零，GAT 部分不贡献梯度，融合应正常输出

    print("\n" + "=" * 60)
    print("模块二测试通过！")
    print("=" * 60)
