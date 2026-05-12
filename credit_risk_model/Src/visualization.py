"""
==============================================================================
可视化模块 - 结果可视化与图表生成
==============================================================================

模块说明:
    本模块负责生成各种可视化图表，包括：
    1. 模型性能对比图
    2. ROC曲线
    3. 特征重要性图
    4. 预测分布图
    5. 目标分布图

注意:
    macOS 中文显示需要设置正确的字体。
    本模块使用 matplotlib 生成图表。

使用方法:
    from visualization import plot_model_comparison, plot_feature_importance

作者: 信用风险模型课程作业版
日期: 2024
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict, Optional

# ==============================================================================
# macOS 中文字体设置
# ==============================================================================

def setup_chinese_font():
    """
    设置 matplotlib 支持中文显示

    macOS 常用中文字体:
        - PingFang SC (苹方)
        - Heiti SC (黑体)
        - STHeiti
        - Songti SC (宋体)

    如果找不到字体，会尝试多个备选方案。
    """
    # 尝试的中文字体列表（按优先级排序）
    chinese_fonts = [
        'PingFang SC',      # macOS 默认中文字体
        'Heiti SC',         # 黑体
        'STHeiti',          # 华文黑体
        'Songti SC',        # 宋体
        'Arial Unicode MS', # Unicode字体
        'SimHei',           # 黑体（Windows/Linux）
        'Microsoft YaHei',  # 微软雅黑（Windows）
    ]

    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 找到第一个可用的中文字体
    font_found = None
    for font in chinese_fonts:
        if font in available_fonts:
            font_found = font
            break

    if font_found:
        # 设置字体
        plt.rcParams['font.sans-serif'] = [font_found]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"已设置中文字体: {font_found}")
    else:
        # 如果找不到中文字体，使用系统默认方法
        print("警告: 未找到合适的中文字体，尝试其他方法...")
        try:
            # 使用 matplotlib 的字体管理器查找
            font_path = None
            for font in fm.fontManager.ttflist:
                if 'PingFang' in font.name or 'Heiti' in font.name or 'SC' in font.name:
                    font_path = font.fname
                    break

            if font_path:
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                print(f"使用字体文件: {font_path}")
        except Exception as e:
            print(f"字体设置失败: {e}")
            print("图表可能无法正确显示中文")


# ==============================================================================
# 模型性能可视化
# ==============================================================================

def plot_model_comparison(lgb_scores: List[float], cat_scores: List[float],
                         output_path: Path, title: str = "模型性能对比"):
    """
    绘制 LightGBM 和 CatBoost 各折性能对比图

    参数:
        lgb_scores: LightGBM 各折 AUC 分数
        cat_scores: CatBoost 各折 AUC 分数
        output_path: 输出图片路径
        title: 图表标题
    """
    # 设置字体
    setup_chinese_font()

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 左图: 各折 AUC 对比 ==========
    ax1 = axes[0]
    folds = range(1, len(lgb_scores) + 1)

    # 绘制柱状图
    bars1 = ax1.bar([f - 0.2 for f in folds], lgb_scores, width=0.4,
                    label='LightGBM', color='#3498db', alpha=0.8)
    bars2 = ax1.bar([f + 0.2 for f in folds], cat_scores, width=0.4,
                    label='CatBoost', color='#e74c3c', alpha=0.8)

    # 添加数值标签
    for bar, score in zip(bars1, lgb_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)
    for bar, score in zip(bars2, cat_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    ax1.set_xlabel('折数', fontsize=12)
    ax1.set_ylabel('AUC 分数', fontsize=12)
    ax1.set_title('各折交叉验证 AUC', fontsize=14)
    ax1.set_xticks(folds)
    ax1.set_xticklabels([f'Fold {f}' for f in folds])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(min(min(lgb_scores), min(cat_scores)) - 0.02,
                 max(max(lgb_scores), max(cat_scores)) + 0.02)
    ax1.grid(axis='y', alpha=0.3)

    # ========== 右图: 平均 AUC 对比 ==========
    ax2 = axes[1]

    lgb_mean = np.mean(lgb_scores)
    cat_mean = np.mean(cat_scores)
    lgb_std = np.std(lgb_scores)
    cat_std = np.std(cat_scores)

    models = ['LightGBM', 'CatBoost', '融合模型']
    means = [lgb_mean, cat_mean, (lgb_mean + cat_mean) / 2]
    stds = [lgb_std, cat_std, min(lgb_std, cat_std)]
    colors = ['#3498db', '#e74c3c', '#9b59b6']

    bars = ax2.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=10)

    ax2.set_ylabel('AUC 分数', fontsize=12)
    ax2.set_title('平均 AUC 对比（含标准差）', fontsize=14)
    ax2.set_ylim(0.5, max(means) + 0.05)
    ax2.grid(axis='y', alpha=0.3)

    # 添加水平参考线
    ax2.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='良好基准线(0.75)')
    ax2.axhline(y=0.70, color='orange', linestyle='--', alpha=0.5, label='一般基准线(0.70)')
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"模型对比图已保存: {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray,
                   output_path: Path, title: str = "ROC曲线"):
    """
    绘制 ROC 曲线

    参数:
        y_true: 真实标签
        y_pred: 预测概率
        output_path: 输出图片路径
        title: 图表标题
    """
    from sklearn.metrics import roc_curve, auc

    setup_chinese_font()

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制 ROC 曲线
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')

    # 绘制对角线（随机预测）
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机预测 (AUC = 0.5)')

    # 填充 ROC 曲线下区域
    ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"ROC曲线已保存: {output_path}")


# ==============================================================================
# 特征重要性可视化
# ==============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path,
                            top_n: int = 20, title: str = "特征重要性排名"):
    """
    绘制特征重要性柱状图

    参数:
        importance_df: 特征重要性 DataFrame
        output_path: 输出图片路径
        top_n: 显示前 N 个特征
        title: 图表标题
    """
    setup_chinese_font()

    if importance_df.empty:
        print("特征重要性数据为空，无法绘图")
        return

    # 获取前 top_n 个特征
    top_features = importance_df.head(top_n)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制水平柱状图
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color='#2ecc71', alpha=0.8)

    # 设置标签
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax.invert_yaxis()  # 最重要的在顶部

    # 添加数值标签
    for bar, importance in zip(bars, top_features['importance'].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', ha='left', va='center', fontsize=9)

    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_ylabel('特征名称', fontsize=12)
    ax.set_title(f'{title} (Top {top_n})', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"特征重要性图已保存: {output_path}")


# ==============================================================================
# 预测分布可视化
# ==============================================================================

def plot_prediction_distribution(predictions: np.ndarray, output_path: Path,
                                 title: str = "预测概率分布"):
    """
    绘制预测概率分布图

    参数:
        predictions: 预测概率数组
        output_path: 输出图片路径
        title: 图表标题
    """
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 左图: 直方图 ==========
    ax1 = axes[0]

    # 绘制直方图
    ax1.hist(predictions, bins=50, color='#3498db', alpha=0.7, edgecolor='white')

    # 添加统计信息
    mean_val = np.mean(predictions)
    median_val = np.median(predictions)

    ax1.axvline(mean_val, color='red', linestyle='--', lw=2, label=f'均值: {mean_val:.4f}')
    ax1.axvline(median_val, color='green', linestyle='--', lw=2, label=f'中位数: {median_val:.4f}')

    ax1.set_xlabel('预测概率', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('预测概率直方图', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)

    # ========== 右图: 箱线图 ==========
    ax2 = axes[1]

    # 绘制箱线图
    box = ax2.boxplot(predictions, patch_artist=True, vert=True)
    box['boxes'][0].set_facecolor('#3498db')
    box['boxes'][0].set_alpha(0.7)

    ax2.set_ylabel('预测概率', fontsize=12)
    ax2.set_title('预测概率箱线图', fontsize=14)
    ax2.grid(alpha=0.3)

    # 添加统计信息文本
    stats_text = f"""
    最小值: {np.min(predictions):.4f}
    最大值: {np.max(predictions):.4f}
    均值: {mean_val:.4f}
    中位数: {median_val:.4f}
    标准差: {np.std(predictions):.4f}
    """
    ax2.text(1.3, median_val, stats_text, fontsize=10, va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"预测分布图已保存: {output_path}")


def plot_target_distribution(y: pd.Series, output_path: Path,
                             title: str = "目标变量分布"):
    """
    绘制目标变量（违约/不违约）分布图

    参数:
        y: 目标变量 Series
        output_path: 输出图片路径
        title: 图表标题
    """
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ========== 左图: 柱状图 ==========
    ax1 = axes[0]

    target_counts = y.value_counts()
    labels = ['不违约 (0)', '违约 (1)']
    colors = ['#2ecc71', '#e74c3c']

    bars = ax1.bar(labels, target_counts.values, color=colors, alpha=0.8)

    # 添加数值标签
    for bar, count in zip(bars, target_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count}\n({count/len(y)*100:.1f}%)', ha='center', va='bottom', fontsize=11)

    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('目标变量分布', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    # ========== 右图: 饼图 ==========
    ax2 = axes[1]

    ax2.pie(target_counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0, 0.1), shadow=True)

    ax2.set_title('目标变量比例', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"目标分布图已保存: {output_path}")


# ==============================================================================
# 综合可视化函数
# ==============================================================================

def generate_all_visualizations(lgb_scores: List[float], cat_scores: List[float],
                                predictions: np.ndarray, importance_df: pd.DataFrame,
                                output_dir: Path, y_train: Optional[pd.Series] = None):
    """
    生成所有可视化图表

    参数:
        lgb_scores: LightGBM 各折 AUC
        cat_scores: CatBoost 各折 AUC
        predictions: 预测概率
        importance_df: 特征重要性 DataFrame
        output_dir: 输出目录
        y_train: 训练集目标变量（可选）
    """
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 模型性能对比图
    plot_model_comparison(
        lgb_scores, cat_scores,
        output_dir / "model_comparison.png",
        "LightGBM vs CatBoost 性能对比"
    )

    # 2. 特征重要性图
    if not importance_df.empty:
        plot_feature_importance(
            importance_df,
            output_dir / "feature_importance.png",
            top_n=20,
            title="特征重要性排名"
        )

    # 3. 预测分布图
    plot_prediction_distribution(
        predictions,
        output_dir / "prediction_distribution.png",
        title="测试集预测概率分布"
    )

    # 4. 目标分布图（如果有训练集目标）
    if y_train is not None:
        plot_target_distribution(
            y_train,
            output_dir / "target_distribution.png",
            title="训练集目标变量分布"
        )

    print(f"\n所有图表已保存到: {output_dir}")
    print("生成的图表:")
    print("  - model_comparison.png: 模型性能对比")
    print("  - feature_importance.png: 特征重要性排名")
    print("  - prediction_distribution.png: 预测概率分布")
    if y_train is not None:
        print("  - target_distribution.png: 目标变量分布")


# ==============================================================================
# 测试字体设置
# ==============================================================================

def test_chinese_display():
    """
    测试中文显示是否正常
    """
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, '中文测试：信用风险预测模型',
            fontsize=20, ha='center', va='center')
    ax.set_title('中文标题测试', fontsize=14)
    ax.set_xlabel('X轴中文', fontsize=12)
    ax.set_ylabel('Y轴中文', fontsize=12)

    plt.tight_layout()
    test_path = Path('/Users/huzhichao/Desktop/credit_risk_model/simple_version/results/chinese_test.png')
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"中文测试图已保存: {test_path}")


if __name__ == "__main__":
    # 测试字体设置
    test_chinese_display()