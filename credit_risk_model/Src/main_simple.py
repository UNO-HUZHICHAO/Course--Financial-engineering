"""
==============================================================================
主程序 - 信用风险预测完整流程（简化版）
==============================================================================

模块说明:
    本模块是项目的入口文件，执行完整建模流程：
    1. 数据加载：从 CSV 文件加载训练集和测试集
    2. 数据预处理：过滤、转换、填充缺失值
    3. 模型训练：3折交叉验证，LightGBM + CatBoost
    4. 模型融合：VotingModel 组合 6 个模型
    5. 预测提交：对测试集预测，生成提交文件

执行方法:
    python main_simple.py

流程图:
    加载CSV → 合并多表 → 预处理 → 分离特征 → 训练模型 → 预测 → 提交

完整流程说明:
    Step 1: 数据加载
        - 加载 train_base.csv（基础表，含目标变量）
        - 加载 train_static_*.csv（静态表）
        - 合并为单一 DataFrame

    Step 2: 数据预处理
        - 删除高缺失列（>95%）
        - 删除低效类别列（唯一值过多）
        - 转换日期特征为相对天数
        - 填充剩余缺失值
        - 转换类别特征类型

    Step 3: 特征分离
        - X: 特征数据
        - y: 目标变量
        - case_id: 客户标识符
        - week_num: 时间分组

    Step 4: 模型训练
        - 3折交叉验证
        - 每折训练 LightGBM + CatBoost
        - 共 6 个模型

    Step 5: 预测和提交
        - 对测试集预测
        - 生成 submission.csv

作者: 信用风险模型课程作业版
日期: 2024
==============================================================================
"""

import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from pathlib import Path

from config_simple import (
    TRAIN_DIR, TEST_DIR, TRAIN_TABLES, TEST_TABLES,
    SAMPLE_SUB_FILE, OUTPUT_FILE, OUTPUT_DIR, FEATURE_DEF_FILE,
    N_SPLITS
)
from preprocessing_simple import (
    load_and_merge_data, preprocess_data, split_features_target, get_test_features
)
from model_simple import (
    train_models, VotingModel, predict, evaluate_model, print_evaluation_results,
    get_feature_importance, print_top_features
)
from visualization import generate_all_visualizations


def main():
    """
    ==================================================================
    功能：主函数 - 执行完整建模流程
    ==================================================================

    执行流程:
        1. 加载训练数据
        2. 预处理训练数据
        3. 加载测试数据
        4. 预处理测试数据
        5. 分离特征和目标
        6. 训练模型（交叉验证）
        7. 预测测试集
        8. 生成提交文件
        9. 分析特征重要性
    ==================================================================
    """

    # 打印项目信息
    print("\n" + "=" * 60)
    print("信用风险预测模型（简化版）")
    print("=" * 60)
    print("\n项目信息:")
    print("  模型类型: LightGBM + CatBoost")
    print("  融合策略: 投票平均")
    print("  交叉验证: 3折 StratifiedKFold")
    print("  数据格式: CSV（Pandas）")

    # ==============================================================
    # Step 1: 加载训练数据
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 1: 加载训练数据")
    print("=" * 60)

    # 加载并合并多个数据表
    # 使用 TRAIN_TABLES 配置中定义的表
    train_raw = load_and_merge_data(TRAIN_DIR, TRAIN_TABLES)

    # 清理内存
    gc.collect()

    # ==============================================================
    # Step 2: 预处理训练数据
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 2: 预处理训练数据")
    print("=" * 60)

    # 预处理数据（过滤、转换、填充）
    # 返回处理后的数据框和类别列列表
    train_processed, cat_cols = preprocess_data(train_raw, is_train=True)

    # 清理内存（删除原始数据）
    del train_raw
    gc.collect()

    # ==============================================================
    # Step 3: 分离特征和目标变量
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 3: 分离特征和目标变量")
    print("=" * 60)

    # 分离特征（X）、目标（y）、标识符（case_id）、时间分组（week_num）
    X, y, train_ids, week_nums = split_features_target(train_processed)

    # 记录训练集特征列（测试集需要使用相同的列）
    train_columns = X.columns.tolist()

    # 清理内存
    del train_processed
    gc.collect()

    # ==============================================================
    # Step 4: 加载测试数据
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 4: 加载测试数据")
    print("=" * 60)

    # 加载并合并测试数据表
    test_raw = load_and_merge_data(TEST_DIR, TEST_TABLES)

    # ==============================================================
    # Step 5: 预处理测试数据
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 5: 预处理测试数据")
    print("=" * 60)

    # 预处理测试数据
    # 注意：测试集没有 target 列
    test_processed, cat_cols_test = preprocess_data(test_raw, is_train=False)

    # 清理内存
    del test_raw
    gc.collect()

    # ==============================================================
    # Step 6: 准备测试特征
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 6: 准备测试特征")
    print("=" * 60)

    # 准备测试特征（确保与训练集特征一致）
    X_test, test_ids = get_test_features(test_processed, train_columns)

    # 清理内存
    del test_processed
    gc.collect()

    # ==============================================================
    # Step 7: 模型训练
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 7: 模型训练")
    print("=" * 60)

    # 训练模型（3折交叉验证）
    # 返回：融合模型、LightGBM分数列表、CatBoost分数列表
    model, lgb_scores, cat_scores = train_models(X, y, cat_cols)

    # 保存训练集目标变量用于可视化
    y_train_for_visualization = y.copy()

    # 清理内存
    del X
    gc.collect()

    # ==============================================================
    # Step 8: 预测测试集
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 8: 预测测试集")
    print("=" * 60)

    # 使用融合模型预测
    predictions = predict(model, X_test, cat_cols)

    # ==============================================================
    # Step 9: 生成提交文件
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 9: 生成提交文件")
    print("=" * 60)

    # 创建提交 DataFrame
    submission = pd.DataFrame({
        'case_id': test_ids,
        'score': predictions
    })

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存提交文件
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n提交文件已保存: {OUTPUT_FILE}")
    print(f"  文件形状: {submission.shape[0]} 行, {submission.shape[1]} 列")

    # ==============================================================
    # Step 10: 特征重要性分析
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 10: 特征重要性分析")
    print("=" * 60)

    # 获取特征重要性
    # 从融合模型的各个子模型中提取
    importance_df = get_feature_importance(model.estimators, train_columns)

    if not importance_df.empty:
        # 打印最重要的特征
        print_top_features(importance_df, top_n=20)

        # 保存特征重要性到文件
        importance_file = OUTPUT_DIR / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        print(f"\n特征重要性已保存: {importance_file}")

    # ==============================================================
    # Step 11: 生成可视化图表
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 11: 生成可视化图表")
    print("=" * 60)

    # 生成所有可视化图表
    generate_all_visualizations(
        lgb_scores=lgb_scores,
        cat_scores=cat_scores,
        predictions=predictions,
        importance_df=importance_df,
        output_dir=OUTPUT_DIR,
        y_train=y_train_for_visualization
    )

    # ==============================================================
    # Step 12: 保存结果摘要
    # ==============================================================
    print("\n" + "=" * 60)
    print("Step 12: 保存结果摘要")
    print("=" * 60)

    # 创建结果摘要文件
    summary_file = OUTPUT_DIR / "results_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("信用风险预测模型 - 结果摘要\n")
        f.write("=" * 60 + "\n\n")

        f.write("[模型配置]\n")
        f.write(f"  模型类型: LightGBM + CatBoost\n")
        f.write(f"  交叉验证: {N_SPLITS} 折 StratifiedKFold\n")
        f.write(f"  融合策略: 投票平均\n")
        f.write(f"  总模型数: {len(model.estimators)} 个\n\n")

        f.write("[LightGBM 结果]\n")
        f.write(f"  各折 AUC: {[f'{s:.4f}' for s in lgb_scores]}\n")
        f.write(f"  平均 AUC: {np.mean(lgb_scores):.4f}\n")
        f.write(f"  AUC 标准差: {np.std(lgb_scores):.4f}\n\n")

        f.write("[CatBoost 结果]\n")
        f.write(f"  各折 AUC: {[f'{s:.4f}' for s in cat_scores]}\n")
        f.write(f"  平均 AUC: {np.mean(cat_scores):.4f}\n")
        f.write(f"  AUC 标准差: {np.std(cat_scores):.4f}\n\n")

        f.write("[预测统计]\n")
        f.write(f"  测试样本数: {len(predictions)}\n")
        f.write(f"  预测概率范围: [{predictions.min():.4f}, {predictions.max():.4f}]\n")
        f.write(f"  预测概率均值: {predictions.mean():.4f}\n")
        f.write(f"  预测概率中位数: {np.median(predictions):.4f}\n\n")

        f.write("[特征重要性 Top 10]\n")
        if not importance_df.empty:
            for i, row in enumerate(importance_df.head(10).itertuples(), 1):
                f.write(f"  {i}. {row.feature}: {row.importance:.4f}\n")

        f.write("\n[输出文件]\n")
        f.write(f"  提交文件: {OUTPUT_FILE}\n")
        f.write(f"  特征重要性: {OUTPUT_DIR / 'feature_importance.csv'}\n")
        f.write(f"  结果摘要: {summary_file}\n")
        f.write(f"  可视化图表: {OUTPUT_DIR}/*.png\n")

    print(f"结果摘要已保存: {summary_file}")

    # ==============================================================
    # 完成！
    # ==============================================================
    print("\n" + "=" * 60)
    print("建模流程完成！")
    print("=" * 60)

    print("\n[最终结果]")
    print(f"  LightGBM 平均 AUC: {np.mean(lgb_scores):.4f}")
    print(f"  CatBoost 平均 AUC: {np.mean(cat_scores):.4f}")
    print(f"  融合模型数量: {len(model.estimators)} 个")
    print(f"  提交文件: {OUTPUT_FILE}")

    print("\n[下一步]")
    print("  1. 检查提交文件格式")
    print("  2. 上传到 Kaggle 查看得分")
    print("  3. 分析特征重要性，优化模型")

    return model, lgb_scores, cat_scores


# ==============================================================================
# 程序入口
# ==============================================================================

if __name__ == "__main__":
    """
    当直接运行此文件时，执行主函数:
        python main_simple.py

    说明:
        - __name__ == "__main__" 表示文件被直接运行
        - 如果文件被导入（import），则不会执行 main()
    """
    # 执行主函数
    main()