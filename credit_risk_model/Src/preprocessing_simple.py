"""
==============================================================================
数据预处理模块 - 特征清洗、转换和多表合并（简化版）
==============================================================================

模块说明:
    本模块负责数据加载和预处理，包括：
    1. 多表数据加载：从 CSV 文件加载训练和测试数据
    2. 多表合并：将多个表通过 case_id 连接
    3. 特征过滤：删除高缺失和低效特征
    4. 日期处理：将日期转换为相对天数
    5. 缺失值处理：填充剩余缺失值
    6. 类别特征转换：转为适合模型的类型

预处理流程:
    CSV文件 → Pandas DataFrame → 合多表 → 过滤特征 → 处理日期 → 填缺失 → 转类型

为什么需要预处理:
    1. 原始数据不适合直接用于机器学习模型
    2. 数据可能有缺失、噪声、不一致等问题
    3. 合理的预处理可以提高模型性能
    4. 特征工程是机器学习中最关键的步骤之一

使用方法:
    from preprocessing_simple import load_and_merge_data, preprocess_data

    train = load_and_merge_data(TRAIN_DIR, TRAIN_TABLES)
    train, cat_cols = preprocess_data(train)

作者: 信用风险模型课程作业版
日期: 2024
==============================================================================
"""

import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from config_simple import (
    TRAIN_DIR, TEST_DIR, TRAIN_TABLES, TEST_TABLES,
    NULL_THRESHOLD, CAT_UNIQUE_THRESHOLD
)


# ==============================================================================
# 数据加载函数
# ==============================================================================

def load_single_table(directory: Path, filename: str) -> pd.DataFrame:
    """
    ==================================================================
    功能：加载单个 CSV 数据表
    ==================================================================

    参数说明:
        directory: 数据目录路径（如 TRAIN_DIR）
        filename: CSV 文件名（如 "train_base.csv"）

    加载流程:
        1. 构建完整文件路径
        2. 使用 pandas.read_csv() 读取 CSV
        3. 打印加载信息（形状、列数等）

    为什么使用 CSV 而非 Parquet:
        1. CSV 更通用，可以用 Excel 查看
        2. Pandas 对 CSV 支持更简单直接
        3. 课程作业数据量小，CSV 足够高效
        4. 学生更容易理解 CSV 格式

    返回值:
        pd.DataFrame: 加载的数据框
    ==================================================================
    """
    # 构建完整文件路径
    file_path = directory / filename

    # 检查文件是否存在
    if not file_path.exists():
        print(f"警告: 文件不存在 {file_path}")
        return pd.DataFrame()  # 返回空数据框

    # 使用 pandas 读取 CSV
    # low_memory=False: 更准确的类型推断，但占用更多内存
    df = pd.read_csv(file_path, low_memory=False)

    # 打印加载信息
    print(f"  已加载: {filename}")
    print(f"    形状: {df.shape[0]} 行, {df.shape[1]} 列")

    return df


def load_and_merge_data(directory: Path, tables: Dict[str, str]) -> pd.DataFrame:
    """
    ==================================================================
    功能：加载多个数据表并合并
    ==================================================================

    参数说明:
        directory: 数据目录（如 TRAIN_DIR 或 TEST_DIR）
        tables: 表配置字典（键为表名，值为文件名）
               例如: {"base": "train_base.csv", "static": "train_static.csv"}

    合并流程:
        1. 加载基础表（包含 case_id 和 target）
        2. 加载其他静态表
        3. 使用 case_id 进行左连接
        4. 合并后的数据每个客户一行

    左连接说明:
        - how="left": 保留基础表的所有记录
        - on="case_id": 使用 case_id 作为连接键
        - 其他表的记录通过 case_id 匹配连接到基础表

    为什么用左连接而非其他连接:
        1. 基础表是主表，包含所有客户和目标变量
        2. 我们需要保留所有训练样本
        3. 其他表可能没有某些客户的记录（缺失值填充）
        4. 不希望丢失任何训练样本

    多表合并的意义:
        1. 不同表包含不同维度的客户信息
        2. 合并后可以获得更全面的客户画像
        3. 静态表：基本信息、信用局数据等
        4. 合并后特征数量增加，信息更丰富

    返回值:
        pd.DataFrame: 合并后的数据框（每个客户一行）
    ==================================================================
    """
    print("\n" + "=" * 60)
    print("开始加载和合并数据表...")
    print("=" * 60)

    # 存储所有数据表
    dfs = {}

    # ========== Step 1: 加载所有表 ==========
    for table_name, filename in tables.items():
        print(f"\n[加载 {table_name} 表]")
        df = load_single_table(directory, filename)
        if not df.empty:
            dfs[table_name] = df

    # ========== Step 2: 检查基础表 ==========
    # 基础表必须存在，包含 case_id 和 target（训练集）
    if "base" not in dfs:
        raise ValueError("错误: 基础表 (base) 未找到，无法继续!")

    # 以基础表作为合并起点
    merged_df = dfs["base"]
    print(f"\n基础表信息:")
    print(f"  行数: {merged_df.shape[0]}")
    print(f"  列数: {merged_df.shape[1]}")
    print(f"  列名: {list(merged_df.columns)}")

    # ========== Step 3: 合并其他表 ==========
    # 按顺序将其他表连接到基础表
    for table_name, df in dfs.items():
        if table_name == "base":
            continue  # 跳过基础表

        print(f"\n[合并 {table_name} 表]")

        # 检查是否有 case_id 列（连接键）
        if "case_id" not in df.columns:
            print(f"  警告: {table_name} 表没有 case_id 列，跳过合并")
            continue

        # 左连接合并
        # suffixes: 如果列名冲突，添加后缀区分
        merged_df = pd.merge(
            merged_df, df,
            on="case_id",
            how="left",
            suffixes=("", f"_{table_name}")
        )

        print(f"  合并后形状: {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")

    # ========== Step 4: 清理内存 ==========
    # 删除不再需要的单独数据表
    del dfs
    gc.collect()

    print(f"\n合并完成!")
    print(f"  最终形状: {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")

    return merged_df


# ==============================================================================
# 特征过滤函数
# ==============================================================================

def filter_high_missing_cols(df: pd.DataFrame, threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    """
    ==================================================================
    功能：删除高缺失率的特征列
    ==================================================================

    参数说明:
        df: 待处理的数据框
        threshold: 缺失比例阈值（默认 0.95，即 95%）

    过滤逻辑:
        1. 计算每列的缺失比例
        2. 缺失比例 > 阈值的列被删除
        3. 排除关键列（case_id, target, week_num）

    为什么删除高缺失列:
        1. 信息量极少：
           - 如果 95% 的值都缺失，这列几乎没有有用信息
           - 即使填充，也只是猜测

        2. 可能引入噪声：
           - 大量缺失值填充后可能不准确
           - 可能误导模型

        3. 节省计算资源：
           - 删除无用列可以减少内存占用
           - 加快训练速度

    为什么阈值是 95%:
        - 低于 95% 的缺失可以用合理方法填充
        - 高于 95% 的缺失，填充值占比太高，不可信
        - 这是一个经验阈值，可以根据数据调整

    缺失比例计算方法:
        df[col].isna().sum() / len(df)
        - isna(): 检查是否为空值，返回布尔 Series
        - sum(): 计算空值数量（True 视为 1）
        - 除以总行数得到比例

    返回值:
        pd.DataFrame: 过滤后的数据框
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("过滤高缺失列...")
    print("-" * 40)

    # 关键列不进行过滤
    # case_id: 客户标识符
    # target: 目标变量（训练集）
    # week_num: 时间分组
    key_cols = ["case_id", "target", "week_num", "WEEK_NUM"]

    # 记录删除的列
    cols_to_drop = []

    # 遍历所有列
    for col in df.columns:
        if col in key_cols:
            continue  # 跳过关键列

        # 计算缺失比例
        null_ratio = df[col].isna().sum() / len(df)

        # 如果缺失比例超过阈值，标记删除
        if null_ratio > threshold:
            cols_to_drop.append(col)
            print(f"  删除: {col} (缺失率: {null_ratio:.2%})")

    # 删除列
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"\n删除了 {len(cols_to_drop)} 个高缺失列")
    else:
        print("\n没有需要删除的高缺失列")

    return df


def filter_low_effective_cat_cols(df: pd.DataFrame, threshold: int = CAT_UNIQUE_THRESHOLD) -> pd.DataFrame:
    """
    ==================================================================
    功能：删除低效的类别特征列
    ==================================================================

    参数说明:
        df: 待处理的数据框
        threshold: 唯一值数量阈值（默认 100）

    过滤逻辑:
        1. 找到所有类别列（object 或 category 类型）
        2. 计算每个类别列的唯一值数量
        3. 删除唯一值 > threshold 或唯一值 = 1 的列

    为什么删除唯一值过多的类别列:
        1. 防止特征爆炸:
           - 如果用 One-Hot 编码，每个唯一值变成一列
           - 100 个唯一值 = 100 列，消耗大量内存

        2. 统计不稳定:
           - 每个类别的样本太少
           - 模型难以学到每个类别的规律

        3. 容易过拟合:
           - 模型可能记住某些罕见类别的噪声
           - 泛化能力下降

    为什么删除唯一值为 1 的类别列:
        1. 无区分度:
           - 如果所有样本的类别值相同
           - 这列对预测没有帮助

        2. 白白占用内存:
           - 删除后不影响模型性能

    唯一值数量计算:
        df[col].nunique()
        - nunique(): 计算不同值的数量（排除 NaN）

    返回值:
        pd.DataFrame: 过滤后的数据框
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("过滤低效类别列...")
    print("-" * 40)

    # 关键列不进行过滤
    key_cols = ["case_id", "target", "week_num", "WEEK_NUM"]

    # 记录删除的列
    cols_to_drop = []

    # 遍历所有列
    for col in df.columns:
        if col in key_cols:
            continue  # 跳过关键列

        # 只处理类别类型的列
        # object: Python 字符串类型
        # category: Pandas 类别类型
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            # 计算唯一值数量
            n_unique = df[col].nunique()

            # 判断是否需要删除
            # 条件1: 唯一值过多 (> threshold)
            # 条件2: 唯一值为 1（无区分度）
            if n_unique > threshold or n_unique == 1:
                cols_to_drop.append(col)
                print(f"  删除: {col} (唯一值数量: {n_unique})")

    # 删除列
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"\n删除了 {len(cols_to_drop)} 个低效类别列")
    else:
        print("\n没有需要删除的低效类别列")

    return df


# ==============================================================================
# 日期处理函数
# ==============================================================================

def handle_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ==================================================================
    功能：处理日期特征
    ==================================================================

    参数说明:
        df: 待处理的数据框
        必须包含 date_decision 列作为基准日期

    处理逻辑:
        1. 将 date_decision 转换为 datetime 类型
        2. 从 date_decision 提取 year、month 信息
        3. 对于其他日期列（以 D 结尾），转换为相对天数
        4. 删除原始日期列

    为什么将日期转换为相对天数:
        1. 时间稳定性:
           - 绝对日期随时间变化（如 "2024-01-15"）
           - 模型可能记住特定日期，导致过拟合
           - 相对天数（如 "距决策30天"）更稳定

        2. 语义更清晰:
           - "距决策30天" 比 "2024-01-15" 更有含义
           - 直接反映申请、还款等行为的时间距离

        3. 模型更友好:
           - 数值特征更适合机器学习模型
           - 可以计算统计量（平均值、方差等）

    日期转换示例:
        原始数据:
        - first_app_D: 2024-01-15
        - date_decision: 2024-02-15

        转换后:
        - first_app_D_days: -30（申请在决策前30天）

    返回值:
        pd.DataFrame: 处理后的数据框
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("处理日期特征...")
    print("-" * 40)

    # 检查是否有决策日期列
    if "date_decision" not in df.columns:
        print("  没有找到 date_decision 列，跳过日期处理")
        return df

    # ========== Step 1: 转换基准日期 ==========
    # 将决策日期转换为 datetime 类型
    df["date_decision"] = pd.to_datetime(df["date_decision"], errors="coerce")

    # 提取年份和月份作为新特征
    df["decision_year"] = df["date_decision"].dt.year
    df["decision_month"] = df["date_decision"].dt.month

    print(f"  提取了 decision_year 和 decision_month")

    # ========== Step 2: 转换其他日期列 ==========
    # 找到所有以 D 结尾的日期列（排除 date_decision）
    date_cols = [col for col in df.columns
                 if col.endswith("D") and col != "date_decision"]

    if date_cols:
        print(f"  找到 {len(date_cols)} 个日期特征列")

        for col in date_cols:
            # 转换为 datetime
            df[col] = pd.to_datetime(df[col], errors="coerce")

            # 计算相对天数
            # 结果 = 该日期 - 决策日期
            # 正数表示在决策之后，负数表示在决策之前
            df[f"{col}_days"] = (df[col] - df["date_decision"]).dt.days

            # 删除原始日期列
            df = df.drop(columns=[col])
            print(f"    转换: {col} -> {col}_days")

    # ========== Step 3: 删除基准日期列 ==========
    # 已经提取了 year 和 month，原始日期不再需要
    df = df.drop(columns=["date_decision"])
    print("  删除了 date_decision 列")

    return df


# ==============================================================================
# 缺失值处理函数
# ==============================================================================

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    ==================================================================
    功能：填充缺失值
    ==================================================================

    参数说明:
        df: 待处理的数据框

    填充策略:
        - 数值列：使用中位数填充
        - 类别列：使用 "missing" 字符串填充

    为什么数值列用中位数而非平均值:
        1. 抗极端值:
           - 平均值受极端值影响大
           - 中位数不受极端值影响

        2. 更稳健:
           - 信用数据常有极端值（如超大收入）
           - 中位数更代表"典型值"

        3. 经验证明:
           - 中位数填充通常效果更好

    为什么类别列用 "missing" 字符串:
        1. 明确标记:
           - "missing" 表示这是填充值，不是真实值
           - 模型可以学习"缺失"本身可能是一个信号

        2. 类别完整:
           - 类别特征不能用数值填充
           - 需要保持类别类型一致性

    填充方法:
        df[col].fillna(value)
        - fillna(): 将 NaN 替换为指定值

    返回值:
        pd.DataFrame: 填充后的数据框
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("填充缺失值...")
    print("-" * 40)

    # ========== Step 1: 数值列填充 ==========
    # 选择所有数值类型的列
    # include=["number"]: 包含 int、float 等数值类型
    num_cols = df.select_dtypes(include=["number"]).columns

    num_filled = 0
    for col in num_cols:
        # 检查是否有缺失值
        if df[col].isna().any():
            # 计算中位数
            median_val = df[col].median()

            # 填充缺失值
            df[col] = df[col].fillna(median_val)
            num_filled += 1

    print(f"  数值列填充: {num_filled} 列（使用中位数）")

    # ========== Step 2: 类别列填充 ==========
    # 选择所有类别类型的列
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    cat_filled = 0
    for col in cat_cols:
        # 检查是否有缺失值
        if df[col].isna().any():
            # 填充为 "missing"
            df[col] = df[col].fillna("missing")
            cat_filled += 1

    print(f"  类别列填充: {cat_filled} 列（使用 'missing'）")

    # ========== Step 3: 检查填充结果 ==========
    # 确保没有剩余缺失值
    remaining_null = df.isna().sum().sum()
    if remaining_null > 0:
        print(f"  警告: 仍有 {remaining_null} 个缺失值")
    else:
        print("  缺失值填充完成!")

    return df


# ==============================================================================
# 类别特征类型转换函数
# ==============================================================================

def convert_category_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    ==================================================================
    功能：转换类别特征的数据类型
    ==================================================================

    参数说明:
        df: 待处理的数据框

    转换逻辑:
        1. 找到所有 object 类型的类别列
        2. 将它们转换为 category 类型
        3. 返回类别列列表

    为什么需要转换为 category 类型:
        1. LightGBM 支持:
           - LightGBM 可以直接处理 category 类型
           - 自动进行高效的类别编码
           - 不需要手动 One-Hot 编码

        2. CatBoost 支持:
           - CatBoost 可以通过指定 cat_features 参数
           - 处理 category 或 object 类型的类别特征

        3. 内存效率:
           - category 类型比 object 类型内存效率更高
           - 特别是当唯一值较少时

        4. 语义明确:
           - 明确告诉模型这是类别特征
           - 区分于普通字符串

    为什么不用 One-Hot 编码:
        1. 特征爆炸:
           - 每个唯一值变成一列
           - 高基数类别会产生大量列

        2. 稀疏矩阵:
           - One-Hot 结果是稀疏矩阵
           - 某些模型处理效率低

        3. 信息丢失:
           - One-Hot 丢失了类别之间的可能关系
           - 梯度提升树的类别编码更智能

    返回值:
        tuple: (转换后的数据框, 类别列列表)
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("转换类别特征类型...")
    print("-" * 40)

    # 找到所有类别列（object 类型）
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        print(f"  找到 {len(cat_cols)} 个类别列")

        # 转换为 category 类型
        for col in cat_cols:
            df[col] = df[col].astype("category")

        # 显示部分类别列名称
        if len(cat_cols) > 10:
            print(f"  类别列: {cat_cols[:10]}... (共 {len(cat_cols)} 列)")
        else:
            print(f"  类别列: {cat_cols}")
    else:
        print("  没有找到类别列")

    return df, cat_cols


# ==============================================================================
# 综合预处理函数
# ==============================================================================

def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    ==================================================================
    功能：综合预处理流程
    ==================================================================

    参数说明:
        df: 待处理的数据框
        is_train: 是否为训练数据（训练集有 target 列）

    处理流程:
        1. 过滤高缺失列
        2. 过滤低效类别列
        3. 处理日期特征
        4. 填充缺失值
        5. 转换类别类型

    流程设计原则:
        1. 先过滤再填充:
           - 先删除无用列，减少处理量
           - 避免在无用列上浪费计算

        2. 日期处理在填充前:
           - 日期转换可能产生新缺失值
           - 在最后统一填充

        3. 类别转换在最后:
           - 其他步骤可能改变列类型
           - 最后统一转为 category

    返回值:
        tuple: (预处理后的数据框, 类别列列表)
    ==================================================================
    """
    print("\n" + "=" * 60)
    print(f"开始预处理 {'训练' if is_train else '测试'} 数据...")
    print("=" * 60)

    print(f"\n输入数据形状: {df.shape[0]} 行, {df.shape[1]} 列")

    # ========== Step 1: 过滤高缺失列 ==========
    df = filter_high_missing_cols(df)

    # ========== Step 2: 过滤低效类别列 ==========
    df = filter_low_effective_cat_cols(df)

    # ========== Step 3: 处理日期特征 ==========
    df = handle_date_features(df)

    # ========== Step 4: 填充缺失值 ==========
    df = fill_missing_values(df)

    # ========== Step 5: 转换类别类型 ==========
    df, cat_cols = convert_category_types(df)

    # ========== Step 6: 汇总信息 ==========
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)
    print(f"\n输出数据形状: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"类别特征数量: {len(cat_cols)}")

    # 检查 target 列（仅训练集）
    if is_train and "target" in df.columns:
        target_dist = df["target"].value_counts()
        print(f"\n目标变量分布:")
        print(f"  不违约 (0): {target_dist[0]} ({target_dist[0]/len(df):.2%})")
        print(f"  违约 (1): {target_dist[1]} ({target_dist[1]/len(df):.2%})")

    return df, cat_cols


# ==============================================================================
# 特征和目标分离函数
# ==============================================================================

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    ==================================================================
    功能：分离特征、目标变量和辅助信息
    ==================================================================

    参数说明:
        df: 训练数据框（包含 target 列）

    分离逻辑:
        X: 特征数据（排除 target, case_id, week_num）
        y: 目标变量（target 列）
        case_ids: 客户标识符（用于提交文件）
        week_nums: 时间分组（用于验证）

    为什么排除这些列:
        1. target:
           - 这是我们要预测的目标
           - 不能作为特征使用

        2. case_id:
           - 客户标识符
           - 只是编号，没有预测意义
           - 但需要保留用于提交

        3. week_num:
           - 时间分组信息
           - 用于验证策略（GroupKFold）
           - 不直接作为特征

    返回值:
        tuple: (特征X, 目标y, case_id列表, week_num列表)
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("分离特征和目标变量...")
    print("-" * 40)

    # 排除列列表
    drop_cols = ["target", "case_id", "week_num", "WEEK_NUM"]

    # 特征数据：排除目标、标识符、时间分组
    # 只保留 drop_cols 中存在的列
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=actual_drop_cols)

    # 目标变量
    if "target" in df.columns:
        y = df["target"]
    else:
        raise ValueError("训练数据缺少 target 列")

    # 标识符和时间分组
    case_ids = df["case_id"] if "case_id" in df.columns else pd.Series()

    # week_num 可能叫 WEEK_NUM
    if "week_num" in df.columns:
        week_nums = df["week_num"]
    elif "WEEK_NUM" in df.columns:
        week_nums = df["WEEK_NUM"]
    else:
        week_nums = pd.Series()

    print(f"  特征数量: {X.shape[1]}")
    print(f"  样本数量: {X.shape[0]}")
    print(f"  目标分布: {y.value_counts().to_dict()}")

    return X, y, case_ids, week_nums


def get_test_features(df: pd.DataFrame, train_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    ==================================================================
    功能：准备测试集特征
    ==================================================================

    参数说明:
        df: 测试数据框（没有 target 列）
        train_columns: 训练集使用的特征列列表

    处理逻辑:
        1. 确保测试集使用与训练集相同的特征
        2. 如果测试集缺少某些特征，填充为 0
        3. 提取 case_id 用于提交

    为什么需要确保特征一致:
        1. 模型期望固定的特征数量和顺序
        2. 测试集和训练集特征不一致会导致预测错误
        3. 某些特征在训练集存在，测试集可能不存在

    返回值:
        tuple: (测试特征X, case_id列表)
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("准备测试集特征...")
    print("-" * 40)

    # 提取 case_id
    case_ids = df["case_id"] if "case_id" in df.columns else pd.Series()

    # 排除非特征列
    drop_cols = ["case_id", "week_num", "WEEK_NUM", "target"]
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    df_features = df.drop(columns=actual_drop_cols)

    # 确保列顺序和训练集一致
    # 添加缺失的列（填充为 0）
    missing_cols = [col for col in train_columns if col not in df_features.columns]
    for col in missing_cols:
        df_features[col] = 0

    # 按训练集列顺序排列
    X_test = df_features[train_columns]

    print(f"  测试样本数: {X_test.shape[0]}")
    print(f"  特征数量: {X_test.shape[1]}")
    if missing_cols:
        print(f"  缺失特征（已填充0）: {len(missing_cols)} 个")

    return X_test, case_ids