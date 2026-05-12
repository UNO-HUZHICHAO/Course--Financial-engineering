"""
==============================================================================
模型训练模块 - LightGBM 和 CatBoost 训练与融合（简化版）
==============================================================================

模块说明:
    本模块负责模型训练、评估和融合，包括：
    1. VotingModel: 投票融合模型类
    2. train_models: 交叉验证训练两种模型
    3. predict: 多模型预测融合
    4. evaluate_model: 模型评估函数

模型选择说明:
    1. LightGBM:
       - 微软开源的梯度提升树框架
       - 特点：速度快、内存低、支持类别特征
       - 适合大规模数据训练

    2. CatBoost:
       - Yandex 开源的梯度提升树框架
       - 特点：类别特征处理出色、Ordered Boosting
       - 适合类别特征丰富的数据

    3. 两者融合:
       - 利用各自优势
       - 提高预测稳定性
       - 通常优于单一模型

交叉验证说明:
    - 使用 StratifiedKFold 3 折交叉验证
    - 每折保持目标分布一致
    - 每折训练一个 LightGBM + 一个 CatBoost
    - 共 6 个模型进行投票融合

使用方法:
    from model_simple import train_models, VotingModel, predict

    model, scores = train_models(X, y, cat_cols)
    predictions = predict(model, X_test)

作者: 信用风险模型课程作业版
日期: 2024
==============================================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Tuple, Dict

from config_simple import LGBM_PARAMS, CATBOOST_PARAMS, N_SPLITS, EARLY_STOPPING_ROUNDS, LOG_EVALUATION_PERIOD


# ==============================================================================
# 投票融合模型
# ==============================================================================

class VotingModel(BaseEstimator, ClassifierMixin):
    """
    ==================================================================
    投票融合模型 - 组合多个模型的预测结果
    ==================================================================

    类说明:
        VotingModel 是一个投票集成模型，将多个基模型的预测结果平均。
        这是模型融合（Ensemble）的一种简单有效方法。

    工作原理:
        1. 训练阶段:
           - 接收已经训练好的模型列表
           - 不进行任何训练（基模型已训练好）

        2. 预测阶段:
           - 每个基模型分别预测
           - 对所有预测结果取平均

    为什么使用模型融合:
        1. 降低偏差:
           - 不同模型有不同的偏差
           - 平均可以降低整体偏差

        2. 提高稳定性:
           - 单一模型可能不稳定
           - 多模型平均更稳定

        3. 减少过拟合:
           - 组合多个模型，降低过拟合风险

        4. 提高泛化能力:
           - 利用不同模型的优势
           - 通常优于单一模型

    融合方法对比:
        1. 简单平均（本项目使用）:
           - 对所有模型预测取平均
           - 简单有效

        2. 加权平均:
           - 根据模型性能分配权重
           - 好模型权重更大

        3. 堆叠法（Stacking）:
           - 使用元模型学习如何组合
           - 更复杂但可能效果更好

    继承说明:
        - BaseEstimator: 提供 sklearn 估计器接口（get_params, set_params）
        - ClassifierMixin: 提供分类器接口（score 方法）
        - 继承这两个类使 VotingModel 兼容 sklearn 工具

    使用示例:
        # 假设 models 是训练好的模型列表
        voting_model = VotingModel(models)

        # 预测概率
        proba = voting_model.predict_proba(X_test)
    ==================================================================
    """

    def __init__(self, estimators: List[BaseEstimator]):
        """
        ==================================================================
        功能：初始化投票模型
        ==================================================================

        参数说明:
            estimators: 已训练好的基模型列表
                       例如: [lgb_model1, cat_model1, lgb_model2, ...]

        初始化过程:
            保存所有基模型，用于后续预测。

        为什么基模型已经训练好:
            在交叉验证流程中:
            1. 数据分为 N 折
            2. 每折训练一个模型
            3. 所有训练好的模型保存到 VotingModel
            4. VotingModel 组合所有模型进行预测

        模型多样性原则:
            - 不同类型模型（LightGBM, CatBoost）
            - 不同训练数据（不同折）
            - 不同参数配置
            - 增加多样性可以提高融合效果
        ==================================================================
        """
        # 调用父类初始化
        super().__init__()

        # 保存基模型列表
        self.estimators = estimators

        # 记录模型数量
        self.n_estimators = len(estimators)

        print(f"VotingModel 创建完成，包含 {self.n_estimators} 个模型")

    def fit(self, X, y=None):
        """
        ==================================================================
        功能：拟合方法（实际不进行训练）
        ==================================================================

        参数说明:
            X: 特征数据（不使用）
            y: 目标标签（不使用）

        为什么不训练:
            1. VotingModel 是投票融合模型
            2. 基模型已经在交叉验证中训练好
            3. fit 方法只是一个接口占位
            4. 保持 sklearn API 兼容

        这种设计的好处:
            1. 兼容 sklearn 的 Pipeline
            2. 可以使用 sklearn 的 cross_val_score 等工具
            3. 接口统一，便于集成到各种框架

        返回值:
            self: 返回模型实例本身
                  这是 sklearn fit 方法的标准返回格式
        ==================================================================
        """
        # 直接返回 self，不做任何训练
        return self

    def predict(self, X):
        """
        ==================================================================
        功能：预测类别标签
        ==================================================================

        参数说明:
            X: 特征数据（numpy array 或 pandas DataFrame）

        预测流程:
            1. 遍历所有基模型
            2. 每个模型调用 predict(X) 得到类别预测
            3. 对所有预测结果取平均

        注意:
            对于二分类问题:
            - predict 通常返回 0 或 1
            - 平均后可能是 0 到 1 之间的值
            - 需要根据阈值判断最终类别

        这里返回的是平均值:
            返回值范围: 0 到 1
            可以理解为"平均预测概率"

        返回值:
            numpy.ndarray: 所有模型预测的平均值
        ==================================================================
        """
        # 获取所有基模型的类别预测
        y_preds = [estimator.predict(X) for estimator in self.estimators]

        # 对所有预测取平均
        # axis=0 表示按列（样本）平均
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        """
        ==================================================================
        功能：预测类别概率（主要预测方法）
        ==================================================================

        参数说明:
            X: 特征数据（numpy array 或 pandas DataFrame）

        预测流程:
            1. 遍历所有基模型
            2. 每个模型调用 predict_proba(X) 得到概率预测
            3. 对所有概率预测取平均

        predict_proba 输出格式:
            对于二分类问题:
            - predict_proba 返回形状为 (n_samples, 2) 的数组
            - 第 0 列: 类别 0 的概率（不违约）
            - 第 1 列: 类别 1 的概率（违约）
            - 两列之和 = 1

        为什么取平均:
            1. 简单有效:
               - 平均可以减少单一模型的偏差

            2. 稳定性:
               - 多个模型的平均比单模型更稳定

            3. 泛化能力:
               - 组合不同模型可以提高泛化

        提交结果:
            通常使用第 1 列（正类概率）作为提交分数:
            y_pred = model.predict_proba(X_test)[:, 1]

        返回值:
            numpy.ndarray: 所有模型概率预测的平均值
            形状: (n_samples, n_classes)
        ==================================================================
        """
        # 获取所有基模型的概率预测
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]

        # 对所有概率预测取平均
        # axis=0 表示按列（样本）平均
        return np.mean(y_preds, axis=0)


# ==============================================================================
# 交叉验证训练函数
# ==============================================================================

def train_models(X: pd.DataFrame, y: pd.Series, cat_cols: List[str]) -> Tuple[VotingModel, List[float], List[float]]:
    """
    ==================================================================
    功能：使用交叉验证训练 LightGBM 和 CatBoost 模型
    ==================================================================

    参数说明:
        X: 特征数据（Pandas DataFrame）
        y: 目标变量（Pandas Series，0 或 1）
        cat_cols: 类别特征列名列表

    训练流程:
        1. 创建 StratifiedKFold 交叉验证对象
        2. 遍历每一折：
           a. 划分训练集和验证集
           b. 训练 LightGBM，计算验证 AUC
           c. 训练 CatBoost，计算验证 AUC
           d. 保存训练好的模型
        3. 创建 VotingModel 融合所有模型

    StratifiedKFold 说明:
        - n_splits=3: 将数据分为 3 份
        - shuffle=True: 打乱数据顺序
        - Stratified: 每折保持目标变量分布一致

        为什么使用 StratifiedKFold:
        1. 保持分布一致:
           - 每折中违约和非违约的比例相同
           - 避免某折全是非违约样本

        2. 更稳定的评估:
           - 每折评估结果更可比
           - 减少随机划分的影响

        为什么使用 3 折而非 5 折:
        1. 课程作业简化:
           - 3 折训练更快
           - 模型数量更少（6个）

        2. 数据量考虑:
           - 如果数据量小，折太多每折样本太少

    模型训练过程:

    LightGBM 训练:
        1. 将类别特征转为 category 类型
        2. 创建 LGBMClassifier
        3. 训练并使用 early_stopping
        4. 验证集预测和计算 AUC

    CatBoost 训练:
        1. 创建 Pool 对象（包含类别特征信息）
        2. 创建 CatBoostClassifier
        3. 训练并评估
        4. 验证集预测和计算 AUC

    返回值:
        tuple: (VotingModel, LightGBM AUC列表, CatBoost AUC列表)
    ==================================================================
    """
    print("\n" + "=" * 60)
    print("开始模型训练...")
    print("=" * 60)

    print(f"\n数据信息:")
    print(f"  样本数量: {X.shape[0]}")
    print(f"  特征数量: {X.shape[1]}")
    print(f"  类别特征: {len(cat_cols)} 个")
    print(f"  交叉验证: {N_SPLITS} 折")

    # ========== 创建交叉验证对象 ==========
    # StratifiedKFold: 分层 K 折交叉验证
    # n_splits: 折数
    # shuffle=True: 打乱数据顺序
    # random_state: 随机种子，确保可复现
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    # ========== 存储模型和分数 ==========
    fitted_models_lgb = []  # LightGBM 模型列表
    fitted_models_cat = []  # CatBoost 模型列表
    cv_scores_lgb = []      # LightGBM AUC 分数列表
    cv_scores_cat = []      # CatBoost AUC 分数列表

    # ========== 遍历每一折 ==========
    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y), 1):
        print(f"\n{'='*20} Fold {fold}/{N_SPLITS} {'='*20}")

        # 划分训练集和验证集
        # iloc: 按位置索引
        X_train, X_valid = X.iloc[idx_train], X.iloc[idx_valid]
        y_train, y_valid = y.iloc[idx_train], y.iloc[idx_valid]

        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_valid)} 样本")
        print(f"  验证集违约率: {y_valid.mean():.2%}")

        # ========== LightGBM 训练 ==========
        print("\n[LightGBM 训练]")

        # 复制数据（避免修改原数据）
        X_train_lgb = X_train.copy()
        X_valid_lgb = X_valid.copy()

        # 类别特征转为 category 类型
        # LightGBM 需要 category 类型来正确处理类别特征
        for col in cat_cols:
            if col in X_train_lgb.columns:
                X_train_lgb[col] = X_train_lgb[col].astype("category")
                X_valid_lgb[col] = X_valid_lgb[col].astype("category")

        # 创建 LightGBM 分类器
        # 使用配置文件中的参数
        model_lgb = lgb.LGBMClassifier(**LGBM_PARAMS)

        # 训练模型
        # eval_set: 验证集，用于评估和早停
        # callbacks:
        #   - log_evaluation: 每N轮输出日志
        #   - early_stopping: 早停防止过拟合
        model_lgb.fit(
            X_train_lgb, y_train,
            eval_set=[(X_valid_lgb, y_valid)],
            callbacks=[
                lgb.log_evaluation(LOG_EVALUATION_PERIOD),
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)
            ]
        )

        # 验证集预测
        # predict_proba 返回概率，[:, 1] 取正类概率（违约概率）
        y_pred_lgb = model_lgb.predict_proba(X_valid_lgb)[:, 1]

        # 计算 AUC
        # roc_auc_score: 计算 ROC 曲线下面积
        auc_lgb = roc_auc_score(y_valid, y_pred_lgb)

        # 保存模型和分数
        fitted_models_lgb.append(model_lgb)
        cv_scores_lgb.append(auc_lgb)

        print(f"  LightGBM Fold {fold} AUC: {auc_lgb:.4f}")
        print(f"  LightGBM 最佳迭代: {model_lgb.best_iteration_}")

        # ========== CatBoost 训练 ==========
        print("\n[CatBoost 训练]")

        # CatBoost 使用 Pool 对象
        # Pool: CatBoost 的数据容器，优化类别特征处理
        # cat_features: 指定哪些列是类别特征（用列索引）
        # Pool 会自动处理类别特征的编码

        # 获取类别特征的列索引
        # get_loc: 获取列名对应的索引位置
        cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]

        # 创建 Pool 对象
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)

        # 创建 CatBoost 分类器
        # 使用配置文件中的参数
        model_cat = CatBoostClassifier(**CATBOOST_PARAMS)

        # 训练模型
        # eval_set: 验证集
        # verbose=False: 不输出详细日志
        model_cat.fit(train_pool, eval_set=valid_pool, verbose=False)

        # 验证集预测
        # predict_proba 返回概率，[:, 1] 取正类概率
        y_pred_cat = model_cat.predict_proba(X_valid)[:, 1]

        # 计算 AUC
        auc_cat = roc_auc_score(y_valid, y_pred_cat)

        # 保存模型和分数
        fitted_models_cat.append(model_cat)
        cv_scores_cat.append(auc_cat)

        print(f"  CatBoost Fold {fold} AUC: {auc_cat:.4f}")
        print(f"  CatBoost 最佳迭代: {model_cat.best_iteration_}")

    # ========== 创建融合模型 ==========
    # 合并所有模型
    # LightGBM(N_SPLITS 个) + CatBoost(N_SPLITS 个) = 共 2*N_SPLITS 个
    all_models = fitted_models_lgb + fitted_models_cat
    voting_model = VotingModel(all_models)

    # ========== 打印训练总结 ==========
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    print(f"\n[LightGBM 结果]")
    print(f"  各折 AUC: {[f'{s:.4f}' for s in cv_scores_lgb]}")
    print(f"  平均 AUC: {np.mean(cv_scores_lgb):.4f}")
    print(f"  AUC 标准差: {np.std(cv_scores_lgb):.4f}")

    print(f"\n[CatBoost 结果]")
    print(f"  各折 AUC: {[f'{s:.4f}' for s in cv_scores_cat]}")
    print(f"  平均 AUC: {np.mean(cv_scores_cat):.4f}")
    print(f"  AUC 标准差: {np.std(cv_scores_cat):.4f}")

    print(f"\n[融合模型]")
    print(f"  模型数量: {len(all_models)} 个")
    print(f"  LightGBM: {len(fitted_models_lgb)} 个")
    print(f"  CatBoost: {len(fitted_models_cat)} 个")

    return voting_model, cv_scores_lgb, cv_scores_cat


# ==============================================================================
# 预测函数
# ==============================================================================

def predict(model: VotingModel, X_test: pd.DataFrame, cat_cols: List[str] = None) -> np.ndarray:
    """
    ==================================================================
    功能：使用融合模型预测
    ==================================================================

    参数说明:
        model: 训练好的 VotingModel
        X_test: 测试特征 DataFrame
        cat_cols: 类别特征列名列表

    预测流程:
        1. 转换类别特征为 category 类型
        2. 调用 VotingModel.predict_proba()
        3. 取正类概率（违约概率）
        4. 返回预测结果

    返回值:
        numpy.ndarray: 预测概率（违约概率）
    ==================================================================
    """
    print("\n" + "-" * 40)
    print("开始预测...")
    print("-" * 40)

    print(f"  测试样本数: {X_test.shape[0]}")

    X_pred = X_test.copy()

    if cat_cols:
        for col in cat_cols:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category")

    # 使用融合模型预测概率
    # predict_proba 返回 (n_samples, 2)
    # [:, 1] 取正类概率（违约概率）
    y_pred = model.predict_proba(X_pred)[:, 1]

    # 打印预测统计
    print(f"  预测概率范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  预测概率均值: {y_pred.mean():.4f}")
    print(f"  预测概率中位数: {np.median(y_pred):.4f}")

    return y_pred


# ==============================================================================
# 模型评估函数
# ==============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    ==================================================================
    功能：评估模型性能
    ==================================================================

    参数说明:
        y_true: 真实标签（0 或 1）
        y_pred: 预测概率（0 到 1）
        threshold: 分类阈值（默认 0.5）

    评估指标:
        1. AUC: ROC 曲线下面积
        2. Accuracy: 准确率
        3. Precision: 精确率
        4. Recall: 召回率
        5. F1 Score: F1 分数

    各指标说明:

    AUC (Area Under ROC Curve):
        - 衡量模型区分能力的指标
        - 范围 0-1，越大越好
        - 不受阈值影响

    Accuracy (准确率):
        - 正确预测的比例
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - 在不平衡数据中可能误导

    Precision (精确率):
        - 预测为正中真正为正的比例
        - Precision = TP / (TP + FP)
        - 关注预测正类的准确性

    Recall (召回率):
        - 真正为正中被预测为正的比例
        - Recall = TP / (TP + FN)
        - 关注能否找出所有正类

    F1 Score:
        - 精确率和召回率的调和平均
        - F1 = 2 * Precision * Recall / (Precision + Recall)
        - 综合评估指标

    信用风险场景:
        - 违约（正类）较少，数据不平衡
        - Recall 重要：不想漏掉违约客户
        - Precision 重要：不想误判好客户
        - 需要权衡两者

    返回值:
        dict: 各评估指标的值
    ==================================================================
    """
    # 计算各指标
    auc = roc_auc_score(y_true, y_pred)

    # 将概率转换为类别（使用阈值）
    y_pred_class = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, zero_division=0)
    recall = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)

    # 返回结果
    metrics = {
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return metrics


def print_evaluation_results(metrics: Dict, title: str = "模型评估结果"):
    """
    ==================================================================
    功能：打印评估结果
    ==================================================================

    参数说明:
        metrics: 评估指标字典
        title: 结果标题
    ==================================================================
    """
    print("\n" + "-" * 40)
    print(title)
    print("-" * 40)

    print(f"  AUC: {metrics['AUC']:.4f}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1']:.4f}")


# ==============================================================================
# 特征重要性分析
# ==============================================================================

def get_feature_importance(models: List, feature_names: List[str]) -> pd.DataFrame:
    """
    ==================================================================
    功能：获取特征重要性
    ==================================================================

    参数说明:
        models: 模型列表
        feature_names: 特征名称列表

    特征重要性说明:
        - 梯度提升树模型会计算特征重要性
        - 反映特征对预测的贡献程度
        - 可以用于特征选择和解释模型

    计算方法:
        - LightGBM: feature_importances_ 属性
        - CatBoost: get_feature_importance() 方法
        - 对多个模型的特征重要性取平均

    返回值:
        pd.DataFrame: 特征重要性排名表
    ==================================================================
    """
    # 存储所有模型的特征重要性
    all_importances = []

    for model in models:
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            # LightGBM
            importance = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            # CatBoost
            importance = model.get_feature_importance()
        else:
            continue

        all_importances.append(importance)

    # 对所有模型的特征重要性取平均
    if all_importances:
        avg_importance = np.mean(all_importances, axis=0)

        # 创建 DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        })

        # 按重要性降序排列
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    return pd.DataFrame()


def print_top_features(importance_df: pd.DataFrame, top_n: int = 20):
    """
    ==================================================================
    功能：打印最重要的特征
    ==================================================================

    参数说明:
        importance_df: 特征重要性 DataFrame
        top_n: 显示前 N 个特征
    ==================================================================
    """
    print("\n" + "-" * 40)
    print(f"最重要的 {top_n} 个特征:")
    print("-" * 40)

    # 显示前 top_n 个特征
    top_features = importance_df.head(top_n)

    for i, row in enumerate(top_features.itertuples(), 1):
        print(f"{i:2d}. {row.feature}: {row.importance:.4f}")