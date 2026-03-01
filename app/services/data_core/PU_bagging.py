import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, Optional

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BaggingPULeaning:
    def __init__(self, n_estimators=200, imbalance_ratio=0.2, random_seed=42):
        self.n_estimators = n_estimators
        self.imbalance_ratio = imbalance_ratio
        self.random_seed = random_seed
        self.models = []
        self.feature_names = []
        self.feature_importance_df = None

    def fit(self, X_p, X_u, y_p, y_u):
        self.feature_names = X_p.columns.tolist()
        n_p = len(X_p)
        n_u_sample = int(n_p * self.imbalance_ratio)  # 按比例采样未标记样本
        print("开始训练 Bagging PU 模型（共{}个子模型）".format(self.n_estimators))
        print("Positive 样本数: {}, 每次迭代Unlabeled采样数: {}".format(n_p, n_u_sample))

        importances = np.zeros(len(self.feature_names))

        for i in range(self.n_estimators):
            # 数据采样优化：随机种子随迭代变化，有放回采样（样本量小时）
            replace = True if n_u_sample > len(X_u) else False
            y_u_subset = y_u.sample(n_u_sample, random_state=self.random_seed + i, replace=replace)
            X_u_subset = X_u.loc[y_u_subset.index]

            # 拼接训练集，正负样本比例1:1，期望能从U集中找到更多P集合
            X_train = pd.concat([X_p, X_u_subset])
            y_train = pd.concat([y_p, y_u_subset])

            params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'verbosity': -1,
                'learning_rate': 0.05,
                'num_leaves': 20,
                'n_jobs': -1,
                'scale_pos_weight': 2,
                'max_depth': 4,
                'min_child_samples': 50,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'boosting_type': 'gbdt',
                'seed': self.random_seed + i
            }

            # 训练LightGBM
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, dtrain, num_boost_round=1200)  # 迭代轮数提升至1200
            self.models.append(model)
            
            # 累加特征重要性 (split gain)
            importances += model.feature_importance(importance_type='gain')

            if (i + 1) % 10 == 0:
                print(f"已完成 {i + 1}/{self.n_estimators} 个模型")
        
        # 计算平均特征重要性
        importances /= self.n_estimators
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df


    def predict_proba(self, X):
        """
        预测每个样本的违约风险概率（0~1，值越大违约风险越高）
        X: 待预测样本特征（需与训练特征一致）
        return: 每个样本的违约概率数组
        """
        if not self.models:
            raise ValueError("模型未训练，请先调用fit()方法")

        # 确保特征顺序一致
        X = X[self.feature_names]

        # 收集所有子模型的预测概率
        all_preds = []
        for model in self.models:
            # LightGBM预测正类（违约）概率
            pred = model.predict(X, num_iteration=model.best_iteration)
            all_preds.append(pred)

        # 对所有子模型的概率取平均（Bagging融合）
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds

def run_pu_learning_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = 'label',
    output_dir: str = 'data/results/pu_learning',
    n_estimators: int = 200,
    imbalance_ratio: float = 0.3
) -> Dict:
    """
    运行完整的 PU Learning 流水线
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    # 训练集中的正样本 (P)
    X_p = train_df[train_df[label_col] == 1].drop(columns=[label_col])
    y_p = train_df[train_df[label_col] == 1][label_col]
    
    # 训练集中的未标记样本 (U) - 这里假设所有非正样本都是未标记样本
    X_u = train_df[train_df[label_col] == 0].drop(columns=[label_col])
    y_u = train_df[train_df[label_col] == 0][label_col]
    
    print(f"PU场景构建:")
    print(f"已知风险客户(P): {len(X_p)} 个")
    print(f"未标记数据(U): {len(X_u)} 个")
    
    # 训练模型
    pu_model = BaggingPULeaning(n_estimators=n_estimators, imbalance_ratio=imbalance_ratio)
    pu_model.fit(X_p, X_u, y_p, y_u)
    
    # 预测测试集
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]
    
    test_probs = pu_model.predict_proba(X_test)
    
    # 计算评估指标 (如果有真实标签)
    auc_score = roc_auc_score(y_test, test_probs)
    print(f"测试集 AUC: {auc_score:.4f}")
    
    # 保存预测结果
    test_df_result = test_df.copy()
    test_df_result['prob'] = test_probs
    test_df_result.to_csv(f'{output_dir}/test_predictions.csv', index=False)
    
    # 保存特征重要性
    feature_importance = pu_model.get_feature_importance()
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    return {
        'auc': auc_score,
        'feature_importance_path': f'{output_dir}/feature_importance.csv',
        'predictions_path': f'{output_dir}/test_predictions.csv'
    }
