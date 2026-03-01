import pandas as pd
import numpy as np

def analyze_dataset(df, label_col='label'):
    """
    对数据集进行统计分析
    :param df: pandas DataFrame
    :param label_col: 标签列名
    :return: 统计结果字典
    """
    stats = {}
    
    # 1. 特征数量
    stats['feature_count'] = df.shape[1]
    stats['sample_count'] = df.shape[0]
    
    # 2. 正负样本比例
    if label_col in df.columns:
        value_counts = df[label_col].value_counts().to_dict()
        stats['label_distribution'] = value_counts
        # 计算比例
        total = df.shape[0]
        stats['label_ratio'] = {k: f"{v/total:.2%}" for k, v in value_counts.items()}
    else:
        stats['label_distribution'] = "Label column not found"
        stats['label_ratio'] = "N/A"

    # 3. 特征类型统计
    dtypes = df.dtypes.value_counts().to_dict()
    stats['dtypes'] = {str(k): v for k, v in dtypes.items()}
    
    # 4. 缺失情况
    missing = df.isnull().sum()
    missing_count = missing[missing > 0].count()
    stats['missing_features_count'] = int(missing_count)
    
    # 全量缺失统计 (特征名: 缺失数量)
    full_missing = missing[missing > 0].sort_values(ascending=False)
    stats['full_missing_stats'] = full_missing.to_dict()
    
    # Top 5 缺失特征
    if missing_count > 0:
        top_missing = full_missing.head(5)
        stats['top_missing_features'] = top_missing.to_dict()
    else:
        stats['top_missing_features'] = {}

    # 5. IQR 分布 (仅针对数值型特征)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if label_col in numeric_cols:
        numeric_cols = numeric_cols.drop(label_col)
    
    iqr_stats = {}
    # 为了性能，只计算前 5 个数值特征的 IQR 示例，或者汇总统计
    # 这里计算所有数值特征的 IQR，但只返回异常值较多的 Top 5
    outlier_counts = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers > 0:
            outlier_counts[col] = outliers

    # 全量异常值统计 (特征名: 异常值数量)
    # 按异常值数量降序排列
    sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
    stats['full_outlier_stats'] = {k: v for k, v in sorted_outliers}

    # 排序并取 Top 5 异常值最多的特征
    top_outliers = sorted_outliers[:5]
    stats['top_outlier_features'] = {k: v for k, v in top_outliers}
    
    return stats
