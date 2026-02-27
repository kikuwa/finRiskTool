import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import json
import os

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

def preprocess_dataframe(df,
                         categorical_mappings=None,
                         binary_mappings=None,
                         text_columns=None,
                         custom_transforms=None):
    """
    通用数据预处理函数
    参数:
    df: 原始DataFrame
    categorical_mappings: 分类列映射配置
    binary_mappings: 二值列映射配置
    text_columns: 文本列配置（将被删除）
    custom_transforms: 自定义转换配置
    返回:
    预处理后的DataFrame
    """
    df_processed = df.copy()

    # 1. 处理分类变量（因子化编码）
    if categorical_mappings:
        for col_config in categorical_mappings:
            col_name = col_config.get('column')
            drop_original = col_config.get('drop_original', False)

            if col_name in df_processed.columns:
                # 创建因子化列
                codes, uniques = pd.factorize(df_processed[col_name])
                df_processed[f"{col_name}_encoded"] = codes

                # 如果需要，删除原始列
                if drop_original:
                    df_processed = df_processed.drop(columns=[col_name])

    # 2. 处理二值变量
    if binary_mappings:
        for col_config in binary_mappings:
            col_name = col_config.get('column')
            mapping = col_config.get('mapping')
            default_value = col_config.get('default', np.nan)

            if col_name in df_processed.columns:
                # 应用映射
                df_processed[col_name] = df_processed[col_name].map(mapping).fillna(default_value)
                # 转换为数值类型
                df_processed[col_name] = pd.to_numeric(df_processed[col_name], errors='coerce')

    # 3. 处理自定义转换
    if custom_transforms:
        for transform_config in custom_transforms:
            col_name = transform_config.get('column')
            transform_type = transform_config.get('type')
            transform_params = transform_config.get('params', {})

            if col_name in df_processed.columns:
                df_processed = apply_custom_transform(
                    df_processed, col_name, transform_type, transform_params
                )

    # 4. 删除文本列
    if text_columns:
        text_cols_to_drop = [col for col in text_columns if col in df_processed.columns]
        if text_cols_to_drop:
            df_processed = df_processed.drop(columns=text_cols_to_drop)

    return df_processed

def apply_custom_transform(df, column, transform_type, params):
    """应用自定义转换"""
    df_copy = df.copy()

    if transform_type == 'string_replace_convert':
        # 字符串替换并转换为数值
        search_str = params.get('search', "")
        replace_str = params.get('replace', "")
        divisor = params.get('divisor', 1)

        df_copy[column] = (
            df_copy[column]
            .astype(str)
            .str.replace(search_str, replace_str, regex=False)
        )
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce') / divisor

    elif transform_type == 'categorical_to_numeric':
        # 分类映射到数值
        mapping = params.get('mapping', {})
        df_copy[column] = df_copy[column].map(mapping)
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')

    elif transform_type == 'year_imputation':
        # 年份数据处理（转换并填充中位数）
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        median_value = df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(median_value)

    elif transform_type == 'custom_mapping':
        # 通用映射转换
        mapping_func = params.get('mapping_func')
        if callable(mapping_func):
            df_copy[column] = df_copy[column].apply(mapping_func)

    return df_copy

# 辅助函数：自动检测列类型
def detect_column_types(df, sample_threshold=0.3):
    """自动检测列的类型"""
    results = {
        'categorical': [],
        'binary': [],
        'text': [],
        'numeric': [],
        'date': []
    }

    for col in df.columns:
        # 跳过全空列
        if df[col].isna().all():
            continue

        # 检测数据类型
        dtype = str(df[col].dtype)

        # 检测日期类型
        if 'date' in dtype.lower() or 'time' in dtype.lower():
            results['date'].append(col)
            continue

        # 检测数值类型
        if pd.api.types.is_numeric_dtype(df[col]):
            # 检查是否是二值变量
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2:
                results['binary'].append(col)
            else:
                results['numeric'].append(col)
            continue

        # 检测分类/文本类型
        unique_count = df[col].nunique()
        total_count = df[col].count()

        if total_count > 0:
            unique_ratio = unique_count / total_count

            # 少量唯一值 => 分类变量
            if unique_ratio < sample_threshold and unique_count < 50:
                results['categorical'].append(col)
            else:
                # 大量唯一值或长文本 => 文本变量
                sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(sample_value, str) and len(str(sample_value)) > 50:
                    results['text'].append(col)
                else:
                    results['categorical'].append(col)

    print('categorical len:' + str(len(results['categorical'])))
    print('binary len:' + str(len(results['binary'])))
    print('numeric len:' + str(len(results['numeric'])))
    print('date len:' + str(len(results['date'])))

    with open('config/pu_config.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

# 自动配置生成
def generate_config_from_data(df, unique_ratio_threshold=0.3):
    """根据数据自动生成预处理配置"""
    print('自动生成数据格式配置文件')
    column_types = detect_column_types(df, unique_ratio_threshold)

    config = {
        'categorical_mappings': [
            {'column': col, 'drop_original': True}
            for col in column_types['categorical']
        ],
        'binary_mappings': [
            {
                'column': col,
                'mapping': create_binary_mapping(df[col]),
                'default': 0
            }
            for col in column_types['binary']
        ],
        'text_columns': column_types['text'],
        'custom_transforms': []
    }

    return config

def create_binary_mapping(series):
    """为二值列创建映射"""
    unique_values = series.dropna().unique()
    mapping = {}

    for i, val in enumerate(sorted(unique_values)):
        if i < 2:  # 只处理前两个唯一值
            mapping[val] = i

    return mapping

# 主处理流程
def process_pipeline(df, auto_config=False, custom_config=None):
    """
    完整的预处理流水线
    参数:
    df: 原始数据
    auto_config: 是否自动生成配置
    custom_config: 自定义配置（如果提供则优先使用）
    """
    # 步骤1: 生成或使用配置
    if custom_config:
        config = custom_config
    elif auto_config:
        config = generate_config_from_data(df)
    else:
        # 使用默认配置
        config = {}

    # 步骤2: 数据预处理
    processed_df = preprocess_dataframe(df, **config)

    # 步骤3: 可选的质量检查
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {processed_df.shape}")

    # 检查缺失值
    missing_info = processed_df.isnull().sum()
    if missing_info.sum() > 0:
        print("\n缺失值统计:")
        for col, count in missing_info[missing_info > 0].items():
            print(f"  {col}: {count}个缺失值 ({count/len(processed_df):.2%})")

    return processed_df

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(r'data/train.csv')
    print(f"加载数据: {df.shape}")
    print(f"列名: {list(df.columns)}")

    df_processed = df.copy()
    processed_df1 = process_pipeline(df_processed, auto_config=True)
    print(f"\n处理后列名: {list(processed_df1.columns)}")
    print("\n处理后的数据类型:")
    print(processed_df1.dtypes.value_counts())

    # 构建PU场景
    delU = processed_df1[processed_df1['label'] != 3]
    y = delU['label'].copy()
    X = delU.drop(columns=['label'])
    print(len(y))

    positive_indices = y[y == 1].index
    hidden_positive_indices = np.random.choice(positive_indices, int(len(positive_indices) * 0.2), replace=False)
    known_positive_indices = list(set(positive_indices) - set(hidden_positive_indices))

    X_p = X.loc[known_positive_indices]
    y_p = pd.Series(1, index=X_p.index)

    u_indices = list(set(X.index) - set(known_positive_indices))
    X_u = X.loc[u_indices]
    y_u = pd.Series(0, index=X_u.index)

    print(f"\nPU场景构建:")
    print(f"已知风险客户(P): {len(X_p)} 个")
    print(f"未标记数据(U): {len(X_u)} 个 (包含 {len(hidden_positive_indices)} 个隐藏风险客户)")

    # 训练PU模型
    pu_model = BaggingPULeaning(n_estimators=200, imbalance_ratio=0.3)
    pu_model.fit(X_p, X_u, y_p, y_u)

    # 全量预测
    all_X = processed_df1.drop('label', axis=1)  # 全量待预测样本
    risk_proba = pu_model.predict_proba(all_X)  # 每个样本的违约概率
    processed_df1['违约风险概率'] = risk_proba

    # 保存预测结果
    output_dir = 'data/results/pu_learning'
    os.makedirs(output_dir, exist_ok=True)
    processed_df1.to_csv(f'{output_dir}/pu_predictions.csv', index=False)
    print(f"预测结果已保存到: {output_dir}/pu_predictions.csv")
    
    # 保存特征重要性
    feature_importance = pu_model.get_feature_importance()
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    print(f"特征重要性已保存到: {output_dir}/feature_importance.csv")
