import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os
from typing import Dict, List, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnsembleFeatureSelector:
    def __init__(self, weights=[0.3, 0.4, 0.3], top_k=50, random_state=42):
        self.weights = weights
        self.top_k = top_k
        self.random_state = random_state
        self.feature_map = {}

    def load_feature_mapping(self, feature_file='config/全部特征.txt'):
        """从特征文件中加载英文特征名到中文特征名的映射"""
        if not os.path.exists(feature_file):
            print(f"警告: 特征映射文件 {feature_file} 不存在，将使用英文特征名")
            return {}
            
        print(f"\n读取特征映射文件: {feature_file}")
        
        # 创建映射字典
        feature_map = {}
        
        try:
            # 手动读取文件并解析，处理数据类型字段中的逗号
            with open(feature_file, 'r', encoding='utf-8') as f:
                # 跳过第一行
                next(f)
                
                # 逐行读取
                for line_num, line in enumerate(f, 2):  # 行号从2开始
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 分割字段，只在第一个和第二个逗号处分割
                    parts = line.split(',', 2)
                    if len(parts) < 3:
                        continue  # 跳过格式不正确的行
                    
                    field_name = parts[0]
                    chinese_name = parts[1]
                    
                    feature_map[field_name] = chinese_name
            
            print(f"成功加载 {len(feature_map)} 个特征映射")
            self.feature_map = feature_map
            return feature_map
        except Exception as e:
            print(f"加载特征映射失败: {e}")
            return {}

    def get_chinese_feature_name(self, feature_name):
        """根据特征映射获取中文特征名，若无映射则返回原英文名"""
        return self.feature_map.get(feature_name, feature_name)

    def preprocess_data(self, df, label_col='label'):
        """预处理数据：编码类别特征，填充缺失值"""
        # 分离特征和标签
        if label_col in df.columns:
            X = df.drop(label_col, axis=1)
            y = df[label_col].astype(int)
        else:
            X = df.copy()
            y = None
        
        # 处理类别特征
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].fillna('Missing')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # 处理数值特征缺失
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numerical_cols:
            X[col] = X[col].fillna(-999)
        
        return X, y

    def select_features(self, X, y, feature_names):
        """集成特征选择算法：MI、XGBoost、RF"""
        
        # MI特征重要性
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_ranks = np.argsort(mi_scores)[::-1]  # 降序排序的索引
        mi_rank_dict = {feature_names[i]: len(mi_ranks) - rank for rank, i in enumerate(mi_ranks)}
        
        # XGBoost特征重要性
        xgb = XGBClassifier(random_state=self.random_state, n_jobs=-1)
        xgb.fit(X, y)
        xgb_scores = xgb.feature_importances_
        xgb_ranks = np.argsort(xgb_scores)[::-1]
        xgb_rank_dict = {feature_names[i]: len(xgb_ranks) - rank for rank, i in enumerate(xgb_ranks)}
        
        # RF特征重要性
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        rf_ranks = np.argsort(rf_scores)[::-1]
        rf_rank_dict = {feature_names[i]: len(rf_ranks) - rank for rank, i in enumerate(rf_ranks)}
        
        # 计算集成分数
        ensemble_scores = {}
        for feature in feature_names:
            ensemble_score = (mi_rank_dict[feature] * self.weights[0] + 
                             xgb_rank_dict[feature] * self.weights[1] + 
                             rf_rank_dict[feature] * self.weights[2])
            ensemble_scores[feature] = ensemble_score
        
        # 按集成分数降序排序，选择top_k特征
        top_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [feature for feature, score in top_features], ensemble_scores

    def visualize_feature_comparison(self, feature_sets, set_names, output_dir):
        """可视化三个特征集的对比"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 特征排名对比
        print("\n生成特征排名对比图...")
        plt.figure(figsize=(15, 10))
        
        for i, (features, name) in enumerate(zip(feature_sets, set_names)):
            # 绘制前20个特征的排名
            top_features = features[:20]
            ranks = range(1, len(top_features) + 1)
            plt.plot(ranks, [i*5 + rank for rank in ranks], 'o-', label=name, markersize=8)
        
        plt.xlabel('特征排名', fontsize=14)
        plt.ylabel('不同数据集的排名偏移', fontsize=14)
        plt.title('特征排名对比（前20名）', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/feature_rank_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特征频次统计
        print("生成特征频次统计图...")
        from collections import Counter
        all_features_list = [feature for features in feature_sets for feature in features]
        feature_counts = Counter(all_features_list)
        
        # 绘制出现次数最多的前20个特征
        top_freq_features = feature_counts.most_common(20)
        if top_freq_features:
            features, counts = zip(*top_freq_features)
            features_chinese = [self.get_chinese_feature_name(f) for f in features]
            
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(features)), counts, color=['skyblue', 'lightgreen', 'salmon'][:len(features)])
            plt.xticks(range(len(features)), features_chinese, rotation=45, ha='right', fontsize=10)
            plt.xlabel('特征名称', fontsize=14)
            plt.ylabel('出现次数', fontsize=14)
            plt.title('特征在三个数据集上的出现频次（前20名）', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_frequency.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 保存特征列表，使用中文特征名
        for i, (features, name) in enumerate(zip(feature_sets, set_names)):
            with open(f'{output_dir}/{name}_top{self.top_k}_features.txt', 'w', encoding='utf-8') as f:
                for j, feature in enumerate(features):
                    chinese_name = self.get_chinese_feature_name(feature)
                    f.write(f'{j+1}. {chinese_name} ({feature})\n')
        
        # 4. 保存特征排名对比表，使用中文特征名
        print("生成特征排名对比表...")
        feature_rank_df = pd.DataFrame()
        for i, (features, name) in enumerate(zip(feature_sets, set_names)):
            for rank, feature in enumerate(features):
                chinese_name = self.get_chinese_feature_name(feature)
                if feature not in feature_rank_df.index:
                    feature_rank_df.loc[feature, '特征名称（中文）'] = chinese_name
                    feature_rank_df.loc[feature, '特征名称（英文）'] = feature
                feature_rank_df.loc[feature, f'{name}_排名'] = rank + 1
        
        # 按平均排名排序
        if not feature_rank_df.empty:
            feature_rank_df['平均排名'] = feature_rank_df[[f'{name}_排名' for name in set_names]].mean(axis=1)
            feature_rank_df = feature_rank_df.sort_values('平均排名')
            
            # 调整列顺序，将中文名称放在前面
            columns = ['特征名称（中文）', '特征名称（英文）'] + [f'{name}_排名' for name in set_names] + ['平均排名']
            feature_rank_df = feature_rank_df[columns]
            
            feature_rank_df.to_csv(f'{output_dir}/feature_rank_comparison.csv', index=False, encoding='utf-8-sig')
        
        print("\n特征对比可视化完成！")

def run_feature_selection_pipeline(
    train_df: pd.DataFrame,
    label_col: str = 'label',
    output_dir: str = 'data/results/feature_selection',
    top_k: int = 50,
    pu_predictions_path: Optional[str] = None
) -> Dict:
    """
    运行完整的特征选择流水线
    """
    print("开始执行集成学习特征选择算法...")
    os.makedirs(output_dir, exist_ok=True)
    
    selector = EnsembleFeatureSelector(top_k=top_k)
    selector.load_feature_mapping()
    
    # 准备训练集列表
    training_sets = []
    set_names = []
    
    # 1. 原始数据
    training_sets.append(train_df.copy())
    set_names.append('原始数据')
    
    # 如果提供了PU预测结果，可以生成更多训练集
    if pu_predictions_path and os.path.exists(pu_predictions_path):
        try:
            pu_predictions = pd.read_csv(pu_predictions_path)
            # 确保长度一致
            if len(pu_predictions) == len(train_df):
                # 训练集2：高置信度负转正
                train2 = train_df.copy()
                # 假设 pu_predictions 有 'prob' 列
                prob_col = 'prob' if 'prob' in pu_predictions.columns else '违约风险概率'
                if prob_col in pu_predictions.columns:
                    train2['prob'] = pu_predictions[prob_col].values
                    high_conf_neg = train2[(train2[label_col] == 0) & (train2['prob'] > 0.9)]
                    train2.loc[high_conf_neg.index, label_col] = 1
                    train2 = train2.drop('prob', axis=1)
                    training_sets.append(train2)
                    set_names.append('高置信负转正')
        except Exception as e:
            print(f"加载PU预测结果失败: {e}")
    
    # 对每个训练集进行特征选择
    all_top_features = []
    
    for i, (train, name) in enumerate(zip(training_sets, set_names)):
        print(f"\n=== 处理{name} ===")
        # 预处理数据
        X, y = selector.preprocess_data(train, label_col)
        # 执行集成特征选择
        top_features, _ = selector.select_features(X, y, X.columns.tolist())
        all_top_features.append(top_features)
        
        print(f"{name}的Top {top_k}特征：")
        for j, feature in enumerate(top_features[:10]):
            print(f"  {j+1}. {selector.get_chinese_feature_name(feature)}")
        print("  ...")
    
    # 可视化特征对比
    selector.visualize_feature_comparison(all_top_features, set_names, output_dir)
    
    return {
        'output_dir': output_dir,
        'feature_rank_path': f'{output_dir}/feature_rank_comparison.csv',
        'top_features': all_top_features[0]  # 返回第一个数据集的特征
    }
