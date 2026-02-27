import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = 'data/results/feature_selection'
os.makedirs(output_dir, exist_ok=True)

# 读取特征映射，获取中文特征名
def load_feature_mapping(feature_file='config/全部特征.txt'):
    """从特征文件中加载英文特征名到中文特征名的映射"""
    print(f"\n读取特征映射文件: {feature_file}")
    
    # 创建映射字典
    feature_map = {}
    
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
            # data_type = parts[2]  # 数据类型字段，这里暂时不需要
            
            feature_map[field_name] = chinese_name
    
    print(f"成功加载 {len(feature_map)} 个特征映射")
    return feature_map

# 获取中文特征名，若没有映射则返回原英文名
def get_chinese_feature_name(feature_name, feature_map):
    """根据特征映射获取中文特征名，若无映射则返回原英文名"""
    return feature_map.get(feature_name, feature_name)

# 1. 读取数据
def load_data():
    print("读取数据...")
    train_df = pd.read_csv('data/train.csv')
    pu_predictions = pd.read_csv('data/results/pu_learning/pu_predictions.csv')
    return train_df, pu_predictions

# 2. 数据预处理
def preprocess_data(df):
    """预处理数据：编码类别特征，填充缺失值"""
    # 分离特征和标签
    X = df.drop('label', axis=1)
    y = df['label'].astype(int)
    
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

# 3. 生成三种不同的训练集
def generate_training_sets(train_df, pu_predictions):
    """生成三种不同的训练集"""
    
    # 训练集1：原始train.csv，负样本中置信度>0.9的视为正样本
    print("\n生成训练集1...")
    train1 = train_df.copy()
    # 添加违约风险概率列，按索引对齐
    train1['违约风险概率'] = pu_predictions['违约风险概率'].values
    # 找出原始负样本中置信度>0.9的样本
    high_conf_neg = train1[(train1['label'] == 0) & (train1['违约风险概率'] > 0.9)]
    # 将这些样本的标签改为1
    train1.loc[high_conf_neg.index, 'label'] = 1
    # 移除临时添加的列
    train1 = train1.drop('违约风险概率', axis=1)
    print(f"训练集1：正样本数量={train1['label'].sum()}, 总样本数={len(train1)}, 正样本比例={train1['label'].sum()/len(train1):.4f}")
    
    # 训练集2：调整正样本比例到10%
    print("\n生成训练集2...")
    train2 = train_df.copy()
    current_pos_count = train2['label'].sum()
    current_total = len(train2)
    target_pos_ratio = 0.1
    target_pos_count = int(current_total * target_pos_ratio)
    needed_pos = target_pos_count - current_pos_count
    
    if needed_pos > 0:
        # 添加违约风险概率列
        train2_with_prob = train2.copy()
        train2_with_prob['违约风险概率'] = pu_predictions['违约风险概率'].values
        # 筛选出原始负样本中置信度高的样本
        candidate_pseudo_pos = train2_with_prob[train2_with_prob['label'] == 0]
        # 按置信度降序排序，选择需要的数量
        candidate_pseudo_pos = candidate_pseudo_pos.sort_values('违约风险概率', ascending=False)
        selected_pseudo_pos = candidate_pseudo_pos.head(needed_pos)
        # 将这些样本的标签改为1
        train2.loc[selected_pseudo_pos.index, 'label'] = 1
    
    print(f"训练集2：正样本数量={train2['label'].sum()}, 总样本数={len(train2)}, 正样本比例={train2['label'].sum()/len(train2):.4f}")
    
    # 训练集3：直接使用原始train.csv
    print("\n生成训练集3...")
    train3 = train_df.copy()
    print(f"训练集3：正样本数量={train3['label'].sum()}, 总样本数={len(train3)}, 正样本比例={train3['label'].sum()/len(train3):.4f}")
    
    return train1, train2, train3

# 4. 特征选择集成算法
def ensemble_feature_selection(X, y, feature_names, weights=[0.3, 0.4, 0.3], top_k=50):
    """集成特征选择算法：MI、XGBoost、RF，权重分别为0.3、0.4、0.3"""
    
    # MI特征重要性
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranks = np.argsort(mi_scores)[::-1]  # 降序排序的索引
    mi_rank_dict = {feature_names[i]: len(mi_ranks) - rank for rank, i in enumerate(mi_ranks)}
    
    # XGBoost特征重要性
    xgb = XGBClassifier(random_state=42, n_jobs=-1)
    xgb.fit(X, y)
    xgb_scores = xgb.feature_importances_
    xgb_ranks = np.argsort(xgb_scores)[::-1]
    xgb_rank_dict = {feature_names[i]: len(xgb_ranks) - rank for rank, i in enumerate(xgb_ranks)}
    
    # RF特征重要性
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_scores = rf.feature_importances_
    rf_ranks = np.argsort(rf_scores)[::-1]
    rf_rank_dict = {feature_names[i]: len(rf_ranks) - rank for rank, i in enumerate(rf_ranks)}
    
    # 计算集成分数
    ensemble_scores = {}
    for feature in feature_names:
        ensemble_score = (mi_rank_dict[feature] * weights[0] + 
                         xgb_rank_dict[feature] * weights[1] + 
                         rf_rank_dict[feature] * weights[2])
        ensemble_scores[feature] = ensemble_score
    
    # 按集成分数降序排序，选择top_k特征
    top_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [feature for feature, score in top_features], ensemble_scores

# 5. 可视化特征对比
def visualize_feature_comparison(feature_sets, set_names, output_dir, feature_map):
    """可视化三个特征集的对比"""
    
    # 1. 特征排名对比
    print("\n生成特征排名对比图...")
    plt.figure(figsize=(15, 10))
    
    for i, (features, name) in enumerate(zip(feature_sets, set_names)):
        # 绘制前20个特征的排名
        top_features = features[:20]
        top_features_chinese = [get_chinese_feature_name(f, feature_map) for f in top_features]
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
    features, counts = zip(*top_freq_features)
    features_chinese = [get_chinese_feature_name(f, feature_map) for f in features]
    
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
        with open(f'{output_dir}/{name}_top50_features.txt', 'w', encoding='utf-8') as f:
            for j, feature in enumerate(features):
                chinese_name = get_chinese_feature_name(feature, feature_map)
                f.write(f'{j+1}. {chinese_name} ({feature})\n')
    
    # 4. 保存特征排名对比表，使用中文特征名
    print("生成特征排名对比表...")
    feature_rank_df = pd.DataFrame()
    for i, (features, name) in enumerate(zip(feature_sets, set_names)):
        for rank, feature in enumerate(features):
            chinese_name = get_chinese_feature_name(feature, feature_map)
            if feature not in feature_rank_df.index:
                feature_rank_df.loc[feature, '特征名称（中文）'] = chinese_name
                feature_rank_df.loc[feature, '特征名称（英文）'] = feature
            feature_rank_df.loc[feature, f'{name}_排名'] = rank + 1
    
    # 按平均排名排序
    feature_rank_df['平均排名'] = feature_rank_df[[f'{name}_排名' for name in set_names]].mean(axis=1)
    feature_rank_df = feature_rank_df.sort_values('平均排名')
    
    # 调整列顺序，将中文名称放在前面
    columns = ['特征名称（中文）', '特征名称（英文）'] + [f'{name}_排名' for name in set_names] + ['平均排名']
    feature_rank_df = feature_rank_df[columns]
    
    feature_rank_df.to_csv(f'{output_dir}/feature_rank_comparison.csv', index=False, encoding='utf-8-sig')
    
    print("\n特征对比可视化完成！")

# 主函数
def main():
    print("开始执行集成学习特征选择算法...")
    
    # 0. 加载特征映射
    feature_map = load_feature_mapping()
    
    # 1. 读取数据
    train_df, pu_predictions = load_data()
    
    # 2. 生成三种训练集
    train1, train2, train3 = generate_training_sets(train_df, pu_predictions)
    training_sets = [train1, train2, train3]
    set_names = ['训练集1_高置信负转正', '训练集2_伪正样本补充', '训练集3_原始数据']
    
    # 3. 对每个训练集进行特征选择
    all_top_features = []
    for i, (train, name) in enumerate(zip(training_sets, set_names)):
        print(f"\n=== 处理{name} ===")
        # 预处理数据
        X, y = preprocess_data(train)
        # 执行集成特征选择
        top_features, _ = ensemble_feature_selection(X, y, X.columns.tolist())
        all_top_features.append(top_features)
        print(f"{name}的Top 50特征：")
        for j, feature in enumerate(top_features[:10]):
            print(f"  {j+1}. {get_chinese_feature_name(feature, feature_map)}")
        print("  ...")
    
    # 4. 可视化特征对比
    visualize_feature_comparison(all_top_features, set_names, output_dir, feature_map)
    
    print("\n=== 执行完成 ===")
    print(f"特征选择结果已保存到目录：{output_dir}")

if __name__ == '__main__':
    main()