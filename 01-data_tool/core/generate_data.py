import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 读取特征文件
def read_features(file_path):
    features = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split(',')
            if parts:
                features.append(parts[0])
    return features

# 生成随机日期
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# 生成随机字符串
def random_string(length=10):
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(letters) for i in range(length))

# 生成随机数值
def random_decimal(min_val=0, max_val=1000000, precision=6):
    return round(random.uniform(min_val, max_val), precision)

# 生成随机整数
def random_int(min_val=0, max_val=1000):
    return random.randint(min_val, max_val)

# 生成数据
def generate_data(features, n_samples=10000, positive_rate=0.007):
    # 计算正样本数量
    n_positive = int(n_samples * positive_rate)
    n_negative = n_samples - n_positive
    
    # 创建标签
    labels = [1] * n_positive + [0] * n_negative
    random.shuffle(labels)
    
    # 创建数据字典
    data = {'label': labels}
    
    # 为每个特征生成随机数据
    for feature in features:
        feature_data = []
        for _ in range(n_samples):
            # 随机生成缺失值，缺失率约为10%
            if random.random() < 0.1:
                feature_data.append(np.nan)
            else:
                # 根据特征名的后缀或内容生成不同类型的数据
                if 'id' in feature.lower() or 'code' in feature.lower() or 'no' in feature.lower():
                    # 生成随机ID或代码
                    feature_data.append(random_string(16))
                elif 'date' in feature.lower() or 'dt' in feature.lower():
                    # 生成随机日期
                    start_date = datetime(2000, 1, 1)
                    end_date = datetime(2023, 12, 31)
                    feature_data.append(random_date(start_date, end_date).strftime('%Y-%m-%d'))
                elif 'amt' in feature.lower() or 'balance' in feature.lower() or 'credit' in feature.lower() or 'sum' in feature.lower() or 'scale' in feature.lower() or 'exposure' in feature.lower():
                    # 生成随机金额或数值
                    feature_data.append(random_decimal())
                elif 'number' in feature.lower() or 'count' in feature.lower():
                    # 生成随机整数
                    feature_data.append(random_int())
                elif 'desc' in feature.lower() or 'name' in feature.lower() or 'addr' in feature.lower() or 'tel' in feature.lower() or 'industry' in feature.lower() or 'type' in feature.lower():
                    # 生成随机字符串
                    feature_data.append(random_string(20))
                elif 'flag' in feature.lower() or 'ind' in feature.lower() or 'sign' in feature.lower() or 'status' in feature.lower() or 'level' in feature.lower():
                    # 生成随机标志或状态
                    options = ['Y', 'N', '1', '0', 'A', 'B', 'C', 'D']
                    feature_data.append(random.choice(options))
                else:
                    # 默认生成随机字符串
                    feature_data.append(random_string(10))
        
        data[feature] = feature_data
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 调整列顺序，将label列放在最后
    cols = [col for col in df.columns if col != 'label'] + ['label']
    df = df[cols]
    
    return df

if __name__ == '__main__':
    print("开始运行数据生成脚本...")
    
    # 读取特征
    features_file = 'data/全部特征.txt'
    print(f"读取特征文件: {features_file}")
    features = read_features(features_file)
    print(f"成功读取 {len(features)} 个特征")
    
    # 生成数据
    print("开始生成数据...")
    df = generate_data(features, n_samples=10000, positive_rate=0.007)
    print(f"数据生成完成，共{len(df)}条记录")
    
    # 保存为CSV文件
    output_file = 'data/generated_data.csv'
    print(f"保存数据到: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"数据保存成功")
    
    print(f"正样本数量: {sum(df['label'] == 1)}")
    print(f"负样本数量: {sum(df['label'] == 0)}")
    print(f"正样本比例: {sum(df['label'] == 1) / len(df):.4f}")
    print("脚本运行结束")
