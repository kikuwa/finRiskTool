import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
def split_data(input_file, train_output, test_output, test_size=0.3):
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始标签分布:")
    print(df['label'].value_counts())
    
    # 分层抽样，保持原有正负比例
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'], 
        random_state=42
    )
    
    print(f"\n训练集形状: {train_df.shape}")
    print(f"训练集标签分布:")
    print(train_df['label'].value_counts())
    
    print(f"\n测试集形状: {test_df.shape}")
    print(f"测试集标签分布:")
    print(test_df['label'].value_counts())
    
    # 保存文件
    train_df.to_csv(train_output, index=False, encoding='utf-8')
    test_df.to_csv(test_output, index=False, encoding='utf-8')
    
    print(f"\n数据分割完成！")
    print(f"训练集已保存到: {train_output}")
    print(f"测试集已保存到: {test_output}")

if __name__ == '__main__':
    split_data(
        input_file='data/generated_data.csv',
        train_output='data/train.csv',
        test_output='data/test.csv',
        test_size=0.3
    )
