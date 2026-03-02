import pandas as pd
from sklearn.model_selection import train_test_split
import chardet
import os

def _detect_encoding(file_path: str) -> str:
    """
    使用 chardet 检测文件编码
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']

def _load_csv_with_encoding(file_path: str) -> pd.DataFrame:
    """
    加载 CSV 文件，自动检测编码并包含多种编码回退机制
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    
    detected_encoding = _detect_encoding(file_path)
    encodings_to_try = []
    if detected_encoding:
        encodings_to_try.append(detected_encoding)
    
    common_encodings = [
        'utf-8',
        'gbk',
        'gb18030',
        'big5',
        'latin-1',
        'utf-16',
        'cp1252'
    ]
    
    for enc in common_encodings:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    
    last_error = None
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
            continue
    
    try:
        return pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
    except Exception as e:
        raise ValueError(f"无法读取文件。尝试了以下编码: {encodings_to_try}。错误: {last_error}")

# 读取数据
def split_data(input_file, train_output, test_output, test_size=0.3, label_col='label'):
    print(f"读取数据文件: {input_file}")
    df = _load_csv_with_encoding(input_file)
    
    if label_col not in df.columns:
        raise KeyError(f"标签列 '{label_col}' 不存在于数据集中。可用列: {list(df.columns)}")

    print(f"原始数据形状: {df.shape}")
    print(f"原始标签分布:")
    print(df[label_col].value_counts())
    
    # 分层抽样，保持原有正负比例
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[label_col], 
        random_state=42
    )
    
    print(f"\n训练集形状: {train_df.shape}")
    print(f"训练集标签分布:")
    print(train_df[label_col].value_counts())
    
    print(f"\n测试集形状: {test_df.shape}")
    print(f"测试集标签分布:")
    print(test_df[label_col].value_counts())
    
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
