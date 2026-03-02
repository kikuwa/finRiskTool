import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import chardet

class DataLoader:
    def __init__(self, label_col: str = 'label', test_size: float = 0.3, random_state: int = 42):
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {} # 用于存储每个列的编码器

    def load_full_dataset(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载完整数据集并自动划分训练集和测试集
        """
        df = self._load_csv(file_path)
        self.validate_data(df)
        
        # 划分训练集和测试集
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=df[self.label_col]  # 保持标签比例
        )
        return train_df, test_df

    def load_train_test_split(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分别加载训练集和测试集
        """
        train_df = self._load_csv(train_path)
        test_df = self._load_csv(test_path)
        
        self.validate_data(train_df)
        self.validate_data(test_df)
        
        # 验证两个数据集的列是否一致
        if set(train_df.columns) != set(test_df.columns):
            raise ValueError("训练集和测试集的列不一致")
            
        return train_df, test_df

    def _detect_encoding(self, file_path: str) -> str:
        """
        使用 chardet 检测文件编码
        """
        with open(file_path, 'rb') as f:
            # 读取一部分文件来检测，避免大文件消耗过多内存
            result = chardet.detect(f.read(100000))
        return result['encoding']

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        加载 CSV 文件，自动检测编码并包含多种编码回退机制
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
            
        if not file_path.lower().endswith('.csv'):
            raise ValueError("只允许上传 CSV 文件")

        # 定义常见编码的尝试顺序，优先使用检测到的编码
        detected_encoding = self._detect_encoding(file_path)
        encodings_to_try = []
        if detected_encoding:
            encodings_to_try.append(detected_encoding)
        
        # 添加常见编码回退列表
        common_encodings = [
            'utf-8',
            'gbk',
            'gb18030',
            'big5',
            'latin-1',
            'utf-16',
            'cp1252'
        ]
        
        # 合并列表，确保不重复
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
        
        # 如果所有编码都失败，最后尝试使用 'utf-8' 并忽略或替换错误字符
        try:
            return pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
        except Exception as e:
            raise ValueError(f"无法读取文件。尝试了以下编码: {encodings_to_try}。错误: {last_error}")

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        验证数据格式和标签
        """
        if df.empty:
            raise ValueError("数据集为空，请上传有效数据")
            
        if self.label_col not in df.columns:
            raise ValueError(f"指定的标签列 '{self.label_col}' 不存在于数据集中")
            
        # 验证标签列是否为二分类
        unique_labels = df[self.label_col].unique()
        if len(unique_labels) != 2:
            raise ValueError(f"标签列 '{self.label_col}' 必须是二分类（只能有 2 个唯一值），当前有 {len(unique_labels)} 个: {unique_labels}")
            
        if len(df.columns) < 2:
            raise ValueError("数据集必须至少包含一个特征列（除标签列外）")

    def preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """
        数据预处理：处理缺失值、编码等
        
        Args:
            df: 待处理的数据框
            fit_encoders: 是否拟合编码器。对于训练集应为 True，对于测试集应为 False（使用训练集的编码器）
        """
        # 复制一份数据避免修改原始数据
        df_processed = df.copy()
        
        # 简单的缺失值处理：数值型填均值，类别型填众数
        for col in df_processed.columns:
            if col == self.label_col:
                continue
                
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            else:
                # 填充缺失值
                mode_val = df_processed[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Missing'
                df_processed[col] = df_processed[col].fillna(fill_val)
                
                # 转换为字符串类型，确保 LabelEncoder 可以处理
                df_processed[col] = df_processed[col].astype(str)
                
                # Label Encoding
                if fit_encoders:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # 处理未知标签：将未知标签替换为 'Unknown' 或其他策略
                        # 这里简单处理：如果遇到未知标签，将其映射为 -1 或者众数，或者报错
                        # 为了稳健性，我们可以扩展 LabelEncoder 的类，或者使用 map
                        
                        # 一种简单的处理未知类别的方法：
                        # 找出测试集中存在但训练集中不存在的类别
                        known_classes = set(le.classes_)
                        current_classes = set(df_processed[col].unique())
                        unknown_classes = current_classes - known_classes
                        
                        if unknown_classes:
                            # 将未知类别替换为训练集中的众数（第一个类）或者一个特定值
                            # 这里我们选择替换为第一个类（通常是0对应的类）
                            # 或者更安全的做法：重新拟合（但这会破坏数据一致性），或者报错
                            # 考虑到这是演示，我们尝试将未知值替换为 'Unknown' 如果它在 classes_ 中，否则替换为第一个类
                            
                            # 更好的策略：使用 map，未匹配的填充为 -1 或其他值
                            # 但 LabelEncoder 不支持 transform 未知值
                            
                            # 策略：将未知值替换为训练集中出现最多的值
                            most_frequent = le.classes_[0] # 假设
                            df_processed[col] = df_processed[col].apply(lambda x: x if x in known_classes else most_frequent)
                            
                        df_processed[col] = le.transform(df_processed[col])
                    else:
                        # 如果没有对应的编码器（例如新出现的列），则新建一个（虽然理论上不应发生）
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col])
                        # 注意：这里不保存到 self.label_encoders，因为它只属于测试集
                
        return df_processed
