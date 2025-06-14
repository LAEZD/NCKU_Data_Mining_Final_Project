# Metadata Features Extraction Module
# ===================================
# 提取和計算四個核心元數據特徵
# 這些特徵具有強信號且能有效區分回答質量

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

class MetadataFeatures:
    """
    元數據特徵提取類，提供四個核心特徵的計算方法
    """
    
    @staticmethod
    def calculate_jaccard_similarity(text1: str, text2: str) -> float:
        """
        計算 Jaccard Index - 衡量兩個文本的字元重疊程度
        
        Args:
            text1 (str): 第一個文本
            text2 (str): 第二個文本
            
        Returns:
            float: Jaccard 相似度 (0-1之間)
        """        
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        
        # 轉換為字詞集合 (以空格分割)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # 計算交集和聯集
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # 避免除以零
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def count_code_blocks(text: str) -> int:
        """
        計算 markdown 格式的程式碼區塊數量
        
        Args:
            text (str): 輸入文本
            
        Returns:
            int: 程式碼區塊數量
        """
        if not isinstance(text, str):
            return 0
        
        return str(text).count('```') // 2
    
    @staticmethod
    def calculate_code_blocks_diff(text1: str, text2: str) -> int:
        """
        計算 markdown 格式的程式碼區塊數量差值
        
        Args:
            text1 (str): 第一個文本 (response_a)
            text2 (str): 第二個文本 (response_b)
            
        Returns:
            int: 程式碼區塊數量差值 (response_a - response_b)
        """
        count1 = MetadataFeatures.count_code_blocks(text1)
        count2 = MetadataFeatures.count_code_blocks(text2)
        return count1 - count2
    
    @staticmethod
    def calculate_length_diff(text1: str, text2: str) -> int:
        """
        計算回應長度差值
        
        Args:
            text1 (str): 第一個文本 (response_a)
            text2 (str): 第二個文本 (response_b)
            
        Returns:
            int: 長度差值 (response_a - response_b)
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0
        
        return len(text1) - len(text2)
    
    @staticmethod
    def calculate_ttr(text: str) -> float:
        """
        計算 Type-Token Ratio (TTR) - 詞彙豐富度指標
        
        Args:
            text (str): 輸入文本
            
        Returns:
            float: TTR 值 (0-1之間)
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        
        # 分詞並轉為小寫
        words = text.lower().split()
        
        if len(words) == 0:
            return 0.0
        
        # 計算不重複詞彙數量 / 總詞彙數量
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    @staticmethod
    def calculate_ttr_diff(text1: str, text2: str) -> float:
        """
        計算 TTR 差值
        
        Args:
            text1 (str): 第一個文本 (response_a)
            text2 (str): 第二個文本 (response_b)
            
        Returns:
            float: TTR 差值 (response_a - response_b)
        """        
        ttr1 = MetadataFeatures.calculate_ttr(text1)
        ttr2 = MetadataFeatures.calculate_ttr(text2)
        return ttr1 - ttr2
    
    @staticmethod
    def extract_core_features(row: pd.Series) -> Dict[str, float]:
        """
        提取四個核心元數據特徵
        
        Args:
            row (pd.Series): 包含 prompt, response_a, response_b 的數據行
            
        Returns:
            Dict[str, float]: 包含四個核心特徵的字典
        """
        response_a = str(row.get('response_a', ''))
        response_b = str(row.get('response_b', ''))
        
        core_features = {
            'jaccard_index': MetadataFeatures.calculate_jaccard_similarity(response_a, response_b),
            'code_blocks_diff': MetadataFeatures.calculate_code_blocks_diff(response_a, response_b),
            'length_diff': MetadataFeatures.calculate_length_diff(response_a, response_b),
            'ttr_diff': MetadataFeatures.calculate_ttr_diff(response_a, response_b)
        }
        
        return core_features
    
    @staticmethod
    def extract_all_features(row: pd.Series) -> Dict[str, float]:
        """
        提取所有元數據特徵 (目前與核心特徵相同)
        
        Args:
            row (pd.Series): 包含 prompt, response_a, response_b 的數據行
            
        Returns:
            Dict[str, float]: 包含所有元數據特徵的字典
        """
        # 目前所有特徵就是核心特徵
        return MetadataFeatures.extract_core_features(row)
    
    @staticmethod
    def add_metadata_features_to_dataframe(df: pd.DataFrame, feature_type: str = 'core') -> pd.DataFrame:
        """
        將元數據特徵添加到數據框中
        
        Args:
            df (pd.DataFrame): 原始數據框
            feature_type (str): 特徵類型 ('core' 或 'all')
            
        Returns:
            pd.DataFrame: 添加了元數據特徵的數據框
        """
        print(f"  - 開始提取 {feature_type} 元數據特徵...")
        
        df_enhanced = df.copy()
        
        if feature_type == 'core':
            # 提取核心特徵
            features_list = df_enhanced.apply(MetadataFeatures.extract_core_features, axis=1).tolist()
        else:
            # 提取所有特徵
            features_list = df_enhanced.apply(MetadataFeatures.extract_all_features, axis=1).tolist()
        
        # 將特徵字典轉換為數據框列
        features_df = pd.DataFrame(features_list)
        
        # 合併到原始數據框
        df_enhanced = pd.concat([df_enhanced, features_df], axis=1)
        
        print(f"  - 成功添加 {len(features_df.columns)} 個元數據特徵")
        print(f"  - 新增特徵: {list(features_df.columns)}")
        
        return df_enhanced
    
    @staticmethod
    def analyze_feature_distributions(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Dict]:
        """
        分析元數據特徵的分布情況
        
        Args:
            df (pd.DataFrame): 包含特徵的數據框
            feature_names (List[str]): 要分析的特徵名稱列表
            
        Returns:
            Dict[str, Dict]: 每個特徵的統計信息
        """
        print("  - 開始分析元數據特徵分布...")
        
        stats = {}
        
        for feature in feature_names:
            if feature in df.columns:
                feature_data = df[feature].dropna()
                stats[feature] = {
                    'count': len(feature_data),
                    'mean': feature_data.mean(),
                    'median': feature_data.median(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'percentile_25': feature_data.quantile(0.25),
                    'percentile_75': feature_data.quantile(0.75),
                    'null_count': df[feature].isnull().sum()
                }
                
                print(f"    {feature}: 均值={stats[feature]['mean']:.3f}, "
                      f"標準差={stats[feature]['std']:.3f}, "
                      f"範圍=[{stats[feature]['min']:.3f}, {stats[feature]['max']:.3f}]")
        
        return stats
    @staticmethod
    def create_feature_vector(features_dict: Dict[str, float], feature_order: List[str] = None) -> List[float]:
        """
        將特徵字典轉換為有序的特徵向量
        
        Args:
            features_dict (Dict[str, float]): 特徵字典
            feature_order (List[str]): 特徵順序，如果為 None 則使用默認順序
            
        Returns:
            List[float]: 特徵向量
        """
        if feature_order is None:
            # 新的四個核心特徵順序
            feature_order = ['jaccard_index', 'code_blocks_diff', 'length_diff', 'ttr_diff']
        
        feature_vector = []
        for feature_name in feature_order:
            value = features_dict.get(feature_name, 0.0)
            # 處理可能的無窮大值和 NaN
            if np.isinf(value):
                value = 10.0  # 將無窮大值替換為一個合理的大數
            elif np.isnan(value):
                value = 0.0
            feature_vector.append(float(value))
        
        return feature_vector
    
