# Metadata Features Extraction Module
# ===================================
# 提取和計算基於數據分析確認的三個核心元數據特徵
# 這些特徵具有強信號且不受語言限制

import pandas as pd
import numpy as np
import string
from typing import Dict, List, Tuple

class MetadataFeatures:
    """
    元數據特徵提取類，提供三個核心特徵的計算方法
    """
    
    @staticmethod
    def calculate_punctuation_variety(text: str) -> int:
        """
        計算標點符號多樣性 - 統計文本中不同標點符號的種類數量
        
        Args:
            text (str): 輸入文本
            
        Returns:
            int: 不同標點符號的種類數量
        """
        if not isinstance(text, str):
            return 0
        
        # 獲取所有標點符號
        punctuation_chars = set(string.punctuation)
        # 計算文本中出現的不同標點符號
        found_punctuation = set(char for char in text if char in punctuation_chars)
        
        return len(found_punctuation)
    
    @staticmethod
    def calculate_jaccard_similarity(text1: str, text2: str) -> float:
        """
        計算字元級 Jaccard 相似度 - 衡量兩個文本的字元重疊程度
        
        Args:
            text1 (str): 第一個文本
            text2 (str): 第二個文本
            
        Returns:
            float: Jaccard 相似度 (0-1之間)
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        
        # 轉換為字元集合
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        # 計算交集和聯集
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # 避免除以零
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def calculate_length_ratio(text1: str, text2: str) -> float:
        """
        計算長度比例 - 較長文本與較短文本的長度比值
        
        Args:
            text1 (str): 第一個文本
            text2 (str): 第二個文本
            
        Returns:
            float: 長度比例 (>=1.0)
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 1.0
        
        len1 = len(text1)
        len2 = len(text2)
        
        # 避免除以零
        if min(len1, len2) == 0:
            return float('inf') if max(len1, len2) > 0 else 1.0
        
        # 返回較長者除以較短者
        return max(len1, len2) / min(len1, len2)
    
    @staticmethod
    def extract_all_features(row: pd.Series) -> Dict[str, float]:
        """
        提取單行數據的所有元數據特徵
        
        Args:
            row (pd.Series): 包含 prompt, response_a, response_b 的數據行
            
        Returns:
            Dict[str, float]: 包含所有元數據特徵的字典
        """
        prompt = str(row.get('prompt', ''))
        response_a = str(row.get('response_a', ''))
        response_b = str(row.get('response_b', ''))
        
        features = {
            # 標點符號多樣性
            'punc_v_a': MetadataFeatures.calculate_punctuation_variety(response_a),
            'punc_v_b': MetadataFeatures.calculate_punctuation_variety(response_b),
            'punc_v_prompt': MetadataFeatures.calculate_punctuation_variety(prompt),
            
            # Jaccard 相似度
            'resp_jaccard': MetadataFeatures.calculate_jaccard_similarity(response_a, response_b),
            'prompt_resp_a_jaccard': MetadataFeatures.calculate_jaccard_similarity(prompt, response_a),
            'prompt_resp_b_jaccard': MetadataFeatures.calculate_jaccard_similarity(prompt, response_b),
            
            # 長度比例
            'len_ratio': MetadataFeatures.calculate_length_ratio(response_a, response_b),
            'prompt_resp_a_len_ratio': MetadataFeatures.calculate_length_ratio(prompt, response_a),
            'prompt_resp_b_len_ratio': MetadataFeatures.calculate_length_ratio(prompt, response_b),
            
            # 額外的長度特徵
            'prompt_length': len(prompt),
            'response_a_length': len(response_a),
            'response_b_length': len(response_b),
            'total_length': len(prompt) + len(response_a) + len(response_b)
        }
        
        return features
    
    @staticmethod
    def extract_core_features(row: pd.Series) -> Dict[str, float]:
        """
        提取核心的三個元數據特徵（最重要的特徵）
        
        Args:
            row (pd.Series): 包含 prompt, response_a, response_b 的數據行
            
        Returns:
            Dict[str, float]: 包含核心三個特徵的字典
        """
        response_a = str(row.get('response_a', ''))
        response_b = str(row.get('response_b', ''))
        
        core_features = {
            'punc_v_a': MetadataFeatures.calculate_punctuation_variety(response_a),
            'punc_v_b': MetadataFeatures.calculate_punctuation_variety(response_b),
            'resp_jaccard': MetadataFeatures.calculate_jaccard_similarity(response_a, response_b),
            'len_ratio': MetadataFeatures.calculate_length_ratio(response_a, response_b)
        }
        
        return core_features
    
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
            # 默認的核心特徵順序
            feature_order = ['punc_v_a', 'punc_v_b', 'resp_jaccard', 'len_ratio']
        
        feature_vector = []
        for feature_name in feature_order:
            value = features_dict.get(feature_name, 0.0)
            # 處理可能的無窮大值
            if np.isinf(value):
                value = 10.0  # 將無窮大值替換為一個合理的大數
            elif np.isnan(value):
                value = 0.0
            feature_vector.append(float(value))
        
        return feature_vector
    
