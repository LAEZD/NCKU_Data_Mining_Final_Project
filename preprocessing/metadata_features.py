# Metadata Features Extraction Module
# ===================================
# 提取和計算四個核心元數據特徵
# 這些特徵具有強信號且能有效區分回答質量

import os
import hashlib # <--- 新增導入
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
from tqdm import tqdm 
tqdm.pandas(desc="Extracting metadata features") 

CACHE_DIR = "./cache/metadata" # <--- 新增快取目錄定義

class MetadataFeatures:
    """
    元數據特徵提取類，提供四個核心特徵的計算方法
    """

    @staticmethod
    def remove_special_content(text: str, remove_special_blocks: bool = None) -> str:
        """
        如果 remove_special_blocks 為 True，則移除程式碼、數學公式和表格區塊。
        """
        if not remove_special_blocks or not isinstance(text, str):
            return text

        # 移除程式碼區塊
        text = re.sub(r'```.*?```', '[CODE_BLOCK]', text, flags=re.DOTALL)
        # 移除數學公式區塊
        text = re.sub(r'\$\$.*?\$\$', '[MATH_BLOCK]', text, flags=re.DOTALL)
        # 移除 Markdown 表格
        # text = re.sub(r"^\s*\|.*\|.*[\r\n]+^\s*\|[-|: \t]+\|.*[\r\n]+(?:^\s*\|.*\|.*\s*[\r\n]?)+", '[TABLE_BLOCK]', text, flags=re.MULTILINE)
        text = re.sub(r'((?:\|.*\|[\r\n\s]*){2,})', '[TABLE_BLOCK]', text)
        return text

    @staticmethod
    def calculate_jaccard_similarity(text1_processed: str, text2_processed: str) -> float:
        """
        計算 Jaccard Index - 衡量兩個文本的字元重疊程度.
        假設傳入的文本已經根據 REMOVE_SPECIAL_BLOCKS 的設定被適當處理過了。
        
        Args:
            text1_processed (str): 可能已處理的第一個文本
            text2_processed (str): 可能已處理的第二個文本
            
        Returns:
            float: Jaccard 相似度 (0-1之間)
        """        
        if not isinstance(text1_processed, str) or not isinstance(text2_processed, str):
            return 0.0
        
        words1 = set(text1_processed.lower().split())
        words2 = set(text2_processed.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
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
    def count_math_blocks(text: str) -> int:
        """
        計算 markdown 格式的數學公式區塊數量 ($$...$$)
        
        Args:
            text (str): 輸入文本
            
        Returns:
            int: 數學公式區塊數量
        """
        if not isinstance(text, str):
            return 0
        math_pattern = re.compile(r'\\$\\$.*?\\$\\$', re.DOTALL)
        return len(math_pattern.findall(text))

    @staticmethod
    def count_table_blocks(text: str) -> int:
        """
        計算 markdown 格式的表格數量
        
        Args:
            text (str): 輸入文本
            
        Returns:
            int: 表格數量
        """
        if not isinstance(text, str):
            return 0
        table_pattern = re.compile(r'((?:\\|.*?\\|[\\r\\n\\s]*){2,})')
        return len(table_pattern.findall(text))
    
    @staticmethod
    def calculate_content_blocks_diff(text1_raw: str, text2_raw: str) -> int:
        """
        計算 markdown 格式的程式碼、數學公式和表格區塊的總數量差值。
        始終在原始文本上操作。
        Args:
            text1_raw (str): 原始第一個文本 (response_a)
            text2_raw (str): 原始第二個文本 (response_b)
            
        Returns:
            int: 內容區塊總數量差值 (response_a - response_b)
        """
        count1_code = MetadataFeatures.count_code_blocks(text1_raw)
        count2_code = MetadataFeatures.count_code_blocks(text2_raw)
        code_diff = count1_code - count2_code

        count1_math = MetadataFeatures.count_math_blocks(text1_raw)
        count2_math = MetadataFeatures.count_math_blocks(text2_raw)
        math_diff = count1_math - count2_math

        count1_table = MetadataFeatures.count_table_blocks(text1_raw)
        count2_table = MetadataFeatures.count_table_blocks(text2_raw)
        table_diff = count1_table - count2_table
        
        return code_diff + math_diff + table_diff
    
    @staticmethod
    def calculate_length_diff(text1_raw: str, text2_raw: str) -> int:
        """
        計算回應長度差值。始終在原始文本上操作。
        
        Args:
            text1_raw (str): 原始第一個文本 (response_a)
            text2_raw (str): 原始第二個文本 (response_b)
            
        Returns:
            int: 長度差值 (response_a - response_b)
        """
        if not isinstance(text1_raw, str) or not isinstance(text2_raw, str):
            return 0
        
        return len(text1_raw) - len(text2_raw)
    
    @staticmethod
    def calculate_ttr(text_processed: str) -> float:
        """
        計算 Type-Token Ratio (TTR) - 詞彙豐富度指標。
        假設傳入的文本已經根據 REMOVE_SPECIAL_BLOCKS 的設定被適當處理過了。
        
        Args:
            text_processed (str): 可能已處理的輸入文本
            
        Returns:
            float: TTR 值 (0-1之間)
        """
        if not isinstance(text_processed, str) or len(text_processed.strip()) == 0:
            return 0.0
        
        words = text_processed.lower().split()
        
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    @staticmethod
    def calculate_ttr_diff(text1_processed: str, text2_processed: str) -> float:
        """
        計算 TTR 差值。
        假設傳入的文本已經根據 REMOVE_SPECIAL_BLOCKS 的設定被適當處理過了。
        
        Args:
            text1_processed (str): 可能已處理的第一個文本 (response_a)
            text2_processed (str): 可能已處理的第二個文本 (response_b)
            
        Returns:
            float: TTR 差值 (response_a - response_b)
        """        
        ttr1 = MetadataFeatures.calculate_ttr(text1_processed)
        ttr2 = MetadataFeatures.calculate_ttr(text2_processed)
        return ttr1 - ttr2
    
    @staticmethod
    def extract_core_features(row: pd.Series) -> Dict[str, float]:
        """
        提取五個核心元數據特徵（更新為5個特徵以匹配模型期望）
        
        Args:
            row (pd.Series): 包含 prompt, response_a, response_b 的數據行
            
        Returns:
            Dict[str, float]: 包含五個核心特徵的字典
        """
        response_a_raw = str(row.get('response_a', ''))
        response_b_raw = str(row.get('response_b', ''))

        # 根據開關決定是否預處理文本 (僅用於 Jaccard 和 TTR)
        response_a_for_jaccard_ttr = MetadataFeatures.remove_special_content(response_a_raw)
        response_b_for_jaccard_ttr = MetadataFeatures.remove_special_content(response_b_raw)

        # 計算原始差異值
        length_diff = MetadataFeatures.calculate_length_diff(response_a_raw, response_b_raw)
        content_blocks_diff = MetadataFeatures.calculate_content_blocks_diff(response_a_raw, response_b_raw)

        # 使用對數變換來縮放特徵，同時保留正負號
        scaled_length_diff = np.sign(length_diff) * np.log1p(abs(length_diff))
        scaled_content_blocks_diff = np.sign(content_blocks_diff) * np.log1p(abs(content_blocks_diff))

        # 計算TTR值
        ttr_a = MetadataFeatures.calculate_ttr(response_a_for_jaccard_ttr)
        ttr_b = MetadataFeatures.calculate_ttr(response_b_for_jaccard_ttr)
        
        # 計算TTR比值（新增的第5個特徵）
        ttr_ratio = max(ttr_a, ttr_b) / max(min(ttr_a, ttr_b), 0.001) if min(ttr_a, ttr_b) > 0 else 1.0

        core_features = {
            # Jaccard 和 TTR 使用可能被處理過的文本
            'jaccard_index': MetadataFeatures.calculate_jaccard_similarity(
                                response_a_for_jaccard_ttr, 
                                response_b_for_jaccard_ttr
                             ),
            'ttr_diff': MetadataFeatures.calculate_ttr_diff(
                                response_a_for_jaccard_ttr, 
                                response_b_for_jaccard_ttr
                        ),
            
            # Content blocks diff 和 Length diff 使用縮放後的值
            'content_blocks_diff': scaled_content_blocks_diff,
            'length_diff': scaled_length_diff,
            
            # 新增的第5個特徵：TTR比值
            'ttr_ratio': ttr_ratio
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
        # 目前所有特徵就是核心特徵, 呼叫更新後的 extract_core_features
        return MetadataFeatures.extract_core_features(row)

    @staticmethod
    def _generate_cache_key(df: pd.DataFrame, feature_type: str) -> str:
        """
        為 DataFrame 和特徵類型生成一個唯一的快取鍵。
        """
        # 使用行數和前幾行的部分內容來生成一個相對穩定的雜湊值
        # 注意：這不是一個完美的雜湊，如果 DataFrame 內容有細微變化但行數和開頭不變，可能會導致快取誤判
        # 更健壯的方法可能需要雜湊整個 DataFrame 的內容，但會更耗時
        hasher = hashlib.md5()
        hasher.update(str(len(df)).encode())
        if not df.empty:
            # 取樣前5行，每行取前100個字元來計算雜湊
            sample_data = "".join(df.head(5).to_string(index=False, header=False, max_colwidth=100).split())
            hasher.update(sample_data.encode())
        hasher.update(feature_type.encode())
        return hasher.hexdigest()

    @staticmethod
    def add_metadata_features_to_dataframe(df: pd.DataFrame, feature_type: str = 'core') -> pd.DataFrame:
        """
        將元數據特徵添加到數據框中，並使用快取機制。
        
        Args:
            df (pd.DataFrame): 原始數據框
            feature_type (str): 特徵類型 (\'core\' 或 \'all\')
            
        Returns:
            pd.DataFrame: 添加了元數據特徵的數據框
        """
        print(f"  - 開始提取 {feature_type} 元數據特徵...")

        # 確保快取目錄存在
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            print(f"  - 創建快取目錄: {CACHE_DIR}")

        # 生成快取檔案路徑
        cache_key = MetadataFeatures._generate_cache_key(df, feature_type)
        cache_file_path = os.path.join(CACHE_DIR, f"{cache_key}_{feature_type}.pkl")
        
        # 檢查快取是否存在
        if os.path.exists(cache_file_path):
            try:
                print(f"  - 發現快取檔案: {cache_file_path}，正在載入...")
                # 載入時，只載入特徵列，避免重複載入原始 df 的欄位
                cached_features_df = pd.read_pickle(cache_file_path)
                
                # 進行更可靠的驗證：檢查快取中的欄位是否都是預期的特徵欄位
                # 假設核心特徵是固定的
                expected_core_features = ['jaccard_index','ttr_diff', 'ttr_ratio', 'content_blocks_diff', 'length_diff']
                # 'all' features 目前與 core 相同，如果將來不同，這裡需要調整
                expected_features_set = set(expected_core_features)

                if isinstance(cached_features_df, pd.DataFrame) and not cached_features_df.empty and set(cached_features_df.columns) == expected_features_set and len(cached_features_df) == len(df):
                    # 將快取的特徵 DataFrame 與原始 DataFrame (不包含已存在的特徵列，以防萬一) 合併
                    # 先移除原始 df 中可能已存在的同名特徵列，再合併
                    df_copy = df.copy()
                    for col in expected_features_set:
                        if col in df_copy.columns:
                            df_copy = df_copy.drop(columns=[col])
                    
                    df_enhanced = pd.concat([df_copy.reset_index(drop=True), cached_features_df.reset_index(drop=True)], axis=1)
                    print(f"  - 成功從快取載入 {len(cached_features_df.columns)} 個元數據特徵")
                    print(f"  - 已載入特徵: {list(cached_features_df.columns)}")
                    return df_enhanced
                else:
                    print(f"  - 快取檔案無效 (欄位不符、為空或長度不匹配)，將重新計算特徵。")
                    if not isinstance(cached_features_df, pd.DataFrame) or cached_features_df.empty:
                         print(f"    - 原因: 快取檔案不是有效的 DataFrame 或為空。")
                    elif set(cached_features_df.columns) != expected_features_set:
                         print(f"    - 原因: 快取欄位與預期不符。預期: {expected_features_set}, 實際: {set(cached_features_df.columns)}")
                    elif len(cached_features_df) != len(df):
                         print(f"    - 原因: 快取長度與當前 DataFrame 長度不符。預期: {len(df)}, 實際: {len(cached_features_df)}")

            except Exception as e:
                print(f"  - 載入快取檔案失敗: {e}，將重新計算特徵。")

        df_enhanced = df.copy()
        
        if feature_type == 'core':
            features_list = df_enhanced.progress_apply(MetadataFeatures.extract_core_features, axis=1).tolist()
        else: # 'all'
            features_list = df_enhanced.progress_apply(MetadataFeatures.extract_all_features, axis=1).tolist()
        
        features_df = pd.DataFrame(features_list) # 這只包含新提取的特徵列
        
        # 儲存到快取 (只儲存特徵 DataFrame)
        try:
            features_df.to_pickle(cache_file_path)
            print(f"  - 特徵已計算並儲存到快取: {cache_file_path}")
        except Exception as e:
            print(f"  - 儲存特徵到快取失敗: {e}")
            
        # 合併原始 DataFrame 和新提取的特徵 DataFrame
        # 先移除原始 df 中可能已存在的同名特徵列，再合併
        for col in features_df.columns:
            if col in df_enhanced.columns:
                df_enhanced = df_enhanced.drop(columns=[col])
        df_enhanced = pd.concat([df_enhanced.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        
        print(f"  - 成功提取 {len(features_df.columns)} 個元數據特徵")
        print(f"  - 已提取特徵: {list(features_df.columns)}")
        
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
            feature_order = ['jaccard_index', 'ttr_diff', 'ttr_ratio', 'content_blocks_diff', 'length_diff'] # Adjusted order for clarity
        
        feature_vector = []
        for feature_name in feature_order:
            value = features_dict.get(feature_name, 0.0)
            if np.isinf(value):
                value = 10.0  
            elif np.isnan(value):
                value = 0.0
            feature_vector.append(float(value))
        
        return feature_vector

    @staticmethod
    def _calculate_features(text: str) -> dict:
        """
        計算文本的所有元數據特徵
        
        Args:
            text (str): 輸入文本
            
        Returns:
            dict: 包含所有特徵的字典
        """
        if not isinstance(text, str):
            return {}
        
        # 計算字元數、詞彙數、句子數
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # 計算平均詞彙長度
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # 計算獨特詞彙比例
        unique_word_ratio = MetadataFeatures.calculate_ttr(text)
        
        # 停用詞和標點符號計數
        stopwords = set(['的', '是', '在', '和', '有', '我', '他', '她', '它', '這', '那', '個', '了', '不', '人', '都', '說', '要', '去', '嗎'])
        words = text.split()
        stopword_count = len([word for word in words if word in stopwords])
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # 特殊字元計數 (例如：@, #, $, %, ^, &, *)
        special_char_count = len(re.findall(r'[@#$%^&*]', text))
        
        # URL 計數
        url_count = len(re.findall(r'http[s]?://\S+', text))
        
        # 程式碼區塊、數學公式和表格區塊計數
        code_block_count = MetadataFeatures.count_code_blocks(text) + MetadataFeatures.count_math_blocks(text) + MetadataFeatures.count_table_blocks(text)
        
        # 返回所有特徵的字典
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_word_ratio,
            'stopword_count': stopword_count,
            'punctuation_count': punctuation_count,
            'special_char_count': special_char_count,
            'code_block_count': code_block_count,
            'url_count': url_count
        }

    @staticmethod
    def get_feature_columns(feature_type: str = 'core') -> list:
        """Returns the list of metadata feature column names."""
        # 定義核心特徵列表，這與 extract_core_features 中實際計算的特徵順序一致
        # 順序必須與 extract_core_features 返回的字典鍵順序完全匹配
        core_features = ['jaccard_index', 'ttr_diff', 'ttr_ratio', 'content_blocks_diff', 'length_diff']

        # 根據 `extract_all_features` 當前的實現，'all' 和 'core' 返回相同的特徵。
        # 因此，我們統一返回 core_features 列表以確保一致性。
        if feature_type == 'core':
            return core_features
        elif feature_type == 'all':
            # `extract_all_features` 目前直接調用 `extract_core_features`，
            # 所以 'all' 類型也應該返回相同的特徵列表。
            return core_features
        else:
            return []

    @staticmethod
    def add_metadata_features(row, feature_type: str = 'core') -> pd.Series:
        """Calculates metadata features for a single row (prompt, response_a, response_b)."""
        response_a = str(row.get('response_a', ''))
        response_b = str(row.get('response_b', ''))
        
        # 計算TTR比值
        ttr_a = MetadataFeatures.calculate_ttr(MetadataFeatures.remove_special_content(response_a))
        ttr_b = MetadataFeatures.calculate_ttr(MetadataFeatures.remove_special_content(response_b))
        ttr_ratio = max(ttr_a, ttr_b) / max(min(ttr_a, ttr_b), 0.001) if min(ttr_a, ttr_b) > 0 else 1.0
        
        return pd.Series({
            'jaccard_index': MetadataFeatures.calculate_jaccard_similarity(response_a, response_b),
            'ttr_diff': MetadataFeatures.calculate_ttr_diff(response_a, response_b),
            'content_blocks_diff': MetadataFeatures.calculate_content_blocks_diff(response_a, response_b),
            'length_diff': MetadataFeatures.calculate_length_diff(response_a, response_b),
            'ttr_ratio': ttr_ratio
        })

