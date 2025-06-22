# Kaggle Offline Inference Script - kaggle_last.py
# ==================================================
# 專門用於 Kaggle 離線環境的推理腳本
# 使用預訓練的模型進行預測，無需重新訓練

import os
import json
import numpy as np
import pandas as pd
import torch
import hashlib
import re
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm

# >>>>>>> ADD >>>>>>>
# 使用與訓練階段完全相同的 MetadataFeatures 來避免特徵計算與順序不一致
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


# <<<<<<< ADD <<<<<<<

# --------------------------------------------------------------------------
# Embedded DualTowerPairClassifier (與 fine_tuning.py 完全一致)
# --------------------------------------------------------------------------
class DualTowerPairClassifier(nn.Module):
    """
    Dual-Encoder / Two-Tower 模型 + Metadata 特徵處理
    - 共用一座 Transformer Encoder，各自編碼 Prompt、Response A、Response B  
    - 特徵向量: [v_p, v_a, |v_p − v_a|, v_b, |v_p − v_b|, metadata_features]
    - metadata 透過 meta_path 從4維升到768維
    - 最終特徵: 768 × 6 = 4608 維
    - 3-類 softmax → 0=A wins, 1=B wins, 2=Tie
    """

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.2,
        metadata_feature_size: int = 5,  # metadata 原始維度（修正為5）
        metadata_fusion: str = 'dual_path',  # metadata 融合方式
        config_dict: dict = None,  # 直接傳入配置字典用於離線載入
    ):
        super().__init__()
        
        # 如果提供了配置字典，直接使用它創建 encoder
        if config_dict is not None:
            from transformers import DistilBertConfig, DistilBertModel
            encoder_config = DistilBertConfig(**config_dict)
            self.encoder = DistilBertModel(encoder_config)
            print(f"    💡 使用提供的配置創建 DistilBert encoder")
        else:
            # 否則使用標準方式（可能需要網絡）
            try:
                self.encoder = AutoModel.from_pretrained(base_model, local_files_only=True)
                print(f"    💡 使用本地緩存載入 {base_model}")
            except Exception as e:
                print(f"    💡 本地載入失敗，嘗試使用預設配置: {e}")
                # 如果本地載入失敗，使用預設的 DistilBert 配置
                from transformers import DistilBertConfig, DistilBertModel
                encoder_config = DistilBertConfig(
                    vocab_size=30522,
                    dim=hidden_size,
                    n_layers=6,
                    n_heads=12,
                    hidden_dim=3072,
                    dropout=0.1,
                    attention_dropout=0.1,
                    max_position_embeddings=512,
                    initializer_range=0.02
                )
                self.encoder = DistilBertModel(encoder_config)
                print(f"    💡 使用預設 DistilBert 配置創建 encoder")
        
        self.dropout = nn.Dropout(dropout)
        
        # Metadata 處理路徑 - 與 fine_tuning.py 完全一致
        self.meta_path = None
        if metadata_fusion == 'dual_path' and metadata_feature_size > 0:
            self.meta_path = nn.Sequential(
                nn.Linear(metadata_feature_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            print(f"    💡 創建 metadata 處理路徑: {metadata_feature_size} -> {hidden_size} -> {hidden_size}")
        
        # 分類器輸入維度: 基礎特徵(768×5) + metadata特徵(768) = 768×6 = 4608
        classifier_input_dim = hidden_size * 6 if self.meta_path else hidden_size * 5
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )
        
        print(f"    💡 分類器輸入維度: {classifier_input_dim} (期望 4608)")

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """回傳 [CLS] embedding (batch, hidden)"""
        out = self.encoder(
            input_ids=ids, attention_mask=mask, return_dict=True
        )
        return out.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        p_input_ids: torch.Tensor = None,
        p_attention_mask: torch.Tensor = None,
        a_input_ids: torch.Tensor = None,
        a_attention_mask: torch.Tensor = None,
        b_input_ids: torch.Tensor = None,
        b_attention_mask: torch.Tensor = None,
        metadata_features: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        支持兩種輸入格式：
        1. 聯合輸入：input_ids + attention_mask + metadata_features (推理時使用)
        2. 分離輸入：p_input_ids, a_input_ids, b_input_ids + metadata_features (訓練時使用)
        """
        
        if input_ids is not None and attention_mask is not None:
            # 聯合輸入格式（推理時使用）
            cls_embedding = self.encode(input_ids, attention_mask)
            feat = cls_embedding
                
        else:
            # 分離輸入格式（訓練時使用）
            v_p = self.encode(p_input_ids, p_attention_mask)
            v_a = self.encode(a_input_ids, a_attention_mask)
            v_b = self.encode(b_input_ids, b_attention_mask)

            # 拼接基礎特徵向量：[v_p, v_a, |v_p - v_a|, v_b, |v_p - v_b|] = 768×5
            feat = torch.cat(
                [v_p, v_a, torch.abs(v_p - v_a), v_b, torch.abs(v_p - v_b)], dim=-1
            )
        
        # 處理 metadata 特徵
        if self.meta_path and metadata_features is not None:
            # 通過 meta_path 升維：4 -> 768 -> 768
            meta_feat = self.meta_path(metadata_features)
            # 拼接：基礎特徵(768×5) + metadata特徵(768) = 768×6 = 4608
            feat = torch.cat([feat, meta_feat], dim=-1)
        elif self.meta_path:
            # 如果模型有 meta_path 但沒有提供 metadata，用零填充
            batch_size = feat.shape[0]
            zero_meta = torch.zeros(batch_size, self.encoder.config.dim, device=feat.device)
            feat = torch.cat([feat, zero_meta], dim=-1)
        
        logits = self.classifier(self.dropout(feat))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
    
    @classmethod
    def from_pretrained(cls, model_path, local_files_only=True):
        """從預訓練路徑載入模型 (完全離線，兼容 Kaggle 環境)"""
        print(f"    🔍 從路徑載入模型: {model_path}")
        
        # 檢查必要文件
        safetensors_path = os.path.join(model_path, 'model.safetensors')
        config_path = os.path.join(model_path, 'config.json')
        tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
        
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"未找到模型權重文件: {safetensors_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
        # 載入權重
        try:
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            print(f"    ✅ 使用 SafeTensors 格式載入權重")
        except ImportError:
            raise ImportError("需要安裝 safetensors: pip install safetensors")
        
        # 分析權重結構，確定模型配置
        has_meta_path = any('meta_path' in key for key in state_dict.keys())
        classifier_input_dim = None
        if 'classifier.0.weight' in state_dict:
            classifier_input_dim = state_dict['classifier.0.weight'].shape[1]
        
        print(f"    🔍 分析模型結構:")
        print(f"      - 包含 meta_path: {has_meta_path}")
        print(f"      - 分類器輸入維度: {classifier_input_dim}")
        
        # 讀取配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 檢查tokenizer配置以確定模型類型
        model_type = 'distilbert'  # 默認
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
            tokenizer_class = tokenizer_config.get('tokenizer_class', '')
            if 'Deberta' in tokenizer_class:
                model_type = 'deberta'
                print(f"    🔍 檢測到 DeBERTa 模型")
            elif 'DistilBert' in tokenizer_class:
                model_type = 'distilbert'
                print(f"    🔍 檢測到 DistilBERT 模型")
        
        # 確定模型參數
        if has_meta_path and classifier_input_dim == 4608:
            hidden_size = 768
            metadata_feature_size = 5
            print(f"      - 檢測到帶metadata的dual-tower模型")
        elif classifier_input_dim == 3840:
            # 沒有 metadata：3840 = 768×5
            hidden_size = 768
            metadata_feature_size = 0
            has_meta_path = False
            print(f"      - 檢測到基礎dual-tower模型")
        else:
            # 通用情況
            hidden_size = classifier_input_dim // 6 if has_meta_path else classifier_input_dim // 5
            metadata_feature_size = 5 if has_meta_path else 0
            print(f"      - 通用配置: hidden_size={hidden_size}, meta_path={has_meta_path}")
        
        # 根據模型類型構建配置字典
        if model_type == 'deberta':
            # DeBERTa 配置
            encoder_config_dict = {
                'vocab_size': config.get('vocab_size', 128100),
                'hidden_size': hidden_size,
                'num_hidden_layers': config.get('num_hidden_layers', 12),
                'num_attention_heads': config.get('num_attention_heads', 12),
                'intermediate_size': config.get('intermediate_size', 3072),
                'hidden_act': config.get('hidden_act', 'gelu'),
                'hidden_dropout_prob': config.get('hidden_dropout_prob', 0.1),
                'attention_probs_dropout_prob': config.get('attention_probs_dropout_prob', 0.1),
                'max_position_embeddings': config.get('max_position_embeddings', 512),
                'type_vocab_size': config.get('type_vocab_size', 0),
                'initializer_range': config.get('initializer_range', 0.02),
                'relative_attention': config.get('relative_attention', True),
                'max_relative_positions': config.get('max_relative_positions', -1),
                'pad_token_id': config.get('pad_token_id', 0),
                'position_biased_input': config.get('position_biased_input', False)
            }
            
            # 創建DeBERTa模型
            try:
                from transformers import DebertaV2Config, DebertaV2Model
                encoder_config = DebertaV2Config(**encoder_config_dict)
                encoder = DebertaV2Model(encoder_config)
                print(f"    💡 使用 DeBERTa 配置創建 encoder")
            except ImportError:
                raise ImportError("需要安裝 transformers 支持 DeBERTa")
        else:
            # DistilBERT 配置
            encoder_config_dict = {
                'vocab_size': config.get('vocab_size', 30522),
                'dim': hidden_size,  # DistilBert 使用 'dim' 而不是 'hidden_size'
                'n_layers': config.get('n_layers', 6),
                'n_heads': config.get('n_heads', 12),
                'hidden_dim': config.get('hidden_dim', 3072),
                'dropout': config.get('dropout', 0.1),
                'attention_dropout': config.get('attention_dropout', 0.1),
                'max_position_embeddings': config.get('max_position_embeddings', 512),
                'initializer_range': config.get('initializer_range', 0.02)
            }
            
            # 創建DistilBERT模型
            from transformers import DistilBertConfig, DistilBertModel
            encoder_config = DistilBertConfig(**encoder_config_dict)
            encoder = DistilBertModel(encoder_config)
            print(f"    💡 使用 DistilBERT 配置創建 encoder")
        
        # 創建模型實例
        metadata_fusion = 'dual_path' if has_meta_path else None
        
        # 正確初始化PyTorch模型
        model = object.__new__(cls)  # 創建實例但不調用__init__
        torch.nn.Module.__init__(model)  # 手動調用父類初始化
        
        # 手動設置屬性
        model.encoder = encoder
        model.dropout = torch.nn.Dropout(0.2)
        
        # Metadata 處理路徑
        model.meta_path = None
        if metadata_fusion == 'dual_path' and metadata_feature_size > 0:
            model.meta_path = torch.nn.Sequential(
                torch.nn.Linear(metadata_feature_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU()
            )
            print(f"    💡 創建 metadata 處理路徑: {metadata_feature_size} -> {hidden_size} -> {hidden_size}")
        
        # 分類器輸入維度: 基礎特徵(768×5) + metadata特徵(768) = 768×6 = 4608
        classifier_input_dim = hidden_size * 6 if model.meta_path else hidden_size * 5
        
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 3),
        )
        
        print(f"    💡 分類器輸入維度: {classifier_input_dim} (期望 4608)")
        
        # 智能權重過濾 - 只載入匹配的權重
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"      ⚠️  形狀不匹配，跳過: {key} ({model_state_dict[key].shape} vs {value.shape})")
            else:
                print(f"      💡 模型中無此層，跳過: {key}")
        
        # 載入權重
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"      ⚠️  缺少的權重 (將使用隨機初始化): {len(missing_keys)} 個層")
        if unexpected_keys:
            print(f"      💡 未使用的權重: {len(unexpected_keys)} 個層")
        
        print(f"    ✅ 模型載入完成")
        return model

# --------------------------------------------------------------------------
# 1. Kaggle Environment Configuration
# --------------------------------------------------------------------------
class KaggleConfig:
    """Kaggle 環境配置"""
    def __init__(self):
        # 檢測運行環境
        self.IS_KAGGLE = os.path.exists('/kaggle/input')
        
        if self.IS_KAGGLE:
            # Kaggle 路徑設置 (離線環境)
            self.MODEL_PATH = "/kaggle/input/global-best-model"  # 預訓練模型路徑
            self.TEST_PATH = "/kaggle/input/llm-classification-finetuning/test.csv"  # 測試數據路徑
            self.OUTPUT_PATH = "/kaggle/working/submission.csv"  # 輸出路徑
            print(f"INFO: Kaggle 離線環境配置")
        else:
            # 本地環境路徑設置
            self.MODEL_PATH = "./global_best_model"  # 本地模型路徑
            # 優先使用測試文件，如果不存在則使用訓練數據的一部分
            if os.path.exists("./test.csv"):
                self.TEST_PATH = "./test.csv"
            elif os.path.exists("./test_final_ttr.csv"):
                self.TEST_PATH = "./test_final_ttr.csv"
            else:
                self.TEST_PATH = "./train.csv"  # 使用訓練數據進行測試
                print(f"    💡 測試文件不存在，將使用訓練數據的前100行進行功能驗證")
            self.OUTPUT_PATH = "./submission.csv"  # 本地輸出路徑
            print(f"INFO: 本地環境配置")
        
        # 設備配置
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 批處理設置
        self.BATCH_SIZE = 32
        self.MAX_LENGTH = 512
        
        print(f"  - 模型路徑: {self.MODEL_PATH}")
        print(f"  - 測試數據: {self.TEST_PATH}")
        print(f"  - 輸出路徑: {self.OUTPUT_PATH}")
        print(f"  - 使用設備: {self.DEVICE}")

# --------------------------------------------------------------------------
# 2. Enhanced Metadata Features with Dynamic Loading (與 fine_tuning.py 一致)
# --------------------------------------------------------------------------
class SimpleMetadataFeatures:
    """
    增強的元數據特徵提取，與 fine_tuning.py 中的 MetadataFeatures 核心邏輯一致
    支持從保存的模型中載入統計參數，確保推理時與訓練時完全一致
    """
    
    # 預設統計參數（作為備份）
    DEFAULT_FEATURE_STATS = {
        'jaccard_index': {'mean': 0.119287, 'std': 0.108648},
        'ttr_diff': {'mean': 0.120716, 'std': 0.114099},
        'content_blocks_diff': {'mean': -0.000184, 'std': 0.328517},
        'length_diff': {'mean': -0.004185, 'std': 5.951747},
        'ttr_ratio': {'mean': 1.257098, 'std': 1.679197},
    }
    
    # 將在載入模型時設置
    FEATURE_STATS = None
    
    @classmethod
    def load_metadata_stats(cls, model_path: str):
        """從模型目錄載入metadata統計參數"""
        stats_path = os.path.join(model_path, 'metadata_stats.json')
        
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                # 將訓練時的4特徵映射到推理時的5特徵格式
                cls.FEATURE_STATS = {}
                if 'jaccard_index' in stats:
                    cls.FEATURE_STATS['jaccard_index'] = {
                        'mean': stats['jaccard_index']['mean'],
                        'std': stats['jaccard_index']['std']
                    }
                
                if 'ttr_diff' in stats:
                    # 訓練時和推理時都計算原始差值 response_a - response_b，無需修改
                    cls.FEATURE_STATS['ttr_diff'] = {
                        'mean': stats['ttr_diff']['mean'],  # 直接使用原始均值
                        'std': stats['ttr_diff']['std']
                    }
                
                if 'content_blocks_diff' in stats:
                    cls.FEATURE_STATS['content_blocks_diff'] = {
                        'mean': stats['content_blocks_diff']['mean'],
                        'std': stats['content_blocks_diff']['std']
                    }
                
                if 'length_diff' in stats:
                    cls.FEATURE_STATS['length_diff'] = {
                        'mean': stats['length_diff']['mean'],
                        'std': stats['length_diff']['std']
                    }
                
                # 添加 ttr_ratio（如果訓練時沒有，使用預設值）
                if 'ttr_ratio' in stats:
                    cls.FEATURE_STATS['ttr_ratio'] = {
                        'mean': stats['ttr_ratio']['mean'],
                        'std': stats['ttr_ratio']['std']
                    }
                else:
                    cls.FEATURE_STATS['ttr_ratio'] = cls.DEFAULT_FEATURE_STATS['ttr_ratio']
                
                print(f"    ✅ 載入metadata統計參數: {stats_path}")
                print(f"    📊 載入的特徵: {list(cls.FEATURE_STATS.keys())}")
                return True
                
            except Exception as e:
                print(f"    ⚠️  載入metadata統計參數失敗: {e}")
        
        # 如果載入失敗，使用預設參數
        print(f"    💡 使用預設metadata統計參數")
        cls.FEATURE_STATS = cls.DEFAULT_FEATURE_STATS.copy()
        return False
    
    @staticmethod
    def remove_special_content(text: str, remove_special_blocks: bool = None) -> str:
        """
        移除特殊內容區塊 - 與fine_tuning.py邏輯完全一致
        如果 remove_special_blocks 為 True，則移除程式碼、數學公式和表格區塊。
        默認（None）不移除特殊內容。
        """
        if not remove_special_blocks or not isinstance(text, str):
            return text

        # 移除程式碼區塊
        text = re.sub(r'```.*?```', '[CODE_BLOCK]', text, flags=re.DOTALL)
        # 移除數學公式區塊
        text = re.sub(r'\$\$.*?\$\$', '[MATH_BLOCK]', text, flags=re.DOTALL)
        # 移除 Markdown 表格
        text = re.sub(r'((?:\|.*\|[\r\n\s]*){2,})', '[TABLE_BLOCK]', text)
        return text

    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """計算 Jaccard 相似度 - 與fine_tuning.py邏輯一致"""
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def count_code_blocks(text: str) -> int:
        """計算markdown格式的程式碼區塊數量 - 與fine_tuning.py邏輯一致"""
        if not isinstance(text, str):
            return 0
        return str(text).count('```') // 2

    @staticmethod
    def count_math_blocks(text: str) -> int:
        """計算markdown格式的數學公式區塊數量 - 與fine_tuning.py邏輯一致"""
        if not isinstance(text, str):
            return 0
        math_pattern = re.compile(r'\\$\\$.*?\\$\\$', re.DOTALL)
        return len(math_pattern.findall(text))

    @staticmethod
    def count_table_blocks(text: str) -> int:
        """計算markdown格式的表格數量 - 與fine_tuning.py邏輯一致"""
        if not isinstance(text, str):
            return 0
        table_pattern = re.compile(r'((?:\\|.*?\\|[\\r\\n\\s]*){2,})')
        return len(table_pattern.findall(text))
    
    @staticmethod
    def calculate_content_blocks_diff(text1_raw: str, text2_raw: str) -> int:
        """
        計算markdown格式的程式碼、數學公式和表格區塊的總數量差值 - 與fine_tuning.py邏輯一致
        
        Args:
            text1_raw (str): 原始第一個文本 (response_a)
            text2_raw (str): 原始第二個文本 (response_b)
            
        Returns:
            int: 內容區塊總數量差值 (response_a - response_b)
        """
        count1_code = SimpleMetadataFeatures.count_code_blocks(text1_raw)
        count2_code = SimpleMetadataFeatures.count_code_blocks(text2_raw)
        code_diff = count1_code - count2_code

        count1_math = SimpleMetadataFeatures.count_math_blocks(text1_raw)
        count2_math = SimpleMetadataFeatures.count_math_blocks(text2_raw)
        math_diff = count1_math - count2_math

        count1_table = SimpleMetadataFeatures.count_table_blocks(text1_raw)
        count2_table = SimpleMetadataFeatures.count_table_blocks(text2_raw)
        table_diff = count1_table - count2_table
        
        return code_diff + math_diff + table_diff

    @staticmethod
    def count_special_blocks(text: str) -> int:
        """計算特殊區塊數量 - 與fine_tuning.py邏輯一致（已廢棄，改用calculate_content_blocks_diff）"""
        if not isinstance(text, str):
            return 0
        code_blocks = text.count('```') // 2
        math_blocks = len(re.findall(r'\$\$.*?\$\$', text, re.DOTALL))
        table_blocks = len(re.findall(r'((?:\|.*?\|[\r\n\s]*){2,})', text))
        return code_blocks + math_blocks + table_blocks
    
    @staticmethod
    def calculate_ttr(text: str) -> float:
        """計算詞彙豐富度 (TTR) - 與fine_tuning.py邏輯一致"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)
    
    @staticmethod
    def standardize_features(features_dict: Dict[str, float]) -> Dict[str, float]:
        """使用載入的統計參數進行z-score標準化"""
        if SimpleMetadataFeatures.FEATURE_STATS is None:
            raise ValueError("尚未載入metadata統計參數，請先調用load_metadata_stats()")
        
        standardized = {}
        for feature_name, value in features_dict.items():
            if feature_name in SimpleMetadataFeatures.FEATURE_STATS:
                mean = SimpleMetadataFeatures.FEATURE_STATS[feature_name]['mean']
                std = SimpleMetadataFeatures.FEATURE_STATS[feature_name]['std']
                standardized[feature_name] = (value - mean) / std if std > 0 else 0.0
            else:
                standardized[feature_name] = value
        return standardized
    
    @staticmethod
    def extract_features(row: pd.Series) -> Dict[str, float]:
        """
        提取核心5維元數據特徵 - 完全匹配fine_tuning.py的特徵處理邏輯
        
        關鍵要點：
        1. jaccard_index: response_a vs response_b（使用處理過的文本）
        2. ttr_diff: response_a - response_b（原始差值，不取絕對值）
        3. ttr_ratio: max(ttr_a, ttr_b) / max(min(ttr_a, ttr_b), 0.001) （max/min形式，與訓練完全一致）
        4. 其他特徵保持一致的log縮放和處理邏輯
        """
        response_a_raw = str(row['response_a'])
        response_b_raw = str(row['response_b'])
        
        # 處理過的文本（用於 jaccard, ttr, ttr_ratio）
        response_a_for_jaccard_ttr = SimpleMetadataFeatures.remove_special_content(response_a_raw)
        response_b_for_jaccard_ttr = SimpleMetadataFeatures.remove_special_content(response_b_raw)
        
        # TTR計算
        ttr_a = SimpleMetadataFeatures.calculate_ttr(response_a_for_jaccard_ttr)
        ttr_b = SimpleMetadataFeatures.calculate_ttr(response_b_for_jaccard_ttr)
        ttr_diff = ttr_a - ttr_b  # 原始差值（不取絕對值）
        
        # TTR比值計算 - 與訓練時完全一致：max/min 形式
        ttr_ratio = max(ttr_a, ttr_b) / max(min(ttr_a, ttr_b), 0.001) if min(ttr_a, ttr_b) > 0 else 1.0
        
        # 原始差異值（在原始文本上計算）
        length_diff = len(response_a_raw) - len(response_b_raw)
        
        # 特殊區塊差異 - 使用與訓練時完全一致的邏輯
        content_blocks_diff_raw = SimpleMetadataFeatures.calculate_content_blocks_diff(response_a_raw, response_b_raw)
        
        # 對數縮放（保留符號）
        scaled_length_diff = np.sign(length_diff) * np.log1p(abs(length_diff))
        scaled_content_blocks_diff = np.sign(content_blocks_diff_raw) * np.log1p(abs(content_blocks_diff_raw))
        
        # 收集原始特徵值（與fine_tuning.py完全一致）
        features_dict = {
            'jaccard_index': SimpleMetadataFeatures.jaccard_similarity(
                response_a_for_jaccard_ttr, response_b_for_jaccard_ttr),
            'ttr_diff': ttr_diff,
            'ttr_ratio': ttr_ratio,
            'content_blocks_diff': scaled_content_blocks_diff,
            'length_diff': scaled_length_diff
        }
        
        # 直接返回原始特徵值（與訓練時一致，不應用標準化）
        return features_dict

# --------------------------------------------------------------------------
# 3. Test Dataset Class
# --------------------------------------------------------------------------
class KaggleTestDataset(Dataset):
    """
    Kaggle 測試數據集 - dual-tower 模型專用
    完全匹配 fine_tuning.py 中的 DualTowerPairDataset 輸入格式
    """
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, config: KaggleConfig):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.config = config
        
        print(f"  - 創建測試數據集，樣本數: {len(self.df)}")
        print(f"  - 模型架構: dual-tower")
        
        # 提取元數據特徵
        print("  - 提取元數據特徵...")
        tqdm.pandas(desc="Extracting features")
        # 使用與訓練完全相同的提取邏輯 (優先使用外部模組，否則退回內建 SimpleMetadataFeatures)
        if MetadataFeatures is not None:
            self.features = self.df.progress_apply(MetadataFeatures.extract_core_features, axis=1)
        else:
            self.features = self.df.progress_apply(SimpleMetadataFeatures.extract_features, axis=1)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        metadata = self.features.iloc[idx]
        
        # 先做 z-score 標準化（依照 SimpleMetadataFeatures.FEATURE_STATS）
        metadata_std = SimpleMetadataFeatures.standardize_features(metadata)
        
        # 生成與訓練相同順序的特徵向量
        if MetadataFeatures is not None:
            feature_order = MetadataFeatures.get_feature_columns('core')
            metadata_values = [float(metadata_std[col]) for col in feature_order]
        else:
            # Fallback: 使用固定順序
            metadata_values = [
                float(metadata_std['jaccard_index']),
                float(metadata_std['ttr_diff']),
                float(metadata_std['ttr_ratio']),
                float(metadata_std['content_blocks_diff']),
                float(metadata_std['length_diff'])
            ]
        metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32)
        
        # dual-tower 模型：使用分離輸入（與 fine_tuning.py 訓練時一致）
        prompt = str(row['prompt'])
        response_a = str(row['response_a'])
        response_b = str(row['response_b'])
        
        # 與訓練時一致：在 Tokenize 前移除特殊區塊
        prompt_clean = SimpleMetadataFeatures.remove_special_content(prompt, remove_special_blocks=True)
        response_a_clean = SimpleMetadataFeatures.remove_special_content(response_a, remove_special_blocks=True)
        response_b_clean = SimpleMetadataFeatures.remove_special_content(response_b, remove_special_blocks=True)

        p_encoded = self.tokenizer(
            prompt_clean,
            add_special_tokens=True,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        a_encoded = self.tokenizer(
            response_a_clean,
            add_special_tokens=True,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        b_encoded = self.tokenizer(
            response_b_clean,
            add_special_tokens=True,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'p_input_ids': p_encoded['input_ids'].squeeze(),
            'p_attention_mask': p_encoded['attention_mask'].squeeze(),
            'a_input_ids': a_encoded['input_ids'].squeeze(),
            'a_attention_mask': a_encoded['attention_mask'].squeeze(),
            'b_input_ids': b_encoded['input_ids'].squeeze(),
            'b_attention_mask': b_encoded['attention_mask'].squeeze(),
            'metadata_features': metadata_tensor
        }

# --------------------------------------------------------------------------
# 4. Model Inference Class
# --------------------------------------------------------------------------
class KaggleInference:
    """Kaggle 推理類"""
    
    def __init__(self, config: KaggleConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """載入預訓練模型"""
        print(f"\n載入預訓練模型...")
        
        try:
            # 檢查模型路徑
            if not os.path.exists(self.config.MODEL_PATH):
                raise FileNotFoundError(f"找不到模型路徑: {self.config.MODEL_PATH}")
            
            # 載入metadata統計參數（必須在創建dataset之前）
            print("  - 載入metadata統計參數...")
            SimpleMetadataFeatures.load_metadata_stats(self.config.MODEL_PATH)
            
            # 載入基本模型信息
            metrics_path = os.path.join(self.config.MODEL_PATH, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                print(f"  - 模型架構: {metrics.get('model_arch', 'dual')}")
                print(f"  - 訓練時間: {metrics.get('timestamp', 'Unknown')}")
                print(f"  - 驗證性能:")
                print(f"    * Log Loss: {metrics.get('log_loss', 'N/A'):.6f}")
                print(f"    * Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                
                # 顯示主要超參數
                hyperparams = metrics.get('hyperparams', {})
                if hyperparams:
                    print(f"  - 訓練配置:")
                    print(f"    * Epochs: {hyperparams.get('epochs', 'N/A')}")
                    print(f"    * Learning Rate: {hyperparams.get('learning_rate', 'N/A')}")
                    print(f"    * Batch Size: {hyperparams.get('batch_size', 'N/A')}")
                
                # 顯示preprocessing配置
                preprocessing_config = metrics.get('preprocessing_config', {})
                if preprocessing_config:
                    print(f"  - 預處理配置:")
                    print(f"    * Extract Metadata: {preprocessing_config.get('extract_metadata', 'N/A')}")
                    print(f"    * Metadata Type: {preprocessing_config.get('metadata_type', 'N/A')}")
            
            # 載入 tokenizer
            print("  - 載入 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_PATH, 
                local_files_only=True
            )
            
            # 載入 dual-tower 模型
            print(f"  - 載入 dual-tower 架構模型...")
            self.model = DualTowerPairClassifier.from_pretrained(
                self.config.MODEL_PATH,
                local_files_only=True
            )
            
            # 移動到設備
            self.model.to(self.config.DEVICE)
            self.model.eval()
            
            print(f"✅ 模型載入成功，使用設備: {self.config.DEVICE}")
            print(f"  - 重要：dual-tower 模型使用分離輸入格式（與訓練時一致）")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """對測試數據進行預測"""
        print(f"\n開始預測...")
        
        # 創建數據集
        test_dataset = KaggleTestDataset(test_df, self.tokenizer, self.config)
        
        # 創建數據加載器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0  # Kaggle 環境建議設為 0
        )
        
        # 預測
        all_predictions = []
        
        print(f"  - 開始批次預測，批次大小: {self.config.BATCH_SIZE}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
                # dual-tower 模型：使用分離輸入
                p_input_ids = batch['p_input_ids'].to(self.config.DEVICE)
                p_attention_mask = batch['p_attention_mask'].to(self.config.DEVICE)
                a_input_ids = batch['a_input_ids'].to(self.config.DEVICE)
                a_attention_mask = batch['a_attention_mask'].to(self.config.DEVICE)
                b_input_ids = batch['b_input_ids'].to(self.config.DEVICE)
                b_attention_mask = batch['b_attention_mask'].to(self.config.DEVICE)
                metadata_features = batch['metadata_features'].to(self.config.DEVICE)
                
                outputs = self.model(
                    p_input_ids=p_input_ids,
                    p_attention_mask=p_attention_mask,
                    a_input_ids=a_input_ids,
                    a_attention_mask=a_attention_mask,
                    b_input_ids=b_input_ids,
                    b_attention_mask=b_attention_mask,
                    metadata_features=metadata_features
                )
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                
                # 計算概率
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                all_predictions.append(probabilities.cpu().numpy())
                
                # 清理記憶體
                if (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 合併所有預測結果
        predictions = np.vstack(all_predictions)
        
        # 創建提交文件
        submission = pd.DataFrame({
            'id': test_df['id'].values,
            'winner_model_a': predictions[:, 0],
            'winner_model_b': predictions[:, 1],
            'winner_tie': predictions[:, 2]
        })
        
        # 確保概率總和為 1
        row_sums = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
        submission['winner_model_a'] /= row_sums
        submission['winner_model_b'] /= row_sums
        submission['winner_tie'] /= row_sums
        
        print(f"✅ 預測完成，共處理 {len(submission)} 個樣本")
        
        # 顯示預測分布
        final_preds = np.argmax(predictions, axis=1)
        print(f"\n預測分布:")
        print(f"  - Model A 勝利: {np.sum(final_preds == 0)} ({np.sum(final_preds == 0)/len(final_preds)*100:.1f}%)")
        print(f"  - Model B 勝利: {np.sum(final_preds == 1)} ({np.sum(final_preds == 1)/len(final_preds)*100:.1f}%)")
        print(f"  - 平局: {np.sum(final_preds == 2)} ({np.sum(final_preds == 2)/len(final_preds)*100:.1f}%)")
        
        return submission

# --------------------------------------------------------------------------
# 5. Main Execution
# --------------------------------------------------------------------------
def main():
    """主執行函數"""
    print("=" * 60)
    print("Kaggle 離線推理腳本 - kaggle_last.py")
    print("完全兼容 fine_tuning.py 中的 dual-tower 模型")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = KaggleConfig()
        
        # 2. 載入測試數據
        print(f"\n載入測試數據...")
        test_df = pd.read_csv(config.TEST_PATH)
        
        # 如果使用訓練數據進行測試，只取前100行並創建id欄位
        if config.TEST_PATH.endswith('train.csv'):
            print(f"  - 使用訓練數據進行功能驗證，只處理前100行")
            test_df = test_df.head(100).copy()
            # 創建id欄位（如果不存在）
            if 'id' not in test_df.columns:
                test_df['id'] = range(len(test_df))
            print(f"  - 創建了 {len(test_df)} 個測試樣本")
        
        print(f"  - 測試數據形狀: {test_df.shape}")
        print(f"  - 欄位: {list(test_df.columns)}")
        
        # 檢查必要欄位
        required_columns = ['id', 'prompt', 'response_a', 'response_b']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要欄位: {missing_columns}")
        
        # 3. 初始化推理器
        inference = KaggleInference(config)
        
        # 4. 進行預測
        submission = inference.predict(test_df)
        
        # 5. 保存結果
        print(f"\n保存提交文件...")
        submission.to_csv(config.OUTPUT_PATH, index=False)
        print(f"✅ 提交文件已保存: {config.OUTPUT_PATH}")
        
        # 6. 顯示提交文件預覽
        print(f"\n提交文件預覽:")
        print(submission.head(10))
        
        print("\n" + "=" * 60)
        print("🎉 Kaggle 離線推理完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 