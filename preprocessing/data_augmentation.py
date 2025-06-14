# Data Augmentation Module
# ========================
# 實現通過響應順序交換和相應標籤調整進行數據增強的策略
# 這種技術使訓練數據翻倍，同時確保模型學習位置不變表示

import pandas as pd
import numpy as np

class DataAugmentation:
    """
    數據增強類，專門處理響應順序交換以減少位置偏差
    """
    
    @staticmethod
    def swap_responses_and_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        通過交換 response_a 和 response_b 的順序來增強數據
        
        Args:
            df (pd.DataFrame): 原始數據集，包含 prompt, response_a, response_b, winner_model_a, winner_model_b
            
        Returns:
            pd.DataFrame: 增強後的數據集（原始數據 + 交換後的數據）
        """
        print("  - 開始執行響應順序交換數據增強...")
        
        # 複製原始數據框
        df_original = df.copy()
        df_swapped = df.copy()
        
        # 交換 response_a 和 response_b
        df_swapped['response_a'] = df['response_b'].copy()
        df_swapped['response_b'] = df['response_a'].copy()
        
        # 相應地交換標籤
        df_swapped['winner_model_a'] = df['winner_model_b'].copy()
        df_swapped['winner_model_b'] = df['winner_model_a'].copy()
        
        # 如果存在其他相關欄位也需要交換（如模型名稱等）
        if 'model_a' in df.columns and 'model_b' in df.columns:
            df_swapped['model_a'] = df['model_b'].copy()
            df_swapped['model_b'] = df['model_a'].copy()
        
        # 合併原始數據和交換後的數據
        df_augmented = pd.concat([df_original, df_swapped], ignore_index=True)
        
        print(f"  - 數據增強完成：原始數據 {len(df)} 筆 → 增強後數據 {len(df_augmented)} 筆")
        print(f"  - 增強倍數：{len(df_augmented) / len(df):.1f}x")
        
        return df_augmented
    
    @staticmethod
    def apply_augmentation_with_validation(df: pd.DataFrame, apply_augmentation: bool = True) -> pd.DataFrame:
        """
        應用數據增強，並進行驗證檢查
        
        Args:
            df (pd.DataFrame): 原始數據集
            apply_augmentation (bool): 是否應用數據增強
            
        Returns:
            pd.DataFrame: 處理後的數據集
        """
        if not apply_augmentation:
            print("  - 跳過數據增強")
            return df
        
        # 檢查必要的欄位是否存在
        required_columns = ['prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  - 警告：缺少必要欄位 {missing_columns}，跳過數據增強")
            return df
        
        # 驗證標籤一致性
        label_check = df.apply(lambda row: (row['winner_model_a'] + row['winner_model_b']) <= 1, axis=1)
        if not label_check.all():
            print(f"  - 警告：發現 {(~label_check).sum()} 筆標籤不一致的數據")
        
        # 執行數據增強
        df_augmented = DataAugmentation.swap_responses_and_labels(df)
        
        # 驗證增強後的數據
        original_label_dist = df.apply(DataAugmentation._get_label, axis=1).value_counts().sort_index()
        augmented_label_dist = df_augmented.apply(DataAugmentation._get_label, axis=1).value_counts().sort_index()
        
        print("  - 標籤分布比較：")
        print(f"    原始數據: {dict(original_label_dist)}")
        print(f"    增強數據: {dict(augmented_label_dist)}")
        
        return df_augmented
    
    @staticmethod
    def _get_label(row):
        """輔助函數：從 winner_model_a 和 winner_model_b 計算標籤"""
        if row["winner_model_a"] == 1: 
            return 0  # A wins
        if row["winner_model_b"] == 1: 
            return 1  # B wins
        return 2  # Tie
    
    @staticmethod
    def create_balanced_augmentation(df: pd.DataFrame, target_balance_ratio: float = 0.3) -> pd.DataFrame:
        """
        創建平衡的數據增強，特別增加少數類別的樣本
        
        Args:
            df (pd.DataFrame): 原始數據集
            target_balance_ratio (float): 目標平衡比例（最少類別佔總數的比例）
            
        Returns:
            pd.DataFrame: 平衡增強後的數據集
        """
        print("  - 開始執行平衡數據增強...")
        
        # 計算當前標籤分布
        df['temp_label'] = df.apply(DataAugmentation._get_label, axis=1)
        label_counts = df['temp_label'].value_counts().sort_index()
        total_samples = len(df)
        
        print(f"  - 原始標籤分布: {dict(label_counts)}")
        
        # 找出需要增強的類別
        min_count = label_counts.min()
        target_min_count = int(total_samples * target_balance_ratio)
        
        if min_count >= target_min_count:
            print("  - 數據已經足夠平衡，執行標準增強")
            df = df.drop('temp_label', axis=1)
            return DataAugmentation.swap_responses_and_labels(df)
        
        # 對少數類別進行額外採樣
        df_balanced = df.copy()
        
        for label_value in label_counts.index:
            current_count = label_counts[label_value]
            if current_count < target_min_count:
                # 計算需要增加的樣本數
                needed_samples = target_min_count - current_count
                label_samples = df[df['temp_label'] == label_value]
                
                # 重複採樣
                additional_samples = label_samples.sample(
                    n=min(needed_samples, len(label_samples)), 
                    replace=True, 
                    random_state=42
                ).reset_index(drop=True)
                
                df_balanced = pd.concat([df_balanced, additional_samples], ignore_index=True)
                print(f"  - 為標籤 {label_value} 增加了 {len(additional_samples)} 個樣本")
        
        df_balanced = df_balanced.drop('temp_label', axis=1)
        
        # 最後應用響應交換增強
        df_final = DataAugmentation.swap_responses_and_labels(df_balanced)
        
        final_label_dist = df_final.apply(DataAugmentation._get_label, axis=1).value_counts().sort_index()
        print(f"  - 最終標籤分布: {dict(final_label_dist)}")
        
        return df_final
