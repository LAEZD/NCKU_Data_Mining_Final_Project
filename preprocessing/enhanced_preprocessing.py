# Enhanced Preprocessing Module
# ============================
# 整合數據增強、動態預算分配和元數據特徵提取的統一預處理模組
# 取代原有的簡單預處理方式，提供更強大和靈活的數據處理能力

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Dict, Any, Tuple, List, Optional

# 導入我們創建的專用模組
from .data_augmentation import DataAugmentation
from .metadata_features import MetadataFeatures
from .unified_input_builder import UnifiedInputBuilder

class EnhancedPreprocessing:
    """
    增強型預處理類，整合所有數據處理改進
    """
    
    @staticmethod
    def load_and_enhance_data(
        config,
        apply_augmentation: bool = True,
        extract_metadata: bool = True,
        metadata_type: str = 'core'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        載入並增強訓練和測試數據
        
        Args:
            config: 配置對象
            apply_augmentation (bool): 是否應用數據增強
            extract_metadata (bool): 是否提取元數據特徵
            metadata_type (str): 元數據特徵類型 ('core' 或 'all')
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: 增強後的訓練數據和測試數據
        """
        print("\n[Enhanced Module 1/5] Loading and enhancing data...")
        
        
        # 載入訓練數據
        try:
            df = pd.read_csv(config.TRAIN_PATH)
            print(f"  - Training data loaded. Shape: {df.shape}")
        except FileNotFoundError:
            print(f"ERROR: Training file not found at {config.TRAIN_PATH}")
            raise
        
        # Quick test mode
        if config.QUICK_TEST and len(df) > config.QUICK_TEST_SIZE:
            df = df.sample(n=config.QUICK_TEST_SIZE, random_state=config.RANDOM_STATE).reset_index(drop=True)
            print(f"  - Quick test mode enabled. Sampled to {df.shape}")
          # 創建標籤（在分割前先創建，但不做數據增強）
        def get_label(row):
            if row["winner_model_a"] == 1: return 0
            if row["winner_model_b"] == 1: return 1
            return 2
        df["label"] = df.apply(get_label, axis=1)
        
        # 注意：數據增強將在 train/val split 之後進行，只對訓練集增強
        
        # 提取元數據特徵
        if extract_metadata:
            df = MetadataFeatures.add_metadata_features_to_dataframe(df, feature_type=metadata_type)
            
            # 分析特徵分布
            # Ensure this list matches the actual features produced by MetadataFeatures
            # Based on the update, 'code_blocks_diff' is now 'content_blocks_diff'
            core_features_to_analyze = ['jaccard_index', 'content_blocks_diff', 'length_diff', 'ttr_diff']
            existing_features = [f for f in core_features_to_analyze if f in df.columns]
            if existing_features:
                MetadataFeatures.analyze_feature_distributions(df, existing_features)
        
        print("  - Labels and enhanced features have been processed.")
        
        # 載入測試數據
        df_test = None
        if os.path.exists(config.TEST_PATH):
            file_size_mb = os.path.getsize(config.TEST_PATH) / (1024**2)
            if file_size_mb > 100:
                print(f"  - Large test file ({file_size_mb:.1f} MB) detected. Will process in batches.")
            else:
                df_test = pd.read_csv(config.TEST_PATH)
                
                # 為測試數據也提取元數據特徵
                if extract_metadata:
                    df_test = MetadataFeatures.add_metadata_features_to_dataframe(df_test, feature_type=metadata_type)
                
                print(f"  - Test data loaded and enhanced. Shape: {df_test.shape}")
        else:
            print("  - WARNING: Test file not found. Inference will be skipped.")
        return df, df_test
    
    @staticmethod
    def analyze_data_characteristics(df: pd.DataFrame, tokenizer):
        """
        分析數據特徵以優化處理策略
        
        Args:
            df (pd.DataFrame): 數據框
            tokenizer: 分詞器
        """
        print("\n[Enhanced Analysis] Analyzing data characteristics...")
        
        # 使用 Unified Input Builder 進行分析
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        token_stats = {
            'prompt_lengths': [],
            'response_a_lengths': [],
            'response_b_lengths': [],
            'total_lengths': []
        }
        
        for _, row in sample_df.iterrows():
            # 分析 token 使用情況
            prompt_tokens = len(tokenizer.encode(str(row['prompt']), add_special_tokens=False))
            response_a_tokens = len(tokenizer.encode(str(row['response_a']), add_special_tokens=False))
            response_b_tokens = len(tokenizer.encode(str(row['response_b']), add_special_tokens=False))
            
            token_stats['prompt_lengths'].append(prompt_tokens)
            token_stats['response_a_lengths'].append(response_a_tokens)
            token_stats['response_b_lengths'].append(response_b_tokens)
            token_stats['total_lengths'].append(prompt_tokens + response_a_tokens + response_b_tokens)
        
        # 計算統計信息
        for key in token_stats:
            lengths = token_stats[key]
            token_stats[key] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'max': np.max(lengths),
                'min': np.min(lengths)
            }
        
        print(f"  - Token 使用分析完成，樣本數: {sample_size}")
        print(f"  - 平均 prompt 長度: {token_stats['prompt_lengths']['mean']:.1f} tokens")
        print(f"  - 平均 response A 長度: {token_stats['response_a_lengths']['mean']:.1f} tokens")
        print(f"  - 平均 response B 長度: {token_stats['response_b_lengths']['mean']:.1f} tokens")
        print(f"  - 平均總長度: {token_stats['total_lengths']['mean']:.1f} tokens")
        
        # 返回優化建議
        optimized_allocation = {
            'recommended_max_len': min(512, int(token_stats['total_lengths']['mean'] * 1.5)),
            'metadata_budget_ratio': 0.15,  # 元數據佔15%
            'prompt_budget_ratio': 0.35,    # prompt 佔35%
            'response_budget_ratio': 0.25    # 每個 response 佔25%
        }
        
        return token_stats, optimized_allocation

class EnhancedLLMDataset(Dataset):
    """
    增強型 LLM 數據集類，使用統一的輸入構建策略
    將元數據作為特權文本注入輸入序列，確保模型能夠利用元數據信息
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer,
        include_metadata: bool = True,
        metadata_type: str = 'core'
        # REMOVED: FastLexRank params are no longer passed here
    ):
        """
        初始化增強型數據集
        
        Args:
            dataframe (pd.DataFrame): 數據框
            tokenizer: 分詞器
            include_metadata (bool): 是否包含元數據特徵
            metadata_type (str): 元數據類型 ('core' 或 'all')
        """
        self.df = dataframe
        self.tokenizer = tokenizer
        self.include_metadata = include_metadata
        self.metadata_type = metadata_type
        
        # 檢查可用的元數據特徵
        # CORRECTED: Use the actual feature names produced by MetadataFeatures
        if metadata_type == 'core':
            self.expected_features = ['jaccard_index', 'content_blocks_diff', 'length_diff', 'ttr_diff'] # Updated
        else: # Assuming 'all' currently means the same as 'core' based on metadata_features.py
            self.expected_features = ['jaccard_index', 'content_blocks_diff', 'length_diff', 'ttr_diff'] # Updated
        
        self.available_metadata_features = [
            f for f in self.expected_features if f in self.df.columns
        ]
        
        if self.include_metadata and self.available_metadata_features:
            print(f"  - Dataset 將使用統一輸入策略，包含元數據特徵: {self.available_metadata_features}")
        else:
            print("  - Dataset 將使用統一輸入策略，不包含元數據特徵")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 提取元數據（如果啟用）
        metadata_dict = None
        if self.include_metadata and self.available_metadata_features:
            metadata_dict = UnifiedInputBuilder.extract_metadata_from_row(row, self.metadata_type)
        
        # 使用統一輸入構建器創建輸入
        # MODIFIED: Removed FastLexRank specific args from call
        encoded = UnifiedInputBuilder.create_unified_input(
            tokenizer=self.tokenizer,
            prompt=str(row['prompt']),
            response_a=str(row['response_a']),
            response_b=str(row['response_b']),
            metadata_dict=metadata_dict
            # Assuming max_len and include_prompt are handled by other logic or defaults
        )
        
        # 添加標籤（如果存在）
        if 'label' in row:
            encoded['labels'] = torch.tensor(row['label'], dtype=torch.long)
        
        return encoded

class EnhancedPipelineModules:
    """
    增強型管道模組，整合所有改進
    """
    
    @staticmethod
    def create_dynamic_budgeted_input(tokenizer, prompt: str, response_a: str, response_b: str, max_len: int = 512):
        """
        使用統一輸入構建策略創建輸入
        這個方法現在使用 UnifiedInputBuilder 實現
        """
        # MODIFIED: Removed FastLexRank specific args from call
        return UnifiedInputBuilder.create_unified_input(
            tokenizer=tokenizer,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            metadata_dict=None,  # 此方法不包含元數據
            max_len=max_len
        )
    @staticmethod
    def load_and_preprocess_data(config):
        """
        載入和預處理數據的修正版本 - 不在此階段進行數據增強
        數據增強將在 train/val split 之後針對訓練集進行
        """
        return EnhancedPreprocessing.load_and_enhance_data(
            config,
            apply_augmentation=False,  # 關鍵修正：在此階段不進行數據增強
            extract_metadata=True,
            metadata_type='core'
        )
    @staticmethod
    def create_enhanced_datasets(df, tokenizer, config): # config is still passed for other settings
        """
        創建增強型數據集 - 修正版：先分割，再對訓練集進行數據增強
        """
        print("\n[Enhanced Module 3/5] Creating enhanced datasets...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        
        # 分析數據特徵
        token_stats, optimized_allocation = EnhancedPreprocessing.analyze_data_characteristics(df, tokenizer)
        
        # ===== 步驟1: 固定驗證集每個類別的數量 =====
        print("  - Step 1: Creating validation set with fixed number per class...")
        
        # 每個類別固定使用的驗證樣本數
        FIXED_VAL_SAMPLES_PER_CLASS = 2000  # 每個類別固定200個樣本
        
        # 獲取類別分布
        class_counts = df['label'].value_counts().sort_index()
        print(f"  - Original class distribution: {dict(class_counts)}")
        
        val_indices = []
        train_indices = []
        
        # 為每個類別分別選擇固定數量的驗證樣本
        for label in sorted(df['label'].unique()):
            class_indices = df[df['label'] == label].index.tolist()
            
            if len(class_indices) < FIXED_VAL_SAMPLES_PER_CLASS:
                print(f"  - Warning: Class {label} has only {len(class_indices)} samples, using all for validation")
                val_samples_for_class = len(class_indices)
            else:
                val_samples_for_class = FIXED_VAL_SAMPLES_PER_CLASS
            
            # 隨機選擇固定數量的驗證樣本
            np.random.seed(config.RANDOM_STATE + label)  # 確保可重現性
            selected_val_indices = np.random.choice(
                class_indices, 
                size=val_samples_for_class, 
                replace=False
            ).tolist()
            
            val_indices.extend(selected_val_indices)
            
            # 剩餘的作為訓練集
            remaining_indices = [idx for idx in class_indices if idx not in selected_val_indices]
            train_indices.extend(remaining_indices)
            
            print(f"  - Class {label}: {val_samples_for_class} validation, {len(remaining_indices)} training")
        
        # 創建訓練和驗證數據框
        train_df_original = df.loc[train_indices].reset_index(drop=True)
        val_df = df.loc[val_indices].reset_index(drop=True)
        
        # 提取標籤
        train_labels = train_df_original["label"].tolist()
        val_labels = val_df["label"].tolist()
        
        print(f"  - Final validation set class distribution: {dict(pd.Series(val_labels).value_counts().sort_index())}")
        print(f"  - Total training samples: {len(train_df_original)}")
        print(f"  - Total validation samples: {len(val_df)} (fixed per class)")
        
        # ===== 步驟2: 只對訓練集進行數據增強 =====
        if config.APPLY_AUGMENTATION:
            print("  - Step 2: Applying data augmentation ONLY to training set...")
            train_df_final = DataAugmentation.apply_augmentation_with_validation(
                train_df_original, apply_augmentation=True
            )
            print(f"  - Final training samples after augmentation: {len(train_df_final)}")
        else:
            print("  - Step 2: Skipping data augmentation (APPLY_AUGMENTATION = False)")
            train_df_final = train_df_original
        
        # ===== 步驟3: 重新計算訓練集標籤（因為數據增強可能改變了數量） =====
        final_train_labels = train_df_final["label"].tolist()
        
        print("\n  - Data leakage prevention check:")
        print(f"    * Training set size: {len(train_df_final)}")
        print(f"    * Validation set size: {len(val_df)}")
        print(f"    * No overlap between train and val: ✓ Guaranteed by proper splitting order")
        
        # 創建增強型數據集
        train_dataset = EnhancedLLMDataset(
            train_df_final, 
            tokenizer, 
            include_metadata=config.EXTRACT_METADATA, # This still comes from config
            metadata_type=config.METADATA_TYPE      # This still comes from config
            # FastLexRank params are no longer passed
        )
        val_dataset = EnhancedLLMDataset(
            val_df, 
            tokenizer, 
            include_metadata=config.EXTRACT_METADATA,   # This still comes from config
            metadata_type=config.METADATA_TYPE        # This still comes from config
            # FastLexRank params are no longer passed
        )
        
        # 計算類別權重（基於最終的訓練集）
        class_weights = compute_class_weight('balanced', classes=np.unique(final_train_labels), y=final_train_labels)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"  - Class weights calculated: {class_weights_dict}")
        
        return train_dataset, val_dataset, val_labels, val_indices, class_weights_dict

class EnhancedTestDataset(Dataset):
    """
    增強型測試數據集類，使用統一輸入構建策略
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer, 
        include_metadata: bool = True,
        metadata_type: str = 'core'
        # REMOVED: FastLexRank params are no longer passed here
    ):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.include_metadata = include_metadata
        self.metadata_type = metadata_type
        
        # 檢查可用的元數據特徵
        # CORRECTED: Use the actual feature names produced by MetadataFeatures
        if metadata_type == 'core':
            self.expected_features = ['jaccard_index', 'content_blocks_diff', 'length_diff', 'ttr_diff'] # Updated
        else: # Assuming 'all' currently means the same as 'core'
            self.expected_features = ['jaccard_index', 'content_blocks_diff', 'length_diff', 'ttr_diff'] # Updated
        
        self.available_metadata_features = [
            f for f in self.expected_features if f in self.df.columns
        ]
        # No print statement here in the original code for test dataset, maintaining that

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 提取元數據（如果啟用）
        metadata_dict = None
        if self.include_metadata and self.available_metadata_features:
            metadata_dict = UnifiedInputBuilder.extract_metadata_from_row(row, self.metadata_type)
        
        # 使用統一輸入構建器
        # MODIFIED: Removed FastLexRank specific args from call
        encoded = UnifiedInputBuilder.create_unified_input(
            tokenizer=self.tokenizer,
            prompt=str(row['prompt']),
            response_a=str(row['response_a']),
            response_b=str(row['response_b']),
            metadata_dict=metadata_dict
            # Assuming max_len and include_prompt are handled by other logic or defaults
        )
        
        return encoded
