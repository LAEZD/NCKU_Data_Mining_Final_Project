# Preprocessing Package
# ====================
# 數據預處理模組包，包含以下子模組：
# - data_augmentation: 數據增強（響應順序交換）
# - dynamic_budgeting: 動態預算分配策略
# - metadata_features: 元數據特徵提取
# - enhanced_preprocessing: 整合的增強預處理模組

from .data_augmentation import DataAugmentation
from .metadata_features import MetadataFeatures
from .enhanced_preprocessing import (
    EnhancedPreprocessing,
    EnhancedLLMDataset,
    EnhancedPipelineModules,
    EnhancedTestDataset
)
from .unified_input_builder import UnifiedInputBuilder

__all__ = [
    'DataAugmentation',
    'DynamicBudgeting', 
    'MetadataFeatures',
    'EnhancedPreprocessing',
    'EnhancedLLMDataset',
    'EnhancedPipelineModules',
    'EnhancedTestDataset',
    'UnifiedInputBuilder'
]

__version__ = '1.0.0'
__author__ = 'NCKU Data Mining Final Project Team'
