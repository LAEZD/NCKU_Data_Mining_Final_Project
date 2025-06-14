# Unified Input Builder Module
# ============================
# 實現統一的、帶有預算感知能力的輸入構建策略
# 將元數據作為特權文本無縫注入輸入序列，並實施帶優先級的截斷規則

import torch
from typing import Dict, Any, Union, List
import numpy as np

class UnifiedInputBuilder:
    """
    統一輸入構建器類，負責將元數據、prompt 和 responses 整合為最終的模型輸入
    """
    
    @staticmethod
    def create_unified_input(
        tokenizer,
        prompt: str,
        response_a: str,
        response_b: str,
        metadata_dict: Dict[str, Union[float, int]] = None,
        max_len: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        統一的輸入構建函數 - 將元數據、prompt 和 responses 整合為模型輸入
        
        優先級策略:
        P0 (最高): 元數據字串 - 必須完整保留
        P1 (次高): Prompt - 重要上下文，設定預算上限
        P2 (普通): Responses - 使用剩餘空間，平均分配
        
        Args:
            tokenizer: HuggingFace tokenizer
            prompt (str): 提示文本
            response_a (str): 回應A
            response_b (str): 回應B
            metadata_dict (Dict): 元數據字典，例如 {'punc_v_a': 5, 'resp_jaccard': 0.8}
            max_len (int): 最大序列長度
            
        Returns:
            Dict[str, torch.Tensor]: 包含 'input_ids' 和 'attention_mask' 的字典
        """
        
        # === Step 1: 將元數據轉換為緊湊的字串格式 ===
        metadata_string = UnifiedInputBuilder._format_metadata_string(metadata_dict)
          # === Step 2: 智能特殊 token 空間計算 ===
        # 計算確切需要的特殊 token 數量
        special_tokens_needed = 4  # [CLS] + 3個 [SEP]
        if metadata_string:
            special_tokens_needed += 1  # 額外的分隔符
        
        available_space = max_len - special_tokens_needed
        
        # === Step 3: 元數據字串處理（最高優先級 P0） ===
        metadata_text = f"meta:{metadata_string}" if metadata_string else ""
        metadata_ids = tokenizer(metadata_text, add_special_tokens=False)['input_ids'] if metadata_text else []
        
        # 智能元數據截斷策略
        metadata_used_space = len(metadata_ids)
        max_metadata_budget = min(available_space // 4, 50)  # 最多占用1/4空間或50個token
        
        if metadata_used_space > max_metadata_budget:
            # 如果元數據過長，智能截斷而不是簡單切割
            print(f"警告: 元數據過長 ({metadata_used_space} tokens)，智能截斷到 {max_metadata_budget} tokens")
            # 保留最重要的元數據（通常是前面的）
            metadata_ids = metadata_ids[:max_metadata_budget]
            metadata_used_space = len(metadata_ids)
        
        remaining_space = available_space - metadata_used_space
        
        if remaining_space <= 10:  # 留一些安全邊距
            print(f"錯誤: 可用空間不足 ({remaining_space} tokens)")
            # 緊急模式：只保留核心元數據
            emergency_budget = 20
            metadata_ids = metadata_ids[:emergency_budget]
            metadata_used_space = len(metadata_ids)
            remaining_space = available_space - metadata_used_space
          # === Step 4: 智能 Prompt 處理（次高優先級 P1） ===
        prompt_text = f"Q:{prompt}"  # 使用更短的前綴
        prompt_ids_full = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        
        # 動態 prompt 預算分配：基於 prompt 和 responses 的相對長度
        response_a_text = f"A:{response_a}"  # 使用更短的前綴
        response_b_text = f"B:{response_b}"
        
        response_a_ids_full = tokenizer(response_a_text, add_special_tokens=False)['input_ids']
        response_b_ids_full = tokenizer(response_b_text, add_special_tokens=False)['input_ids']
        
        total_content_length = len(prompt_ids_full) + len(response_a_ids_full) + len(response_b_ids_full)
        
        if total_content_length <= remaining_space:
            # 空間充足，不需要截斷
            prompt_ids_final = prompt_ids_full
            response_a_ids_final = response_a_ids_full
            response_b_ids_final = response_b_ids_full
        else:
            # 需要智能分配空間
            # 動態分配：prompt 佔 30-50%，responses 平分剩餘
            prompt_ratio = max(0.3, min(0.5, len(prompt_ids_full) / total_content_length))
            prompt_budget = int(remaining_space * prompt_ratio)
            response_budget_total = remaining_space - prompt_budget
            response_budget_each = response_budget_total // 2
            
            # 應用預算
            prompt_ids_final = prompt_ids_full[:prompt_budget]
            response_a_ids_final = response_a_ids_full[:response_budget_each]
            response_b_ids_final = response_b_ids_full[:response_budget_each]
            
            # 如果還有剩餘空間，智能分配給需要更多空間的部分
            used_space = len(prompt_ids_final) + len(response_a_ids_final) + len(response_b_ids_final)
            leftover = remaining_space - used_space
            
            if leftover > 0:
                # 優先級：response > prompt（因為 response 是比較的核心）
                a_deficit = len(response_a_ids_full) - len(response_a_ids_final)
                b_deficit = len(response_b_ids_full) - len(response_b_ids_final)
                p_deficit = len(prompt_ids_full) - len(prompt_ids_final)
                
                if a_deficit > 0 and leftover > 0:
                    extra_a = min(a_deficit, leftover // 2)
                    response_a_ids_final = response_a_ids_full[:len(response_a_ids_final) + extra_a]
                    leftover -= extra_a
                
                if b_deficit > 0 and leftover > 0:
                    extra_b = min(b_deficit, leftover // 2)
                    response_b_ids_final = response_b_ids_full[:len(response_b_ids_final) + extra_b]
                    leftover -= extra_b
                
                if p_deficit > 0 and leftover > 0:
                    extra_p = min(p_deficit, leftover)
                    prompt_ids_final = prompt_ids_full[:len(prompt_ids_final) + extra_p]
        
        # === Step 6: 組裝最終序列 ===
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        
        # 構建序列：[CLS] + metadata + [SEP] + prompt + [SEP] + response_a + [SEP] + response_b
        final_input_ids = [cls_id]
        
        if metadata_ids:
            final_input_ids.extend(metadata_ids)
            final_input_ids.append(sep_id)
        
        final_input_ids.extend(prompt_ids_final)
        final_input_ids.append(sep_id)
        final_input_ids.extend(response_a_ids_final)
        final_input_ids.append(sep_id)
        final_input_ids.extend(response_b_ids_final)
          # === Step 7: 最終長度控制和填充 ===
        current_length = len(final_input_ids)
        
        if current_length > max_len:
            # 極端情況：仍然超長，從尾部緊急截斷
            print(f"緊急警告: 最終序列超長 ({current_length} > {max_len})，執行尾部截斷")
            final_input_ids = final_input_ids[:max_len]
            current_length = max_len
            padding_length = 0
        else:
            padding_length = max_len - current_length
        
        # 添加填充
        if padding_length > 0:
            final_input_ids.extend([pad_id] * padding_length)
            
        attention_mask = [1] * current_length + [0] * padding_length       
        # === Step 8: 生成詳細統計信息（調試用） ===
        # if metadata_string:  # 只在有元數據時打印統計
        #     UnifiedInputBuilder._print_allocation_stats(
        #         metadata_used_space, len(prompt_ids_final),
        #         len(response_a_ids_final), len(response_b_ids_final),
        #         current_length, max_len
        #     )
        
        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
    
    @staticmethod
    def _format_metadata_string(metadata_dict: Dict[str, Union[float, int]]) -> str:
        """
        將元數據字典轉換為極度緊湊的字串格式
        
        使用極短鍵名和智能數值格式化來最大化信息密度
        
        Args:
            metadata_dict: 元數據字典
            
        Returns:
            str: 格式化的元數據字串，例如 "pa:5 pb:3 jc:.75 lr:1.2"
        """
        if not metadata_dict:
            return ""
        
        # 使用極短鍵名映射，每個鍵名最多2-3字符
        key_mapping = {
            # Corrected mappings for the actual core features
            'jaccard_index': 'ji',
            'code_blocks_diff': 'cd',
            'length_diff': 'ld',
            'ttr_diff': 'td',

            # Mappings for other potential features if 'all' type were to include more
            'punc_v_a': 'pa',
            'punc_v_b': 'pb',
            'punc_v_prompt': 'pp',
            'prompt_resp_a_jaccard': 'aj',
            'prompt_resp_b_jaccard': 'bj',
            'prompt_resp_a_len_ratio': 'alr',
            'prompt_resp_b_len_ratio': 'blr',
            'prompt_length': 'pl',
            'response_a_length': 'ral',
            'response_b_length': 'rbl',
            'total_length': 'tl'
        }
        
        formatted_parts = []
        for key, value in metadata_dict.items():
            # 使用極短鍵名，如果沒有映射則取前2個字符
            short_key = key_mapping.get(key, key[:2])
            
            # 處理異常值
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            
            # 極度緊湊的數值格式化
            if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
                # 整數直接顯示
                formatted_parts.append(f"{short_key}:{int(value)}")
            else:
                # 浮點數智能格式化
                if abs(value) < 1:
                    # 小於1的數去掉前導零，例如 0.75 -> .75
                    formatted_value = f"{value:.2f}".lstrip('0')
                    if formatted_value.startswith('.'):
                        formatted_value = formatted_value
                    else:
                        formatted_value = f"{value:.2f}"
                elif abs(value) < 10:
                    # 小於10保留2位小數但去掉尾隨零
                    formatted_value = f"{value:.2f}".rstrip('0').rstrip('.')
                else:
                    # 大於等於10保留1位小數但去掉尾隨零
                    formatted_value = f"{value:.1f}".rstrip('0').rstrip('.')
                
                formatted_parts.append(f"{short_key}:{formatted_value}")
        
        return " ".join(formatted_parts)
    
    @staticmethod
    def _print_allocation_stats(
        metadata_tokens: int,
        prompt_tokens: int, 
        response_a_tokens: int,
        response_b_tokens: int,
        total_used: int,
        max_len: int
    ):
        """
        打印 token 分配統計（調試用）
        """
        print(f"    Token 分配: Meta={metadata_tokens}, Prompt={prompt_tokens}, "
              f"Resp_A={response_a_tokens}, Resp_B={response_b_tokens}, "
              f"Total={total_used}/{max_len}")
    
    @staticmethod
    def extract_metadata_from_row(row, feature_type: str = 'core') -> Dict[str, Union[float, int]]:
        """
        從數據框行中提取元數據特徵
        
        Args:
            row: pandas Series 或類似的行對象
            feature_type: 'core' 只提取核心特徵，'all' 提取所有特徵
            
        Returns:
            Dict: 元數據字典
        """
        if feature_type == 'core':
            # 只提取最重要的核心特徵
            core_features = ['jaccard_index', 'code_blocks_diff', 'length_diff', 'ttr_diff'] # CORRECTED
            metadata = {}
            for feature in core_features:
                if hasattr(row, feature) or feature in row:
                    metadata[feature] = row.get(feature, 0.0)
            return metadata
        else: # feature_type == 'all'
            # 提取所有可用的元數據特徵
            # CORRECTED to reflect what metadata_features.py actually produces for 'all' (which is currently core)
            all_possible_features = ['jaccard_index', 'code_blocks_diff', 'length_diff', 'ttr_diff']
            metadata = {}
            for feature in all_possible_features:
                if hasattr(row, feature) or feature in row:
                    metadata[feature] = row.get(feature, 0.0)
            return metadata
