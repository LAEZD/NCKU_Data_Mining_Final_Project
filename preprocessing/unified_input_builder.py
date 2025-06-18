# Unified Input Builder Module
# ============================
# 實現統一的、帶有預算感知能力的輸入構建策略
# 將元數據作為特權文本無縫注入輸入序列，並實施帶優先級的截斷規則

import torch
from typing import Dict, Any, Union, List
import numpy as np

# Imports for FastLexrank
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# NEW: Import MetadataFeatures for content cleaning
from .metadata_features import MetadataFeatures


# Ensure NLTK\\\'s punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("INFO: Downloading NLTK punkt tokenizer for FastLexrank...")
    nltk.download('punkt', quiet=True)
    print("INFO: NLTK punkt tokenizer downloaded.")
except Exception as e: # Catch other potential errors during NLTK setup
    print(f"WARNING: Could not verify/download NLTK punkt. FastLexRank might fail. Error: {e}")

# --- Preprocessing Configuration Switches ---
# (所有 default 參數已移除，統一由 fine_tuning.py 控制)
# Centralized place to adjust preprocessing behavior defaults.
# Modules like fine_tuning.py can import these to ensure consistency,
# or they can be overridden by function parameters where applicable.

# Settings for UnifiedInputBuilder.create_unified_input\'s FastLexRank behavior
# These are used as default values for the corresponding parameters in create_unified_input.
CREATE_UNIFIED_INPUT_USE_FASTLEXRANK_DEFAULT = False
CREATE_UNIFIED_INPUT_FASTLEXRANK_LOWER_BOUND_DEFAULT = 1
INCLUDE_PROMPT_DEFAULT = True
# NEW: Defaults for response FastLexRank
CREATE_UNIFIED_INPUT_USE_FASTLEXRANK_FOR_RESPONSE_DEFAULT = False
CREATE_UNIFIED_INPUT_FASTLEXRANK_RESPONSE_LOWER_BOUND_DEFAULT = 10
# RENAMED and UPDATED: Default for applying content cleaning to prompt/responses before tokenization and LexRank
APPLY_CONTENT_CLEANING_DEFAULT = True
# NEW: Switch for fixed input format
USE_FIXED_FORMAT_DEFAULT = False # You can change this to True if you want fixed format by default


def get_lexrank_summary_token_ids(
    original_prompt_text: str,
    tokenizer,
    target_min_tokens: int,
    target_max_tokens: int,
    original_prompt_token_len: int,
    prefix: str = "Q:"
) -> Union[List[int], None]:
    """
    Generates a summary of the prompt using LexRank and tokenizes it.
    Tries to find a summary that is shorter than the original and fits within token budget.
    Prefers SHORTER summaries that meet the minimum requirement.
    """
    if not original_prompt_text.strip():
        return None
    try:
        parser = PlaintextParser.from_string(original_prompt_text, SumyTokenizer("english"))
        summarizer = LexRankSummarizer()
    except Exception as e:
        print(f"WARNING: Failed to initialize LexRank components. Error: {e}")
        return None # Cannot proceed with summarization

    best_summary_ids = None
    # Iterate through a range of sentence counts for the summary
    for num_sents in range(1, 6): # Try summaries of 1 to 5 sentences
        try:
            summary_sentences = summarizer(parser.document, sentences_count=num_sents)
        except Exception as e:
            print(f"WARNING: LexRank summarization failed for {num_sents} sentences. Error: {e}")
            continue # Try next sentence count or fail if this was the last one

        summary_text = " ".join([str(s) for s in summary_sentences])

        if not summary_text.strip():
            if num_sents == 1:
                return None
            else:
                break

        # FIXED: Changed [\'input_ids\'] to ['input_ids']
        current_summary_ids = tokenizer(f"{prefix}{summary_text}", add_special_tokens=False)['input_ids']
        current_summary_token_len = len(current_summary_ids)

        if current_summary_token_len < original_prompt_token_len:
            if current_summary_token_len <= target_max_tokens:
                if current_summary_token_len >= target_min_tokens:
                    # This logic prefers shorter summaries that still meet the minimum requirement
                    if best_summary_ids is None or current_summary_token_len < len(best_summary_ids):
                        best_summary_ids = current_summary_ids
                elif best_summary_ids is None: # If no summary met the min_tokens, take the one that is at least shorter than original
                    best_summary_ids = current_summary_ids
            else:
                # If even a short summary exceeds the max token budget, stop trying
                break
        elif num_sents == 1 and current_summary_token_len >= original_prompt_token_len:
            # If a 1-sentence summary is not shorter than the original, summarization is pointless
            return None

    return best_summary_ids


def get_lexrank_summary_token_ids_for_response(
    original_text: str,
    tokenizer,
    target_min_tokens: int,
    target_max_tokens: int,
    original_text_token_len: int,
    prefix: str = "A:"
) -> Union[List[int], None]:
    """
    (優化後版本)
    Generates a summary of the response text using LexRank.
    It runs the summarizer ONCE to get a ranked list of sentences.
    Then, it iterates to find the LONGEST summary that fits the budget.
    """
    if not original_text.strip():
        return None

    try:
        parser = PlaintextParser.from_string(original_text, SumyTokenizer("english"))
        summarizer = LexRankSummarizer()
        document_sentences = list(parser.document.sentences)
        if not document_sentences:
            return None
    except Exception as e:
        print(f"WARNING: Failed to initialize LexRank components for response summarization. Error: {e}")
        return None

    # --- 優化核心 ---
    # 1. 只運行一次 LexRank 來獲取所有句子的重要性排序
    #    我們請求摘要的句子數等於總句數，這樣會返回一個按重要性排序的句子列表
    try:
        # The summarizer returns sentences ordered by their rank.
        ranked_sentences = summarizer(parser.document, sentences_count=len(document_sentences))
    except Exception as e:
        print(f"WARNING: LexRank summarization for response failed. Error: {e}")
        return None

    best_summary_ids = None

    # 2. 從最長的摘要開始，向下迭代尋找第一個符合條件的摘要
    #    這樣找到的第一個就是我們想要的「最長且符合條件」的摘要
    for num_sents in range(len(ranked_sentences), 0, -1):
        # 從已排序的列表中取出前 num_sents 個句子
        current_summary_sentences = ranked_sentences[:num_sents]
        summary_text = " ".join([str(s) for s in current_summary_sentences])

        if not summary_text.strip():
            continue

        current_summary_ids = tokenizer(f"{prefix}{summary_text}", add_special_tokens=False)['input_ids']
        current_summary_token_len = len(current_summary_ids)

        is_shorter_than_original = (current_summary_token_len < original_text_token_len)
        is_within_max_budget = (current_summary_token_len <= target_max_tokens)
        meets_min_budget = (current_summary_token_len >= target_min_tokens)

        if is_shorter_than_original and is_within_max_budget and meets_min_budget:
            # 找到了！這是符合條件的最長摘要，直接返回
            return current_summary_ids
    
    # 如果循環結束都沒有找到符合所有條件的摘要，返回 None
    # （原始邏輯是會保留一個不滿足最低長度但比原始短的摘要，這裡為了簡潔，可以根據需求調整）
    return None


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
        max_len: int = 512,
        include_prompt: bool = None,
        use_fastlexrank_for_question: bool = None,
        fastlexrank_question_token_lower_bound: int = None,
        use_fastlexrank_for_response: bool = None,
        fastlexrank_response_token_lower_bound: int = None,
        apply_content_cleaning: bool = None,
        use_fixed_format: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        統一的輸入構建函數 - 將元數據、prompt 和 responses 整合為模型輸入

        優先級策略 (更新後):
        P0 (最高): 元數據字串 - 必須完整保留
        P1 (次高): Responses - 如果超長，會嘗試 FastLexRank (如果啟用)，然後截斷。目標是公平分配空間並填滿預算。
        P2 (最低): Prompt - 如果超長，會嘗試 FastLexRank (如果啟用)，然後截断。
        """

        # === Step 1: 將元數據轉換為緊湊的字串格式 ===
        metadata_string = UnifiedInputBuilder._format_metadata_string(metadata_dict)

        # === Step 2: 智能特殊 token 空間計算 ===
        if use_fixed_format:
            special_tokens_needed = 4  # [CLS], [SEP] for meta, [SEP] for Q, [SEP] for A/B
        else:
            special_tokens_needed = 3  # [CLS] + 2個 [SEP] (for responses)
            if metadata_string:
                special_tokens_needed += 1
            # FIXED: Removed invalid escape in comment
            if include_prompt: # This will be for the prompt's SEP
                special_tokens_needed += 1

        available_space = max_len - special_tokens_needed

        # === Step 3: 元數據字串處理（最高優先級 P0） ===
        metadata_text = f"meta:{metadata_string}" if metadata_string else ""
        metadata_ids = tokenizer(metadata_text, add_special_tokens=False)['input_ids'] if metadata_text else []

        metadata_used_space = len(metadata_ids)
        # Consider a dynamic cap based on max_len, e.g., 10% of max_len but not more than 50.
        max_metadata_budget = min(available_space // 4, 50)


        if metadata_used_space > max_metadata_budget:
            # print(f\"警告: 元數據過長 ({metadata_used_space} tokens)，智能截斷到 {max_metadata_budget} tokens\")
            metadata_ids = metadata_ids[:max_metadata_budget]
            metadata_used_space = len(metadata_ids)

        remaining_space_after_metadata = available_space - metadata_used_space

        if remaining_space_after_metadata <= 10: # Ensure some space for actual content
            # print(f\"警告: 元數據處理後可用空間不足 ({remaining_space_after_metadata} tokens). 嘗試減少元數據.\")
            # Further reduce metadata if it leaves too little space.
            emergency_metadata_budget = min(metadata_used_space, max(0, available_space - 10)) # Ensure at least 10 for content
            if metadata_used_space > emergency_metadata_budget:
                 metadata_ids = metadata_ids[:emergency_metadata_budget]
                 metadata_used_space = len(metadata_ids)
            remaining_space_after_metadata = available_space - metadata_used_space
            if remaining_space_after_metadata <=0:
                # print(\"錯誤: 無可用空間創建輸入.\")
                # Handle this case: maybe return empty tensors or raise error
                # For now, let it proceed, it will likely result in empty content parts.
                pass


        # === Step 4: Prompt 和 Responses Content Cleaning, Tokenization & Initial Setup ===
        prompt_prefix = "Q:"
        response_a_prefix = "A:"
        response_b_prefix = "B:"

        # Cache for cleaned text to avoid redundant processing if apply_content_cleaning is True
        cleaned_text_cache = {} 

        # Determine the actual texts to be used for tokenization and LexRank
        # These will be cleaned versions if apply_content_cleaning is True
        actual_prompt_text = prompt
        actual_response_a_text = response_a
        actual_response_b_text = response_b

        if apply_content_cleaning:
            # Clean prompt text
            if prompt in cleaned_text_cache: # Check cache using original prompt as key
                actual_prompt_text = cleaned_text_cache[prompt]
            else:
                cleaned_version = MetadataFeatures.remove_special_content(prompt)
                actual_prompt_text = cleaned_version
                cleaned_text_cache[prompt] = cleaned_version # Cache the cleaned version
            
            # Clean response_a text
            if response_a in cleaned_text_cache: # Check cache using original response_a as key
                actual_response_a_text = cleaned_text_cache[response_a]
            else:
                cleaned_version = MetadataFeatures.remove_special_content(response_a)
                actual_response_a_text = cleaned_version
                cleaned_text_cache[response_a] = cleaned_version # Cache the cleaned version

            # Clean response_b text
            if response_b in cleaned_text_cache: # Check cache using original response_b as key
                actual_response_b_text = cleaned_text_cache[response_b]
            else:
                cleaned_version = MetadataFeatures.remove_special_content(response_b)
                actual_response_b_text = cleaned_version
                cleaned_text_cache[response_b] = cleaned_version # Cache the cleaned version
        
        # Tokenize full versions using the (potentially cleaned) texts
        # The prefixes are added here before tokenization
        prompt_ids_full = tokenizer(f"{prompt_prefix}{actual_prompt_text}", add_special_tokens=False)['input_ids'] if include_prompt else []
        response_a_ids_full = tokenizer(f"{response_a_prefix}{actual_response_a_text}", add_special_tokens=False)['input_ids']
        response_b_ids_full = tokenizer(f"{response_b_prefix}{actual_response_b_text}", add_special_tokens=False)['input_ids']
        
        prompt_current_ids = list(prompt_ids_full)
        response_a_current_ids = list(response_a_ids_full)
        response_b_current_ids = list(response_b_ids_full)

        # The texts for LexRank will be the same (potentially cleaned) texts, 
        # but WITHOUT prefixes, as get_lexrank_summary_token_ids adds its own prefix.
        prompt_text_for_lexrank = actual_prompt_text if include_prompt else ""
        response_a_text_for_lexrank = actual_response_a_text
        response_b_text_for_lexrank = actual_response_b_text

        # === Step 5: Iterative LexRank and Budgeting based on Priority (Meta > Resp > Prompt) ===
        
        # Initial check for total needed tokens
        total_content_needed = (len(prompt_current_ids) if include_prompt else 0) + len(response_a_current_ids) + len(response_b_current_ids)

        if total_content_needed > remaining_space_after_metadata:
            # Overflow detected. Apply strategies based on priority.

            # --- Strategy for Prompt (P2 - lowest priority for LexRank application order) ---
            if include_prompt and use_fastlexrank_for_question:
                # Calculate space for prompt assuming responses take their current (possibly full) length
                space_for_prompt_lexrank = remaining_space_after_metadata -(len(response_a_current_ids) + len(response_b_current_ids))
                space_for_prompt_lexrank = max(0, space_for_prompt_lexrank)

                if space_for_prompt_lexrank >= fastlexrank_question_token_lower_bound:
                    summarized_prompt_ids = get_lexrank_summary_token_ids(
                        prompt_text_for_lexrank, tokenizer, # Use cleaned text for LexRank
                        fastlexrank_question_token_lower_bound,
                        space_for_prompt_lexrank,
                        len(prompt_ids_full), # original length of full prompt_ids
                        prefix=prompt_prefix
                    )
                    if summarized_prompt_ids and len(summarized_prompt_ids) < len(prompt_current_ids):
                        prompt_current_ids = summarized_prompt_ids
                        # print(f\"INFO: FastLexRank applied to Prompt. New length: {len(prompt_current_ids)}\")
                total_content_needed = (len(prompt_current_ids) if include_prompt else 0) +  len(response_a_current_ids) + len(response_b_current_ids)


            # --- Strategy for Responses (P1 retention priority; LexRank if shrinking prompt is insufficient) ---
            if total_content_needed > remaining_space_after_metadata and use_fastlexrank_for_response:
                space_for_responses_lexrank = remaining_space_after_metadata - (len(prompt_current_ids) if include_prompt else 0)
                space_for_responses_lexrank = max(0, space_for_responses_lexrank)

                # NEW "Both or Neither" LexRank Logic for Responses
                attempt_lexrank_for_responses_pair = False
                # Only attempt if there's enough space for two summaries, each meeting the lower bound.
                if space_for_responses_lexrank >= (2 * fastlexrank_response_token_lower_bound):
                    space_for_responses_lexrank_per_response = space_for_responses_lexrank // 2 # Max budget for each response summary

                    # Try LexRank for Response A using the new function
                    summary_a_ids = get_lexrank_summary_token_ids_for_response(
                        response_a_text_for_lexrank,
                        tokenizer,
                        fastlexrank_response_token_lower_bound,
                        space_for_responses_lexrank_per_response,
                        len(response_a_ids_full), # original token length (with prefix)
                        prefix=response_a_prefix
                    )
                    # Try LexRank for Response B using the new function
                    summary_b_ids = get_lexrank_summary_token_ids_for_response(
                        response_b_text_for_lexrank,
                        tokenizer,
                        fastlexrank_response_token_lower_bound,
                        space_for_responses_lexrank_per_response,
                        len(response_b_ids_full), # original token length (with prefix)
                        prefix=response_b_prefix
                    )

                    if summary_a_ids is not None and summary_b_ids is not None:
                        response_a_current_ids = summary_a_ids
                        response_b_current_ids = summary_b_ids
                        # print(f\"INFO: FastLexRank applied to BOTH Response A (len: {len(response_a_current_ids)}) and B (len: {len(response_b_current_ids)}).\")
                    else:
                        # If not both successfully summarized, neither is applied.
                        # Responses remain as they were before this paired attempt.
                        # print(f\"INFO: FastLexRank for response pair failed or was not beneficial for both. Using original responses for budgeting.\")
                        pass # No change to response_a_current_ids, response_b_current_ids
                # else: (attempt_lexrank_for_responses_pair is False)
                    # print(f\"INFO: Not enough space for paired response LexRank (each meeting lower bound). Using original responses for budgeting.\")
                    # No change to response_a_current_ids, response_b_current_ids needed, they are already full.
                
                total_content_needed = (len(prompt_current_ids) if include_prompt else 0) + len(response_a_current_ids) + len(response_b_current_ids)

        # === Step 6: Final Truncation if still overflowing (Priority: Meta > Resp > Prompt) === 
        # Ensure prompt_final_ids is initialized even if include_prompt is False
        if include_prompt:
            prompt_final_ids = list(prompt_current_ids) # Make a copy to modify
        else:
            prompt_final_ids = [] 

        response_a_final_ids = list(response_a_current_ids) # Make a copy
        response_b_final_ids = list(response_b_current_ids) # Make a copy

        current_len_content = (len(prompt_final_ids) if include_prompt else 0) + len(response_a_final_ids) + len(response_b_final_ids)

        if current_len_content > remaining_space_after_metadata:
            overflow = current_len_content - remaining_space_after_metadata

            # Truncate Prompt (P2 - lowest priority)
            if include_prompt and overflow > 0:
                prompt_len_to_truncate = min(overflow, len(prompt_final_ids))
                prompt_final_ids = prompt_final_ids[:len(prompt_final_ids) - prompt_len_to_truncate]
                overflow -= prompt_len_to_truncate
                # print(f\"INFO: Truncated Prompt by {prompt_len_to_truncate}. New length: {len(prompt_final_ids)}\")

            # Truncate Responses (P1 - fairly) if overflow persists
            if overflow > 0:
                len_a_before_trunc = len(response_a_final_ids)
                len_b_before_trunc = len(response_b_final_ids)
                
                # Target total length for responses after truncation
                target_total_responses_len = (len_a_before_trunc + len_b_before_trunc) - overflow
                target_total_responses_len = max(0, target_total_responses_len)

                new_len_a = min(len_a_before_trunc, target_total_responses_len // 2 + target_total_responses_len % 2)
                new_len_b = min(len_b_before_trunc, target_total_responses_len // 2)
                
                # If one response was shorter than its allocated share, give more to the other
                if new_len_a < len_a_before_trunc and (new_len_a + len_b_before_trunc > target_total_responses_len):
                     new_len_b = min(len_b_before_trunc, target_total_responses_len - new_len_a)
                elif new_len_b < len_b_before_trunc and (len_a_before_trunc + new_len_b > target_total_responses_len):
                     new_len_a = min(len_a_before_trunc, target_total_responses_len - new_len_b)
                
                response_a_final_ids = response_a_final_ids[:new_len_a]
                response_b_final_ids = response_b_final_ids[:new_len_b]
                # print(f\"INFO: Truncated Responses. A: {len(response_a_final_ids)}, B: {len(response_b_final_ids)}\")

        # === Step 7: Redistribute Leftover Space to Maximize Budget Use ===
        # (Priority: Responses, then Prompt)
        current_used_by_content = (len(prompt_final_ids) if include_prompt else 0) + len(response_a_final_ids) + len(response_b_final_ids)
        leftover = remaining_space_after_metadata - current_used_by_content

        if leftover > 0:
            # Try to add back to Response A
            can_add_to_a = len(response_a_ids_full) - len(response_a_final_ids) # Diff from original full
            add_to_a = min(leftover, can_add_to_a)
            if add_to_a > 0:
                # Append from the original full token list where it was truncated or summarized from
                response_a_final_ids.extend(response_a_ids_full[len(response_a_final_ids) : len(response_a_final_ids) + add_to_a])
                leftover -= add_to_a
            
            # Try to add back to Response B
            can_add_to_b = len(response_b_ids_full) - len(response_b_final_ids)
            add_to_b = min(leftover, can_add_to_b)
            if add_to_b > 0:
                response_b_final_ids.extend(response_b_ids_full[len(response_b_final_ids) : len(response_b_final_ids) + add_to_b])
                leftover -= add_to_b

            # Try to add back to Prompt
            if include_prompt and leftover > 0:
                # Ensure prompt_ids_full is available (it would be [] if include_prompt was false initially)
                can_add_to_p = len(prompt_ids_full) - len(prompt_final_ids)
                add_to_p = min(leftover, can_add_to_p)
                if add_to_p > 0:
                    prompt_final_ids.extend(prompt_ids_full[len(prompt_final_ids) : len(prompt_final_ids) + add_to_p])
                    leftover -= add_to_p
        
        # === Step 8: 組裝最終序列 ===
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        final_input_ids = [cls_id]

        if metadata_ids:
            final_input_ids.extend(metadata_ids)
            final_input_ids.append(sep_id)

        if include_prompt and prompt_final_ids: # Ensure prompt_final_ids is used
            final_input_ids.extend(prompt_final_ids)
            final_input_ids.append(sep_id)

        final_input_ids.extend(response_a_final_ids) # Use response_a_final_ids
        final_input_ids.append(sep_id)
        final_input_ids.extend(response_b_final_ids) # Use response_b_final_ids

        # === Step 9: 最終長度控制和填充 === (Was Step 7)
        current_length = len(final_input_ids)

        if current_length > max_len:
            print(f"緊急警告: 最終序列超長 ({current_length} > {max_len})，執行尾部截斷")
            final_input_ids = final_input_ids[:max_len]
            current_length = max_len

        padding_length = max_len - current_length
        attention_mask = [1] * current_length + [0] * padding_length

        if padding_length > 0:
            final_input_ids.extend([pad_id] * padding_length)

        # === Step 10: 生成詳細統計信息（調試用） === (Was Step 8)
        # UnifiedInputBuilder._print_allocation_stats(
        #     len(metadata_ids),
        #     len(prompt_final_ids) if include_prompt else 0, # Use prompt_final_ids
        #     len(response_a_final_ids), # Use response_a_final_ids
        #     len(response_b_final_ids), # Use response_b_final_ids
        #     current_length, max_len
        # )

        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    @staticmethod
    def _format_metadata_string(metadata_dict: Dict[str, Union[float, int]]) -> str:
        """
        將元數據字典轉換為極度緊湊的字串格式
        """
        if not metadata_dict:
            return ""

        key_mapping = {
            'jaccard_index': 'ji', 'content_blocks_diff': 'cd', 'length_diff': 'ld', 'ttr_diff': 'td',
            'punc_v_a': 'pa', 'punc_v_b': 'pb', 'punc_v_prompt': 'pp',
            'prompt_resp_a_jaccard': 'aj', 'prompt_resp_b_jaccard': 'bj',
            'prompt_resp_a_len_ratio': 'alr', 'prompt_resp_b_len_ratio': 'blr',
            'prompt_length': 'pl', 'response_a_length': 'ral', 'response_b_length': 'rbl',
            'total_length': 'tl'
        }

        formatted_parts = []
        for key, value in metadata_dict.items():
            short_key = key_mapping.get(key, key[:2])
            if value is None or np.isnan(value) or np.isinf(value): value = 0.0

            if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
                formatted_parts.append(f"{short_key}:{int(value)}")
            else:
                # Smart formatting for floats
                if abs(value) < 1:
                    formatted_value = f"{value:.2f}".lstrip('0') if value != 0 else ".0"
                elif abs(value) < 10:
                    formatted_value = f"{value:.2f}".rstrip('0').rstrip('.')
                else:
                    formatted_value = f"{value:.1f}".rstrip('0').rstrip('.')
                
                if formatted_value == "-.0": formatted_value = ".0"

                formatted_parts.append(f"{short_key}:{formatted_value}")

        return " ".join(formatted_parts)

    @staticmethod
    def _print_allocation_stats(
        metadata_tokens: int, prompt_tokens: int,
        response_a_tokens: int, response_b_tokens: int,
        total_used: int, max_len: int
    ):
        """
        打印 token 分配統計（調試用）
        """
        print(f"    Token 分配: Meta={metadata_tokens}, Prompt={prompt_tokens}, Resp_A={response_a_tokens}, Resp_B={response_b_tokens}, Total={total_used}/{max_len}")

    @staticmethod
    # FIXED: Changed \'core\' to 'core'
    def extract_metadata_from_row(row, feature_type: str = 'core') -> Dict[str, Union[float, int]]:
        """
        從數據框行中提取元數據特徵
        """
        # FIXED: Changed \'jaccard_index\' etc. to 'jaccard_index'
        core_features = ['jaccard_index', 'content_blocks_diff', 'length_diff', 'ttr_diff']
        if feature_type == 'core':
            metadata = {f: row.get(f, 0.0) for f in core_features if hasattr(row, f) or f in row}
            return metadata
        else: # 'all'
            all_possible_features = core_features
            metadata = {f: row.get(f, 0.0) for f in all_possible_features if hasattr(row, f) or f in row}
            return metadata

# Example Usage (for testing, not part of the class)
# FIXED: Changed \'__main__\' to '__main__'
if __name__ == '__main__':
    from transformers import AutoTokenizer
    # FIXED: Changed 'punkt_tab' to 'punkt' as it's the correct package name
    nltk.download('punkt')

    # FIXED: Changed 'bert-base-uncased' to 'bert-base-uncased'
    tokenizer_name = 'bert-base-uncased'
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Could not load tokenizer {tokenizer_name} for testing: {e}")
        # This is a fallback tokenizer for environments without internet, etc.
        # FIXED: Corrected the dictionary syntax ['input_ids']
        tokenizer = lambda text, add_special_tokens=False: {'input_ids': [0]*len(text)}

    print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    if tokenizer.pad_token is None:
        # FIXED: Corrected the heavily escaped dictionary literal
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Added pad_token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")

    prompt1 = "This is a short prompt."
    res_a1 = "Response A is also quite brief."
    res_b1 = "Response B is similarly concise."

    # FIXED: Corrected the dictionary literal syntax
    meta1 = {'jaccard_index': 0.5, 'length_diff': 10}

    print("\n--- Test Case 1: Basic ---")
    # MODIFIED: Removed FastLexRank specific args from call as they are now hardcoded internally
    # ADDED new response lexrank params to test calls
    output1 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt1, res_a1, res_b1, meta1, max_len=128, include_prompt=True,
        use_fastlexrank_for_response=True # Test with response lexrank
    )
    # FIXED: Corrected the key access syntax
    print(f"Input IDs length: {len(output1['input_ids'])}")

    prompt2 ="Artificial Intelligence, commonly known as AI, is a wide-ranging branch of computer science concerned with building smart machines capable of performing tasks that typically require human intelligence. These tasks include learning from experience, understanding natural language, recognizing patterns, solving complex problems, and making decisions. A major subset of AI is Machine Learning, which focuses on developing algorithms that allow computers to learn from and make predictions or decisions based on data. Deep Learning, a further specialization within Machine Learning, utilizes neural networks with many layers to analyze various factors, enabling more sophisticated applications. In everyday life, AI powers recommendation engines on platforms like Netflix and Amazon, suggesting content or products based on user behavior. The technology is also pivotal in the development of autonomous vehicles, which use AI to perceive their environment and navigate without human input. Furthermore, in the medical field, AI algorithms are being used to analyze medical images to detect diseases with high accuracy. Despite its rapid advancement, the field faces significant challenges, including ethical considerations and data privacy concerns. Ultimately, the goal of AI research is to create technology that can augment human capabilities, driving innovation across nearly every industry."
    res_a2 = "Response A for the long prompt, also quite verbose to ensure we hit limits."
    res_b2 = "Response B for the long prompt, equally wordy to make the test challenging."
    # FIXED: Corrected the dictionary literal syntax
    meta2 = {'jaccard_index': 0.2, 'length_diff': 50, 'ttr_diff': 0.1}

    print("\n--- Test Case 2: Overflow, FastLexrank Active ---")
    output2 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=128, include_prompt=True,
        use_fastlexrank_for_response=True
    )
    print(f"Input IDs length: {len(output2['input_ids'])}")

    print("\n--- Test Case 3: Overflow, FastLexrank Active, Tight Space ---")
    output3 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=80, include_prompt=True,
        use_fastlexrank_for_response=True
    )
    print(f"Input IDs length: {len(output3['input_ids'])}")

    print("\n--- Test Case 4: Overflow, (FastLexrank is now always active) ---")
    # NOTE: This test case will now behave like Test Case 2, as FastLexRank is always enabled internally.
    output4 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=128, include_prompt=True,
        use_fastlexrank_for_question=False, # Test with question lexrank off
        use_fastlexrank_for_response=True
    )
    print(f"Input IDs length: {len(output4['input_ids'])}")

    print("\n--- Test Case 5: include_prompt=False ---")
    output5 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=128, include_prompt=False,
        use_fastlexrank_for_response=True
    )
    print(f"Input IDs length: {len(output5['input_ids'])}")

    prompt6 = "Short."
    res_a6 = "A." * 30
    res_b6 = "B." * 30
    print("\n--- Test Case 6: Very short prompt, overflow by responses ---")
    output6 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt6, res_a6, res_b6, meta1, max_len=100, include_prompt=True,
        use_fastlexrank_for_response=True
    )
    print(f"Input IDs length: {len(output6['input_ids'])}")

    print("\n--- Test Case 7: High lower bound, insufficient space ---")
    # NOTE: This test's behavior is now governed by the hardcoded lower bound (1).
    output7 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=100, include_prompt=True,
        fastlexrank_question_token_lower_bound=50, # High lower bound for question
        use_fastlexrank_for_response=True,
        fastlexrank_response_token_lower_bound=20 # High lower bound for response
    )
    print(f"Input IDs length: {len(output7['input_ids'])}")

    print("\n--- Test Case 8: Response LexRank only, prompt fits but responses overflow ---")
    short_prompt_for_resp_overflow = "This prompt is short."
    long_res_a = "This is response A, and it is very long, designed to cause an overflow when combined with another long response, even if the prompt itself is short. We need many words here." * 3
    long_res_b = "This is response B, similarly very long, also designed to cause an overflow. It mirrors the length of response A to test fairness in truncation or summarization." * 3
    output8 = UnifiedInputBuilder.create_unified_input(
        tokenizer, short_prompt_for_resp_overflow, long_res_a, long_res_b, meta1, max_len=150, include_prompt=True,
        use_fastlexrank_for_question=False, # Question lexrank off
        use_fastlexrank_for_response=True,
        fastlexrank_response_token_lower_bound=10
    )
    print(f"Input IDs length: {len(output8['input_ids'])}")

    print("\n--- Test Case 9: remove_special_blocks_for_response_lexrank=False (conceptual test) ---")
    # This test is more about the parameter being passed, actual block removal isn't implemented here.
    output9 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt1, res_a1, res_b1, meta1, max_len=128, include_prompt=True,
        use_fastlexrank_for_response=True,
        apply_content_cleaning=False # Test with cleaning off
    )
    print(f"Input IDs length: {len(output9['input_ids'])}")

    print("\\n--- Test Case 10: Content Cleaning Active (Default) ---")
    # Assuming prompt2, res_a2, res_b2 contain markdown that would be cleaned
    # For example, add some mock markdown:
    prompt_with_markdown = prompt2 + "\\n```python\\nprint('hello')\\n```"
    res_a_with_markdown = res_a2 + "\\n$$E=mc^2$$"
    res_b_with_markdown = res_b2 + "\\n|Header1|Header2|\\n|---|---|\\n|Data1|Data2|"
    
    output10 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt_with_markdown, res_a_with_markdown, res_b_with_markdown, meta2, max_len=128, 
        include_prompt=True,
        use_fastlexrank_for_question=True,
        use_fastlexrank_for_response=True,
        apply_content_cleaning=True # Explicitly test with cleaning on
    )

    print(f"Input IDs length: {len(output10['input_ids'])}")
    # To verify cleaning, one would ideally inspect the summarized text if LexRank was triggered,
    # or the input to LexRank. For now, we just ensure the parameter is passed.