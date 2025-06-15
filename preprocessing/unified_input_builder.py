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

# Ensure NLTK\'s punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("INFO: Downloading NLTK punkt tokenizer for FastLexrank...")
    nltk.download('punkt', quiet=True)
    print("INFO: NLTK punkt tokenizer downloaded.")
except Exception as e: # Catch other potential errors during NLTK setup
    print(f"WARNING: Could not verify/download NLTK punkt. FastLexRank might fail. Error: {e}")

# --- Preprocessing Configuration Switches ---
# Centralized place to adjust preprocessing behavior defaults.
# Modules like fine_tuning.py can import these to ensure consistency,
# or they can be overridden by function parameters where applicable.

# Settings for UnifiedInputBuilder.create_unified_input\'s FastLexRank behavior
# These are used as default values for the corresponding parameters in create_unified_input.
CREATE_UNIFIED_INPUT_USE_FASTLEXRANK_DEFAULT = True
CREATE_UNIFIED_INPUT_FASTLEXRANK_LOWER_BOUND_DEFAULT = 1

# General preprocessing flags (primarily for logic outside UnifiedInputBuilder,
# e.g., in enhanced_preprocessing.py, often controlled by fine_tuning.py\'s Config)
# Defined here for user convenience to see all related flags together.
# Consider importing these into fine_tuning.py\'s Config if you want this file to be the source of truth.
APPLY_AUGMENTATION_CONFIG_DEFAULT = False # Corresponds to Config.APPLY_AUGMENTATION in fine_tuning.py
EXTRACT_METADATA_CONFIG_DEFAULT = True    # Corresponds to Config.EXTRACT_METADATA in fine_tuning.py
METADATA_TYPE_CONFIG_DEFAULT = 'core'     # Corresponds to Config.METADATA_TYPE in fine_tuning.py
# INCLUDE_PROMPT is a direct parameter to create_unified_input, typically from Config.

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
        include_prompt: bool = True,
        # NEW parameters for FastLexRank, using module-level defaults
        use_fastlexrank_for_question: bool = CREATE_UNIFIED_INPUT_USE_FASTLEXRANK_DEFAULT,
        fastlexrank_question_token_lower_bound: int = CREATE_UNIFIED_INPUT_FASTLEXRANK_LOWER_BOUND_DEFAULT
    ) -> Dict[str, torch.Tensor]:
        """
        統一的輸入構建函數 - 將元數據、prompt 和 responses 整合為模型輸入

        優先級策略:
        P0 (最高): 元數據字串 - 必須完整保留
        P1 (次高): Prompt - 重要上下文，設定預算上限 (如果 include_prompt 為 True)
                     (可能會被 FastLexrank 壓縮)
        P2 (普通): Responses - 使用剩餘空間，平均分配
        """

        # === Step 1: 將元數據轉換為緊湊的字串格式 ===
        metadata_string = UnifiedInputBuilder._format_metadata_string(metadata_dict)

        # === Step 2: 智能特殊 token 空間計算 ===
        special_tokens_needed = 3  # [CLS] + 2個 [SEP] (for responses)
        if metadata_string:
            special_tokens_needed += 1
        # FIXED: Removed invalid escape in comment
        if include_prompt: # This will be for the prompt's SEP
            special_tokens_needed += 1

        available_space = max_len - special_tokens_needed

        # === Step 3: 元數據字串處理（最高優先級 P0） ===
        metadata_text = f"meta:{metadata_string}" if metadata_string else ""
        # FIXED: Changed [\'input_ids\'] to ['input_ids']
        metadata_ids = tokenizer(metadata_text, add_special_tokens=False)['input_ids'] if metadata_text else []

        metadata_used_space = len(metadata_ids)
        max_metadata_budget = min(available_space // 4, 50)

        if metadata_used_space > max_metadata_budget:
            print(f"警告: 元數據過長 ({metadata_used_space} tokens)，智能截斷到 {max_metadata_budget} tokens")
            metadata_ids = metadata_ids[:max_metadata_budget]
            metadata_used_space = len(metadata_ids)

        remaining_space_after_metadata = available_space - metadata_used_space

        if remaining_space_after_metadata <= 10:
            print(f"錯誤: 元數據處理後可用空間不足 ({remaining_space_after_metadata} tokens)")
            emergency_budget = min(metadata_used_space, 20)
            metadata_ids = metadata_ids[:emergency_budget]
            metadata_used_space = len(metadata_ids)
            remaining_space_after_metadata = available_space - metadata_used_space


        # === Step 4: Prompt 和 Responses Tokenization & FastLexRank (if applicable) ===
        prompt_prefix = "Q:"
        response_a_prefix = "A:"
        response_b_prefix = "B:"

        # FIXED: Changed [\'input_ids\'] to ['input_ids']
        response_a_ids_full = tokenizer(f"{response_a_prefix}{response_a}", add_special_tokens=False)['input_ids']
        # FIXED: Changed [\'input_ids\'] to ['input_ids']
        response_b_ids_full = tokenizer(f"{response_b_prefix}{response_b}", add_special_tokens=False)['input_ids']

        prompt_ids_for_budgeting = []

        if include_prompt:
            original_prompt_text_content = prompt
            # FIXED: Changed [\\\'input_ids\\\'] to ['input_ids']
            current_prompt_ids = tokenizer(f"{prompt_prefix}{original_prompt_text_content}", add_special_tokens=False)['input_ids']
            prompt_ids_for_budgeting = current_prompt_ids

            if use_fastlexrank_for_question:
                estimated_tokens_with_original_prompt = len(current_prompt_ids) + len(response_a_ids_full) + len(response_b_ids_full)

                if estimated_tokens_with_original_prompt > remaining_space_after_metadata:
                    # print(f"INFO: Token overflow detected ({estimated_tokens_with_original_prompt} > {remaining_space_after_metadata}). Attempting FastLexRank for question.")
                    # ADDED: Print original prompt before summarization attempt
                    # print(f"      Original prompt (first 150 chars): '{original_prompt_text_content[:150]}...'")

                    # Calculate available space for prompt *after* allocating for full responses
                    max_tokens_for_summarized_prompt = remaining_space_after_metadata - (len(response_a_ids_full) + len(response_b_ids_full))
                    max_tokens_for_summarized_prompt = max(0, max_tokens_for_summarized_prompt)

                    min_tokens_for_summarized_prompt = fastlexrank_question_token_lower_bound

                    if max_tokens_for_summarized_prompt >= min_tokens_for_summarized_prompt:
                        summarized_ids = get_lexrank_summary_token_ids(
                            original_prompt_text_content,
                            tokenizer,
                            min_tokens_for_summarized_prompt,
                            max_tokens_for_summarized_prompt,
                            len(current_prompt_ids),
                            prefix=prompt_prefix
                        )
                        if summarized_ids is not None and len(summarized_ids) < len(current_prompt_ids):
                            prompt_ids_for_budgeting = summarized_ids
                            # ADDED: Decode and print summarized prompt
                            summarized_prompt_text = tokenizer.decode(summarized_ids, skip_special_tokens=True)
                            # print(f"INFO: FastLexRank applied. Prompt tokens reduced from {len(current_prompt_ids)} to {len(prompt_ids_for_budgeting)}.")
                            # print(f"      Summarized prompt (first 150 chars): '{summarized_prompt_text[:150]}...'")
                        else:
                            # print("INFO: FastLexRank did not produce a shorter or suitable summary. Original prompt (or its truncation by later logic) will be used.")
                            # ADDED: Print original prompt if summarization failed or was not better
                            # print(f"      Using original prompt (first 150 chars): '{original_prompt_text_content[:150]}...'")
                            pass
                    else:
                        pass
                        # print(f"INFO: Not enough space for effective FastLexRank summarization of prompt (max_allowable: {max_tokens_for_summarized_prompt}, min_target: {min_tokens_for_summarized_prompt}). Prompt will be truncated by budgeting logic.")
                        # ADDED: Print original prompt if not enough space for summarization
                        # print(f"      Using original prompt (first 150 chars): '{original_prompt_text_content[:150]}...'")

        # === Step 5: 智能 Prompt 和 Responses 預算分配 ===
        prompt_ids_final = []

        total_content_length_for_allocation = len(response_a_ids_full) + len(response_b_ids_full)
        if include_prompt:
            total_content_length_for_allocation += len(prompt_ids_for_budgeting)

        if total_content_length_for_allocation <= remaining_space_after_metadata:
            if include_prompt:
                prompt_ids_final = prompt_ids_for_budgeting
            response_a_ids_final = response_a_ids_full
            response_b_ids_final = response_b_ids_full
        else:
            if include_prompt:
                # Give prompt a slightly higher priority in allocation
                prompt_ratio_numerator = len(prompt_ids_for_budgeting)
                prompt_ratio = max(0.3, min(0.5, prompt_ratio_numerator / total_content_length_for_allocation if total_content_length_for_allocation > 0 else 0.3))

                prompt_budget = int(remaining_space_after_metadata * prompt_ratio)
                response_budget_total = remaining_space_after_metadata - prompt_budget

                prompt_ids_final = prompt_ids_for_budgeting[:prompt_budget]
            else:
                response_budget_total = remaining_space_after_metadata

            response_budget_each = response_budget_total // 2

            response_a_ids_final = response_a_ids_full[:response_budget_each]
            response_b_ids_final = response_b_ids_full[:response_budget_each]

            # Re-distribute leftover space after integer division
            used_space_by_content = len(response_a_ids_final) + len(response_b_ids_final)
            if include_prompt:
                used_space_by_content += len(prompt_ids_final)

            leftover = remaining_space_after_metadata - used_space_by_content

            if leftover > 0:
                a_deficit = len(response_a_ids_full) - len(response_a_ids_final)
                b_deficit = len(response_b_ids_full) - len(response_b_ids_final)

                can_give_a = min(a_deficit, leftover)
                if can_give_a > 0:
                    response_a_ids_final.extend(response_a_ids_full[len(response_a_ids_final) : len(response_a_ids_final) + can_give_a])
                    leftover -= can_give_a

                can_give_b = min(b_deficit, leftover)
                if can_give_b > 0:
                    response_b_ids_final.extend(response_b_ids_full[len(response_b_ids_final) : len(response_b_ids_final) + can_give_b])
                    leftover -= can_give_b

                if include_prompt and leftover > 0:
                    p_deficit = len(prompt_ids_for_budgeting) - len(prompt_ids_final)
                    can_give_p = min(p_deficit, leftover)
                    if can_give_p > 0:
                         prompt_ids_final.extend(prompt_ids_for_budgeting[len(prompt_ids_final) : len(prompt_ids_final) + can_give_p])

        # === Step 6: 組裝最終序列 ===
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        final_input_ids = [cls_id]

        if metadata_ids:
            final_input_ids.extend(metadata_ids)
            final_input_ids.append(sep_id)

        if include_prompt and prompt_ids_final:
            final_input_ids.extend(prompt_ids_final)
            final_input_ids.append(sep_id)

        final_input_ids.extend(response_a_ids_final)
        final_input_ids.append(sep_id)
        final_input_ids.extend(response_b_ids_final)

        # === Step 7: 最終長度控制和填充 ===
        current_length = len(final_input_ids)

        if current_length > max_len:
            print(f"緊急警告: 最終序列超長 ({current_length} > {max_len})，執行尾部截斷")
            final_input_ids = final_input_ids[:max_len]
            current_length = max_len

        padding_length = max_len - current_length
        attention_mask = [1] * current_length + [0] * padding_length

        if padding_length > 0:
            final_input_ids.extend([pad_id] * padding_length)

        # === Step 8: 生成詳細統計信息（調試用） ===
        # UnifiedInputBuilder._print_allocation_stats(
        #     len(metadata_ids),
        #     len(prompt_ids_final) if include_prompt else 0,
        #     len(response_a_ids_final),
        #     len(response_b_ids_final),
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
    output1 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt1, res_a1, res_b1, meta1, max_len=128, include_prompt=True
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
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=128, include_prompt=True
    )
    print(f"Input IDs length: {len(output2['input_ids'])}")

    print("\n--- Test Case 3: Overflow, FastLexrank Active, Tight Space ---")
    output3 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=80, include_prompt=True
    )
    print(f"Input IDs length: {len(output3['input_ids'])}")

    print("\n--- Test Case 4: Overflow, (FastLexrank is now always active) ---")
    # NOTE: This test case will now behave like Test Case 2, as FastLexRank is always enabled internally.
    output4 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=128, include_prompt=True
    )
    print(f"Input IDs length: {len(output4['input_ids'])}")

    print("\n--- Test Case 5: include_prompt=False ---")
    output5 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=128, include_prompt=False
    )
    print(f"Input IDs length: {len(output5['input_ids'])}")

    prompt6 = "Short."
    res_a6 = "A." * 30
    res_b6 = "B." * 30
    print("\n--- Test Case 6: Very short prompt, overflow by responses ---")
    output6 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt6, res_a6, res_b6, meta1, max_len=100, include_prompt=True
    )
    print(f"Input IDs length: {len(output6['input_ids'])}")

    print("\n--- Test Case 7: High lower bound, insufficient space ---")
    # NOTE: This test's behavior is now governed by the hardcoded lower bound (1).
    output7 = UnifiedInputBuilder.create_unified_input(
        tokenizer, prompt2, res_a2, res_b2, meta2, max_len=100, include_prompt=True
    )
    print(f"Input IDs length: {len(output7['input_ids'])}")