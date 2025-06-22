from torch.utils.data import Dataset
import torch
from preprocessing.metadata_features import MetadataFeatures
import numpy as np
try:
    from fastlexrank import FastLexRank
    FASTLEXRANK_AVAILABLE = True
except ImportError:
    FASTLEXRANK_AVAILABLE = False

class DualTowerPairDataset(Dataset):
    """
    (修正後版本)
    為雙塔模型 (Dual-Encoder) 產生獨立、正確編碼的輸入。
    - Prompt, Response A, Response B 會被獨立 tokenize。
    - Metadata 作為獨立的數值特徵向量。
    - 可選地加入 LexRank 進行文本摘要。
    - 可選地包含/排除 Prompt。
    """

    def __init__(self, dataframe, tokenizer, max_len: int = 512,
                 apply_content_cleaning: bool = True,
                 include_metadata: bool = False,
                 metadata_type: str = 'core',
                 include_prompt: bool = True,
                 use_lexrank_q: bool = False,
                 lexrank_q_lower_bound: int = 1,
                 use_lexrank_r: bool = False,
                 lexrank_r_lower_bound: int = 10,
                 standardize_metadata: bool = False,
                 metadata_stats: dict = None):
        self.df = dataframe.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.apply_content_cleaning = apply_content_cleaning
        self.include_metadata = include_metadata
        self.metadata_type = metadata_type
        self.metadata_cols = []
        
        self.include_prompt = include_prompt
        self.use_lexrank_q = use_lexrank_q and FASTLEXRANK_AVAILABLE
        self.lexrank_q_lower_bound = lexrank_q_lower_bound
        self.use_lexrank_r = use_lexrank_r and FASTLEXRANK_AVAILABLE
        self.lexrank_r_lower_bound = lexrank_r_lower_bound
        self.standardize_metadata = standardize_metadata
        self.metadata_stats = metadata_stats or {}

        if self.include_metadata:
            self.metadata_cols = MetadataFeatures.get_feature_columns(self.metadata_type)
            if not all(col in self.df.columns for col in self.metadata_cols):
                 raise ValueError(f"DataFrame is missing required metadata columns for type '{self.metadata_type}'")

        if (self.use_lexrank_q or self.use_lexrank_r):
            if FASTLEXRANK_AVAILABLE:
                self.lexrank = FastLexRank()
                print("  - INFO: LexRank will be used for summarization in DualTowerPairDataset.")
            else:
                print("  - WARNING: fastlexrank is not installed. LexRank features will be disabled.")


    def __len__(self):
        return len(self.df)

    def _summarize(self, text, lower_bound):
        """Helper to apply LexRank summarization."""
        if len(self.tok.encode(text, add_special_tokens=False)) > lower_bound:
            # A simple sentence tokenizer
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                # Summarize to a number of words that should be less than max_len tokens
                summary_sentences = self.lexrank.summarize(sentences, max_words=int(self.max_len * 1.5))
                return " ".join(summary_sentences)
        return text

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        p_text = str(row.get("prompt", "")) if self.include_prompt else ""
        a_text = str(row.get("response_a", ""))
        b_text = str(row.get("response_b", ""))
        label = int(row["label"]) if "label" in row else -1

        # 1. (可選) 文本清理
        if self.apply_content_cleaning:
            # 傳遞 remove_special_blocks=True 來確保特殊區塊被移除
            p_text = MetadataFeatures.remove_special_content(p_text, remove_special_blocks=True)
            a_text = MetadataFeatures.remove_special_content(a_text, remove_special_blocks=True)
            b_text = MetadataFeatures.remove_special_content(b_text, remove_special_blocks=True)

        # 1.5 (可選) LexRank 摘要
        if self.use_lexrank_q and p_text:
            p_text = self._summarize(p_text, self.lexrank_q_lower_bound)
        
        if self.use_lexrank_r:
            if a_text:
                a_text = self._summarize(a_text, self.lexrank_r_lower_bound)
            if b_text:
                b_text = self._summarize(b_text, self.lexrank_r_lower_bound)

        # 2. 獨立 Tokenization
        #    每段文本都被獨立地截斷和填充到 max_len
        p_tok = self.tok(
            p_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        a_tok = self.tok(
            a_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        b_tok = self.tok(
            b_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # 3. 組裝樣本
        sample = {
            "p_input_ids": p_tok["input_ids"].squeeze(0),
            "p_attention_mask": p_tok["attention_mask"].squeeze(0),
            "a_input_ids": a_tok["input_ids"].squeeze(0),
            "a_attention_mask": a_tok["attention_mask"].squeeze(0),
            "b_input_ids": b_tok["input_ids"].squeeze(0),
            "b_attention_mask": b_tok["attention_mask"].squeeze(0),
        }

        # 4. (可選) 加入 Metadata
        if self.include_metadata:
            metadata_values = row[self.metadata_cols].values.astype(np.float32)
            if self.standardize_metadata and self.metadata_stats:
                standardized = []
                for val, col in zip(metadata_values, self.metadata_cols):
                    stats = self.metadata_stats.get(col, None)
                    if stats and stats.get('std', 0) > 1e-8:
                        val = (val - stats['mean']) / stats['std']
                    standardized.append(val)
                metadata_values = np.array(standardized, dtype=np.float32)
            sample["metadata_features"] = torch.tensor(metadata_values, dtype=torch.float)

        # 5. 加入標籤
        if label != -1:
            sample["labels"] = torch.tensor(label, dtype=torch.long)

        return sample
