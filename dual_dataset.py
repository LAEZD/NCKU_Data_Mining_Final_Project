from torch.utils.data import Dataset
import torch

class DualTowerPairDataset(Dataset):
    """Dataset that yields (Prompt, Response A, Response B) token tensors for Dual‑Encoder."""

    def __init__(self, dataframe, tokenizer, max_len: int = 512):
        self.df = dataframe.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _tok(self, text: str):
        return self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p, a, b = str(row["prompt"]), str(row["response_a"]), str(row["response_b"])
        label = int(row["label"]) if "label" in row else -1

        ptok = self._tok(p)
        atok = self._tok(a)
        btok = self._tok(b)

        sample = {
            "p_input_ids": ptok["input_ids"].squeeze(0),
            "p_attention_mask": ptok["attention_mask"].squeeze(0),
            "a_input_ids": atok["input_ids"].squeeze(0),
            "a_attention_mask": atok["attention_mask"].squeeze(0),
            "b_input_ids": btok["input_ids"].squeeze(0),
            "b_attention_mask": btok["attention_mask"].squeeze(0)
        }
        if label >= 0:
            sample["labels"] = torch.tensor(label, dtype=torch.long)
        return sample
