import torch
from torch import nn
from transformers import AutoModel
from typing import Optional

class DualTowerPairClassifier(nn.Module):
    """
    Dual-Encoder / Two-Tower 模型  
    - 共用一座 Transformer Encoder，各自編碼 Prompt、Response A、Response B  
    - 特徵向量: [v_p, v_a, |v_p − v_a|, v_b, |v_p − v_b|]  
    - 3-類 softmax → 0=A wins, 1=B wins, 2=Tie
    """

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """回傳 [CLS] embedding (batch, hidden)"""
        out = self.encoder(
            input_ids=ids, attention_mask=mask, return_dict=True
        )
        return out.last_hidden_state[:, 0]

    def forward(
        self,
        p_input_ids: torch.Tensor,
        p_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
        b_input_ids: torch.Tensor,
        b_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        v_p = self.encode(p_input_ids, p_attention_mask)
        v_a = self.encode(a_input_ids, a_attention_mask)
        v_b = self.encode(b_input_ids, b_attention_mask)

        feat = torch.cat(
            [v_p, v_a, torch.abs(v_p - v_a), v_b, torch.abs(v_p - v_b)], dim=-1
        )
        logits = self.classifier(self.dropout(feat))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
