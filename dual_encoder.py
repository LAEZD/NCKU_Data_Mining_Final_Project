import torch
from torch import nn
from transformers import AutoModel
from typing import Optional

class DualTowerPairClassifier(nn.Module):
    """
    Dual-Encoder / Two-Tower 模型
    - 支援三種 meta data 融合方式（concat, dual_path, residual_inject）
    - 支援 include_prompt 控制向量組合方式
    """
    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.2,
        metadata_feature_size: int = 0,
        metadata_fusion: str = 'concat',  # 新增
        include_prompt: bool = True,      # 新增
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.metadata_fusion = metadata_fusion
        self.include_prompt = include_prompt
        self.hidden_size = hidden_size
        self.metadata_feature_size = metadata_feature_size

        # meta data 殘差注入用
        if metadata_fusion == 'residual_inject' and metadata_feature_size > 0:
            self.metadata_injector_for_diff = nn.Sequential(
                nn.Linear(metadata_feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_size)
            )
        # dual path
        if metadata_fusion == 'dual_path' and metadata_feature_size > 0:
            self.meta_path = nn.Sequential(
                nn.Linear(metadata_feature_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
        # 分類器輸入維度（根據 include_prompt 動態調整）
        if metadata_fusion == 'concat':
            base = 5 if include_prompt else 4
            classifier_input_size = hidden_size * base + metadata_feature_size
        elif metadata_fusion == 'dual_path':
            base = 5 if include_prompt else 4
            classifier_input_size = hidden_size * base + hidden_size  # 語義+meta路徑
        elif metadata_fusion == 'residual_inject':
            classifier_input_size = hidden_size * 4
        else:
            classifier_input_size = hidden_size * 5
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        metadata_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        v_p = self.encode(p_input_ids, p_attention_mask)
        v_a = self.encode(a_input_ids, a_attention_mask)
        v_b = self.encode(b_input_ids, b_attention_mask)
        # include_prompt 決定向量組合
        if self.include_prompt:
            feat_base = torch.cat(
                [v_p, v_a, torch.abs(v_p - v_a), v_b, torch.abs(v_p - v_b)], dim=-1
            )
        else:
            # 只比三個向量
            feat_base = torch.cat(
                [v_a, v_b, v_a - v_b, v_a * v_b], dim=-1
            )
        # meta data 融合
        if self.metadata_fusion == 'concat':
            if metadata_features is not None:
                feat = torch.cat([feat_base, metadata_features], dim=-1)
            else:
                feat = feat_base
        elif self.metadata_fusion == 'dual_path' and metadata_features is not None:
            meta_vec = self.meta_path(metadata_features)
            feat = torch.cat([feat_base, meta_vec], dim=-1)
        elif self.metadata_fusion == 'residual_inject' and metadata_features is not None:
            semantic_diff = v_a - v_b
            semantic_interaction = v_a * v_b
            statistical_diff_vector = self.metadata_injector_for_diff(metadata_features)
            combined_diff = semantic_diff + statistical_diff_vector
            feat = torch.cat([v_a, v_b, combined_diff, semantic_interaction], dim=-1)
        else:
            feat = feat_base
        logits = self.classifier(self.dropout(feat))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
