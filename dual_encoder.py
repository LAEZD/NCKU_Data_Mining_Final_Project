import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig
from typing import Optional

class DualTowerConfig(PretrainedConfig):
    """DualTower模型的配置類"""
    model_type = "dual_tower"
    
    def __init__(
        self,
        base_model="distilbert-base-uncased",
        hidden_size=768,
        dropout=0.2,
        metadata_feature_size=5,
        metadata_fusion="dual_path",
        num_labels=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.metadata_feature_size = metadata_feature_size
        self.metadata_fusion = metadata_fusion
        self.num_labels = num_labels
        self.has_meta_path = metadata_feature_size > 0 and metadata_fusion == 'dual_path'
        self.classifier_input_dim = hidden_size * 6 if self.has_meta_path else hidden_size * 5

class DualTowerPairClassifier(nn.Module):
    """
    Dual-Encoder / Two-Tower 模型 + Metadata 特徵處理
    - 共用一座 Transformer Encoder，各自編碼 Prompt、Response A、Response B  
    - 特徵向量: [v_p, v_a, |v_p − v_a|, v_b, |v_p − v_b|, metadata_features]
    - metadata 透過 meta_path 從5維升到768維
    - 最終特徵: 768 × 6 = 4608 維
    - 3-類 softmax → 0=A wins, 1=B wins, 2=Tie
    """

    config_class = DualTowerConfig

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.2,
        metadata_feature_size: int = 5,  # metadata 原始維度
        metadata_fusion: str = 'dual_path',  # metadata 融合方式
        config: DualTowerConfig = None,
    ):
        super().__init__()
        
        # 創建或使用提供的config
        if config is None:
            self.config = DualTowerConfig(
                base_model=base_model,
                hidden_size=hidden_size,
                dropout=dropout,
                metadata_feature_size=metadata_feature_size,
                metadata_fusion=metadata_fusion
            )
        else:
            self.config = config
            
        self.encoder = AutoModel.from_pretrained(self.config.base_model)
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Metadata 處理路徑 - 與 kaggle_last.py 完全一致
        self.meta_path = None
        if self.config.has_meta_path:
            self.meta_path = nn.Sequential(
                nn.Linear(self.config.metadata_feature_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU()
            )
            print(f"    💡 創建 metadata 處理路徑: {self.config.metadata_feature_size} -> {self.config.hidden_size} -> {self.config.hidden_size}")
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.classifier_input_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.num_labels),
        )
        
        print(f"    💡 分類器輸入維度: {self.config.classifier_input_dim} (期望 4608 如果有metadata)")

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
        metadata_features: Optional[torch.Tensor] = None, # metadata 輸入
        labels: Optional[torch.Tensor] = None,
    ):
        v_p = self.encode(p_input_ids, p_attention_mask)
        v_a = self.encode(a_input_ids, a_attention_mask)
        v_b = self.encode(b_input_ids, b_attention_mask)

        # 拼接基礎特徵向量：[v_p, v_a, |v_p - v_a|, v_b, |v_p - v_b|] = 768×5
        feat = torch.cat(
            [v_p, v_a, torch.abs(v_p - v_a), v_b, torch.abs(v_p - v_b)], dim=-1
        )
        
        # 處理 metadata 特徵
        if self.meta_path and metadata_features is not None:
            # 通過 meta_path 升維：5 -> 768 -> 768
            meta_feat = self.meta_path(metadata_features)
            # 拼接：基礎特徵(768×5) + metadata特徵(768) = 768×6 = 4608
            feat = torch.cat([feat, meta_feat], dim=-1)
        elif self.meta_path:
            # 如果模型有 meta_path 但沒有提供 metadata，用零填充
            batch_size = feat.shape[0]
            zero_meta = torch.zeros(batch_size, self.encoder.config.dim, device=feat.device)
            feat = torch.cat([feat, zero_meta], dim=-1)
            
        logits = self.classifier(self.dropout(feat))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
    
    def save_pretrained(self, save_directory):
        """保存模型和配置"""
        import os
        import json
        
        # 保存config.json
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # 保存模型權重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"    💾 模型和配置已保存到: {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """從保存的路徑載入模型"""
        import os
        import json
        
        # 載入config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = DualTowerConfig(**config_dict)
        else:
            # 如果沒有config.json，使用預設配置
            config = DualTowerConfig()
        
        # 創建模型
        model = cls(config=config)
        
        # 載入權重
        model_path_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_path_file):
            state_dict = torch.load(model_path_file, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model
