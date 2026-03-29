from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class VisualEncoder(nn.Module):
    """视觉编码器，后续可扩展并行文本分支。"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MLPHead(nn.Module):
    """简单稳定的任务头。"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChangeUnderstandingBaseline(nn.Module):
    """双时相纯视觉 baseline。"""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.visual_encoder = VisualEncoder(pretrained=pretrained)
        feat_dim = self.visual_encoder.feature_dim
        fusion_dim = feat_dim * 3

        self.change_head = MLPHead(fusion_dim, 1, hidden_dim=256, dropout=dropout)
        self.object_head = MLPHead(fusion_dim, 3, hidden_dim=256, dropout=dropout)
        self.action_head = MLPHead(fusion_dim, 4, hidden_dim=256, dropout=dropout)
        self.location_head = MLPHead(fusion_dim, 10, hidden_dim=256, dropout=dropout)

    def encode_visual_pair(self, image_t1: torch.Tensor, image_t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = self.visual_encoder(image_t1)
        f2 = self.visual_encoder(image_t2)
        return f1, f2

    @staticmethod
    def fuse_features(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        return torch.cat([f1, f2, torch.abs(f2 - f1)], dim=1)

    def forward(self, image_t1: torch.Tensor, image_t2: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1, f2 = self.encode_visual_pair(image_t1, image_t2)
        fusion = self.fuse_features(f1, f2)
        return {
            "change_logits": self.change_head(fusion).squeeze(-1),
            "object_logits": self.object_head(fusion),
            "action_logits": self.action_head(fusion),
            "location_logits": self.location_head(fusion),
        }
