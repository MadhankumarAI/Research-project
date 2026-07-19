
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from src.trust.edge_features import FEATURE_DIM, EdgeTrustFeatures


class TrustScorer(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        
        logits = self.net(edge_feats).squeeze(-1)
        return torch.sigmoid(logits)

    @staticmethod
    def features_to_tensor(feats_list: list[EdgeTrustFeatures]) -> torch.Tensor:
        arr = np.stack([f.to_vector() for f in feats_list])
        return torch.from_numpy(arr).float()
