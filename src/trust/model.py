
from __future__ import annotations

import torch
import torch.nn as nn

from src.trust.trust_module import TrustScorer
from src.trust.aggregation import HeterophilyAwareAggregation


class EvAGNNTrustModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, trust_hidden: int = 16):
        super().__init__()
        self.trust_scorer = TrustScorer(hidden_dim=trust_hidden)
        self.aggregation = HeterophilyAwareAggregation(in_dim=in_dim, out_dim=hidden_dim)
        # aggregation output is hidden_dim * 2 (node_repr concat evidence_proj)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        is_celebrity_target: torch.Tensor,
        use_trust: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if use_trust:
            tau = self.trust_scorer(edge_features)
        else:
            tau = torch.ones(edge_features.shape[0], device=edge_features.device)

        node_repr = self.aggregation(h0, edge_index, tau, is_celebrity_target)
        logits = self.classifier(node_repr).squeeze(-1)
        return logits, tau
