
from __future__ import annotations

import torch
import torch.nn as nn

LOW_TRUST_THRESHOLD = 0.3

DISTRUST_SUMMARY_DIM = 4 


class HeterophilyAwareAggregation(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        
        self.w_evidence = nn.Linear(DISTRUST_SUMMARY_DIM, out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        h: torch.Tensor, # (N,in_dim) current node embeddings
        edge_index: torch.Tensor,# (2,E) long tensor: [source_idx, target_idx]
        tau: torch.Tensor,  # (E,) trust scores in [0,1], aligned with edge_index columns
        is_celebrity_target: torch.Tensor,# (E,) 0/1 float, aligned with edge_index columns
    ) -> torch.Tensor:

        n = h.shape[0]
        src, tgt = edge_index[0], edge_index[1]

        all_recipients = torch.cat([tgt, src], dim=0)     
        all_senders = torch.cat([src, tgt], dim=0)        
        all_tau = torch.cat([tau, tau], dim=0)             

        messages = self.w_neigh(h[all_senders]) * all_tau.unsqueeze(-1)   # (2E, out_dim)
        agg = torch.zeros(n, self.out_dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, all_recipients, messages)

        weight_sum = torch.zeros(n, device=h.device, dtype=h.dtype)
        weight_sum.index_add_(0, all_recipients, all_tau)
        weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(-1)
        agg = agg / weight_sum   # normalized trust-weighted mean

        self_term = self.w_self(h)
        node_repr = self.act(self_term + agg)   # (N, out_dim)

        all_is_celeb = torch.cat([is_celebrity_target, is_celebrity_target], dim=0)
        is_low_trust = (all_tau < LOW_TRUST_THRESHOLD).float()

        n_low_trust = torch.zeros(n, device=h.device, dtype=h.dtype)
        n_low_trust.index_add_(0, all_recipients, is_low_trust)

        tau_sum = torch.zeros(n, device=h.device, dtype=h.dtype)
        tau_sum.index_add_(0, all_recipients, all_tau)
        edge_count = torch.zeros(n, device=h.device, dtype=h.dtype)
        edge_count.index_add_(0, all_recipients, torch.ones_like(all_tau))
        mean_tau = tau_sum / edge_count.clamp(min=1e-6)

        low_trust_celebrity = is_low_trust * all_is_celeb
        n_low_trust_celeb = torch.zeros(n, device=h.device, dtype=h.dtype)
        n_low_trust_celeb.index_add_(0, all_recipients, low_trust_celebrity)

        # placeholder 4th slot: reserved for your burst-intensity signal from ξ_u(t), so the evidence vector can eventually be fused
        # with the synergy module's output without changing this layer's shape.
        burst_placeholder = torch.zeros(n, device=h.device, dtype=h.dtype)

        evidence = torch.stack([
            torch.log1p(n_low_trust),
            mean_tau,
            torch.log1p(n_low_trust_celeb),
            burst_placeholder,
        ], dim=1)   # (N, DISTRUST_SUMMARY_DIM)

        evidence_proj = self.act(self.w_evidence(evidence))   # (N, out_dim)

        return torch.cat([node_repr, evidence_proj], dim=1)   # (N, out_dim * 2)

