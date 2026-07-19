
from __future__ import annotations

import torch

from src.pipeline.schema import UnifiedGraph
from src.trust.edge_features import build_reverse_index, compute_edge_trust_features
from src.trust.trust_module import TrustScorer


def build_trust_inputs(graph: UnifiedGraph) -> dict:
    """
    Returns a dict with everything needed to run TrustScorer + HeterophilyAwareAggregation:
      - node_ids: list[str], stable ordering, index i- this node's row in h
      - node_id_to_idx: dict[str, int]
      - h0: (N, feat_dim) float tensor of initial node features x_u (must be pre-populated)
      - edge_index: (2, E) long tensor [source_idx, target_idx]
      - edge_features: (E, FEATURE_DIM) float tensor, raw trust features per edge
      - is_celebrity_target: (E,) float tensor, aligned with edge_index columns
      - edges: the original list[Edge], same order as edge_index columns (for the
               explanation head to map back to human-readable edge info)
    """
    node_ids = list(graph.nodes.keys())
    node_id_to_idx = {uid: i for i, uid in enumerate(node_ids)}

    missing_x = [uid for uid in node_ids if graph.nodes[uid].x is None]
    if missing_x:
        raise ValueError(
            f"{len(missing_x)} nodes have no x_u populated — run build_node_features() "
            f"from the pipeline before calling build_trust_inputs()."
        )
    h0 = torch.tensor([graph.nodes[uid].x for uid in node_ids], dtype=torch.float32)

    reverse_index = build_reverse_index(graph.edges)

    src_idx, tgt_idx = [], []
    feat_rows = []
    celeb_flags = []
    kept_edges = []
    for e in graph.edges:
        if e.u not in node_id_to_idx or e.v not in node_id_to_idx:
            continue  # dangling edge (shouldn't normally happen, but guard anyway)
        feats = compute_edge_trust_features(e, graph.nodes, reverse_index)
        src_idx.append(node_id_to_idx[e.u])
        tgt_idx.append(node_id_to_idx[e.v])
        feat_rows.append(feats.to_vector())
        celeb_flags.append(feats.is_celebrity_target)
        kept_edges.append(e)

    import numpy as np
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_features = torch.from_numpy(np.stack(feat_rows)).float()
    is_celebrity_target = torch.tensor(celeb_flags, dtype=torch.float32)

    return {
        "node_ids": node_ids,
        "node_id_to_idx": node_id_to_idx,
        "h0": h0,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "is_celebrity_target": is_celebrity_target,
        "edges": kept_edges,
    }


def compute_tau(trust_scorer: TrustScorer, edge_features: torch.Tensor) -> torch.Tensor:
    
    return trust_scorer(edge_features)
