
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.pipeline.schema import UnifiedGraph, Edge
from src.trust.edge_features import EdgeTrustFeatures, compute_edge_trust_features, build_reverse_index

TOP_K_DEFAULT = 5


@dataclass
class EdgeExplanation:
    neighbor_id: str
    relation: str
    tau: float
    reciprocity: float
    behavioral_similarity: float
    is_celebrity_target: bool
    t_observed: bool

    def to_sentence(self) -> str:
        recip_str = "reciprocated" if self.reciprocity > 0.5 else "one-way, not reciprocated"
        celeb_str = " (high-profile account)" if self.is_celebrity_target else ""
        conf_str = "" if self.t_observed else " [approximate timing]"
        return (
            f"{self.relation} -> {self.neighbor_id}{celeb_str}: trust={self.tau:.2f}, "
            f"{recip_str}, behavioral similarity={self.behavioral_similarity:.2f}{conf_str}"
        )


@dataclass
class NodeExplanation:
    user_id: str
    predicted_prob_bot: float
    top_distrusted: list[EdgeExplanation] = field(default_factory=list)
    top_trusted: list[EdgeExplanation] = field(default_factory=list)
    n_low_trust_edges: int = 0
    mean_tau: float = 0.0
    evasion_event_note: str | None = None  # hook for  ξ_u(t) output

    def to_report(self) -> str:
        lines = [
            f"=== Explanation for {self.user_id} ===",
            f"Predicted bot probability: {self.predicted_prob_bot:.2f}",
            f"Low-trust edges: {self.n_low_trust_edges}  |  Mean trust across all edges: {self.mean_tau:.2f}",
        ]
        if self.evasion_event_note:
            lines.append(f"Evasion-event signal: {self.evasion_event_note}")
        lines.append("\nMost distrusted connections (evidence of camouflage):")
        if self.top_distrusted:
            for e in self.top_distrusted:
                lines.append(f"  - {e.to_sentence()}")
        else:
            lines.append("  (none below the low-trust threshold)")
        lines.append("\nMost trusted connections (for context):")
        if self.top_trusted:
            for e in self.top_trusted:
                lines.append(f"  - {e.to_sentence()}")
        else:
            lines.append("  (no edges found)")
        return "\n".join(lines)


def explain_node(
    user_id: str,
    graph: UnifiedGraph,
    tau_by_edge: dict[int, float],   # edge index (position in graph.edges) -> tau value
    predicted_prob_bot: float,
    top_k: int = TOP_K_DEFAULT,
    low_trust_threshold: float = 0.3,
    evasion_event_note: str | None = None,
) -> NodeExplanation:
    
    reverse_index = build_reverse_index(graph.edges)

    incident: list[tuple[Edge, float, EdgeTrustFeatures]] = []
    for idx, e in enumerate(graph.edges):
        if e.u != user_id and e.v != user_id:
            continue
        if idx not in tau_by_edge:
            continue
        neighbor_side_edge = e  # could be outgoing (u==user_id) or incoming (v==user_id)
        feats = compute_edge_trust_features(neighbor_side_edge, graph.nodes, reverse_index)
        incident.append((neighbor_side_edge, tau_by_edge[idx], feats))

    def to_explanation(e: Edge, tau: float, feats: EdgeTrustFeatures) -> EdgeExplanation:
        neighbor = e.v if e.u == user_id else e.u
        return EdgeExplanation(
            neighbor_id=neighbor,
            relation=e.r.value,
            tau=tau,
            reciprocity=feats.reciprocity,
            behavioral_similarity=feats.behavioral_similarity,
            is_celebrity_target=bool(feats.is_celebrity_target),
            t_observed=e.t_observed,
        )

    sorted_by_tau_asc = sorted(incident, key=lambda x: x[1])
    distrusted = [to_explanation(e, t, f) for e, t, f in sorted_by_tau_asc if t < low_trust_threshold][:top_k]
    trusted = [to_explanation(e, t, f) for e, t, f in sorted_by_tau_asc[::-1] if t >= low_trust_threshold][:top_k]

    n_low_trust = sum(1 for _, t, _ in incident if t < low_trust_threshold)
    mean_tau = sum(t for _, t, _ in incident) / len(incident) if incident else 0.0

    return NodeExplanation(
        user_id=user_id,
        predicted_prob_bot=predicted_prob_bot,
        top_distrusted=distrusted,
        top_trusted=trusted,
        n_low_trust_edges=n_low_trust,
        mean_tau=mean_tau,
        evasion_event_note=evasion_event_note,
    )


def explain_batch(
    graph: UnifiedGraph,
    node_ids: list[str],
    edge_index: torch.Tensor,
    tau: torch.Tensor,
    edges_list: list[Edge],
    logits: torch.Tensor,
    node_id_to_idx: dict[str, int],
    user_ids_to_explain: list[str],
    top_k: int = TOP_K_DEFAULT,
) -> dict[str, NodeExplanation]:
    
    tau_by_edge = {i: tau[i].item() for i in range(len(edges_list))}
    probs = torch.sigmoid(logits)

    results = {}
    for uid in user_ids_to_explain:
        if uid not in node_id_to_idx:
            continue
        idx = node_id_to_idx[uid]
        results[uid] = explain_node(
            user_id=uid,
            graph=graph,
            tau_by_edge=tau_by_edge,
            predicted_prob_bot=probs[idx].item(),
            top_k=top_k,
        )
    return results
