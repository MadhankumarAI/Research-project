
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch

from src.pipeline.schema import UnifiedGraph, Edge, RelationType
from src.trust.build_trust_graph import build_trust_inputs
from src.trust.model import EvAGNNTrustModel


@dataclass
class RobustnessResult:
    burst_sizes: list[int]
    prob_bot_with_trust: list[float]
    prob_bot_without_trust: list[float]

    def burst_to_flip(self, probs: list[float], threshold: float = 0.5) -> "int | None":
        
        #Returns None if it never drops below threshold within the tested range.
        for b, p in zip(self.burst_sizes, probs):
            if p < threshold:
                return b
        return None

    def resistance_ratio(self) -> "float | None":

        flip_wt = self.burst_to_flip(self.prob_bot_with_trust)
        flip_wot = self.burst_to_flip(self.prob_bot_without_trust)
        if flip_wt is None or flip_wot is None or flip_wot == 0:
            return None
        return flip_wt / max(flip_wot, 1e-6)

    def kill_condition_met(self, drop_threshold: float = 0.15) -> bool:

        if len(self.burst_sizes) < 2:
            return False
        drop_with_trust = self.prob_bot_with_trust[0] - self.prob_bot_with_trust[-1]
        drop_without_trust = self.prob_bot_without_trust[0] - self.prob_bot_without_trust[-1]
        return (drop_without_trust - drop_with_trust) >= drop_threshold

    def summary(self) -> str:
        lines = ["burst_size | P(bot) w/ trust | P(bot) w/o trust"]
        for b, wt, wot in zip(self.burst_sizes, self.prob_bot_with_trust, self.prob_bot_without_trust):
            lines.append(f"{b:10d} | {wt:15.3f} | {wot:16.3f}")
        lines.append(f"\nKill condition met (endpoint-drop comparison): {self.kill_condition_met()}")
        flip_wt = self.burst_to_flip(self.prob_bot_with_trust)
        flip_wot = self.burst_to_flip(self.prob_bot_without_trust)
        lines.append(f"Burst size needed to flip to 'human' -- with trust: {flip_wt}, without trust: {flip_wot}")
        ratio = self.resistance_ratio()
        if ratio is not None:
            lines.append(f"Resistance ratio (bigger = more protective): {ratio:.2f}x")
        return "\n".join(lines)


def inject_camouflage_burst(
    graph: UnifiedGraph,
    attacker_id: str,
    n_edges: int,
    seed: int = 0,
) -> UnifiedGraph:
   
    #Returns a NEW UnifiedGraph (deep-copied edges list; nodes are shared references since we don't mutate them) with n_edges synthetic one-way follow edges added

    rng = random.Random(seed)
    new_graph = copy.copy(graph)
    new_graph.edges = list(graph.edges) 

    candidates = [
        uid for uid, node in graph.nodes.items()
        if uid != attacker_id and float(node.metadata.get("followers_count", 0) or 0) > 0
    ]
    
    candidates.sort(key=lambda uid: float(graph.nodes[uid].metadata.get("followers_count", 0) or 0), reverse=True)
    pool = candidates[:max(len(candidates) // 2, 1)] or candidates

    if not pool:
        raise ValueError("No valid injection targets found in graph.")

    targets = [rng.choice(pool) for _ in range(n_edges)]
    injected = [
        Edge(u=attacker_id, v=tgt, r=RelationType.FOLLOW, t=None, t_observed=False)
        for tgt in targets
    ]
    new_graph.edges = new_graph.edges + injected
    return new_graph


def run_robustness_experiment(
    graph: UnifiedGraph,
    model_with_trust: EvAGNNTrustModel,
    model_without_trust: EvAGNNTrustModel,
    attacker_id: str,
    burst_sizes: list[int] = (0, 5, 10, 25, 50, 100),
    seed: int = 0,
) -> RobustnessResult:
    
    prob_with_trust = []
    prob_without_trust = []

    model_with_trust.eval()
    model_without_trust.eval()
    for burst in burst_sizes:
        attacked_graph = inject_camouflage_burst(graph, attacker_id, burst, seed=seed) if burst > 0 else graph
        inputs = build_trust_inputs(attacked_graph)
        attacker_idx = inputs["node_id_to_idx"][attacker_id]

        with torch.no_grad():
            logits_wt, _ = model_with_trust(
                inputs["h0"], inputs["edge_index"], inputs["edge_features"],
                inputs["is_celebrity_target"], use_trust=True,
            )
            logits_wot, _ = model_without_trust(
                inputs["h0"], inputs["edge_index"], inputs["edge_features"],
                inputs["is_celebrity_target"], use_trust=False,
            )
        prob_with_trust.append(torch.sigmoid(logits_wt[attacker_idx]).item())
        prob_without_trust.append(torch.sigmoid(logits_wot[attacker_idx]).item())

    return RobustnessResult(
        burst_sizes=list(burst_sizes),
        prob_bot_with_trust=prob_with_trust,
        prob_bot_without_trust=prob_without_trust,
    )
