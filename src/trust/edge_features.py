
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.pipeline.schema import Edge, UserNode, RelationType


@dataclass
class EdgeTrustFeatures:
    reciprocity: float  # 1.0 if a reverse edge exists between v->u, else 0.0
    behavioral_similarity: float  # cosine sim of x_u, x_v in [-1, 1]
    degree_disparity: float # log(followers_v + 1) - log(followers_u + 1), signed
    is_celebrity_target: float  # 1.0 if v's follower count is in the top percentile (heuristic threshold)
    relation_onehot: np.ndarray  # one-hot over RelationType, len == len(RelationType)
    temporal_confidence: float  # 1.0 if t_observed else 0.3 (soft discount, not zero — proxy still has some info)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            np.array([
                self.reciprocity,
                self.behavioral_similarity,
                self.degree_disparity,
                self.is_celebrity_target,
                self.temporal_confidence,
            ], dtype=np.float32),
            self.relation_onehot.astype(np.float32),
        ])


_RELATIONS = list(RelationType)
_REL_INDEX = {r: i for i, r in enumerate(_RELATIONS)}
FEATURE_DIM = 5 + len(_RELATIONS)

CELEBRITY_FOLLOWER_THRESHOLD = 50_000


def _reverse_edge_exists(u: str, v: str, reverse_index: dict[tuple[str, str], bool]) -> bool:
    return reverse_index.get((v, u), False)


def build_reverse_index(edges: list[Edge]) -> dict[tuple[str, str], bool]:
  #Precompute (u,v) -> True for every edge, so reciprocity checks are O(1).
    idx: dict[tuple[str, str], bool] = {}
    for e in edges:
        idx[(e.u, e.v)] = True
    return idx


def _cosine_sim(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def compute_edge_trust_features(
    e: Edge,
    nodes: dict[str, UserNode],
    reverse_index: dict[tuple[str, str], bool],
) -> EdgeTrustFeatures:
    u_node = nodes.get(e.u)
    v_node = nodes.get(e.v)

    reciprocity = 1.0 if _reverse_edge_exists(e.u, e.v, reverse_index) else 0.0
    behavioral_similarity = _cosine_sim(
        u_node.x if u_node else None,
        v_node.x if v_node else None,
    )

    followers_u = float((u_node.metadata.get("followers_count", 0) if u_node else 0) or 0)
    followers_v = float((v_node.metadata.get("followers_count", 0) if v_node else 0) or 0)
    degree_disparity = float(np.log1p(followers_v) - np.log1p(followers_u))
    is_celebrity_target = 1.0 if followers_v >= CELEBRITY_FOLLOWER_THRESHOLD else 0.0

    onehot = np.zeros(len(_RELATIONS), dtype=np.float32)
    onehot[_REL_INDEX[e.r]] = 1.0

    temporal_confidence = 1.0 if e.t_observed else 0.3

    return EdgeTrustFeatures(
        reciprocity=reciprocity,
        behavioral_similarity=behavioral_similarity,
        degree_disparity=degree_disparity,
        is_celebrity_target=is_celebrity_target,
        relation_onehot=onehot,
        temporal_confidence=temporal_confidence,
    )
