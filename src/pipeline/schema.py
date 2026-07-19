
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import datetime as dt


class RelationType(str, Enum):
    FOLLOW = "follow"       # unifies dataset-specific strings: 'friend' (TB20/Cresci-15), 'followers'/'following' (TB22)
    MENTION = "mention"
    RETWEET = "retweet"
    REPLY = "reply"
    QUOTE = "quote"


# The reference preprocessing code for ALL THREE datasets (TwiBot-20, TwiBot-22, Cresci-2015) builds graph edges from FOLLOW-type relations ONLY:
#TwiBot-20 / Cresci-2015 edge.csv: relation == 'friend' -> follow edge
#(relation == 'post' is a user->TWEET edge, used only to gather tweet text, not part of the user-user social graph; at least one other relation string exists in TB20/Cresci edge.csv that we have not yet identified)
#TwiBot-22 edge.csv: relation in {'followers', 'following'} -> follow edge


TIMESTAMPED_RELATIONS: set[RelationType] = set()


@dataclass(frozen=True)
class Edge:
    """e = (u, v, r, t)"""
    u: str # source user id 
    v: str  # target user id 
    r: RelationType
    t: Optional[dt.datetime]# timestamp 
    t_observed: bool # True = real event time, False = estimated proxy

    def __post_init__(self):
        if self.r in TIMESTAMPED_RELATIONS and self.t is not None and not self.t_observed:
            raise ValueError(
                f"Relation {self.r} should carry an OBSERVED timestamp, got t_observed=False. "
                f"This likely indicates a bug in the loader."
            )
        if self.r == RelationType.FOLLOW and self.t_observed:
            raise ValueError(
                "Follow edges should never be marked t_observed=True until we've confirmed "
                "the raw files actually contain follow timestamps (see notation.md). "
                "If you've confirmed this, update TIMESTAMPED_RELATIONS and this check together."
            )


@dataclass
class UserNode:
    
    user_id: str # global id
    dataset: str # "tb20" | "tb22" | "cresci15"
    label: Optional[int] # 1 = bot, 0 = human, None = unlabeled
    created_at: Optional[dt.datetime]   
    metadata: dict = field(default_factory=dict)  
    text_blob: str = ""  # concatenated bio/tweets for embedding
    # populated later by the feature extraction stage
    x: Optional["list[float]"] = None   


@dataclass
class UnifiedGraph:
    
    nodes: dict[str, UserNode]  # user_id-UserNode
    edges: list[Edge]
    split: dict[str, str] # user_id- "train" | "val" | "test" | "support"
    source_datasets: list[str]

    def summary(self) -> dict:
        rel_counts: dict[str, int] = {}
        for e in self.edges:
            rel_counts[e.r.value] = rel_counts.get(e.r.value, 0) + 1
        n_labeled = sum(1 for n in self.nodes.values() if n.label is not None)
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_labeled": n_labeled,
            "edges_by_relation": rel_counts,
            "source_datasets": self.source_datasets,
        }
