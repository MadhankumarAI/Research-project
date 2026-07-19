
from __future__ import annotations

import datetime as dt
from typing import Optional

from src.pipeline.schema import Edge, RelationType, UserNode


def estimate_follow_time_proxy(
    u: str,
    v: str,
    user_nodes: dict[str, UserNode],
    timestamped_edges_by_pair: dict[tuple[str, str], dt.datetime],
) -> Optional[dt.datetime]:
    
    candidates = []

    earliest_interaction = timestamped_edges_by_pair.get((u, v))
    if earliest_interaction is not None:
        candidates.append(earliest_interaction)

    u_node = user_nodes.get(u)
    if u_node is not None and u_node.created_at is not None:
        candidates.append(u_node.created_at)

    if not candidates:
        return None
    return min(candidates)


def build_timestamped_pair_index(edges: list[Edge]) -> dict[tuple[str, str], dt.datetime]:
   
    index: dict[tuple[str, str], dt.datetime] = {}
    for e in edges:
        if not e.t_observed or e.t is None:
            continue
        key = (e.u, e.v)
        if key not in index or e.t < index[key]:
            index[key] = e.t
    return index
