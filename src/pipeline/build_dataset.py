

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.pipeline.schema import UnifiedGraph, Edge, RelationType
from src.pipeline.loaders.twibot import load_twibot
from src.pipeline.loaders.cresci15 import load_cresci15
from src.pipeline.time_proxy import build_timestamped_pair_index, estimate_follow_time_proxy
from src.pipeline.features import build_node_features, RobertaTextEncoder


def fill_follow_time_proxies(graph: UnifiedGraph) -> UnifiedGraph:
    #Replaces follow edges' t=None with a proxy timestamp where possible.
    
    pair_index = build_timestamped_pair_index(graph.edges)
    new_edges = []
    for e in graph.edges:
        if e.r == RelationType.FOLLOW and e.t is None:
            proxy_t = estimate_follow_time_proxy(e.u, e.v, graph.nodes, pair_index)
            new_edges.append(Edge(u=e.u, v=e.v, r=e.r, t=proxy_t, t_observed=False))
        else:
            new_edges.append(e)
    graph.edges = new_edges
    return graph


def build(
    dataset: str,
    raw_dir: str,
    out_path: str,
    with_text_embeddings: bool = False,
) -> UnifiedGraph:
    if dataset in ("tb20", "tb22"):
        graph = load_twibot(raw_dir, dataset_prefix=dataset)
    elif dataset == "cresci15":
        graph = load_cresci15(raw_dir, dataset_prefix=dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    graph = fill_follow_time_proxies(graph)

    encoder = RobertaTextEncoder() if with_text_embeddings else None
    build_node_features(graph.nodes, text_encoder=encoder)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)

    print(f"[{dataset}] {graph.summary()}")
    return graph


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tb20", "tb22", "cresci15"])
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_text_embeddings", action="store_true")
    args = ap.parse_args()
    build(args.dataset, args.raw_dir, args.out, args.with_text_embeddings)
