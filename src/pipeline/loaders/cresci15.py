
from __future__ import annotations

import csv
import json
import datetime as dt
from pathlib import Path

from src.pipeline.schema import Edge, RelationType, UserNode, UnifiedGraph


RELATION_MAP = {
    "friend": RelationType.FOLLOW,
    
}

_DEFAULT_AVATAR_URLS = {
    
    f"http://a0.twimg.com/sticky/default_profile_images/default_profile_{i}_normal.png"
    for i in range(7)
}


def _parse_created_at(raw) -> dt.datetime | None:
    
    if raw is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(raw), tz=dt.timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _is_default_avatar(profile_image_url: str | None) -> bool:
    if not profile_image_url:
        return True
    return profile_image_url in _DEFAULT_AVATAR_URLS


def load_cresci15(raw_dir: str | Path, dataset_prefix: str = "cresci15") -> UnifiedGraph:
    raw_dir = Path(raw_dir)

    node_file = raw_dir / "node.json"
    with open(node_file) as f:
        raw_nodes = json.load(f)

    nodes: dict[str, UserNode] = {}
    for rn in raw_nodes:
        uid_raw = rn.get("id")
        if uid_raw is None:
            continue
        uid = f"{dataset_prefix}:{uid_raw}"

        public_metrics = rn.get("public_metrics") or {}
        followers_count = public_metrics.get("followers_count", 0) or 0
        following_count = public_metrics.get("following_count", 0) or 0
        listed_count = public_metrics.get("listed_count", 0) or 0

        created_at = _parse_created_at(rn.get("created_at"))
        default_profile_image = _is_default_avatar(rn.get("profile_image_url"))
        username = rn.get("username") or ""

        nodes[uid] = UserNode(
            user_id=uid,
            dataset=dataset_prefix,
            label=None,  # filled from label.csv below
            created_at=created_at,
            metadata={
                "followers_count": followers_count,
                "following_count": following_count,
                "listed_count": listed_count,
                "default_profile_image": default_profile_image,
                
            },
            text_blob=username,
        )

    
    label_file = raw_dir / "label.csv"
    if label_file.exists():
        with open(label_file) as f:
            for row in csv.DictReader(f):
                uid = f"{dataset_prefix}:{row['id']}"
                if uid in nodes:
                    raw_label = row["label"].strip().lower()
                    nodes[uid].label = 0 if raw_label == "human" else 1

    
    split: dict[str, str] = {}
    split_file = raw_dir / "split.csv"
    if split_file.exists():
        with open(split_file) as f:
            for row in csv.DictReader(f):
                uid = f"{dataset_prefix}:{row['id']}"
                split[uid] = row["split"]

   
    edges: list[Edge] = []
    edge_file = raw_dir / "edge.csv"
    skipped_relations: set[str] = set()
    with open(edge_file) as f:
        for row in csv.DictReader(f):
            rel_raw = row["relation"].strip().lower()
            rel = RELATION_MAP.get(rel_raw)
            if rel is None:
                skipped_relations.add(rel_raw)
                continue
            u = f"{dataset_prefix}:{row['source_id']}"
            v = f"{dataset_prefix}:{row['target_id']}"
            edges.append(Edge(u=u, v=v, r=rel, t=None, t_observed=False))

    if skipped_relations:
        print(f"[loaders.cresci15] Skipped unmapped edge relations: {skipped_relations}")

    return UnifiedGraph(
        nodes=nodes,
        edges=edges,
        split=split,
        source_datasets=[dataset_prefix],
    )
