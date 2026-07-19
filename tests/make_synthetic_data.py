
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path


def _fake_unix_timestamp(rng: random.Random) -> int:
    
    start = int(time.mktime((2010, 1, 1, 0, 0, 0, 0, 0, 0)))
    end = int(time.mktime((2023, 1, 1, 0, 0, 0, 0, 0, 0)))
    return rng.randint(start, end)


def _make_users(n_users: int, id_offset: int, rng: random.Random) -> list[dict]:
    users = []
    for i in range(n_users):
        uid = str(id_offset + i)
        users.append({
            "id": uid,
            "username": f"user_{uid}",
            "description": rng.choice([
                "Just a regular person tweeting about my day.",
                "Crypto news and updates, follow for alpha!",
                "Sports fan, dog lover, coffee addict.",
                "",
            ]),
            "created_at": _fake_unix_timestamp(rng),
            "verified": rng.random() < 0.05,
            "protected": rng.random() < 0.02,
            "profile_image_url": "" if rng.random() < 0.1 else "https://example.com/avatar.jpg",
            "public_metrics": {
                "followers_count": rng.randint(0, 5000),
                "following_count": rng.randint(0, 3000),
                "listed_count": rng.randint(0, 50),
            },
        })
    return users


def make_synthetic_twibot(out_dir: str | Path, dataset: str, n_users: int = 50, seed: int = 0) -> None:
    
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = _make_users(n_users, id_offset=1000, rng=rng)
    ids = [u["id"] for u in users]

    node_filename = "node.json" if dataset == "tb20" else "user.json"
    with open(out_dir / node_filename, "w") as f:
        json.dump(users, f)

    labels, splits = [], []
    for uid in ids:
        label = "bot" if rng.random() < 0.3 else "human"
        labels.append({"id": uid, "label": label})
        splits.append({"id": uid, "split": rng.choice(["train", "train", "val", "test"])})

    with open(out_dir / "label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label"])
        w.writeheader()
        w.writerows(labels)
    with open(out_dir / "split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split"])
        w.writeheader()
        w.writerows(splits)

    
    follow_relations = ["friend"] if dataset == "tb20" else ["followers", "following"]
    edges = []
    for _ in range(n_users * 3):
        src, tgt = rng.sample(ids, 2)
        edges.append({"source_id": src, "relation": rng.choice(follow_relations), "target_id": tgt})
   
    for src in ids[:5]:
        edges.append({"source_id": src, "relation": "post", "target_id": f"tweet_{src}_0"})

    
    bot_id = ids[0]
    for tgt in ids[1:11]:
        edges.append({"source_id": bot_id, "relation": follow_relations[0], "target_id": tgt})

    with open(out_dir / "edge.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "relation", "target_id"])
        w.writeheader()
        w.writerows(edges)


def make_synthetic_cresci15(out_dir: str | Path, n_users: int = 40, seed: int = 1) -> None:

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = _make_users(n_users, id_offset=2000, rng=rng)
    ids = [u["id"] for u in users]

    with open(out_dir / "node.json", "w") as f:
        json.dump(users, f)

    labels, splits = [], []
    for i, uid in enumerate(ids):
        
        label = "human" if i < n_users // 2 else "bot"
        labels.append({"id": uid, "label": label})
        splits.append({"id": uid, "split": rng.choice(["train", "train", "val", "test"])})

    with open(out_dir / "label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label"])
        w.writeheader()
        w.writerows(labels)
    with open(out_dir / "split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split"])
        w.writeheader()
        w.writerows(splits)

    edges = []
    for _ in range(n_users * 3):
        src, tgt = rng.sample(ids, 2)
        edges.append({"source_id": src, "relation": "friend", "target_id": tgt})
    for src in ids[:5]:
        edges.append({"source_id": src, "relation": "post", "target_id": f"tweet_{src}_0"})

    with open(out_dir / "edge.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "relation", "target_id"])
        w.writeheader()
        w.writerows(edges)


def make_synthetic_twibot_with_signal(
    out_dir: str | Path,
    dataset: str = "tb20",
    n_humans: int = 150,
    n_celebrities: int = 20,
    n_quiet_bots: int = 50,
    n_camouflage_bots: int = 50,
    camouflage_burst_size: int = 15,
    seed: int = 0,
) -> dict:
   
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_pop(n, id_start, followers_range, following_range, listed_range, recent_days_range):
        pop = []
        for i in range(n):
            uid = str(id_start + i)
            pop.append({
                "id": uid,
                "username": f"user_{uid}",
                "description": "",
                "created_at": int(time.time()) - rng.randint(*recent_days_range) * 86400,
                "verified": False,
                "protected": False,
                "profile_image_url": "https://example.com/avatar.jpg",
                "public_metrics": {
                    "followers_count": rng.randint(*followers_range),
                    "following_count": rng.randint(*following_range),
                    "listed_count": rng.randint(*listed_range),
                },
            })
        return pop

    humans = make_pop(n_humans, 5000, (10, 500), (10, 300), (0, 5), (100, 3000))
    celebrities = make_pop(n_celebrities, 6000, (50_000, 500_000), (10, 300), (500, 5000), (500, 4000))
    
    quiet_bots = make_pop(n_quiet_bots, 7000, (10, 500), (10, 300), (0, 5), (100, 3000))
    camouflage_bots = make_pop(n_camouflage_bots, 8000, (10, 500), (10, 300), (0, 5), (100, 3000))

    all_users = humans + celebrities + quiet_bots + camouflage_bots
    human_ids = [u["id"] for u in humans + celebrities]
    celeb_ids = [u["id"] for u in celebrities]
    quiet_bot_ids = [u["id"] for u in quiet_bots]
    camo_bot_ids = [u["id"] for u in camouflage_bots]

    node_filename = "node.json" if dataset == "tb20" else "user.json"
    with open(out_dir / node_filename, "w") as f:
        json.dump(all_users, f)

    labels, splits = [], []
    for uid in human_ids:
        labels.append({"id": uid, "label": "human"})
    for uid in quiet_bot_ids + camo_bot_ids:
        labels.append({"id": uid, "label": "bot"})
    for uid in [u["id"] for u in all_users]:
        splits.append({"id": uid, "split": rng.choice(["train", "train", "train", "val", "test"])})

    with open(out_dir / "label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label"])
        w.writeheader()
        w.writerows(labels)
    with open(out_dir / "split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split"])
        w.writeheader()
        w.writerows(splits)

    follow_rel = "friend" if dataset == "tb20" else "followers"
    edges = []

    # humans <-> humans: mutual follows (high reciprocity), the "normal" pattern
    for uid in human_ids:
        n_friends = rng.randint(3, 10)
        friends = rng.sample([h for h in human_ids if h != uid], min(n_friends, len(human_ids) - 1))
        for f in friends:
            edges.append({"source_id": uid, "relation": follow_rel, "target_id": f})
            edges.append({"source_id": f, "relation": follow_rel, "target_id": uid})  # reciprocated

    # quiet bots <-> quiet bots: a mutually-connected cluster among themselves,
   
    for uid in quiet_bot_ids:
        if len(quiet_bot_ids) > 1:
            n_friends = rng.randint(2, 5)
            friends = rng.sample([b for b in quiet_bot_ids if b != uid], min(n_friends, len(quiet_bot_ids) - 1))
            for f in friends:
                edges.append({"source_id": uid, "relation": follow_rel, "target_id": f})
                edges.append({"source_id": f, "relation": follow_rel, "target_id": uid})

    # camouflage bots -> celebrities/humans: the hard case. One-way, no reciprocity, biased toward high-follower celebrity targets.
    for uid in camo_bot_ids:
        targets = rng.sample(celeb_ids, min(camouflage_burst_size // 2, len(celeb_ids))) + \
                  rng.sample(human_ids, min(camouflage_burst_size // 2, len(human_ids)))
        for tgt in targets:
            edges.append({"source_id": uid, "relation": follow_rel, "target_id": tgt})
            
    with open(out_dir / "edge.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "relation", "target_id"])
        w.writeheader()
        w.writerows(edges)

    return {
        "human_ids": human_ids,
        "celebrity_ids": celeb_ids,
        "quiet_bot_ids": quiet_bot_ids,
        "camouflage_bot_ids": camo_bot_ids,
    }


if __name__ == "__main__":
    make_synthetic_twibot("data/raw/tb20_synthetic", dataset="tb20")
    make_synthetic_twibot("data/raw/tb22_synthetic", dataset="tb22")
    make_synthetic_cresci15("data/raw/cresci15_synthetic")
    print("Synthetic data generated (matching verified real schema).")
