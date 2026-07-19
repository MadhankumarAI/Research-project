
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.make_synthetic_data import make_synthetic_twibot, make_synthetic_cresci15
from src.pipeline.build_dataset import build


def run():
    tmp_root = Path("data/raw")
    tb_dir = tmp_root / "tb20_synthetic"
    cresci_dir = tmp_root / "cresci15_synthetic"

    print("Generating synthetic data...")
    make_synthetic_twibot(tb_dir, dataset="tb20", n_users=50)
    make_synthetic_cresci15(cresci_dir, n_users=40)

    print("\nRunning TwiBot-20 pipeline...")
    tb_graph = build("tb20", str(tb_dir), "data/processed/tb20_synthetic.pkl")

    print("\nRunning Cresci-2015 pipeline...")
    cresci_graph = build("cresci15", str(cresci_dir), "data/processed/cresci15_synthetic.pkl")

    assert len(tb_graph.nodes) == 50, f"expected 50 nodes, got {len(tb_graph.nodes)}"
    assert all(n.x is not None for n in tb_graph.nodes.values()), "all nodes should have x populated"
    assert any(e.r.value == "follow" for e in tb_graph.edges), "should have follow edges"

    bot_id = "tb20:1000"
    follow_out = [e for e in tb_graph.edges if e.u == bot_id and e.r.value == "follow"]
    assert len(follow_out) >= 10, f"expected injected camouflage burst, got {len(follow_out)} follow edges"
    print(f"\nCamouflage burst check: user {bot_id} has {len(follow_out)} outgoing follow edges. OK.")

    follow_edges = [e for e in tb_graph.edges if e.r.value == "follow"]
    n_with_proxy = sum(1 for e in follow_edges if e.t is not None)
    print(f"Follow edges with a time proxy filled in: {n_with_proxy}/{len(follow_edges)}")

    assert len(cresci_graph.nodes) == 40, f"expected 40 cresci nodes, got {len(cresci_graph.nodes)}"
    labels = {n.label for n in cresci_graph.nodes.values()}
    assert labels == {0, 1}, f"expected both genuine(0) and fake(1) labels, got {labels}"

    print("\nAll sanity checks passed.")
    print("TB20 summary:", tb_graph.summary())
    print("Cresci15 summary:", cresci_graph.summary())


if __name__ == "__main__":
    run()
