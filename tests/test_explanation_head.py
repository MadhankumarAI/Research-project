
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.make_synthetic_data import make_synthetic_twibot
from src.pipeline.build_dataset import build
from src.trust.train import train_trust_model
from src.explain.explanation_head import explain_batch


def run():
    tb_dir = Path("data/raw/tb20_synthetic")
    make_synthetic_twibot(tb_dir, dataset="tb20", n_users=50)
    graph = build("tb20", str(tb_dir), "data/processed/tb20_synthetic.pkl")

    print("\nTraining model...")
    model, history = train_trust_model(graph, epochs=100, verbose=True)

    inputs = history["inputs"]
    model.eval()
    import torch
    with torch.no_grad():
        logits, tau = model(
            inputs["h0"], inputs["edge_index"], inputs["edge_features"],
            inputs["is_celebrity_target"], use_trust=True,
        )

    targets = ["tb20:1000", inputs["node_ids"][5], inputs["node_ids"][20]]
    explanations = explain_batch(
        graph=graph,
        node_ids=inputs["node_ids"],
        edge_index=inputs["edge_index"],
        tau=tau,
        edges_list=inputs["edges"],
        logits=logits,
        node_id_to_idx=inputs["node_id_to_idx"],
        user_ids_to_explain=targets,
    )

    for uid, exp in explanations.items():
        print()
        print(exp.to_report())

    assert "tb20:1000" in explanations
    assert explanations["tb20:1000"].n_low_trust_edges >= 0  
    print("\n\nExplanation head smoke test passed.")


if __name__ == "__main__":
    run()
