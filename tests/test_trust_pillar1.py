
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from tests.make_synthetic_data import make_synthetic_twibot
from src.pipeline.build_dataset import build
from src.trust.build_trust_graph import build_trust_inputs, compute_tau
from src.trust.trust_module import TrustScorer
from src.trust.aggregation import HeterophilyAwareAggregation
from src.trust.edge_features import compute_edge_trust_features, build_reverse_index


def run():
    tb_dir = Path("data/raw/tb20_synthetic")
    make_synthetic_twibot(tb_dir, dataset="tb20", n_users=50)
    graph = build("tb20", str(tb_dir), "data/processed/tb20_synthetic.pkl")

    print("\nBuilding trust inputs...")
    inputs = build_trust_inputs(graph)
    print(f"  h0 shape: {inputs['h0'].shape}")
    print(f"  edge_index shape: {inputs['edge_index'].shape}")
    print(f"  edge_features shape: {inputs['edge_features'].shape}")

    torch.manual_seed(0)
    trust_scorer = TrustScorer(hidden_dim=16)
    tau = compute_tau(trust_scorer, inputs["edge_features"])
    assert tau.shape[0] == inputs["edge_index"].shape[1]
    assert torch.all((tau >= 0) & (tau <= 1)), "tau must be in [0,1]"
    print(f"\nτ computed for {tau.shape[0]} edges. Range: [{tau.min():.3f}, {tau.max():.3f}], mean: {tau.mean():.3f}")

    in_dim = inputs["h0"].shape[1]
    agg_layer = HeterophilyAwareAggregation(in_dim=in_dim, out_dim=32)
    out = agg_layer(inputs["h0"], inputs["edge_index"], tau, inputs["is_celebrity_target"])
    expected_dim = 32 * 2  # node_repr concat evidence_proj
    assert out.shape == (inputs["h0"].shape[0], expected_dim), f"unexpected output shape {out.shape}"
    print(f"Aggregation output shape: {out.shape} (== [n_nodes, out_dim*2]). OK.")

   
    reverse_index = build_reverse_index(graph.edges)
    bot_id = "tb20:1000"
    burst_feats = [
        compute_edge_trust_features(e, graph.nodes, reverse_index)
        for e in graph.edges if e.u == bot_id and e.r.value == "follow"
    ]
    all_follow_feats = [
        compute_edge_trust_features(e, graph.nodes, reverse_index)
        for e in graph.edges if e.r.value == "follow"
    ]
    burst_reciprocity = sum(f.reciprocity for f in burst_feats) / len(burst_feats)
    overall_reciprocity = sum(f.reciprocity for f in all_follow_feats) / len(all_follow_feats)
    print(f"\nCamouflage burst mean reciprocity: {burst_reciprocity:.3f}")
    print(f"Overall follow-edge mean reciprocity: {overall_reciprocity:.3f}")
    print("(Expect burst <= overall on average, since injected edges are deliberately one-way to strangers.)")

    print("\nAll Pillar 1 smoke checks passed.")


if __name__ == "__main__":
    run()
