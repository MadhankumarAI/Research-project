
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.make_synthetic_data import make_synthetic_twibot
from src.pipeline.build_dataset import build
from src.trust.train import train_trust_model
from src.robustness.camouflage_injection import run_robustness_experiment


def run():
    tb_dir = Path("data/raw/tb20_synthetic")
    make_synthetic_twibot(tb_dir, dataset="tb20", n_users=50)
    graph = build("tb20", str(tb_dir), "data/processed/tb20_synthetic.pkl")

    print("\n--- Training model WITH trust-weighting ---")
    model_with_trust, _ = train_trust_model(graph, epochs=100, use_trust=True, verbose=False, seed=0)

    print("--- Training model WITHOUT trust-weighting (ablation) ---")
    model_without_trust, _ = train_trust_model(graph, epochs=100, use_trust=False, verbose=False, seed=0)

    attacker_id = "tb20:1000"  
    print(f"\nRunning camouflage-injection attack on {attacker_id}...")
    result = run_robustness_experiment(
        graph=graph,
        model_with_trust=model_with_trust,
        model_without_trust=model_without_trust,
        attacker_id=attacker_id,
        burst_sizes=[0, 5, 10, 25, 50, 100],
    )

    print()
    print(result.summary())
    print("\nRobustness experiment smoke test completed.")


if __name__ == "__main__":
    run()
