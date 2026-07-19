
#Run: python -m tests.test_robustness_experiment_v2

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.make_synthetic_data import make_synthetic_twibot_with_signal
from src.pipeline.build_dataset import build
from src.trust.train import train_trust_model
from src.robustness.camouflage_injection import run_robustness_experiment


def run():
    tb_dir = Path("data/raw/tb20_signal_synthetic")
    ids = make_synthetic_twibot_with_signal(
        tb_dir, dataset="tb20",
        n_humans=150, n_celebrities=20, n_quiet_bots=50, n_camouflage_bots=50,
        camouflage_burst_size=15, seed=0,
    )
    graph = build("tb20", str(tb_dir), "data/processed/tb20_signal_synthetic.pkl")

    print("\n--- Training model WITH trust-weighting ---")
    model_with_trust, hist_wt = train_trust_model(graph, epochs=150, use_trust=True, verbose=True, seed=0)

    print("\n--- Training model WITHOUT trust-weighting (ablation) ---")
    model_without_trust, hist_wot = train_trust_model(graph, epochs=150, use_trust=False, verbose=True, seed=0)

    test_targets = ids["camouflage_bot_ids"][:5]
    print(f"\nRunning camouflage-injection attack on {len(test_targets)} camouflage-bot accounts...")

    for attacker_id_raw in test_targets:
        attacker_id = f"tb20:{attacker_id_raw}"
        result = run_robustness_experiment(
            graph=graph,
            model_with_trust=model_with_trust,
            model_without_trust=model_without_trust,
            attacker_id=attacker_id,
            burst_sizes=[0, 5, 10, 25, 50],
        )
        print(f"\n=== Attacker {attacker_id} ===")
        print(result.summary())

    print("\nRobustness experiment v2 (designed-signal data) completed.")


if __name__ == "__main__":
    run()
