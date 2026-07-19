

#Run: python -m tests.test_robustness_multiseed

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import statistics

from tests.make_synthetic_data import make_synthetic_twibot_with_signal
from src.pipeline.build_dataset import build
from src.trust.train import train_trust_model
from src.robustness.camouflage_injection import run_robustness_experiment


def run(n_seeds: int = 5, n_targets_per_seed: int = 5):
    all_ratios = []
    all_flip_wt = []
    all_flip_wot = []
    never_fooled_wt = 0  
    no_benefit_count = 0  

    for seed in range(n_seeds):
        tb_dir = Path(f"data/raw/tb20_signal_seed{seed}")
        ids = make_synthetic_twibot_with_signal(
            tb_dir, dataset="tb20",
            n_humans=150, n_celebrities=20, n_quiet_bots=50, n_camouflage_bots=50,
            camouflage_burst_size=15, seed=seed,
        )
        graph = build("tb20", str(tb_dir), f"data/processed/tb20_signal_seed{seed}.pkl")

        model_with_trust, _ = train_trust_model(graph, epochs=150, use_trust=True, verbose=False, seed=seed)
        model_without_trust, _ = train_trust_model(graph, epochs=150, use_trust=False, verbose=False, seed=seed)

        targets = ids["camouflage_bot_ids"][:n_targets_per_seed]
        for attacker_id_raw in targets:
            attacker_id = f"tb20:{attacker_id_raw}"
            result = run_robustness_experiment(
                graph=graph,
                model_with_trust=model_with_trust,
                model_without_trust=model_without_trust,
                attacker_id=attacker_id,
                burst_sizes=[0, 5, 10, 25, 50],
            )
            flip_wt = result.burst_to_flip(result.prob_bot_with_trust)
            flip_wot = result.burst_to_flip(result.prob_bot_without_trust)
            all_flip_wt.append(flip_wt)
            all_flip_wot.append(flip_wot)

            ratio = result.resistance_ratio()
            if ratio is None:
                never_fooled_wt += 1
                continue
            all_ratios.append(ratio)
            if abs(ratio - 1.0) < 1e-6:
                no_benefit_count += 1

        print(f"seed {seed}: done ({len(targets)} targets)")

    n_total = n_seeds * n_targets_per_seed
    print(f"\n=== Multi-seed results ({n_total} total attacker instances across {n_seeds} seeds) ===")
    print(f"Instances where trust model was never fooled within tested range (burst<=50): {never_fooled_wt}/{n_total}")
    print(f"Instances with resistance ratio == 1.0 (no measurable benefit): {no_benefit_count}/{len(all_ratios)}")

    if all_ratios:
        print(f"\nResistance ratio distribution (n={len(all_ratios)}):")
        print(f"  mean   = {statistics.mean(all_ratios):.2f}")
        print(f"  median = {statistics.median(all_ratios):.2f}")
        print(f"  stdev  = {statistics.stdev(all_ratios) if len(all_ratios) > 1 else 0:.2f}")
        print(f"  min    = {min(all_ratios):.2f}")
        print(f"  max    = {max(all_ratios):.2f}")
        pct_above_1 = sum(1 for r in all_ratios if r > 1.0) / len(all_ratios) * 100
        print(f"  % of instances with ratio > 1.0 (any benefit at all): {pct_above_1:.0f}%")
    else:
        print("\nNo instances produced a valid resistance ratio (both models either never "
              "flipped, or flipped at burst=0 in all cases) -- cannot draw a conclusion.")

    


if __name__ == "__main__":
    run()
