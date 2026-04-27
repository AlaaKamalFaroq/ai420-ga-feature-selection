import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
from ga_core import run_ga

SEEDS = [200, 201, 202, 203, 204]

CROSSOVERS = ["single_point", "two_point", "uniform"]
MUTATIONS  = ["bit_flip", "swap", "inversion"]

results = []

print("=" * 70)
print("  PAPER 4 EXPERIMENTS (WAAD)")
print("  Crossover & Mutation Comparison")
print("=" * 70)

for cx in CROSSOVERS:
    for mut in MUTATIONS:

        print("\n" + "-" * 70)
        print(f"Experiment: Crossover={cx} | Mutation={mut}")
        print("-" * 70)

        for i, seed in enumerate(SEEDS):
            print(f"\n[Run {i+1}/5] seed={seed}")
            np.random.seed(seed)

            result = run_ga(
                selection_method="rank",
                crossover_method=cx,
                mutation_method=mut,
                verbose=False
            )

            results.append({
                "crossover": cx,
                "mutation": mut,
                "seed": seed,
                "best_fitness": float(result["best_fitness"]),
                "num_features": int(np.sum(result["best_individual"]))
            })

            print(f"Best: {result['best_fitness']:.4f}")

# save results
out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'paper4.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, 'w') as f:
    json.dump(results, f, indent=4)

print("\nSaved →", out_path)