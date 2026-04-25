"""
run_paper4.py  –  Paper 4: Roulette Wheel + Uniform Crossover + Inversion Mutation
====================================================================================
Configuration : Roulette Wheel Selection + Uniform Crossover + Inversion Mutation
Runs          : 5  (seeds 42, 7, 13, 99, 2025)
Dataset       : Breast Cancer Wisconsin (30 features, 569 samples)
Classifier    : KNN (k=5)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import json
import csv
from src.ga_core import run_ga
from src.config import NUM_RUNS, SEEDS

# ── Fixed seeds for reproducibility ──────────────────────────────────────────
SEEDS = [42, 7, 13, 99, 2025]

print("=" * 60)
print("  PAPER 4 EXPERIMENTS")
print("  Selection : Roulette Wheel")
print("  Crossover : Uniform")
print("  Mutation  : Inversion")
print("=" * 60)

results = []

for run_id, seed in enumerate(SEEDS):
    print(f"\n[Run {run_id + 1}/5]  seed={seed}")
    np.random.seed(seed)

    result = run_ga(
        selection_method="roulette",
        crossover_method="uniform",
        mutation_method="inversion",
        verbose=True
    )

    record = {
        "run":          run_id + 1,
        "seed":         seed,
        "best_fitness": float(result["best_fitness"]),
        "num_features": int(np.sum(result["best_individual"])),
        "history_best": [float(v) for v in result["history_best"]],
    }
    results.append(record)

    print(f"  → Best Fitness : {record['best_fitness']:.4f}")
    print(f"  → Features Used: {record['num_features']} / 30")

# ── Summary statistics ───────────────────────────────────────────────────────
fitness_vals = [r["best_fitness"]  for r in results]
feature_vals = [r["num_features"]  for r in results]

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Avg Fitness  : {np.mean(fitness_vals):.4f}")
print(f"  Std Fitness  : {np.std(fitness_vals):.4f}")
print(f"  Best Fitness : {np.max(fitness_vals):.4f}")
print(f"  Worst Fitness: {np.min(fitness_vals):.4f}")
print(f"  Avg Features : {np.mean(feature_vals):.1f}")
print(f"  Reduction    : {(1 - np.mean(feature_vals)/30)*100:.1f}%")
print("=" * 60)

# ── Save results ─────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(out_dir, exist_ok=True)

json_path = os.path.join(out_dir, 'results_paper4.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nJSON results saved → {json_path}")

csv_path = os.path.join(out_dir, 'results_paper4.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["run", "seed", "best_fitness", "num_features"])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r[k] for k in ["run", "seed", "best_fitness", "num_features"]})
print(f"CSV results saved → {csv_path}")