"""
run_paper3.py  –  Member 3 (Aliyaa): Selection Operators Experiments
=====================================================================
Configuration : Rank Selection + Two-Point Crossover + Bit-Flip Mutation
Runs          : 5  (seeds 200-204, stored in seeds.txt)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
import csv
from ga_core import run_ga
from config import NUM_RUNS

# ── Fixed seeds for reproducibility (stored for reporting) ──────────────────
SEEDS = [200, 201, 202, 203, 204]

# Save seeds to file
seeds_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'seeds.txt')
os.makedirs(os.path.dirname(seeds_path), exist_ok=True)
with open(seeds_path, 'a') as f:
    f.write("=== Paper 3 Seeds (Member: Aliyaa) ===\n")
    for s in SEEDS:
        f.write(f"{s}\n")
    f.write("\n")

print("=" * 60)
print("  PAPER 3 EXPERIMENTS")
print("  Selection: Rank | Crossover: Two-Point | Mutation: Bit-Flip")
print("=" * 60)

results = []

for run_id, seed in enumerate(SEEDS):
    print(f"\n[Run {run_id + 1}/5]  seed={seed}")
    np.random.seed(seed)

    result = run_ga(
        selection_method="rank",
        crossover_method="two_point",
        mutation_method="bit_flip",
        verbose=True
    )

    record = {
        "run":           run_id + 1,
        "seed":          seed,
        "best_fitness":  float(result["best_fitness"]),
        "num_features":  int(np.sum(result["best_individual"])),
        "history_best":  [float(v) for v in result["history_best"]],
    }
    results.append(record)

    print(f"  → Best Fitness : {record['best_fitness']:.4f}")
    print(f"  → Features Used: {record['num_features']}")


# ── Summary statistics ───────────────────────────────────────────────────────
fitness_vals  = [r["best_fitness"]  for r in results]
feature_vals  = [r["num_features"]  for r in results]

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Avg Fitness  : {np.mean(fitness_vals):.4f}")
print(f"  Std Fitness  : {np.std(fitness_vals):.4f}")
print(f"  Best Fitness : {np.max(fitness_vals):.4f}")
print(f"  Worst Fitness: {np.min(fitness_vals):.4f}")
print(f"  Avg Features : {np.mean(feature_vals):.1f}")
print("=" * 60)

# ── Save JSON results ────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(out_dir, exist_ok=True)

json_path = os.path.join(out_dir, 'results_paper3.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nJSON results saved → {json_path}")

# ── Save CSV results ─────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, 'results_paper3.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["run", "seed", "best_fitness", "num_features"])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r[k] for k in ["run", "seed", "best_fitness", "num_features"]})
print(f"CSV results saved → {csv_path}")