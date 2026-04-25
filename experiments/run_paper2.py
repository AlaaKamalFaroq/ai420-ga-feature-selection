import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import json
from src.ga_core import run_ga
from src.config import NUM_RUNS, SEEDS

results = []

# Run Experiments
for run_id in range(NUM_RUNS):

    print(f"Run {run_id + 1}")

    # Reproducibility per run
    np.random.seed(SEEDS[run_id])

    # Run GA
    result = run_ga(
        selection_method="roulette",
        crossover_method="single_point",
        mutation_method="bit_flip",
        verbose=True
    )

    # Store results
    results.append({
        "Seed": SEEDS[run_id],
        "run": run_id + 1,
        "fitness": float(result["best_fitness"]),
        "num_features": int(np.sum(result["best_individual"]))
    })


# Show Results
print("\nAll Results:\nSelection: Roulette, Crossover: Sing Point, Mutation: Bit Flip")
for r in results:
    print(r)


# Statistics
fitness_values = np.array([r["fitness"] for r in results], dtype=float)

avg_fitness = np.mean(fitness_values)
std_fitness = np.std(fitness_values)

print("\nAverage Fitness:", avg_fitness)
print("Std Fitness:", std_fitness)


# Save Results
with open("results_paper2.json", "w") as f:
    json.dump(results, f, indent=4)