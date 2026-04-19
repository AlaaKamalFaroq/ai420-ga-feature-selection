import numpy as np
import json
from src.ga_core import run_ga
from src.config import NUM_RUNS


results = []

# Run Experiments
for run_id in range(NUM_RUNS):

    print(f"Run {run_id}")

    # Ensure reproducibility per run
    np.random.seed(run_id)

    best_fitness, best_individual = run_ga(
        selection_method="roulette",
        crossover_method="single_point",
        mutation_method="bit_flip"
    )

    result = {
        "run": run_id,
        "fitness": float(best_fitness),
        "num_features": int(np.sum(best_individual))
    }

    results.append(result)


# Show Results
print("\nAll Results:")
for r in results:
    print(r)


# Statistics
fitness_values = [r["fitness"] for r in results]

avg_fitness = np.mean(fitness_values)
std_fitness = np.std(fitness_values)

print("\nAverage Fitness:", avg_fitness)
print("Std Fitness:", std_fitness)


# Save Results
with open("results_paper2.json", "w") as f:
    json.dump(results, f, indent=4)