import time
import csv
import os
import numpy as np
import pandas as pd

from src.data_loader import load_data, preprocess, get_baseline_accuracy
from src.ga_core import run_ga
from src.pso import run_pso
from src.config import SEEDS 

os.makedirs("results", exist_ok=True)

GA_CONFIGS = [
    ("GA_Tournament_Single_Bitflip", "tournament", "single_point", "bit_flip"),
    ("GA_Roulette_Uniform_Swap",     "roulette",   "uniform",      "swap"),
]

def main():
    print("=" * 70)
    print("RESEARCH EXPERIMENT: 5 RUNS FOR GA SETTINGS & 5 RUNS FOR PSO")
    print("=" * 70)

    X_raw, y_raw, feature_names = load_data()
    X_train, X_test, y_train, y_test = preprocess(X_raw, y_raw)
    baseline_acc = get_baseline_accuracy(X_train, X_test, y_train, y_test)

    results_path = "results/research_full_results.csv"
    fieldnames = [
        "algorithm", "config_label", "run", "seed", 
        "best_fitness", "best_accuracy", "num_features", 
        "reduction_pct", "runtime_sec"
    ]

    with open(results_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()

        for (label, sel, cx, mut) in GA_CONFIGS:
            print(f"\n>>> Running GA Config: {label}")
            for run_idx in range(5): 
                seed = SEEDS[run_idx]
                t0 = time.time()

                result = run_ga(
                    selection_method=sel,
                    crossover_method=cx,
                    mutation_method=mut,
                    seed=seed,
                    verbose=False
                )

                elapsed = time.time() - t0
                reduction = (1 - (result["num_features"] / len(feature_names))) * 100
                
                writer.writerow({
                    "algorithm": "GA",
                    "config_label": label,
                    "run": run_idx + 1,
                    "seed": seed,
                    "best_fitness": round(result["best_fitness"], 6),
                    "best_accuracy": round(result["best_accuracy"], 6),
                    "num_features": result["num_features"],
                    "reduction_pct": round(reduction, 2),
                    "runtime_sec": round(elapsed, 2)
                })
                csvfile.flush() 
                print(f"   Run {run_idx + 1}/5 finished.")

        print(f"\n>>> Running PSO Baseline (5 Runs)")
        for run_idx in range(5): 
            seed = SEEDS[run_idx]
            t0 = time.time()
            result = run_pso(X_train, X_test, y_train, y_test, seed=seed, verbose=False)
            elapsed = time.time() - t0
            reduction = (1 - (result["num_features"] / len(feature_names))) * 100

            writer.writerow({
                "algorithm": "PSO",
                "config_label": "Standard_BPSO",
                "run": run_idx + 1,
                "seed": seed,
                "best_fitness": round(result["best_fitness"], 6),
                "best_accuracy": round(result["best_accuracy"], 6),
                "num_features": result["num_features"],
                "reduction_pct": round(reduction, 2),
                "runtime_sec": round(elapsed, 2)
            })
            csvfile.flush() 
            print(f"   Run {run_idx + 1}/5 finished.")

    print(f"\n[SUCCESS] Experiments finished. Saved to {results_path}")
    _print_final_report(results_path, baseline_acc)

def _print_final_report(path, baseline):
    df = pd.read_csv(path)
    summary = df.groupby(['algorithm', 'config_label']).agg({
        'best_accuracy': ['mean', 'std'],
        'reduction_pct': ['mean'],
        'runtime_sec': ['mean']
    }).round(4)
    print("\n--- STATISTICAL SUMMARY (AVERAGE OF RUNS) ---")
    print(summary)
    summary.to_csv("results/summary_report.csv")

if __name__ == "__main__":
    main()
