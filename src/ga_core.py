import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ── Imports ──────────────────────────────────────────────────────────────────
from src.data_loader import load_data, preprocess
from src.selection import select_parents
from src.operators import crossover, mutation  # mutation operators not mutate

from src.config import (
    POPULATION_SIZE, NUM_GENERATIONS, ELITISM_K,
    KNN_NEIGHBORS, ALPHA, CROSSOVER_RATE, MUTATION_RATE
)

# ── Load & preprocess once at module level ───────────────────────────────────
X_raw, y_raw, feature_names = load_data()
X_train, X_test, y_train, y_test = preprocess(X_raw, y_raw)

# ── Population initialisation ────────────────────────────────────────────────
def initialize_population(num_features):
    return np.random.randint(0, 2, (POPULATION_SIZE, num_features))

def fitness(individual):
    indices = np.where(individual == 1)[0]
    if len(indices) == 0:
        return 0.0

    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, n_jobs=-1)
    
  
    model.fit(X_train[:, indices], y_train)
    acc = model.score(X_train[:, indices], y_train) 
    
    feature_ratio = len(indices) / len(individual)
    return ALPHA * acc + (1.0 - ALPHA) * (1.0 - feature_ratio)

# ── Main GA loop ─────────────────────────────────────────────────────────────
def run_ga(
    selection_method="tournament",
    crossover_method="single_point",
    mutation_method="bit_flip",
    fitness_func=fitness,
    selection_params=None,
    seed=None,
    verbose=False,
):
    if seed is not None:
        np.random.seed(seed)

    if selection_params is None:
        selection_params = {}

    num_features = X_train.shape[1]
    population   = initialize_population(num_features)

    best_individual = None
    best_fitness    = -np.inf
    history_best    = []

    if verbose:
        print(f"GA started | selection={selection_method} | seed={seed}")

    for gen in range(NUM_GENERATIONS):
        # 1. Evaluate fitness
        fitness_scores = np.array([fitness_func(ind) for ind in population])

        # 2. Track global best
        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness    = float(fitness_scores[gen_best_idx])
            best_individual = population[gen_best_idx].copy()

        # 3. Save elites 
        elite_indices = np.argsort(fitness_scores)[-ELITISM_K:]
        elites        = population[elite_indices].copy()

        # 4. Selection → crossover → mutation
        parents    = select_parents(population, fitness_scores, method=selection_method, **selection_params)
        children   = crossover(parents, method=crossover_method)
        
        # mutation not mutate
        population = np.array([
            mutation(ind.copy(), method=mutation_method)
            for ind in children
        ])

        # 5. Elitism replacement
        new_fitness   = np.array([fitness_func(ind) for ind in population])
        worst_indices = np.argsort(new_fitness)[:ELITISM_K]
        population[worst_indices] = elites

        # 6. Logging
        history_best.append(float(np.max(fitness_scores)))
        if verbose and gen % 10 == 0:
            n_feats = int(np.sum(best_individual)) if best_individual is not None else 0
            print(f"  Gen {gen:3d} | Best fitness: {best_fitness:.4f} | Features: {n_feats}/{num_features}")

    # Final evaluation
    final_indices  = np.where(best_individual == 1)[0]
    final_model    = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, n_jobs=-1)
    final_model.fit(X_train[:, final_indices], y_train)
    best_accuracy  = final_model.score(X_test[:, final_indices], y_test)

    return {
        "best_fitness":    best_fitness,
        "best_individual": best_individual,
        "best_accuracy":   best_accuracy,
        "num_features":    int(np.sum(best_individual)),
        "history_best":    history_best,
    }

if __name__ == "__main__":
    from src.config import SEEDS
    res = run_ga(verbose=True, seed=SEEDS[0])
    print(f"\nFinal GA Results: Accuracy={res['best_accuracy']:.4f}, Features={res['num_features']}")
