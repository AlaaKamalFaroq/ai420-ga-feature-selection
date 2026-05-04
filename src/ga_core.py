import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ── Imports ──────────────────────────────────────────────────────────────────
from src.data_loader import load_data, preprocess
from src.selection import select_parents
from src.operators import crossover, mutation

from src.config import (
    POPULATION_SIZE, NUM_GENERATIONS, ELITISM_K,
    KNN_NEIGHBORS, ALPHA, CROSSOVER_RATE, MUTATION_RATE
)

# ── Load & preprocess once at module level ───────────────────────────────────
X_raw, y_raw, feature_names = load_data()
X_train, X_test, y_train, y_test = preprocess(X_raw, y_raw)

# ── Population initialisation ────────────────────────────────────────────────
def initialize_population(num_features):
    """Random binary initialisation."""
    return np.random.randint(0, 2, (POPULATION_SIZE, num_features))

# ── Fitness Function (NO data leakage — 3-fold CV on X_train only) ───────────
def fitness(individual):
    """
    Fitness = ALPHA * CV_accuracy + (1 - ALPHA) * (1 - feature_ratio)
    Uses 3-fold cross-validation on X_train only to avoid test-set leakage.
    """
    indices = np.where(individual == 1)[0]
    if len(indices) == 0:
        return 0.0

    model  = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, n_jobs=-1)
    scores = cross_val_score(
        model, X_train[:, indices], y_train,
        cv=3, scoring='accuracy'
    )
    acc          = scores.mean()
    feature_ratio = len(indices) / len(individual)
    return ALPHA * acc + (1.0 - ALPHA) * (1.0 - feature_ratio)


# ── Survivor Selection Methods ────────────────────────────────────────────────

def _elitism_replacement(population, fitness_scores, children, fitness_func, elites):
    """
    Survivor Selection Method 1: Elitism
    ─────────────────────────────────────
    Keep the top ELITISM_K individuals from the previous generation
    and replace the worst ELITISM_K in the new population with them.
    Guarantees the best solution is never lost.
    """
    new_fitness   = np.array([fitness_func(ind) for ind in children])
    worst_indices = np.argsort(new_fitness)[:ELITISM_K]
    children[worst_indices] = elites
    return children


def _generational_replacement(population, fitness_scores, children, fitness_func, elites):
    """
    Survivor Selection Method 2: Generational Replacement
    ───────────────────────────────────────────────────────
    Replace the entire population with children (full generational model).
    Still keeps the single best individual (1 elite) to avoid losing the best.
    This increases diversity compared to Elitism.
    """
    new_pop = children.copy()
    # Keep only the single best from previous generation
    best_idx = np.argmax(fitness_scores)
    worst_new_idx = np.argmin([fitness_func(ind) for ind in new_pop])
    new_pop[worst_new_idx] = population[best_idx].copy()
    return new_pop


# ── Diversity Preservation ────────────────────────────────────────────────────

def _preserve_diversity(population, num_features):
    """
    Diversity Preservation via Duplicate Removal + Random Reinitialisation.
    ────────────────────────────────────────────────────────────────────────
    If more than 50% of the population are duplicates, remove them and
    replace with random individuals to maintain exploration.
    Always guarantees population size = POPULATION_SIZE.
    """
    unique = np.unique(population, axis=0)

    if len(unique) < POPULATION_SIZE * 0.5:
        # Too many duplicates → reinitialise missing slots randomly
        n_new      = POPULATION_SIZE - len(unique)
        random_inds = np.random.randint(0, 2, (n_new, num_features))
        population  = np.vstack([unique, random_inds])
    else:
        population = unique[:POPULATION_SIZE]

    # Final size guarantee
    if len(population) < POPULATION_SIZE:
        extra      = np.random.randint(0, 2, (POPULATION_SIZE - len(population), num_features))
        population = np.vstack([population, extra])

    return population[:POPULATION_SIZE]


# ── Main GA loop ─────────────────────────────────────────────────────────────

def run_ga(
    selection_method="tournament",
    crossover_method="single_point",
    mutation_method="bit_flip",
    survivor_method="elitism",          # "elitism" | "generational"
    fitness_func=fitness,
    selection_params=None,
    seed=None,
    verbose=False,
):
    """
    Run one independent GA experiment.

    Parameters
    ----------
    selection_method : str   "tournament" | "roulette" | "rank"
    crossover_method : str   "single_point" | "two_point" | "uniform"
    mutation_method  : str   "bit_flip" | "swap" | "inversion"
    survivor_method  : str   "elitism" | "generational"
    seed             : int   for reproducibility
    verbose          : bool  print progress every 10 generations

    Returns
    -------
    dict with keys:
        best_fitness, best_individual, best_accuracy,
        num_features, history_best
    """
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
        print(f"GA started | selection={selection_method} | "
              f"crossover={crossover_method} | mutation={mutation_method} | "
              f"survivor={survivor_method} | seed={seed}")

    for gen in range(NUM_GENERATIONS):

        # ── 1. Evaluate fitness ───────────────────────────────────────────
        fitness_scores = np.array([fitness_func(ind) for ind in population])

        # ── 2. Track global best ──────────────────────────────────────────
        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness    = float(fitness_scores[gen_best_idx])
            best_individual = population[gen_best_idx].copy()

        # ── 3. Save elites for survivor selection ─────────────────────────
        elite_indices = np.argsort(fitness_scores)[-ELITISM_K:]
        elites        = population[elite_indices].copy()

        # ── 4. Selection → Crossover → Mutation ──────────────────────────
        parents  = select_parents(
            population, fitness_scores,
            method=selection_method,
            **selection_params
        )
        children = crossover(parents, method=crossover_method)
        children = np.array([
            mutation(ind.copy(), method=mutation_method)
            for ind in children
        ])

        # ── 5. Survivor Selection (2 methods) ─────────────────────────────
        if survivor_method == "generational":
            population = _generational_replacement(
                population, fitness_scores, children, fitness_func, elites
            )
        else:  # default: elitism
            population = _elitism_replacement(
                population, fitness_scores, children, fitness_func, elites
            )

        # ── 6. Diversity Preservation (Duplicate Removal) ─────────────────
        population = _preserve_diversity(population, num_features)

        # ── 7. Logging ────────────────────────────────────────────────────
        history_best.append(float(np.max(fitness_scores)))
        if verbose and gen % 10 == 0:
            n_feats = int(np.sum(best_individual)) if best_individual is not None else 0
            print(f"  Gen {gen:3d} | Best fitness: {best_fitness:.4f} | "
                  f"Features: {n_feats}/{num_features}")

    # ── Final evaluation on test set (ONCE after full run) ────────────────
    final_indices = np.where(best_individual == 1)[0]
    final_model   = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, n_jobs=-1)
    final_model.fit(X_train[:, final_indices], y_train)
    best_accuracy = final_model.score(X_test[:, final_indices], y_test)

    if verbose:
        print(f"\nGA done | Best fitness: {best_fitness:.4f} | "
              f"Test accuracy: {best_accuracy:.4f} | "
              f"Features: {len(final_indices)}/{num_features}")

    return {
        "best_fitness":    best_fitness,
        "best_individual": best_individual,
        "best_accuracy":   best_accuracy,
        "num_features":    int(np.sum(best_individual)),
        "history_best":    history_best,
    }


if __name__ == "__main__":
    from src.config import SEEDS

    print("── Elitism Survivor Selection ──")
    res = run_ga(verbose=True, seed=SEEDS[0], survivor_method="elitism")
    print(f"Accuracy={res['best_accuracy']:.4f} | Features={res['num_features']}")

    print("\n── Generational Survivor Selection ──")
    res = run_ga(verbose=True, seed=SEEDS[0], survivor_method="generational")
    print(f"Accuracy={res['best_accuracy']:.4f} | Features={res['num_features']}")