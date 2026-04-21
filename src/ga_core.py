import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_data, preprocess
from config import POPULATION_SIZE, NUM_GENERATIONS, ELITISM_K, KNN_NEIGHBORS


# ── Load & preprocess once (not inside the loop) ────────────────────────────
X_raw, y_raw, feature_names = load_data()
X_train, X_test, y_train, y_test = preprocess(X_raw, y_raw)


# ── Initialize Population ───────────────────────────────────────────────────
def initialize_population(num_features):
    return np.random.randint(
        0, 2,
        (POPULATION_SIZE, num_features)
    )


# ── Real Fitness Function ───────────────────────────────────────────────────
def fitness(individual):
    """
    Objective: maximise classification accuracy while minimising feature count.
    fitness = accuracy - 0.01 * (num_selected / total_features)
    """
    indices = np.where(individual == 1)[0]
    if len(indices) == 0:
        return 0.0

    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train[:, indices], y_train)
    acc = model.score(X_test[:, indices], y_test)

    # Small penalty to discourage using all features
    penalty = 0.01 * (len(indices) / len(individual))
    return acc - penalty


# ── GA Main Loop ────────────────────────────────────────────────────────────
def run_ga(selection_method,
           crossover_method,
           mutation_method,
           fitness_func=fitness,
           verbose=False):

    from selection import select_parents
    from operators import crossover, mutation

    num_features = X_train.shape[1]
    population   = initialize_population(num_features)

    best_individual = None
    best_fitness    = -np.inf
    history_best    = []

    if verbose:
        print("GA started")

    for gen in range(NUM_GENERATIONS):

        # 1. Fitness Evaluation
        fitness_scores = np.array([
            fitness_func(ind) for ind in population
        ])

        # 2. Elitism (Top-K)
        elite_indices = np.argsort(fitness_scores)[-ELITISM_K:]
        elites        = population[elite_indices].copy()

        # 3. Track Global Best
        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness    = fitness_scores[gen_best_idx]
            best_individual = population[gen_best_idx].copy()

        # 4. Selection
        parents = select_parents(
            population,
            fitness_scores,
            method=selection_method
        )

        # 5. Crossover
        children = crossover(parents, method=crossover_method)

        # 6. Mutation
        population = np.array([
            mutation(ind.copy(), method=mutation_method)
            for ind in children
        ])

        # 7. Inject Elites
        population[:ELITISM_K] = elites

        # 8. Logging
        history_best.append(float(np.max(fitness_scores)))

        if verbose:
            num_feats = int(np.sum(best_individual)) if best_individual is not None else 0
            print(f"Generation {gen:3d} | Best Fitness: {np.max(fitness_scores):.4f} | Features: {num_feats}/30")

    return {
        "best_fitness":    best_fitness,
        "best_individual": best_individual,
        "history_best":    history_best
    }