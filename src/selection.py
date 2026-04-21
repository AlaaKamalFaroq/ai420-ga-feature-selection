import numpy as np


def roulette_wheel_selection(population, fitness_scores):
    """
    Roulette Wheel (Fitness Proportionate) Selection.
    Each individual's probability of selection is proportional to its fitness.
    Handles negative fitness by shifting scores.
    """
    shifted = fitness_scores - fitness_scores.min()
    total = shifted.sum()
    if total == 0:
        probs = np.ones(len(population)) / len(population)
    else:
        probs = shifted / total

    selected_indices = np.random.choice(
        len(population),
        size=len(population),
        replace=True,
        p=probs
    )
    return population[selected_indices]


def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Tournament Selection.
    Randomly pick `tournament_size` individuals and select the best.
    """
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        contestants = np.random.choice(pop_size, size=tournament_size, replace=False)
        winner = contestants[np.argmax(fitness_scores[contestants])]
        selected.append(population[winner])
    return np.array(selected)


def rank_selection(population, fitness_scores):
    """
    Rank-Based Selection.
    Individuals are ranked by fitness; selection probability is proportional to rank,
    not raw fitness. This reduces dominance of super-fit individuals.
    """
    pop_size = len(population)
    ranks = np.argsort(np.argsort(fitness_scores)) + 1  # ranks from 1 to N
    total_rank = ranks.sum()
    probs = ranks / total_rank

    selected_indices = np.random.choice(
        pop_size,
        size=pop_size,
        replace=True,
        p=probs
    )
    return population[selected_indices]


def select_parents(population, fitness_scores, method="tournament", tournament_size=3):
    """
    Unified selection interface.

    Parameters
    ----------
    population      : np.ndarray of shape (pop_size, num_features)
    fitness_scores  : np.ndarray of shape (pop_size,)
    method          : one of {"roulette", "tournament", "rank"}
    tournament_size : used only when method == "tournament"

    Returns
    -------
    selected : np.ndarray same shape as population
    """
    method = method.lower().strip()

    if method == "roulette":
        return roulette_wheel_selection(population, fitness_scores)
    elif method == "tournament":
        return tournament_selection(population, fitness_scores, tournament_size)
    elif method == "rank":
        return rank_selection(population, fitness_scores)
    else:
        raise ValueError(
            f"Unknown selection method '{method}'. "
            "Choose from: 'roulette', 'tournament', 'rank'."
        )