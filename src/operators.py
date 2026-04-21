import numpy as np


# ─────────────────────────────────────────────
#              CROSSOVER OPERATORS
# ─────────────────────────────────────────────

def single_point_crossover(parent1, parent2):
    """
    Single-Point Crossover.
    A random cut-point splits both parents; tails are swapped.
    """
    n = len(parent1)
    point = np.random.randint(1, n)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Two-Point Crossover.
    Two random cut-points define a segment that is swapped between parents.
    """
    n = len(parent1)
    p1, p2 = sorted(np.random.choice(range(1, n), size=2, replace=False))
    child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
    child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])
    return child1, child2


def uniform_crossover(parent1, parent2, prob=0.5):
    """
    Uniform Crossover.
    Each gene is independently inherited from either parent with probability `prob`.
    Produces higher exploration than point-based crossovers.
    """
    mask = np.random.rand(len(parent1)) < prob
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


# ─────────────────────────────────────────────
#              MUTATION OPERATORS
# ─────────────────────────────────────────────

def bit_flip_mutation(individual, mutation_rate=0.05):
    """
    Bit-Flip Mutation.
    Each bit is independently flipped (0→1 or 1→0) with probability `mutation_rate`.
    Standard mutation for binary representations.
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def swap_mutation(individual, mutation_rate=0.05):
    """
    Swap Mutation.
    With probability `mutation_rate`, two randomly chosen positions are swapped.
    Preserves the number of selected features (Hamming weight conserving).
    """
    mutated = individual.copy()
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(mutated), size=2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def inversion_mutation(individual, mutation_rate=0.05):
    """
    Inversion Mutation.
    With probability `mutation_rate`, a randomly chosen segment of the chromosome
    is reversed. Maintains gene content while altering order/context.
    """
    mutated = individual.copy()
    if np.random.rand() < mutation_rate:
        i, j = sorted(np.random.choice(len(mutated), size=2, replace=False))
        mutated[i:j+1] = mutated[i:j+1][::-1]
    return mutated


# ─────────────────────────────────────────────
#              UNIFIED INTERFACES
# ─────────────────────────────────────────────

def crossover(parents, method="single_point", crossover_rate=0.8):
    """
    Apply crossover to pairs of parents to produce a new generation.

    Parameters
    ----------
    parents        : np.ndarray of shape (pop_size, num_features)
    method         : one of {"single_point", "two_point", "uniform"}
    crossover_rate : probability that crossover is actually applied to a pair

    Returns
    -------
    children : np.ndarray same shape as parents
    """
    method = method.lower().strip()
    crossover_fn_map = {
        "single_point": single_point_crossover,
        "two_point":    two_point_crossover,
        "uniform":      uniform_crossover,
    }
    if method not in crossover_fn_map:
        raise ValueError(
            f"Unknown crossover method '{method}'. "
            "Choose from: 'single_point', 'two_point', 'uniform'."
        )
    fn = crossover_fn_map[method]

    children = []
    pop_size = len(parents)
    for i in range(0, pop_size - 1, 2):
        p1, p2 = parents[i], parents[i + 1]
        if np.random.rand() < crossover_rate:
            c1, c2 = fn(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()
        children.extend([c1, c2])

    # If odd population, carry last parent through unchanged
    if pop_size % 2 != 0:
        children.append(parents[-1].copy())

    return np.array(children[:pop_size])


def mutation(individual, method="bit_flip", mutation_rate=0.05):
    """
    Apply mutation to a single individual.

    Parameters
    ----------
    individual    : np.ndarray of shape (num_features,)
    method        : one of {"bit_flip", "swap", "inversion"}
    mutation_rate : per-gene (or per-event) probability of mutation

    Returns
    -------
    mutated : np.ndarray same shape as individual
    """
    method = method.lower().strip()
    mutation_fn_map = {
        "bit_flip":  bit_flip_mutation,
        "swap":      swap_mutation,
        "inversion": inversion_mutation,
    }
    if method not in mutation_fn_map:
        raise ValueError(
            f"Unknown mutation method '{method}'. "
            "Choose from: 'bit_flip', 'swap', 'inversion'."
        )
    return mutation_fn_map[method](individual, mutation_rate)