"""
operators.py
============
Author  : Waad
Branch  : feature/wad-operators
Purpose : Crossover & Mutation operators for GA-based Feature Selection
          (AI420 – Evolutionary Algorithms, Spring 2025-2026)

How to use
----------
Both public functions accept a ``method`` string so the caller (ga_core.py or
main.py) can pick which operator to run without touching this file:

    from operators import crossover, mutation

    children   = crossover(parents, method="uniform")     # or "single_point" / "two_point"
    individual = mutation(individual, method="bit_flip")  # or "swap" / "inversion"

Supported crossover methods : "single_point"  |  "two_point"  |  "uniform"
Supported mutation  methods : "bit_flip"      |  "swap"       |  "inversion"
"""

import numpy as np
from src.config import CROSSOVER_RATE, MUTATION_RATE


# ─────────────────────────────────────────────────────────────────────────────
# CROSSOVER OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """
    Single-Point Crossover
    ----------------------
    Picks one random cut point and swaps the tails of both parents.

    Example (n=8, cut=3):
        P1 : [1 0 1 | 0 0 1 1 0]
        P2 : [0 1 0 | 1 1 0 0 1]
        C1 : [1 0 1   1 1 0 0 1]
        C2 : [0 1 0   0 0 1 1 0]
    """
    n = len(parent1)
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    point = np.random.randint(1, n)          # at least 1 gene from each parent
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def _two_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """
    Two-Point Crossover
    -------------------
    Picks two random cut points; the segment between them is swapped.

    Example (n=8, cuts=2,5):
        P1 : [1 0 | 1 0 0 | 1 1 0]
        P2 : [0 1 | 0 1 1 | 0 0 1]
        C1 : [1 0   0 1 1   1 1 0]
        C2 : [0 1   1 0 0   0 0 1]
    """
    n = len(parent1)
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    pts = sorted(np.random.choice(range(1, n), size=2, replace=False))
    p, q = pts[0], pts[1]

    child1 = np.concatenate([parent1[:p], parent2[p:q], parent1[q:]])
    child2 = np.concatenate([parent2[:p], parent1[p:q], parent2[q:]])
    return child1, child2


def _uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """
    Uniform Crossover
    -----------------
    For every gene position, flip a fair coin to decide which parent
    contributes to each child.  Produces higher exploration than
    single/two-point methods.
    """
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    mask   = np.random.randint(0, 2, size=len(parent1), dtype=bool)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


# Public dispatcher ─────────────────────────────────────────────────────────

def crossover(parents: np.ndarray, method: str = "single_point") -> np.ndarray:
    """
    Apply the chosen crossover operator to a population of parents and
    return a same-size array of children.

    Parameters
    ----------
    parents : np.ndarray, shape (pop_size, n_features)
        Current generation – must have an even number of rows.
    method  : str
        One of  "single_point" | "two_point" | "uniform"

    Returns
    -------
    np.ndarray  (same shape as parents)
    """
    _dispatch = {
        "single_point": _single_point_crossover,
        "two_point":    _two_point_crossover,
        "uniform":      _uniform_crossover,
    }

    if method not in _dispatch:
        raise ValueError(
            f"Unknown crossover method '{method}'. "
            f"Choose from: {list(_dispatch.keys())}"
        )

    op       = _dispatch[method]
    children = []

    # Pair up consecutive individuals; wrap around if odd
    n = len(parents)
    for i in range(0, n - 1, 2):
        c1, c2 = op(parents[i], parents[i + 1])
        children.extend([c1, c2])

    # If pop_size is odd, copy last parent unchanged
    if n % 2 == 1:
        children.append(parents[-1].copy())

    return np.array(children)


# ─────────────────────────────────────────────────────────────────────────────
# MUTATION OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _bit_flip_mutation(individual: np.ndarray) -> np.ndarray:
    """
    Bit-Flip Mutation
    -----------------
    Standard binary mutation: each bit is flipped independently with
    probability MUTATION_RATE.  Well-suited for binary chromosomes
    (feature-present / feature-absent encoding).
    """
    ind = individual.copy()
    flip_mask = np.random.rand(len(ind)) < MUTATION_RATE
    ind[flip_mask] ^= 1          # XOR flip: 0→1 or 1→0
    return ind


def _swap_mutation(individual: np.ndarray) -> np.ndarray:
    """
    Swap Mutation
    -------------
    Selects two random gene positions and swaps their values.
    Preserves the number of selected features (cardinality-preserving),
    which helps when the fitness function penalises large subsets.
    """
    if np.random.rand() > MUTATION_RATE:
        return individual.copy()

    ind  = individual.copy()
    n    = len(ind)
    i, j = np.random.choice(n, size=2, replace=False)
    ind[i], ind[j] = ind[j], ind[i]
    return ind


def _inversion_mutation(individual: np.ndarray) -> np.ndarray:
    """
    Inversion Mutation
    ------------------
    Selects a random sub-sequence and reverses it in place.
    Can escape local optima by restructuring gene order while keeping
    the overall bit count constant.
    """
    if np.random.rand() > MUTATION_RATE:
        return individual.copy()

    ind = individual.copy()
    n   = len(ind)
    pts = sorted(np.random.choice(range(n), size=2, replace=False))
    i, j = pts[0], pts[1]
    ind[i:j + 1] = ind[i:j + 1][::-1]
    return ind


# Public dispatcher ─────────────────────────────────────────────────────────

def mutation(individual: np.ndarray, method: str = "bit_flip") -> np.ndarray:
    """
    Apply the chosen mutation operator to a single individual.

    Parameters
    ----------
    individual : np.ndarray, shape (n_features,)
        Binary chromosome.
    method     : str
        One of  "bit_flip" | "swap" | "inversion"

    Returns
    -------
    np.ndarray  (same shape as individual)
    """
    _dispatch = {
        "bit_flip":  _bit_flip_mutation,
        "swap":      _swap_mutation,
        "inversion": _inversion_mutation,
    }

    if method not in _dispatch:
        raise ValueError(
            f"Unknown mutation method '{method}'. "
            f"Choose from: {list(_dispatch.keys())}"
        )

    return _dispatch[method](individual)