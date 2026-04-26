"""
pso.py
======
Author  :A'laa
Purpose : Binary Particle Swarm Optimisation (BPSO) for feature selection
          on the Malaria Cell Image dataset.

Algorithm reference
-------------------
Kennedy, J. & Eberhart, R.C. (1997)
"A discrete binary version of the particle swarm algorithm"
Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics.

How BPSO works
--------------
Each particle is a binary vector (1 = feature selected, 0 = not selected).
Each particle has a velocity vector of real values; a sigmoid maps velocity
to a probability of the bit being 1 on the next step.

Update equations:
    vel = W * vel
        + C1 * r1 * (pbest - pos)
        + C2 * r2 * (gbest - pos)
    prob = sigmoid(vel)
    pos[i] = 1  if random() < prob[i]  else 0

Compatible with ga_core.py
--------------------------
- Same input: X_train, X_test, y_train, y_test
- Same output dict keys: best_fitness, best_individual, best_accuracy,
  num_features, history_best
- Fitness uses 3-fold CV on X_train only (no test-set leakage, same as GA)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from src.config import (
    POPULATION_SIZE, NUM_GENERATIONS,
    KNN_NEIGHBORS, ALPHA
)
from src.config import TOURNAMENT_SIZE
# ── PSO hyperparameters ───────────────────────────────────────────────────────
W  = 0.7    # inertia weight        (how much old velocity is kept)
C1 = 1.5    # cognitive coefficient  (pull toward personal best)
C2 = 1.5    # social coefficient     (pull toward global best)
V_MAX = 4.0 # velocity clamp        (prevents sigmoid saturation)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(v):
    """Numerically stable sigmoid — maps velocity → probability in (0, 1)."""
    v_clipped = np.clip(v, -V_MAX, V_MAX)
    return 1.0 / (1.0 + np.exp(-v_clipped))


def _fitness_cv(individual, X_train, y_train):
    """
    3-fold CV accuracy on X_train only — identical approach to ga_core.py.
    Returns weighted fitness = ALPHA * accuracy + (1-ALPHA) * (1 - feature_ratio)
    """
    indices = np.where(individual == 1)[0]
    if len(indices) == 0:
        return 0.0

    model  = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    scores = cross_val_score(
        model,
        X_train[:, indices],
        y_train,
        cv=3,
        scoring="accuracy"
    )
    acc           = scores.mean()
    feature_ratio = len(indices) / len(individual)
    return ALPHA * acc + (1.0 - ALPHA) * (1.0 - feature_ratio)


# ── Main PSO loop ─────────────────────────────────────────────────────────────

def run_pso(X_train, X_test, y_train, y_test, seed=None, verbose=False):
    """
    Run one independent Binary PSO experiment.

    Parameters
    ----------
    X_train, X_test : np.ndarray  already preprocessed by data_loader.preprocess()
    y_train, y_test : np.ndarray
    seed            : int   for reproducibility (use values from config.SEEDS)
    verbose         : bool  print progress every 10 generations

    Returns
    -------
    dict with keys:
        "best_fitness"     float
        "best_individual"  np.ndarray  binary, length = num_features
        "best_accuracy"    float  evaluated on X_test ONCE after the run
        "num_features"     int
        "history_best"     list[float]  length = NUM_GENERATIONS
    """
    if seed is not None:
        np.random.seed(seed)

    n_particles = POPULATION_SIZE
    n_features  = X_train.shape[1]

    # ── Initialise positions and velocities ──────────────────────────────
    pos = np.random.randint(0, 2, (n_particles, n_features)).astype(float)
    vel = np.random.uniform(-1.0, 1.0, (n_particles, n_features))

    # ── Personal bests ───────────────────────────────────────────────────
    pbest_pos = pos.copy()
    pbest_fit = np.array([
        _fitness_cv(pos[i], X_train, y_train) for i in range(n_particles)
    ])

    # ── Global best ──────────────────────────────────────────────────────
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    history_best = []

    if verbose:
        print(f"PSO started | particles={n_particles} "
              f"generations={NUM_GENERATIONS} seed={seed}")

    for gen in range(NUM_GENERATIONS):

        # ── Velocity update ───────────────────────────────────────────
        r1  = np.random.rand(n_particles, n_features)
        r2  = np.random.rand(n_particles, n_features)

        vel = (W  * vel
               + C1 * r1 * (pbest_pos - pos)
               + C2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -V_MAX, V_MAX)

        # ── Position update (binary via sigmoid) ──────────────────────
        probs = _sigmoid(vel)
        pos   = (np.random.rand(n_particles, n_features) < probs).astype(float)

        # ── Evaluate new positions ────────────────────────────────────
        fits = np.array([
            _fitness_cv(pos[i], X_train, y_train) for i in range(n_particles)
        ])

        # ── Update personal bests ─────────────────────────────────────
        improved = fits > pbest_fit
        pbest_pos[improved] = pos[improved].copy()
        pbest_fit[improved] = fits[improved]

        # ── Update global best ────────────────────────────────────────
        gen_best_idx = np.argmax(fits)
        if fits[gen_best_idx] > gbest_fit:
            gbest_fit = float(fits[gen_best_idx])
            gbest_pos = pos[gen_best_idx].copy()

        history_best.append(gbest_fit)

        if verbose and gen % 10 == 0:
            n_feats = int(gbest_pos.sum())
            print(f"  Gen {gen:3d} | Best fitness: {gbest_fit:.4f} "
                  f"| Features: {n_feats}/{n_features}")

    # ── Final test-set evaluation (ONCE, after the full run) ─────────────
    final_indices = np.where(gbest_pos == 1)[0]
    if len(final_indices) == 0:
        best_accuracy = 0.0
    else:
        final_model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
        final_model.fit(X_train[:, final_indices], y_train)
        best_accuracy = final_model.score(X_test[:, final_indices], y_test)

    if verbose:
        print(f"PSO done | Best fitness: {gbest_fit:.4f} "
              f"| Test accuracy: {best_accuracy:.4f} "
              f"| Features selected: {len(final_indices)}/{n_features}")

    return {
        "best_fitness":    gbest_fit,
        "best_individual": gbest_pos.astype(int),
        "best_accuracy":   best_accuracy,
        "num_features":    int(gbest_pos.sum()),
        "history_best":    history_best,
    }
if __name__ == "__main__":
    from src.data_loader import load_data, preprocess
    from src.config import SEEDS
    import os

  
    X, y, names = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)

    print("\n" + "="*40)
    print("  RUNNING PARTICLE SWARM OPTIMIZATION")
    print("="*40)

    results = run_pso(X_train, X_test, y_train, y_test, seed=SEEDS[0], verbose=True)

 
    print("\n" + "-"*30)
    print("  PSO FINAL SUMMARY")
    print("-"*30)
    print(f"Initial Features  : {len(names)}")
    print(f"Selected Features : {results['num_features']}")
    print(f"Reduction Rate    : {(1 - results['num_features']/len(names))*100:.1f}%")
    print(f"Test Accuracy     : {results['best_accuracy']:.4f}")
    print("-"*30)
