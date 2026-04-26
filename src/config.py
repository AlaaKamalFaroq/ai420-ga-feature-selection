
# config.py  – Central configuration for the GA Feature Selection project
# -------------------------------------------------------------------------
# All hyperparameters live here so every module imports from one place.

# ── Dataset / Preprocessing ──────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
NORMALIZE      = True
KNN_NEIGHBORS  = 5

# ── GA Core ───────────────────────────────────────────────────────────────────
POPULATION_SIZE  = 50
NUM_GENERATIONS  = 100
ELITISM_K        = 5
NUM_RUNS         = 5

# ── Operator Rates ────────────────────────────────────────────────────────────
CROSSOVER_RATE   = 0.85   # probability that two parents actually swap genes
MUTATION_RATE    = 0.01   # per-bit probability for bit_flip;
                          # trigger probability for swap / inversion
TOURNAMENT_SIZE = 3
ALPHA          = 0.99   # accuracy weight in fitness function


# ── Experiment Seeds (first 5; full list in experiments/seeds.txt) ────────────
SEEDS = [42, 7, 13, 99, 2025]
>>>>>>> 27a19e5aaac250ee235a91e6192fdbaa1c03300f
