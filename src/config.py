# config.py

# ── Dataset / Preprocessing ──────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
NORMALIZE      = True
KNN_NEIGHBORS  = 3

# ── GA Core ───────────────────────────────────────────────────────────────────
POPULATION_SIZE  = 50   # حجم الجمهرة
NUM_GENERATIONS  = 100  # عدد الأجيال (Termination Condition)
ELITISM_K        = 2    # تقليل الـ Elitism لـ 2 بيساعد على الحفاظ على التنوع (Diversity)
NUM_RUNS         = 30   # تم التعديل لـ 30 كما طلب الدكتور

# ── Operator Rates ────────────────────────────────────────────────────────────
CROSSOVER_RATE   = 0.85 
MUTATION_RATE    = 0.05 # رفعنا النسبة قليلاً لضمان التنوع (Exploration)
TOURNAMENT_SIZE  = 3
ALPHA            = 0.95 # التوازن بين الدقة (95%) وتقليل الميزات (5%)

# ── Experiment Seeds (30 Seeds) ───────────────────────────────────────────────
# الدكتور طلب تخزين الـ Seeds وتوفيرها. دي قائمة بـ 30 Seed ثابتين ومختارين بعناية.
SEEDS = [
    42, 7, 13, 99, 2025, 101, 55, 777, 88, 19,
    23, 45, 67, 89, 12, 34, 56, 78, 90, 11,
    22, 33, 44, 555, 666, 77, 888, 999, 123, 456
]

# ── PSO Hyperparameters ───────────────────────────────────────────────────────
W  = 0.7     # Inertia weight
C1 = 1.5     # Cognitive coefficient
C2 = 1.5     # Social coefficient
V_MAX = 4.0  # Velocity clamp
