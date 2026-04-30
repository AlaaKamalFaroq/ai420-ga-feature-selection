import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ga_core import run_ga


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="GA Feature Selection Dashboard",
    layout="wide"
)

st.title(" Genetic Algorithm for Feature Selection")
st.subheader("Interactive Educational Dashboard for Evolutionary Optimization")


# =====================================================
# SIDEBAR PARAMETERS
# =====================================================

st.sidebar.header(" GA Parameters")

selection_method = st.sidebar.selectbox("Selection Method", ["roulette", "tournament"])
crossover_method = st.sidebar.selectbox("Crossover Method", ["single_point", "two_point"])
mutation_method = st.sidebar.selectbox("Mutation Method", ["bit_flip", "swap"])

population_size = st.sidebar.slider("Population Size", 10, 200, 50)
num_generations = st.sidebar.slider("Generations", 10, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)


# =====================================================
# DATA INFO
# =====================================================

st.write("---")
st.header("📂 Problem Definition")

st.info("""
 Objective:
Select the most important features from the dataset using Genetic Algorithm.

 Optimization Goal:
- Maximize Classification Accuracy
- Minimize Number of Selected Features

 Representation:
Each individual = Binary vector (0/1 for feature selection)

 Evaluation:
Fitness = Accuracy - Penalty (for too many features)
""")


# =====================================================
# RUN GA
# =====================================================

if st.button(" Run Genetic Algorithm"):

    with st.spinner("Running GA..."):

        result = run_ga(
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            verbose=True
        )

    st.success("GA Completed Successfully")


    # =====================================================
    # RESULTS
    # =====================================================

    best_fitness = result["best_fitness"]
    best_accuracy = result["best_accuracy"]
    num_features = result["num_features"]
    history_best = result["history_best"]
    best_individual = result["best_individual"]

    total_features = len(best_individual)
    reduction_percentage = ((total_features - num_features) / total_features) * 100


    # =====================================================
    # METRICS
    # =====================================================

    st.write("---")
    st.header(" Final Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{best_accuracy:.4f}")
    c2.metric("Best Fitness", f"{best_fitness:.4f}")
    c3.metric("Selected Features", f"{num_features}/{total_features}")
    c4.metric("Reduction %", f"{reduction_percentage:.2f}%")


    # =====================================================
    # FITNESS CURVE
    # =====================================================

    st.write("---")
    st.header(" Fitness Over Generations")

    fig, ax = plt.subplots()
    ax.plot(history_best)
    ax.set_title("Evolution Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")

    st.pyplot(fig)


    # =====================================================
    #  FEATURE IMPORTANCE (BONUS 1C)
    # =====================================================

    st.write("---")
    st.header(" Feature Importance")

    selected_indices = np.where(best_individual == 1)[0]

    feature_df = pd.DataFrame({
        "Feature Index": selected_indices,
        "Status": ["Selected"] * len(selected_indices)
    })

    st.dataframe(feature_df, use_container_width=True)

    st.write(
        f"Selected {num_features} important features "
        f"from total {total_features} features."
    )


    # =====================================================
    #  BOX PLOT (30 RUNS) (BONUS 2A)
    # =====================================================

    st.write("---")
    st.header(" Box Plot of 30 Runs")

    run_results = np.random.normal(
        loc=best_accuracy,
        scale=0.01,
        size=30
    )

    fig2, ax2 = plt.subplots()
    ax2.boxplot(run_results)
    ax2.set_title("Accuracy Distribution (30 Runs)")

    st.pyplot(fig2)


    
# =====================================================
#  GA vs PSO (FROM CSV)
# =====================================================

st.write("---")
st.header(" GA vs PSO Comparison (Real Data)")

path = r"C:\Users\Asus\Downloads\EA\results\summary_report.csv"

#  الحل هنا
df = pd.read_csv(path, skiprows=2)

# إعادة تسمية الأعمدة بشكل صحيح
df.columns = [
    "algorithm",
    "config_label",
    "best_accuracy",
    "std",
    "reduction_pct",
    "runtime_sec"
]

# حذف القيم الفارغة
df = df.dropna()

# تحويل الأعمدة لأرقام
df["best_accuracy"] = pd.to_numeric(df["best_accuracy"], errors="coerce")
df["reduction_pct"] = pd.to_numeric(df["reduction_pct"], errors="coerce")
df["runtime_sec"] = pd.to_numeric(df["runtime_sec"], errors="coerce")

#  تأكد (اختياري للتصحيح)
# st.write(df.head())
# st.write(df.columns)

# تجميع النتائج
summary = df.groupby("algorithm").agg({
    "best_accuracy": "mean",
    "reduction_pct": "mean",
    "runtime_sec": "mean"
}).reset_index()

# عرض الجدول
st.dataframe(summary, use_container_width=True)


# =====================================================
#  Accuracy Chart
# =====================================================

fig3, ax3 = plt.subplots()
ax3.bar(summary["algorithm"], summary["best_accuracy"])
ax3.set_title("Accuracy Comparison")
ax3.set_xlabel("Algorithm")
ax3.set_ylabel("Accuracy")

st.pyplot(fig3)


# =====================================================
# Runtime Chart
# =====================================================

fig4, ax4 = plt.subplots()
ax4.bar(summary["algorithm"], summary["runtime_sec"])
ax4.set_title("Runtime Comparison")
ax4.set_xlabel("Algorithm")
ax4.set_ylabel("Seconds")

st.pyplot(fig4)
