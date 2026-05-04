"""
ui.py — AI420 GA & PSO Feature Selection Dashboard
====================================================
Run with:  streamlit run ui.py

Changes vs original
-------------------
1. ✅ Fixed cache bug: data_loader now uses .npz (not .pkl)
2. ✅ Experiment results are cached in session_state so re-runs don't re-execute
3. ✅ All GA configs from experiment_runner are exposed in the UI
4. ✅ Added GA config presets (matching experiment_runner.py configs)
5. ✅ Clear cache button to force fresh run
6. ✅ Progress shows per-run timing
7. ✅ Results tab added alongside the plots
"""

import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from src.ga_core import run_ga
from src.pso import run_pso
from src.data_loader import load_data, preprocess
from src.config import SEEDS

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI420 — GA & PSO Feature Selection",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load data — cached so extraction only happens ONCE per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading / extracting features (first time only)...")
def get_data():
    X, y, names = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    return X_train, X_test, y_train, y_test, names


# ─────────────────────────────────────────────────────────────────────────────
# Experiment result cache — stored in st.session_state by a config key
# so clicking "Run" again with the SAME settings reuses previous results.
# ─────────────────────────────────────────────────────────────────────────────
def _make_cache_key(algorithm, cfg: dict, num_runs: int, seed_start: int) -> str:
    payload = json.dumps(
        {"alg": algorithm, "cfg": cfg, "runs": num_runs, "seed": seed_start},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def get_cached_results(key):
    return st.session_state.get(f"results_{key}")


def set_cached_results(key, results):
    st.session_state[f"results_{key}"] = results


# ─────────────────────────────────────────────────────────────────────────────
# GA configuration presets  (mirrors experiment_runner.py GA_CONFIGS)
# ─────────────────────────────────────────────────────────────────────────────
GA_PRESETS = {
    "Custom (manual)": None,
    "📌 Preset 1 — Tournament + Single Point + Bit-Flip": {
        "selection": "tournament",
        "crossover": "single_point",
        "mutation":  "bit_flip",
        "survivor":  "elitism",
    },
    "📌 Preset 2 — Roulette + Uniform + Swap": {
        "selection": "roulette",
        "crossover": "uniform",
        "mutation":  "swap",
        "survivor":  "elitism",
    },
    "📌 Preset 3 — Rank + Two-Point + Inversion (Generational)": {
        "selection": "rank",
        "crossover": "two_point",
        "mutation":  "inversion",
        "survivor":  "generational",
    },
    "📌 Preset 4 — Tournament + Uniform + Swap (Generational)": {
        "selection": "tournament",
        "crossover": "uniform",
        "mutation":  "swap",
        "survivor":  "generational",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Experiment Settings")

algorithm = st.sidebar.radio("Algorithm", ["GA", "PSO"], horizontal=True)

ga_cfg = {}
if algorithm == "GA":
    st.sidebar.markdown("---")
    preset_name = st.sidebar.selectbox("GA Preset", list(GA_PRESETS.keys()))
    preset = GA_PRESETS[preset_name]

    st.sidebar.markdown("**Operators**")
    if preset:
        ga_cfg["selection"] = st.sidebar.selectbox(
            "Selection", ["tournament", "roulette", "rank"],
            index=["tournament", "roulette", "rank"].index(preset["selection"]),
        )
        ga_cfg["crossover"] = st.sidebar.selectbox(
            "Crossover", ["single_point", "two_point", "uniform"],
            index=["single_point", "two_point", "uniform"].index(preset["crossover"]),
        )
        ga_cfg["mutation"] = st.sidebar.selectbox(
            "Mutation", ["bit_flip", "swap", "inversion"],
            index=["bit_flip", "swap", "inversion"].index(preset["mutation"]),
        )
        ga_cfg["survivor"] = st.sidebar.selectbox(
            "Survivor", ["elitism", "generational"],
            index=["elitism", "generational"].index(preset["survivor"]),
        )
    else:
        ga_cfg["selection"] = st.sidebar.selectbox("Selection", ["tournament", "roulette", "rank"])
        ga_cfg["crossover"] = st.sidebar.selectbox("Crossover", ["single_point", "two_point", "uniform"])
        ga_cfg["mutation"]  = st.sidebar.selectbox("Mutation",  ["bit_flip", "swap", "inversion"])
        ga_cfg["survivor"]  = st.sidebar.selectbox("Survivor",  ["elitism", "generational"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Runs & Seeds**")

use_config_seeds = st.sidebar.checkbox("Use config.py SEEDS list", value=True)
num_runs = st.sidebar.slider("Number of runs", 1, 30, 5)

if use_config_seeds:
    seed_start = SEEDS[0]
    seeds_to_use = SEEDS[:num_runs]
    st.sidebar.caption(f"Using seeds: {seeds_to_use}")
else:
    seed_start = st.sidebar.number_input("Seed start", value=42, step=1)
    seeds_to_use = [int(seed_start) + i for i in range(num_runs)]

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear ALL cached results"):
    keys_to_del = [k for k in st.session_state if k.startswith("results_")]
    for k in keys_to_del:
        del st.session_state[k]
    st.sidebar.success(f"Cleared {len(keys_to_del)} cached result(s).")

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧬 GA & PSO Feature Selection Dashboard")
st.caption("AI420 — Evolutionary Algorithms  |  TB Chest X-Ray Dataset")

tab_run, tab_compare, tab_eda = st.tabs(["🚀 Run Experiment", "⚔️ GA vs PSO Comparison", "🔬 EDA & Plots"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Run Experiment
# ═════════════════════════════════════════════════════════════════════════════
with tab_run:
    cache_key = _make_cache_key(algorithm, ga_cfg, num_runs, seeds_to_use[0])
    cached = get_cached_results(cache_key)

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶️ Run Experiment", type="primary", use_container_width=True)
    with col_info:
        if cached:
            st.success(f"✅ Results loaded from cache — click Run to re-execute with same settings, or change settings for a new run.")
        else:
            st.info("Configure settings in the sidebar, then click Run.")

    if run_btn or cached:
        # Load data (cached after first extraction)
        X_train, X_test, y_train, y_test, feature_names = get_data()

        if run_btn and not cached:
            # ── Execute experiment ────────────────────────────────────────
            results = []
            timings = []

            progress_bar  = st.progress(0, text="Starting…")
            status_text   = st.empty()

            for i, seed in enumerate(seeds_to_use):
                t0 = time.time()
                status_text.markdown(f"⏳ Run **{i+1}/{num_runs}** | seed={seed}")

                if algorithm == "GA":
                    res = run_ga(
                        selection_method=ga_cfg["selection"],
                        crossover_method=ga_cfg["crossover"],
                        mutation_method=ga_cfg["mutation"],
                        survivor_method=ga_cfg["survivor"],
                        seed=seed,
                        verbose=False,
                    )
                else:
                    res = run_pso(X_train, X_test, y_train, y_test, seed=seed, verbose=False)

                elapsed = time.time() - t0
                timings.append(elapsed)
                results.append(res)
                progress_bar.progress(
                    (i + 1) / num_runs,
                    text=f"Run {i+1}/{num_runs} done — acc={res['best_accuracy']:.4f} ({elapsed:.1f}s)"
                )

            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Done! Total time: {sum(timings):.1f}s")

            # Cache results in session_state
            set_cached_results(cache_key, {"results": results, "timings": timings})
            cached = get_cached_results(cache_key)

        # ── Display results ───────────────────────────────────────────────
        results = cached["results"]
        timings = cached["timings"]

        acc   = [r["best_accuracy"] for r in results]
        feats = [r["num_features"]  for r in results]
        total = len(results[0]["best_individual"])

        # Metrics row
        st.markdown("---")
        st.subheader("📊 Summary Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mean Accuracy",  f"{np.mean(acc):.4f}")
        m2.metric("Best Accuracy",  f"{np.max(acc):.4f}")
        m3.metric("Worst Accuracy", f"{np.min(acc):.4f}")
        m4.metric("Avg Features",   f"{np.mean(feats):.1f} / {total}")
        m5.metric("Avg Reduction",  f"{(1 - np.mean(feats)/total)*100:.1f}%")

        # Charts
        st.markdown("---")
        st.subheader("📈 Evolution & Distribution")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Fitness Curve (last run)**")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(results[-1]["history_best"], color="#2563EB", linewidth=1.8)
            ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness")
            ax.set_title("Convergence Curve")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.markdown("**Accuracy Distribution**")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.boxplot(acc, patch_artist=True,
                        boxprops=dict(facecolor="#DBEAFE"),
                        medianprops=dict(color="#2563EB", linewidth=2))
            ax2.set_ylabel("Accuracy")
            ax2.set_title(f"Accuracy over {num_runs} runs")
            ax2.grid(alpha=0.3, axis="y")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with c3:
            st.markdown("**All Runs — Accuracy**")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.bar(range(1, num_runs + 1), acc, color="#3B82F6", alpha=0.8)
            ax3.axhline(np.mean(acc), color="#DC2626", linestyle="--", linewidth=1.5,
                        label=f"Mean={np.mean(acc):.4f}")
            ax3.set_xlabel("Run"); ax3.set_ylabel("Accuracy")
            ax3.set_title("Per-Run Accuracy")
            ax3.legend(fontsize=8)
            ax3.grid(alpha=0.3, axis="y")
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        # Results table
        st.markdown("---")
        st.subheader("📋 Per-Run Results")
        df_runs = pd.DataFrame({
            "Run":          list(range(1, num_runs + 1)),
            "Seed":         seeds_to_use[:len(results)],
            "Accuracy":     [f"{a:.4f}" for a in acc],
            "Features":     feats,
            "Reduction %":  [f"{(1 - f/total)*100:.1f}%" for f in feats],
            "Time (s)":     [f"{t:.1f}" for t in timings],
        })
        st.dataframe(df_runs, use_container_width=True, hide_index=True)

        # Selected features from best run
        st.markdown("---")
        st.subheader("🔍 Selected Features (Best Run)")
        best_run = results[int(np.argmax(acc))]
        idx = np.where(best_run["best_individual"] == 1)[0]
        df_feat = pd.DataFrame({"#": idx, "Feature Name": [feature_names[i] for i in idx]})
        st.dataframe(df_feat, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — GA vs PSO Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("⚔️ GA vs PSO Comparison")

    summary_path = Path("results") / "summary_report.csv"

    if summary_path.exists():
        try:
            df_raw = pd.read_csv(summary_path, header=[0, 1])
            df_raw.columns = ["algorithm", "config", "acc_mean", "acc_std", "reduction_mean", "runtime_mean"]
            df_raw = df_raw.dropna()
            df_raw["acc_mean"]       = pd.to_numeric(df_raw["acc_mean"],       errors="coerce")
            df_raw["acc_std"]        = pd.to_numeric(df_raw["acc_std"],        errors="coerce")
            df_raw["reduction_mean"] = pd.to_numeric(df_raw["reduction_mean"], errors="coerce")
            df_raw["runtime_mean"]   = pd.to_numeric(df_raw["runtime_mean"],   errors="coerce")
            df_raw = df_raw.dropna()
        except Exception:
            # Fallback: skip first 2 rows
            df_raw = pd.read_csv(summary_path, skiprows=2, header=None)
            df_raw.columns = ["algorithm", "config", "acc_mean", "acc_std", "reduction_mean", "runtime_mean"]
            df_raw = df_raw.dropna()
            for col in ["acc_mean", "acc_std", "reduction_mean", "runtime_mean"]:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            df_raw = df_raw.dropna()

        summary = (
            df_raw.groupby("algorithm")[["acc_mean", "reduction_mean", "runtime_mean"]]
            .mean()
            .reset_index()
        )

        # KPI cards
        ga_row  = summary[summary["algorithm"] == "GA"]
        pso_row = summary[summary["algorithm"] == "PSO"]

        if not ga_row.empty and not pso_row.empty:
            g = ga_row.iloc[0]
            p = pso_row.iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("GA Mean Accuracy",  f"{g['acc_mean']:.4f}",
                      delta=f"{g['acc_mean'] - p['acc_mean']:+.4f} vs PSO")
            k2.metric("PSO Mean Accuracy", f"{p['acc_mean']:.4f}")
            k3.metric("GA Reduction",      f"{g['reduction_mean']:.1f}%",
                      delta=f"{g['reduction_mean'] - p['reduction_mean']:+.1f}% vs PSO")

        st.markdown("---")
        ca, cb, cc = st.columns(3)

        with ca:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(summary["algorithm"], summary["acc_mean"],
                   color=["#2563EB", "#DC2626"], alpha=0.85)
            ax.set_title("Mean Accuracy"); ax.set_ylabel("Accuracy"); ax.grid(alpha=0.3, axis="y")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with cb:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(summary["algorithm"], summary["reduction_mean"],
                   color=["#2563EB", "#DC2626"], alpha=0.85)
            ax.set_title("Mean Feature Reduction %"); ax.set_ylabel("Reduction %")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with cc:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(summary["algorithm"], summary["runtime_mean"],
                   color=["#2563EB", "#DC2626"], alpha=0.85)
            ax.set_title("Mean Runtime (s)"); ax.set_ylabel("Seconds")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.subheader("Full Summary Table")
        st.dataframe(df_raw, use_container_width=True, hide_index=True)

    else:
        st.warning(
            "⚠️ `results/summary_report.csv` not found. "
            "Run `python experiments/experiment_runner.py` first to generate it."
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA & Static Plots
# ═════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("🔬 EDA & Pre-generated Plots")

    plots_dir = Path("results") / "plots"
    plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

    if plot_files:
        # Show 2 per row
        for i in range(0, len(plot_files), 2):
            row_cols = st.columns(2)
            for j, col in enumerate(row_cols):
                if i + j < len(plot_files):
                    p = plot_files[i + j]
                    col.image(str(p), caption=p.stem, use_container_width=True)
    else:
        st.info(
            "No plots found in `results/plots/`. "
            "Run `python src/visualizer.py` to generate them."
        )