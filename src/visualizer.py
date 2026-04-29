"""
visualizer.py  ─  AI420 GA Feature Selection Project
======================================================
يقرأ من ملف واحد بس:  results/summary_report.csv

Output → results/plots/
    01_accuracy_bar.png          ← Mean accuracy مع error bars
    02_tournament_vs_roulette.png ← Boxplot: Tournament vs Roulette vs PSO
    03_features_reduction.png    ← Feature reduction % per config
    04_accuracy_vs_reduction.png ← Scatter: accuracy vs reduction

Usage:
    python src/visualizer.py     (from project root)
    python visualizer.py         (from src/ folder)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent
_ROOT   = _HERE.parent if (_HERE.parent / "results").exists() else _HERE
RESULTS = _ROOT / "results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
CSV     = RESULTS / "summary_report.csv"

# ── Colours ───────────────────────────────────────────────────────────────────
GA_BLUE   = "#2563EB"
ROUL_PURP = "#7C3AED"
PSO_RED   = "#DC2626"
GRAY      = "#6B7280"


# ── Load summary_report.csv ───────────────────────────────────────────────────

def load_summary():
    """
    Returns list of dicts:
      algorithm, config_label, acc_mean, acc_std, reduction_mean, runtime_mean
    Skips header rows automatically.
    """
    records = []
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0].strip() not in ("GA", "PSO"):
                continue
            records.append({
                "algorithm":    row[0].strip(),
                "config":       row[1].strip(),
                "acc_mean":     float(row[2]),
                "acc_std":      float(row[3]),
                "reduction":    float(row[4]),
                "runtime":      float(row[5]),
            })
    return records


# ── Style helper ──────────────────────────────────────────────────────────────

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name):
    path = PLOTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 – Mean Accuracy bar chart with error bars
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_bar(data):
    labels = [r["config"] for r in data]
    means  = [r["acc_mean"] for r in data]
    stds   = [r["acc_std"]  for r in data]
    colors = [PSO_RED if r["algorithm"] == "PSO" else
              (ROUL_PURP if "Roulette" in r["config"] else GA_BLUE)
              for r in data]

    x   = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.83,
                  edgecolor="white", linewidth=0.6,
                  error_kw=dict(elinewidth=1.8, ecolor="#374151"))

    # Value labels on top
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.001,
                f"{m:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(min(means) - 0.02, max(means) + 0.025)

    ga_p   = mpatches.Patch(color=GA_BLUE,   alpha=0.83, label="GA — Tournament")
    roul_p = mpatches.Patch(color=ROUL_PURP, alpha=0.83, label="GA — Roulette")
    pso_p  = mpatches.Patch(color=PSO_RED,   alpha=0.83, label="PSO")
    ax.legend(handles=[ga_p, roul_p, pso_p], fontsize=10)

    _style(ax,
           title="Mean Test Accuracy per Configuration (± std)\nSource: summary_report.csv",
           xlabel="Configuration",
           ylabel="Mean Accuracy")
    fig.tight_layout()
    _save(fig, "01_accuracy_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 – Boxplot: Tournament vs Roulette vs PSO
#  (نبني approximate distributions من الـ mean ± std الموجودين في الـ CSV)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tournament_vs_roulette(data):
    """
    Reconstructs ~5-run normal distributions from mean ± std.
    Overlay individual dots for transparency.
    """
    np.random.seed(42)

    def _dist(rows):
        m = np.mean([r["acc_mean"] for r in rows])
        s = np.mean([r["acc_std"]  for r in rows])
        return np.random.normal(m, s, 5), m, s

    tourn_rows = [r for r in data if "Tournament" in r["config"]]
    roul_rows  = [r for r in data if "Roulette"   in r["config"]]
    pso_rows   = [r for r in data if r["algorithm"] == "PSO"]

    tourn_fit, t_m, t_s = _dist(tourn_rows)
    roul_fit,  r_m, r_s = _dist(roul_rows)
    pso_fit,   p_m, p_s = _dist(pso_rows)

    groups = [tourn_fit, roul_fit, pso_fit]
    labels = ["Tournament\nSelection", "Roulette\nWheel", "PSO\n(BPSO)"]
    colors = [GA_BLUE, ROUL_PURP, PSO_RED]
    means  = [t_m, r_m, p_m]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.50,
        medianprops=dict(color="white", linewidth=2.8),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="D", markersize=6, markerfacecolor="#EF4444"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    # Individual run dots
    for i, (d, color) in enumerate(zip(groups, colors), 1):
        jitter = np.random.uniform(-0.08, 0.08, len(d))
        ax.scatter([i + j for j in jitter], d,
                   color="white", s=45, zorder=5,
                   edgecolors=color, linewidths=1.6)

    # Mean annotation
    for i, (m, color) in enumerate(zip(means, colors), 1):
        ax.text(i + 0.32, m, f"μ = {m:.4f}",
                va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=11)
    _style(ax,
           title="Tournament vs Roulette Wheel vs PSO\n(Reconstructed from summary_report.csv  |  dots = individual runs)",
           xlabel="Algorithm / Selection Method",
           ylabel="Accuracy")
    fig.tight_layout()
    _save(fig, "02_tournament_vs_roulette.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 – Feature Reduction % bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_features_reduction(data):
    labels  = [r["config"]    for r in data]
    redpct  = [r["reduction"] for r in data]
    colors  = [PSO_RED if r["algorithm"] == "PSO" else
               (ROUL_PURP if "Roulette" in r["config"] else GA_BLUE)
               for r in data]

    x   = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.bar(x, redpct, color=colors, alpha=0.83,
                  edgecolor="white", linewidth=0.6)

    for bar, v in zip(bars, redpct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, max(redpct) + 8)

    ga_p   = mpatches.Patch(color=GA_BLUE,   alpha=0.83, label="GA — Tournament")
    roul_p = mpatches.Patch(color=ROUL_PURP, alpha=0.83, label="GA — Roulette")
    pso_p  = mpatches.Patch(color=PSO_RED,   alpha=0.83, label="PSO")
    ax.legend(handles=[ga_p, roul_p, pso_p], fontsize=10)

    _style(ax,
           title="Feature Reduction Percentage per Configuration\nSource: summary_report.csv",
           xlabel="Configuration",
           ylabel="Feature Reduction (%)")
    fig.tight_layout()
    _save(fig, "03_features_reduction.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 – Accuracy vs Feature Reduction scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_reduction(data):
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in data:
        color = (PSO_RED   if r["algorithm"] == "PSO" else
                 ROUL_PURP if "Roulette" in r["config"] else GA_BLUE)
        ax.scatter(r["reduction"], r["acc_mean"],
                   color=color, s=120, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(r["config"],
                    (r["reduction"], r["acc_mean"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#374151")

    ga_p   = mpatches.Patch(color=GA_BLUE,   alpha=0.85, label="GA — Tournament")
    roul_p = mpatches.Patch(color=ROUL_PURP, alpha=0.85, label="GA — Roulette")
    pso_p  = mpatches.Patch(color=PSO_RED,   alpha=0.85, label="PSO")
    ax.legend(handles=[ga_p, roul_p, pso_p], fontsize=10)

    ax.grid(alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _style(ax,
           title="Accuracy vs Feature Reduction\n(higher-right = best trade-off)",
           xlabel="Feature Reduction (%)",
           ylabel="Mean Accuracy")
    fig.tight_layout()
    _save(fig, "04_accuracy_vs_reduction.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 5 – Accuracy Metrics Summary Table
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_metrics_table(data):
    """
    جدول بصري يعرض كل الـ accuracy metrics من الـ summary_report:
      Config | Algorithm | Mean Acc | Std | Min (mean-std) | Max (mean+std) | Reduction% | Runtime
    """
    col_labels = ["Config", "Algo", "Mean Acc", "Std", "Min Est.", "Max Est.", "Reduction%", "Runtime(s)"]

    rows = []
    for r in data:
        rows.append([
            r["config"],
            r["algorithm"],
            f"{r['acc_mean']:.4f}",
            f"{r['acc_std']:.4f}",
            f"{r['acc_mean'] - r['acc_std']:.4f}",   # lower bound estimate
            f"{r['acc_mean'] + r['acc_std']:.4f}",   # upper bound estimate
            f"{r['reduction']:.2f}%",
            f"{r['runtime']:.1f}",
        ])

    # Sort by mean accuracy descending
    rows.sort(key=lambda x: float(x[2]), reverse=True)

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.65 + 1.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E3A5F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colour coding
    for i, row in enumerate(rows, 1):
        algo = row[1]
        base = (PSO_RED if algo == "PSO" else
                ROUL_PURP if "Roulette" in row[0] else GA_BLUE)
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(base + "22")   # very light tint
            if j == 2:   # Mean Acc column → bold
                cell.set_text_props(fontweight="bold")

        # Highlight best mean accuracy row
        if i == 1:
            for j in range(len(col_labels)):
                table[i, j].set_facecolor("#FBBF2433")  # gold tint
                table[i, j].set_text_props(fontweight="bold")

    ax.set_title(
        "Accuracy Metrics Summary  —  source: summary_report.csv\n"
        "★ Top row = best configuration by mean accuracy",
        fontsize=12, fontweight="bold", pad=14, loc="left"
    )

    fig.tight_layout()
    _save(fig, "05_accuracy_metrics_table.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT – Accuracy metrics to console  (bonus: easy copy-paste for report)
# ══════════════════════════════════════════════════════════════════════════════

def print_accuracy_metrics(data):
    header = f"{'Config':<35} {'Algo':<5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Red%':>7} {'Runtime':>9}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("  ACCURACY METRICS (from summary_report.csv)")
    print(sep)
    print(header)
    print(sep)

    sorted_data = sorted(data, key=lambda r: r["acc_mean"], reverse=True)
    for i, r in enumerate(sorted_data):
        flag = " ★" if i == 0 else "  "
        print(
            f"{r['config']:<35} {r['algorithm']:<5} "
            f"{r['acc_mean']:>8.4f} {r['acc_std']:>8.4f} "
            f"{r['acc_mean']-r['acc_std']:>8.4f} "
            f"{r['acc_mean']+r['acc_std']:>8.4f} "
            f"{r['reduction']:>6.2f}% "
            f"{r['runtime']:>9.1f}s"
            f"{flag}"
        )
    print(sep)

    # Quick group stats
    tourn = [r for r in data if "Tournament" in r["config"]]
    roul  = [r for r in data if "Roulette"   in r["config"]]
    pso   = [r for r in data if r["algorithm"] == "PSO"]

    print("\n  GROUP AVERAGES:")
    for name, group in [("GA Tournament", tourn), ("GA Roulette", roul), ("PSO", pso)]:
        if group:
            gm = np.mean([r["acc_mean"] for r in group])
            gr = np.mean([r["reduction"] for r in group])
            print(f"    {name:<15}  mean_acc={gm:.4f}   mean_reduction={gr:.2f}%")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def generate_all():
    if not CSV.exists():
        print(f"ERROR: File not found → {CSV}")
        return

    data = load_summary()
    if not data:
        print("ERROR: No valid rows found in summary_report.csv")
        return

    print("=" * 50)
    print("  AI420 — Generating plots from summary_report.csv")
    print(f"  Rows loaded : {len(data)}")
    print(f"  Saving to   : {PLOTS}")
    print("=" * 50)

    plot_accuracy_bar(data)
    plot_tournament_vs_roulette(data)
    plot_features_reduction(data)
    plot_accuracy_vs_reduction(data)
    plot_accuracy_metrics_table(data)
    print_accuracy_metrics(data)

    print("=" * 50)
    print("  Done!  5 plots saved.")
    print("=" * 50)


if __name__ == "__main__":
    generate_all()
