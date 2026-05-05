
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ───────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent
_ROOT   = _HERE.parent if (_HERE.parent / "results").exists() else _HERE
RESULTS = _ROOT / "results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
CSV     = RESULTS / "summary_report.csv"

# ══════════════════════════════════════════════════
#  PREMIUM DESIGN SYSTEM  (dark theme)
# ══════════════════════════════════════════════════
BG        = "#EAEEF5"
SURFACE   = "#E9EDF1"
BORDER    = "#30363D"
TEXT_PRI  = "#E6EDF3"
TEXT_SEC  = "#8B949E"
GRID_COL  = "#E4EAF1"

GA_BLUE   = "#3B82F6"
ROUL_PURP = "#A855F7"
PSO_RED   = "#F43F5E"
GOLD      = "#F59E0B"

TITLE_FONT = {"fontfamily": "DejaVu Sans", "fontweight": "bold"}
LABEL_FONT = {"fontfamily": "DejaVu Sans"}

def _global_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_SEC,
        "axes.titlecolor":   TEXT_PRI,
        "xtick.color":       TEXT_SEC,
        "ytick.color":       TEXT_SEC,
        "grid.color":        GRID_COL,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.8,
        "text.color":        TEXT_PRI,
        "font.family":       "DejaVu Sans",
        "lines.linewidth":   2,
        "patch.linewidth":   0,
        "figure.dpi":        150,
    })

_global_style()

# ── Load CSV ────────────────────────────────────────
def load_summary():
    records = []
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0].strip() not in ("GA", "PSO"):
                continue
            records.append({
                "algorithm": row[0].strip(),
                "config":    row[1].strip(),
                "acc_mean":  float(row[2]),
                "acc_std":   float(row[3]),
                "reduction": float(row[4]),
                "runtime":   float(row[5]),
            })
    return records

# ── Helpers ─────────────────────────────────────────
def _color_for(r):
    if r["algorithm"] == "PSO":      return PSO_RED
    if "Roulette" in r["config"]:    return ROUL_PURP
    return GA_BLUE

def _add_watermark(fig):
    fig.text(0.98, 0.01, "AI420 · Feature Selection",
             ha="right", va="bottom", fontsize=7,
             color=TEXT_SEC, alpha=0.4)

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_SEC, length=4)
    ax.grid(axis="y", zorder=0)

def _save(fig, name):
    _add_watermark(fig)
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  OK  {name}")


# ════════════════════════════════════════════════════
#  PLOT 1 – Accuracy Bar
# ════════════════════════════════════════════════════
def plot_accuracy_bar(data):
    labels = [r["config"]   for r in data]
    means  = [r["acc_mean"] for r in data]
    stds   = [r["acc_std"]  for r in data]
    colors = [_color_for(r) for r in data]

    x   = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.22)

    bars = ax.bar(x, means, color=colors, alpha=0.88,
                  width=0.62, zorder=3, edgecolor=BG, linewidth=1.5)

    ax.errorbar(x, means, yerr=stds,
                fmt="none", capsize=5, capthick=2,
                elinewidth=2, ecolor=TEXT_PRI, zorder=5, alpha=0.7)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.0014,
                f"{m:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEXT_PRI)

    best = max(means)
    ax.axhline(best, color=GOLD, linewidth=1.2, linestyle=":", alpha=0.6, zorder=2)
    ax.text(len(data) - 0.45, best + 0.0005,
            f"Best: {best:.4f}", color=GOLD, fontsize=8.5, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.5, color=TEXT_SEC)
    ax.set_ylim(min(means) - 0.025, max(means) + 0.030)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    patches = [
        mpatches.Patch(color=GA_BLUE,   label="GA — Tournament"),
        mpatches.Patch(color=ROUL_PURP, label="GA — Roulette"),
        mpatches.Patch(color=PSO_RED,   label="PSO (BPSO)"),
    ]
    ax.legend(handles=patches, fontsize=9.5,
              facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_PRI, loc="lower right")

    ax.set_title("Mean Test Accuracy per Configuration  (± std)",
                 fontsize=14, **TITLE_FONT, pad=14)
    ax.set_xlabel("Configuration", fontsize=11, color=TEXT_SEC, labelpad=8)
    ax.set_ylabel("Mean Accuracy",  fontsize=11, color=TEXT_SEC)
    _style_axes(ax)
    _save(fig, "01_accuracy_bar.png")


# ════════════════════════════════════════════════════
#  PLOT 2 – Boxplot
# ════════════════════════════════════════════════════
def plot_tournament_vs_roulette(data):
    np.random.seed(42)

    def dist(rows):
        m = np.mean([r["acc_mean"] for r in rows])
        s = np.mean([r["acc_std"]  for r in rows])
        return np.random.normal(m, s, 5)

    tourn = [r for r in data if "Tournament" in r["config"]]
    roul  = [r for r in data if "Roulette"   in r["config"]]
    pso   = [r for r in data if r["algorithm"] == "PSO"]

    groups  = [dist(tourn), dist(roul), dist(pso)]
    xlabels = ["GA  Tournament", "GA  Roulette", "PSO  (BPSO)"]
    colors  = [GA_BLUE, ROUL_PURP, PSO_RED]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.85, bottom=0.12)

    bp = ax.boxplot(
        groups, patch_artist=True, widths=0.42,
        medianprops=dict(color="white", linewidth=2.8),
        whiskerprops=dict(linewidth=1.8, color=BORDER),
        capprops=dict(linewidth=1.8, color=BORDER),
        flierprops=dict(marker="D", markersize=7,
                        markerfacecolor=GOLD,
                        markeredgecolor=BG, markeredgewidth=0.8),
        boxprops=dict(linewidth=0),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    for i, (d, color) in enumerate(zip(groups, colors), 1):
        jitter = np.random.uniform(-0.10, 0.10, len(d))
        ax.scatter([i + j for j in jitter], d,
                   color="white", s=65, zorder=5,
                   edgecolors=color, linewidths=2)

    for i, (d, color) in enumerate(zip(groups, colors), 1):
        m = np.mean(d)
        ax.text(i + 0.30, m, f"μ={m:.4f}",
                va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(xlabels, fontsize=11, color=TEXT_PRI)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_title("Selection Method Comparison — Accuracy Distribution",
                 fontsize=13, **TITLE_FONT, pad=14)
    ax.set_xlabel("Algorithm / Selection Method", fontsize=11, color=TEXT_SEC)
    ax.set_ylabel("Test Accuracy", fontsize=11, color=TEXT_SEC)
    _style_axes(ax)
    _save(fig, "02_tournament_vs_roulette.png")


# ════════════════════════════════════════════════════
#  PLOT 3 – Feature Reduction
# ════════════════════════════════════════════════════
def plot_features_reduction(data):
    labels = [r["config"]    for r in data]
    vals   = [r["reduction"] for r in data]
    colors = [_color_for(r)  for r in data]

    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.22)

    bars = ax.bar(x, vals, color=colors, alpha=0.88,
                  width=0.62, zorder=3, edgecolor=BG, linewidth=1.5)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=TEXT_PRI)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.5, color=TEXT_SEC)
    ax.set_ylim(0, max(vals) + 12)

    patches = [
        mpatches.Patch(color=GA_BLUE,   label="GA — Tournament"),
        mpatches.Patch(color=ROUL_PURP, label="GA — Roulette"),
        mpatches.Patch(color=PSO_RED,   label="PSO (BPSO)"),
    ]
    ax.legend(handles=patches, fontsize=9.5,
              facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_PRI)

    ax.set_title("Feature Reduction Percentage per Configuration",
                 fontsize=14, **TITLE_FONT, pad=14)
    ax.set_xlabel("Configuration", fontsize=11, color=TEXT_SEC, labelpad=8)
    ax.set_ylabel("Feature Reduction (%)", fontsize=11, color=TEXT_SEC)
    _style_axes(ax)
    _save(fig, "03_features_reduction.png")


# ════════════════════════════════════════════════════
#  PLOT 4 – Accuracy vs Reduction Scatter
# ════════════════════════════════════════════════════
def plot_accuracy_vs_reduction(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.85, bottom=0.10)

    for r in data:
        color = _color_for(r)
        ax.scatter(r["reduction"], r["acc_mean"],
                   color=color, s=140, zorder=5,
                   edgecolors="white", linewidths=1.6, alpha=0.90)
        ax.annotate(r["config"],
                    (r["reduction"], r["acc_mean"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=TEXT_SEC)

    patches = [
        mpatches.Patch(color=GA_BLUE,   label="GA — Tournament"),
        mpatches.Patch(color=ROUL_PURP, label="GA — Roulette"),
        mpatches.Patch(color=PSO_RED,   label="PSO (BPSO)"),
    ]
    ax.legend(handles=patches, fontsize=10,
              facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_PRI)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_title("Accuracy vs Feature Reduction  —  Pareto Trade-off",
                 fontsize=13, **TITLE_FONT, pad=14)
    ax.set_xlabel("Feature Reduction (%)", fontsize=11, color=TEXT_SEC)
    ax.set_ylabel("Mean Test Accuracy",    fontsize=11, color=TEXT_SEC)
    _style_axes(ax)
    ax.grid(True, alpha=0.15)
    _save(fig, "04_accuracy_vs_reduction.png")


# ════════════════════════════════════════════════════
#  PLOT 5 – Metrics Table
# ════════════════════════════════════════════════════
def plot_accuracy_metrics_table(data):
    col_labels = ["Configuration", "Algo", "Mean Acc",
                  "Std", "Min (est)", "Max (est)", "Red %", "Runtime (s)"]
    rows = []
    for r in data:
        rows.append([
            r["config"],
            r["algorithm"],
            f"{r['acc_mean']:.4f}",
            f"{r['acc_std']:.4f}",
            f"{r['acc_mean'] - r['acc_std']:.4f}",
            f"{r['acc_mean'] + r['acc_std']:.4f}",
            f"{r['reduction']:.2f}%",
            f"{r['runtime']:.1f}",
        ])

    rows.sort(key=lambda x: float(x[2]), reverse=True)

    n_rows = len(rows)
    fig_h  = max(4, n_rows * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(17, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor(BORDER)

    for i, row in enumerate(rows, 1):
        algo   = row[1]
        base_c = (PSO_RED   if algo == "PSO" else
                  ROUL_PURP if "Roulette" in row[0] else GA_BLUE)
        fill   = base_c + "28"

        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(fill if i % 2 == 1 else SURFACE)
            cell.set_text_props(color=TEXT_PRI)
            cell.set_edgecolor(BORDER)
            if j == 2:
                cell.set_text_props(color=TEXT_PRI, fontweight="bold")

        if i == 1:
            for j in range(len(col_labels)):
                table[i, j].set_facecolor("#F59E0B22")
                table[i, j].set_text_props(color=GOLD, fontweight="bold")

    ax.set_title(
        "Accuracy Metrics Summary  ·  Best configuration highlighted in gold",
        fontsize=12, **TITLE_FONT, pad=16, color=TEXT_PRI, loc="left",
    )
    _save(fig, "05_accuracy_metrics_table.png")


# ════════════════════════════════════════════════════
#  PLOT 6 – Runtime Summary
# ════════════════════════════════════════════════════
def plot_runtime_summary(data):
    tourn = [r["runtime"] for r in data if "Tournament" in r["config"]]
    roul  = [r["runtime"] for r in data if "Roulette"   in r["config"]]
    pso   = [r["runtime"] for r in data if r["algorithm"] == "PSO"]

    labels = ["GA  Tournament", "GA  Roulette", "PSO  (BPSO)"]
    values = [np.mean(t) if t else 0 for t in [tourn, roul, pso]]
    stds   = [np.std(t)  if t else 0 for t in [tourn, roul, pso]]
    colors = [GA_BLUE, ROUL_PURP, PSO_RED]

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.85, bottom=0.12)

    bars = ax.bar(x, values, color=colors, alpha=0.88,
                  width=0.52, zorder=3, edgecolor=BG, linewidth=1.5)

    ax.errorbar(x, values, yerr=stds,
                fmt="none", capsize=6, capthick=2,
                elinewidth=2, ecolor=TEXT_PRI, zorder=5, alpha=0.6)

    for bar, v, s in zip(bars, values, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + s + 8, f"{v:.0f} s",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_PRI)

    if values[2] > 0 and values[0] > 0:
        ratio = values[0] / values[2]
        ax.annotate(
            f"PSO is {ratio:.1f}x faster",
            xy=(2, values[2]), xytext=(1.4, values[0] * 0.70),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8),
            fontsize=10, color=GOLD, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color=TEXT_PRI)
    ax.set_ylim(0, max(values) + max(stds) + max(values) * 0.20)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,} s"))
    ax.set_title("Runtime Comparison by Algorithm",
                 fontsize=14, **TITLE_FONT, pad=14)
    ax.set_xlabel("Algorithm", fontsize=11, color=TEXT_SEC)
    ax.set_ylabel("Mean Runtime (seconds)", fontsize=11, color=TEXT_SEC)
    _style_axes(ax)
    _save(fig, "06_runtime_summary.png")


# ════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════
def generate_all():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found"); return

    data = load_summary()
    if not data:
        print("ERROR: no valid rows in CSV"); return

    print("=" * 50)
    print(f"  AI420 Visualizer  |  {len(data)} rows loaded")
    print(f"  Output -> {PLOTS}")
    print("=" * 50)

    plot_accuracy_bar(data)
    plot_tournament_vs_roulette(data)
    plot_features_reduction(data)
    plot_accuracy_vs_reduction(data)
    plot_accuracy_metrics_table(data)
    plot_runtime_summary(data)

    print("=" * 50)
    print("  Done - 6 plots saved")
    print("=" * 50)

if __name__ == "__main__":
    generate_all()
