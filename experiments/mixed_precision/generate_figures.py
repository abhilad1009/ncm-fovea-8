"""Generate 5 publication-quality figures for the Fovea-8 Inference NCM mixed-precision study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "outputs" / "mixed_precision" / "eval_full" / "summary_full.json"
BENCH_PATH = ROOT / "outputs" / "mixed_precision" / "benchmark" / "summary_full.json"
TRAIN_PATH = ROOT / "outputs" / "mixed_precision" / "train" / "summary.json"
FIG_DIR = ROOT / "outputs" / "mixed_precision" / "report" / "figures"

MODES = [
    "official_bf16",
    "pure_bf16",
    "coarse_bf16_fine_fp8",
    "coarse_bf16_fine_fp8_w8",
    "coarse_fp8_fine_bf16",
    "fp8_all",
]

MODE_SHORT = {
    "official_bf16": "Official BF16",
    "pure_bf16": "Pure BF16",
    "coarse_bf16_fine_fp8": "Fovea-Dual",
    "coarse_bf16_fine_fp8_w8": "Fovea-8 (W8)",
    "coarse_fp8_fine_bf16": "Reverse (FP8-C)",
    "fp8_all": "FP8-All",
}

MODE_COLORS = {
    "official_bf16": "#6C757D",
    "pure_bf16": "#2B5B84",
    "coarse_bf16_fine_fp8": "#00A6A6",
    "coarse_bf16_fine_fp8_w8": "#E66101",
    "coarse_fp8_fine_bf16": "#D95F8E",
    "fp8_all": "#7B3294",
}

TASKS = ["sudoku-easy", "sudoku-medium", "sudoku-hard", "tinystories", "wmt14", "xsum"]
TASK_TITLES = {
    "sudoku-easy": "Sudoku-Easy",
    "sudoku-medium": "Sudoku-Medium",
    "sudoku-hard": "Sudoku-Hard",
    "tinystories": "TinyStories",
    "wmt14": "WMT14 De-En",
    "xsum": "XSum",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def fig1_scale_budget_and_weights():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7), gridspec_kw={"width_ratios": [1.1, 1.05]})

    families = [
        "Sudoku\n(12 levels, T=322)",
        "WMT14 De-En\n(11 levels, T=313)",
        "XSum\n(9 levels, T=200)",
        "TinyStories\n(12 levels, T=641)",
    ]
    coarse_pct = [100 * 35 / 322, 100 * 47 / 313, 100 * 34 / 200, 100 * 121 / 641]
    fine_pct = [100 - c for c in coarse_pct]
    y = np.arange(len(families))

    ax1.barh(y, coarse_pct, height=0.52, color="#2B5B84", label="Coarse Half (BF16 Protected)")
    ax1.barh(y, fine_pct, left=coarse_pct, height=0.52, color="#E66101", label="Fine Half (FP8 E4M3 Accelerated)")
    for i, (c, f) in enumerate(zip(coarse_pct, fine_pct)):
        ax1.text(c / 2, i, f"{c:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
        ax1.text(c + f / 2, i, f"{f:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9.5)
    ax1.set_yticks(y, families)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of Generated Concept Positions (%)")
    ax1.set_title("A. Coarse vs. Fine Position Budget Across NCM Tasks")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=8.8)

    storage_modes = ["official_bf16", "coarse_bf16_fine_fp8", "pure_bf16", "coarse_bf16_fine_fp8_w8", "fp8_all"]
    storage_labels = [
        "Official BF16\n(FP32 Wt)",
        "Fovea-Dual\n(BF16+FP8)",
        "Pure BF16\n(BF16 Wt)",
        "Fovea-8 (W8)\n(1x FP8 Wt)",
        "FP8-All\n(1x FP8 Wt)",
    ]
    bytes_per_param = [4.0, 3.0, 2.0, 1.0, 1.0]
    colors = [MODE_COLORS[m] for m in storage_modes]
    x = np.arange(len(storage_modes))
    bars = ax2.bar(x, bytes_per_param, width=0.52, color=colors, edgecolor="#222222", linewidth=0.6)
    notes = ["4.0 B/p\n(W32A16)", "3.0 B/p\n(Dual)", "2.0 B/p\n(W16A16)", "1.0 B/p\n(W8A16→A8)", "1.0 B/p\n(W8A8)"]
    for bar, note in zip(bars, notes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, note,
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.set_xticks(x, storage_labels, fontsize=8.4)
    ax2.set_ylim(0, 4.85)
    ax2.set_ylabel("Shared Backbone Storage (Bytes / Param)")
    ax2.set_title("B. Shared Backbone Weight Footprint & Compute Path")

    fig.suptitle("Fovea-8 Inference Design: Coarse Scales Need Precision; Fine Scales Dominate Compute", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_scale_budget_and_weights.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_quality_all_tasks(eval_data: dict):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    axes = axes.ravel()

    specs = [
        ("sudoku-easy", "solved_pct", "Solved Rate (%) ↑", (98.5, 100.0)),
        ("sudoku-medium", "solved_pct", "Solved Rate (%) ↑", (80.0, 99.0)),
        ("sudoku-hard", "solved_pct", "Solved Rate (%) ↑", (80.0, 88.5)),
        ("tinystories", "mean_genppl", "GenPPL (GPT-2 Large) ↓", (15.5, 17.6)),
        ("wmt14", "bleu", "SacreBLEU ↑", (20.5, 22.3)),
        ("xsum", "rouge1", "ROUGE-1 ↑", (34.0, 35.4)),
    ]

    x = np.arange(len(MODES))
    short_labels = [MODE_SHORT[m] for m in MODES]

    for ax, (task, key, ylabel, ylim) in zip(axes, specs):
        vals = [eval_data[task][m]["metrics"][key] for m in MODES]
        colors = [MODE_COLORS[m] for m in MODES]
        bars = ax.bar(x, vals, width=0.62, color=colors, edgecolor="#222222", linewidth=0.5)
        bars[3].set_linewidth(1.8)
        bars[3].set_edgecolor("#111111")
        ax.axhline(vals[0], color="#444444", linestyle=":", linewidth=1.1, alpha=0.85)

        span = ylim[1] - ylim[0]
        for bar, v in zip(bars, vals):
            fmt = f"{v:.2f}" if key != "solved_pct" else f"{v:.2f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * span, fmt,
                    ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_xticks(x, short_labels, rotation=24, ha="right", fontsize=8.2)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(TASK_TITLES[task])

    fig.suptitle("Quality Across All 6 Benchmarks: Fovea-8 Inference Preserves BF16 Accuracy While Coarse-FP8 Degrades", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig2_quality_all_tasks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_pareto_frontier(eval_data: dict, bench_data: list[dict]):
    lookup = {(r["task"], r["mode"], r["batch_size"]): r for r in bench_data}
    panels = [
        ("sudoku-easy", 64, "solved_pct", "Solved Rate (%) ↑", False),
        ("sudoku-medium", 64, "solved_pct", "Solved Rate (%) ↑", False),
        ("sudoku-hard", 64, "solved_pct", "Solved Rate (%) ↑", False),
        ("tinystories", 64, "mean_genppl", "GenPPL ↓ (Inverted: Up = Better)", True),
        ("wmt14", 64, "bleu", "SacreBLEU ↑", False),
        ("xsum", 32, "rouge1", "ROUGE-1 ↑", False),
    ]

    # Per-panel non-overlapping label placements: (dx_pt, dy_pt, ha)
    custom_offsets = {
        "sudoku-easy": {
            "official_bf16": (0, 14, "left"),
            "pure_bf16": (0, -18, "center"),
            "coarse_bf16_fine_fp8": (-6, -15, "right"),
            "coarse_bf16_fine_fp8_w8": (-4, 16, "center"),
            "coarse_fp8_fine_bf16": (-8, 10, "right"),
            "fp8_all": (-10, -12, "right"),
        },
        "sudoku-medium": {
            "official_bf16": (0, 14, "left"),
            "pure_bf16": (-4, 15, "center"),
            "coarse_bf16_fine_fp8": (-6, -15, "right"),
            "coarse_bf16_fine_fp8_w8": (-4, 16, "center"),
            "coarse_fp8_fine_bf16": (10, 6, "left"),
            "fp8_all": (-10, 12, "right"),
        },
        "sudoku-hard": {
            "official_bf16": (4, -15, "left"),
            "pure_bf16": (0, 14, "center"),
            "coarse_bf16_fine_fp8": (-8, -16, "right"),
            "coarse_bf16_fine_fp8_w8": (-10, -14, "right"),
            "coarse_fp8_fine_bf16": (10, -10, "left"),
            "fp8_all": (-10, 12, "right"),
        },
        "tinystories": {
            "official_bf16": (6, -15, "left"),
            "pure_bf16": (8, -14, "left"),
            "coarse_bf16_fine_fp8": (8, -16, "left"),
            "coarse_bf16_fine_fp8_w8": (-8, 16, "right"),
            "coarse_fp8_fine_bf16": (-8, 14, "right"),
            "fp8_all": (-10, 12, "right"),
        },
        "wmt14": {
            "official_bf16": (10, -12, "left"),
            "pure_bf16": (8, -14, "left"),
            "coarse_bf16_fine_fp8": (-8, -16, "right"),
            "coarse_bf16_fine_fp8_w8": (-10, 12, "right"),
            "coarse_fp8_fine_bf16": (6, 12, "left"),
            "fp8_all": (-10, 12, "right"),
        },
        "xsum": {
            "official_bf16": (10, -12, "left"),
            "pure_bf16": (-8, 12, "right"),
            "coarse_bf16_fine_fp8": (8, 10, "left"),
            "coarse_bf16_fine_fp8_w8": (-10, 12, "right"),
            "coarse_fp8_fine_bf16": (-8, -14, "right"),
            "fp8_all": (-10, -12, "right"),
        },
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    axes = axes.ravel()

    for ax, (task, bs, mkey, ylabel, invert_y) in zip(axes, panels):
        base_eff = lookup[(task, "official_bf16", bs)]["samples_per_sec_per_gib"]
        xs, ys, raw_ys = [], [], []
        for mode in MODES:
            b = lookup[(task, mode, bs)]
            q = eval_data[task][mode]["metrics"][mkey]
            xs.append(b["samples_per_sec_per_gib"] / base_eff)
            raw_ys.append(q)
            ys.append(-q if invert_y else q)

        # Compute non-dominated Pareto frontier (maximizing x and higher-is-better ys)
        pts = sorted([(xs[i], ys[i], raw_ys[i]) for i in range(len(MODES))], key=lambda p: (-p[0], -p[1]))
        frontier = []
        best_y = -float("inf")
        for px, py, ry in pts:
            if py >= best_y - 1e-12:
                frontier.append((px, ry))
                best_y = py
        frontier.sort(key=lambda p: p[0])
        if len(frontier) >= 2:
            ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                    linestyle="--", color="#E66101", alpha=0.55, linewidth=1.6, zorder=2, label="Pareto Frontier")

        for i, mode in enumerate(MODES):
            is_ours = mode == "coarse_bf16_fine_fp8_w8"
            edge = "#111111" if is_ours else "#333333"
            lw = 2.0 if is_ours else 0.9
            marker = "*" if is_ours else "o"
            s = 320 if is_ours else 105
            ax.scatter(xs[i], raw_ys[i], s=s, color=MODE_COLORS[mode], edgecolors=edge, linewidth=lw, marker=marker, alpha=0.95, zorder=4)

        ax.axvline(1.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
        x_min, x_max = min(xs), max(xs)
        x_span = x_max - x_min
        ax.set_xlim(x_min - 0.14 * x_span, x_max + 0.24 * x_span)

        y_min, y_max = min(raw_ys), max(raw_ys)
        y_span = max(y_max - y_min, 0.05)
        ax.set_ylim(y_min - 0.24 * y_span, y_max + 0.28 * y_span)
        if invert_y:
            ax.invert_yaxis()

        for i, mode in enumerate(MODES):
            dx, dy, ha = custom_offsets[task][mode]
            is_ours = mode == "coarse_bf16_fine_fp8_w8"
            label = f"{MODE_SHORT[mode]} ({xs[i]:.2f}×)"
            ax.annotate(
                label,
                (xs[i], raw_ys[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=7.6,
                fontweight="bold" if is_ours else "normal",
                color="#B84400" if is_ours else "#111111",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.65, shrinkA=0, shrinkB=4),
                zorder=5,
            )

        ax.set_xlabel(f"Relative Throughput / GiB vs. Official BF16 (B={bs}) →")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{TASK_TITLES[task]} (B={bs})")

    fig.suptitle("Quality vs. Memory-Normalized Speed (Throughput / GiB Relative to Official BF16 = 1.00×) — Upper-Right = Better", fontsize=12.8, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig3_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_speed_and_vram(bench_data: list[dict]):
    lookup = {(r["task"], r["mode"], r["batch_size"]): r for r in bench_data}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5))

    plot_modes = ["official_bf16", "pure_bf16", "coarse_bf16_fine_fp8", "coarse_bf16_fine_fp8_w8", "fp8_all"]
    x = np.arange(len(TASKS))
    width = 0.155
    for idx, mode in enumerate(plot_modes):
        speedups = []
        for t in TASKS:
            bs = 32 if t == "xsum" else 64
            base = lookup[(t, "official_bf16", bs)]["samples_per_second"]
            cur = lookup[(t, mode, bs)]["samples_per_second"]
            speedups.append(cur / base)
        offset = (idx - (len(plot_modes) - 1) / 2) * width
        bars = ax1.bar(x + offset, speedups, width=width, color=MODE_COLORS[mode], label=MODE_SHORT[mode], edgecolor="#222222", linewidth=0.45)
        if mode == "coarse_bf16_fine_fp8_w8":
            for b, sp in zip(bars, speedups):
                ax1.text(b.get_x() + b.get_width() / 2, sp + 0.03, f"{sp:.2f}×", ha="center", va="bottom",
                         fontsize=7.6, fontweight="bold", color="#B84400",
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))
    ax1.axhline(1.0, color="#333333", linestyle=":", linewidth=1.0)
    ax1.set_xticks(x, [TASK_TITLES[t] for t in TASKS], rotation=20, ha="right")
    ax1.set_ylim(0, 2.08)
    ax1.set_ylabel("Throughput Speedup vs. Official BF16 (↑)")
    ax1.set_title("A. High-Batch Inference Speedup (B=64; XSum B=32)")
    ax1.legend(loc="upper right", ncol=2, fontsize=8.0)

    for idx, mode in enumerate(plot_modes):
        vrams = []
        for t in TASKS:
            bs = 32 if t == "xsum" else 64
            vrams.append(lookup[(t, mode, bs)]["peak_allocated_mib"])
        offset = (idx - (len(plot_modes) - 1) / 2) * width
        bars = ax2.bar(x + offset, vrams, width=width, color=MODE_COLORS[mode], label=MODE_SHORT[mode], edgecolor="#222222", linewidth=0.45)
        if mode == "coarse_bf16_fine_fp8_w8":
            for b, vr in zip(bars, vrams):
                ax2.text(b.get_x() + b.get_width() / 2, vr + 18, f"{vr:.0f}", ha="center", va="bottom",
                         fontsize=7.5, fontweight="bold", color="#B84400")
    ax2.set_xticks(x, [TASK_TITLES[t] for t in TASKS], rotation=20, ha="right")
    ax2.set_ylim(0, 1260)
    ax2.set_ylabel("Peak Allocated VRAM (MiB) ↓")
    ax2.set_title("B. Peak Allocated GPU Memory During Inference (MiB)")
    ax2.legend(loc="upper left", ncol=2, fontsize=8.0)

    fig.suptitle("Fovea-8 Inference Speedup and Memory Reduction Across All 6 Benchmarks", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_speed_and_vram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_memory_efficiency_and_training(bench_data: list[dict], train_data: dict):
    lookup = {(r["task"], r["mode"], r["batch_size"]): r for r in bench_data}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.3), gridspec_kw={"width_ratios": [1.35, 1.0]})

    eff_modes = ["official_bf16", "pure_bf16", "coarse_bf16_fine_fp8", "coarse_bf16_fine_fp8_w8", "fp8_all"]
    x = np.arange(len(TASKS))
    width = 0.155
    for idx, mode in enumerate(eff_modes):
        ratios = []
        for t in TASKS:
            bs = 32 if t == "xsum" else 64
            base_eff = lookup[(t, "official_bf16", bs)]["samples_per_sec_per_gib"]
            cur_eff = lookup[(t, mode, bs)]["samples_per_sec_per_gib"]
            ratios.append(cur_eff / base_eff)
        offset = (idx - (len(eff_modes) - 1) / 2) * width
        bars = ax1.bar(x + offset, ratios, width=width, color=MODE_COLORS[mode], label=MODE_SHORT[mode], edgecolor="#222222", linewidth=0.45)
        if mode == "coarse_bf16_fine_fp8_w8":
            for b, val in zip(bars, ratios):
                ax1.text(b.get_x() + b.get_width() / 2, val + 0.09, f"{val:.2f}×", ha="center", va="bottom",
                         fontsize=7.6, fontweight="bold", color="#B84400",
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))
    ax1.axhline(1.0, color="#333333", linestyle=":", linewidth=1.0)
    ax1.set_xticks(x, [TASK_TITLES[t] for t in TASKS], rotation=20, ha="right")
    ax1.set_ylim(0, 5.15)
    ax1.set_ylabel("Throughput per GiB Gain vs. Official BF16 (↑)")
    ax1.set_title("A. Memory Efficiency Gain (Samples/s per GiB Peak VRAM)")
    ax1.legend(loc="upper right", ncol=2, fontsize=7.8)

    tr = train_data["exp_5a_from_scratch"]
    tr_modes = ["fp32", "bf16", "coarse_bf16_fine_fp8", "fp8_all"]
    tr_labels = ["FP32", "BF16\n(AMP)", "Fovea-Train\n(Coarse BF16\nFine FP8)", "FP8-All"]
    val_total = [tr[m]["val_loss"] for m in tr_modes]
    val_fine = [tr[m]["val_fine_loss"] for m in tr_modes]

    x2 = np.arange(len(tr_modes))
    w2 = 0.34
    b_tot = ax2.bar(x2 - w2 / 2, val_total, width=w2, color="#2B5B84", label="Total Val Loss ↓", edgecolor="#222222", linewidth=0.5)
    b_fin = ax2.bar(x2 + w2 / 2, val_fine, width=w2, color="#E66101", label="Fine-Level Val Loss (L6–11) ↓", edgecolor="#222222", linewidth=0.5)
    for b in list(b_tot) + list(b_fin):
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_xticks(x2, tr_labels, fontsize=8.5)
    ax2.set_ylim(0.92, 1.13)
    ax2.set_ylabel("Validation Cross-Entropy Loss ↓")
    ax2.set_title("B. From-Scratch Training Loss (Sudoku-Hard, 400 Steps)")
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.suptitle("Inference Memory Efficiency (Up to 4.00× for Fovea-8) & Scale-Adaptive Training Stability", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_memory_efficiency_and_training.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eval_data = json.loads(EVAL_PATH.read_text())
    bench_data = json.loads(BENCH_PATH.read_text())
    train_data = json.loads(TRAIN_PATH.read_text())

    fig1_scale_budget_and_weights()
    fig2_quality_all_tasks(eval_data)
    fig3_pareto_frontier(eval_data, bench_data)
    fig4_speed_and_vram(bench_data)
    fig5_memory_efficiency_and_training(bench_data, train_data)
    print(f"Generated 5 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
