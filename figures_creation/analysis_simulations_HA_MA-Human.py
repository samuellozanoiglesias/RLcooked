#!/usr/bin/env python3
"""
HA vs MA Comparison Analysis — Human data version.

Adapts the CSV-driven raincloud figure (analysis_figure3_HA_MA_csv.py) to
human-subject data, which is supplied as two separate WIDE-format CSVs
(one row per participant, one column per condition) instead of one long
SLURM/simulation table:

  scores_human.csv
      columns: "Open High", "Open Mixed",
               "Partially Blocked High", "Partially Blocked Mixed"

  specialization_index_human.csv
      columns: "Open High Ability", "Open Mixed Ability",
               "Partially Blocked High Ability", "Partially Blocked Mixed Ability"

Each column name encodes the two factors of the design:
  - Map / switching-cost condition : "Open" (low s)  vs  "Partially Blocked" (high s)
  - Ability group                  : "High" (HA)      vs  "Mixed" (MA)

The two wide CSVs are melted into long format internally and plotted with
the same raincloud layout used for the simulated data (half-violin +
boxplot + diamond mean + connecting line), so the human and simulation
figures stay visually comparable:

  Panel A: Game score                   <- scores_human.csv
  Panel B: Specialization index        <- specialization_index_human.csv

Usage:
nohup python3 analysis_simulations_HA_MA-Human.py --scores_csv ./human/scores_human.csv --specialization_csv ./human/specialization_index_human.csv > log_analysis_human_HA_MA.out 2>&1 &

Author: Samuel Lozano
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Global Style Configuration (PNAS requirements) — IDENTICAL TO SIMULATION
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        \usepackage{amssymb}
    """,
    "axes.unicode_minus": False,
    # High resolution settings
    "figure.dpi": 300,       # Screen/default DPI
    "savefig.dpi": 600,      # Publication quality saved DPI
    # Font embedding settings (Type 42 is TrueType embedded)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Group / style constants  (IDENTICAL TO SIMULATION FOR COMPARABILITY)
# ---------------------------------------------------------------------------
_SWITCH_ORDER  = ["low", "high"]
_SWITCH_X      = {"low": 1.0, "high": 2.0}
_SWITCH_LABELS = {"low": "Open", "high": "PB"}

_ABILITY_ORDER = ["HA", "MA"]
_SIDE_OFFSET   = {"HA": -0.18, "MA": +0.18}   # HA drawn left, MA drawn right of each x position
_VIOLIN_DIR    = {"HA": -1, "MA": +1}          # violin fans outward from the box

# High contrast colors Reference Style scheme
_COLOR_HA = '#E24A33' # Reddish-Salmon
_COLOR_MA = '#348ABD' # Teal-Blue
_COLOR        = {"HA": _COLOR_HA, "MA": _COLOR_MA}   # salmon / teal
_COLOR_DARK   = {"HA": "black", "MA": "black"} # Dark black outlines for contrast

_LEGEND_LABELS = {"HA": "High ability", "MA": "Mixed ability"}  # no rho — these are humans

# Increased alpha values based on Reference style requirements
_ALPHA_VIOLIN = 0.55  # Increased density
_ALPHA_BOX    = 0.85  # Much more solid, less pastel
_VIOLIN_WIDTH = 0.30
_BOX_WIDTH    = 0.10
_WHISK_CAP    = 0.06

_FS_AXIS  = 30
_FS_TICK_X  = 30
_FS_TICK_Y = 30
_FS_PANEL = 28
_FS_LEG   = 30


# ---------------------------------------------------------------------------
# Wide -> long loading
# ---------------------------------------------------------------------------
def _parse_column_label(col: str) -> tuple[str, str]:
    """
    Parse a wide-format column name like "Open High" or
    "Partially Blocked Mixed Ability" into (switch_cost_condition, ability_type).
    """
    c = col.strip().lower()

    if "open" in c:
        switch = "low"
    elif "partially" in c or "blocked" in c:
        switch = "high"
    else:
        raise ValueError(f"Could not parse switching condition from column: {col!r}")

    if "mixed" in c:
        ability = "MA"
    elif "high" in c:
        ability = "HA"
    else:
        raise ValueError(f"Could not parse ability type from column: {col!r}")

    return switch, ability


def load_wide_human_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a wide-format human CSV (one row per participant, one column per
    Map x Ability condition) and melt it into long format with columns:
    subject_id, switch_cost_condition (low/high), ability_type (HA/MA), value.
    """
    df = pd.read_csv(csv_path)
    df = df.reset_index(names="subject_id")

    long_frames = []
    for col in df.columns:
        if col == "subject_id":
            continue
        switch, ability = _parse_column_label(col)
        sub = df[["subject_id", col]].rename(columns={col: "value"})
        sub["switch_cost_condition"] = switch
        sub["ability_type"] = ability
        long_frames.append(sub)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.dropna(subset=["value"])


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
def _aggregate(values: np.ndarray):
    """Return (mean, sem, n, finite_values)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0, arr
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, sem, arr.size, arr


# ---------------------------------------------------------------------------
# Plotting primitives ( raincloud style — IDENTICAL TO SIMULATION)
# ---------------------------------------------------------------------------
def _draw_half_violin(ax, x_center, values, color, direction, color_dark='black', y_min=None, y_max=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return
    if np.std(arr) < 1e-10:
        arr = arr + np.random.default_rng(0).normal(0, 1e-8, size=arr.size)
    kde = gaussian_kde(arr, bw_method="scott")
    y_lo = arr.min() if y_min is None else y_min
    y_hi = arr.max() if y_max is None else y_max
    ys = np.linspace(y_lo, y_hi, 200)
    density = kde(ys)
    density_norm = (
        density / density.max() * _VIOLIN_WIDTH * direction
        if density.max() > 0 else density * 0
    )
    xs_outer = x_center + density_norm
    xs_inner = np.full_like(xs_outer, x_center)
    # Area fill (transparent)
    ax.fill_betweenx(ys, xs_inner, xs_outer, color=color, alpha=_ALPHA_VIOLIN, linewidth=0, zorder=2)
    # Add solid dark path outline
    ax.plot(xs_outer, ys, color=color_dark, linewidth=1.0, alpha=1.0, zorder=2.5)


def _draw_boxplot(ax, x_center, values, color, color_dark):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_whisk = max(arr.min(), q1 - 1.5 * iqr)
    hi_whisk = min(arr.max(), q3 + 1.5 * iqr)
    bw = _BOX_WIDTH / 2
    rect = plt.Rectangle(
        (x_center - bw, q1), _BOX_WIDTH, iqr,
        facecolor=color, edgecolor=color_dark,
        linewidth=1.3, alpha=_ALPHA_BOX, zorder=3,
    )
    ax.add_patch(rect)
    # Thicker black median line
    ax.plot([x_center - bw, x_center + bw], [med, med], color=color_dark, linewidth=1.8, zorder=4)
    # Pure black whiskers
    ax.plot([x_center, x_center], [lo_whisk, q1], color=color_dark, linewidth=1.2, zorder=3)
    ax.plot([x_center, x_center], [q3, hi_whisk], color=color_dark, linewidth=1.2, zorder=3)
    cw = _WHISK_CAP / 2
    # Pure black caps
    ax.plot([x_center - cw, x_center + cw], [lo_whisk, lo_whisk], color=color_dark, linewidth=1.2, zorder=3)
    ax.plot([x_center - cw, x_center + cw], [hi_whisk, hi_whisk], color=color_dark, linewidth=1.2, zorder=3)


def _draw_diamond_mean(ax, x_center, mean, color, color_dark, size=10):
    ax.plot(
        x_center, mean, marker="D", markersize=size,
        markerfacecolor=color, markeredgecolor=color_dark,
        markeredgewidth=1.5, zorder=5, linestyle="none", # Thicker black edge
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(panels: list[dict], output_path: Path, title: str | None = None):
    """
    panels: list of dicts, one per subplot, each with keys:
        label       : panel letter, e.g. "A"
        df          : long-format dataframe with columns
                      switch_cost_condition (low/high), ability_type (HA/MA), value
        ylabel      : y-axis label
        floor_zero  : bool, whether to clip the y-axis at 0
    """
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5)) # DPI handled by config
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.24, wspace=0.40)

    if title:
        fig.suptitle(title, fontsize=10, y=0.97, color="#444444")

    for ax, panel in zip(axes, panels):
        panel_label = panel["label"]
        df = panel["df"]
        ylabel = panel["ylabel"]
        floor_zero = panel.get("floor_zero", False)

        # ── draw per switching-cost condition × ability type ─────────────
        mean_pts = {at: [] for at in _ABILITY_ORDER}

        for switch in _SWITCH_ORDER:
            switch_x = _SWITCH_X[switch]
            for at in _ABILITY_ORDER:
                sel = df[(df["switch_cost_condition"] == switch) & (df["ability_type"] == at)]
                mean, sem, n, vals = _aggregate(sel["value"].to_numpy())
                if n == 0:
                    mean_pts[at].append((switch_x + _SIDE_OFFSET[at], np.nan))
                    continue

                x_center = switch_x + _SIDE_OFFSET[at]
                color, color_dark = _COLOR[at], _COLOR_DARK[at]
                direction = _VIOLIN_DIR[at]

                # dynamic scaling logic was buggy here, simplified and applied limits at the end
                _draw_half_violin(ax, x_center, vals, color, direction, color_dark=color_dark)
                _draw_boxplot(ax, x_center, vals, color, color_dark)
                _draw_diamond_mean(ax, x_center, mean, color, color_dark)

                mean_pts[at].append((x_center, mean))

        # ── connecting lines (one per ability group, across switch conditions) ──
        for at in _ABILITY_ORDER:
            pts = mean_pts[at]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if all(not np.isnan(y) for y in ys):
                actual_color = _COLOR_HA if at == "HA" else _COLOR_MA
                ax.plot(xs, ys, color=actual_color, linewidth=2.5,
                         zorder=4, solid_capstyle="round", alpha=0.8)

        # ── x-axis ─────────────────────────────────────────────────────
        ax.set_xticks([_SWITCH_X[s] for s in _SWITCH_ORDER])
        ax.set_xticklabels([_SWITCH_LABELS[s] for s in _SWITCH_ORDER], fontsize=_FS_TICK_X)
        ax.set_xlim(0.4, 2.6)

        # ── y-axis ─────────────────────────────────────────────────────
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=_FS_AXIS) # BOLD Y-Label
        ax.tick_params(axis="y", labelsize=_FS_TICK_Y)

        # ── panel label (commented out per style) ──────────────────────
        #ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes,
        #         fontsize=_FS_PANEL, fontweight="bold", va="bottom")

        # ── grid / spines ──────────────────────────────────────────────
        ax.yaxis.grid(True, color="lightgray", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # ── y-axis Formatting (PNAS Style specific limits) ─────────────────────
        if "Specialization" in ylabel:
            # Force limits to 0 and 100 for Specialization Index
            y_lo, y_hi = 0.0, 100.0
            ax.set_ylim(y_lo, y_hi)
            
            # Format tick labels to include the '%' symbol
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
            
        elif "Game score" in ylabel:
            # Human Game score ranges roughly 0-16
            ax.set_ylim(0, 16) 
            
        else:
            # Standard auto-scaling padding
            all_vals = df["value"].to_numpy(dtype=float)
            all_vals = all_vals[np.isfinite(all_vals)]
            if all_vals.size:
                v_min, v_max = all_vals.min(), all_vals.max()
                pad = (v_max - v_min) * 0.15 if v_max != v_min else 1.0
                y_lo, y_hi = v_min - pad, v_max + pad
                if floor_zero:
                    y_lo = max(0.0, y_lo)
            else:
                y_lo, y_hi = 0.0, 1.0
            ax.set_ylim(y_lo, y_hi)


    # ── shared legend ──────────────────────────────────────────────────
    legend_handles = []
    for at in _ABILITY_ORDER:
        # High contrast handle: solid black edge path
        handle = matplotlib.lines.Line2D(
            [], [], marker="D", markersize=10,
            markerfacecolor=_COLOR[at], markeredgecolor="black",
            markeredgewidth=1.5, linewidth=0, label=_LEGEND_LABELS[at],
            alpha=1.0 # Legend solid
        )
        # HandleTuple style patch for contrast in legend
        box_patch = matplotlib.patches.Patch(
            facecolor=_COLOR[at], edgecolor="black",
            alpha=_ALPHA_BOX, linewidth=1.3,
        )
        legend_handles.append((box_patch, handle))

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Figure saved (High Resolution, PNAS style) → {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(panels: list[dict]):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for panel in panels:
        df = panel["df"]
        print(f"\n-- {panel['label']}: {panel['ylabel']} --")
        header = f"{'Group':<28}{'N':>6}   {'mean ± SEM':>18}"
        print(header)
        print("-" * len(header))
        for switch in _SWITCH_ORDER:
            for at in _ABILITY_ORDER:
                sel = df[(df["switch_cost_condition"] == switch) & (df["ability_type"] == at)]
                label = f"{_SWITCH_LABELS[switch].splitlines()[0]} / {at}"
                mean, sem, n, _ = _aggregate(sel["value"].to_numpy())
                stat_str = f"{mean:.3f} ± {sem:.3f}".rjust(18)
                print(f"{label:<28}{n:>6}   {stat_str}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HA vs MA comparison from human-subject wide-format CSVs."
    )
    parser.add_argument("--scores_csv", default="data/human/scores_human.csv",
                         help="Wide-format CSV with game scores (default: scores_human.csv)")
    parser.add_argument("--specialization_csv", default="data/human/specialization_index_human.csv",
                         help="Wide-format CSV with specialization index "
                              "(default: specialization_index_human.csv)")
    parser.add_argument("--output_dir", default="./figures",
                         help="Where to save the figure (default: current directory)")
    parser.add_argument("--output_name", default="figure3_HA_MA_comparison_Human.png",
                         help="Output filename (default: figure3_HA_MA_comparison_Human.png)")
    parser.add_argument("--title", default=None,
                         help="Optional figure suptitle")
    args = parser.parse_args()

    scores_long = load_wide_human_csv(Path(args.scores_csv))
    spec_long = load_wide_human_csv(Path(args.specialization_csv))

    # --- Percentage Scaling Logic fix ---
    # The figure needs percentage axis for Spec Index, melt format usually gives 0-1, so multiply by 100
    spec_long["value"] = spec_long["value"] #* 100
    # -------------------------------------

    panels = [
        {"label": "A", "df": scores_long, "ylabel": "Game score", "floor_zero": True},
        {"label": "B", "df": spec_long, "ylabel": "Specialization index", "floor_zero": True},
    ]

    print_summary(panels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    make_figure(panels, output_path, title=args.title)


if __name__ == "__main__":
    main()