#!/usr/bin/env python3
"""
HA vs MA Comparison Analysis — CSV version.

Reads a long-format CSV (one row per simulation run) and reproduces the
raincloud-style comparison figure (half-violin + boxplot + diamond mean +
connecting line), but driven directly from `figure3_specialization_index_long.csv`
instead of crawling the SLURM simulation directory tree.

Group definitions (per experiment design):
  Ability   : HA  <->  rho == 1     (High ability)
              MA  <->  rho <  1     (Mixed ability)
  Switching cost (x-axis):
              low s  -> "Open" map
              high s -> "Partially-blocked" map

Panels (axes differ from the original filesystem-based version, since the
metrics available in this CSV are different):
  A: Reward rate
  B: Specialization index (|Δ|)   (model_specialization_index_rate)

Each panel keeps the original raincloud layout: two x-positions
(low s / high s, labelled Open / Partially-blocked), with HA (salmon) and
MA (teal) drawn side-by-side at each position, plus a mean-connecting line
per ability group across the two switching-cost conditions.

Usage:
nohup python3 analysis_simulations_HA_MA-Analytic_Model.py --csv ./analytic_model/figure3_specialization_index_long.csv > log_HA_MA_comparison_Analytic_Model.out 2>&1 & 

Author: Samuel Lozano
"""

import argparse
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
# Global Style Configuration (PNAS requirements)
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
# Group / style constants
# ---------------------------------------------------------------------------
RHO_TOL = 1e-6  # rho == 1.0 (within tolerance) -> HA; otherwise -> MA

_SWITCH_ORDER  = ["low", "high"]
_SWITCH_X      = {"low": 1.0, "high": 2.0}
_SWITCH_LABELS = {"low": "Open\n($s=0.05$)", "high": "PB\n($s=0.5$)"}

_ABILITY_ORDER = ["HA", "MA"]
_SIDE_OFFSET   = {"HA": -0.18, "MA": +0.18}   # HA drawn left, MA drawn right of each x position
_VIOLIN_DIR    = {"HA": -1, "MA": +1}          # violin fans outward from the box

# New color scheme based on panels B and E
_COLOR_HA = '#E24A33' # Reddish-Salmon
_COLOR_MA = '#348ABD' # Teal-Blue
_COLOR        = {"HA": _COLOR_HA, "MA": _COLOR_MA}   # salmon / teal
_COLOR_DARK   = {"HA": "black", "MA": "black"} # Dark outlines for both
_LEGEND_LABELS = {
    "HA": r"\textbf{High ability ($\boldsymbol{\rho = 1}$)}",
    "MA": r"\textbf{Mixed ability ($\boldsymbol{\rho < 1}$)}",
}

# Adjusted alpha values based on Panels B and E for higher contrast
_ALPHA_VIOLIN = 0.55  # Slightly increased density
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
# Data loading
# ---------------------------------------------------------------------------
def classify_ability(rho: float) -> str:
    """HA <-> rho == 1 ; MA <-> rho < 1."""
    return "HA" if abs(rho - 1.0) < RHO_TOL else "MA"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {
        "rho", "switch_cost_condition", "reward_rate",
        "model_specialization_index_rate", "switching_parameter_a",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ability_type"] = df["rho"].apply(classify_ability)

    # --- NEW SCALING LOGIC ---
    df["model_specialization_index_rate"] = df["model_specialization_index_rate"] * 100
# -------------------------

    df["switch_cost_condition"] = (
        df["switch_cost_condition"].astype(str).str.strip().str.lower()
    )
    unknown = set(df["switch_cost_condition"].unique()) - set(_SWITCH_ORDER)
    if unknown:
        raise ValueError(
            f"Unexpected switch_cost_condition values: {sorted(unknown)} "
            f"(expected a subset of {_SWITCH_ORDER})"
        )

    return df


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
# Plotting primitives (raincloud style)
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
    
    # Fill with specified alpha and no linewidth for the area
    ax.fill_betweenx(ys, xs_inner, xs_outer, color=color, alpha=_ALPHA_VIOLIN, linewidth=0, zorder=2)
    # Add a solid dark outline to survive grayscale/projection (matching reference B/E)
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
        linewidth=1.3, alpha=_ALPHA_BOX, zorder=3, # Increased linewidth and alpha
    )
    ax.add_patch(rect)
    # Median is now black with higher zorder
    ax.plot([x_center - bw, x_center + bw], [med, med], color=color_dark, linewidth=1.8, zorder=4)
    # Whiskers are black
    ax.plot([x_center, x_center], [lo_whisk, q1], color=color_dark, linewidth=1.2, zorder=3)
    ax.plot([x_center, x_center], [q3, hi_whisk], color=color_dark, linewidth=1.2, zorder=3)
    cw = _WHISK_CAP / 2
    # Caps are black
    ax.plot([x_center - cw, x_center + cw], [lo_whisk, lo_whisk], color=color_dark, linewidth=1.2, zorder=3)
    ax.plot([x_center - cw, x_center + cw], [hi_whisk, hi_whisk], color=color_dark, linewidth=1.2, zorder=3)


def _draw_diamond_mean(ax, x_center, mean, color, color_dark, size=10):
    ax.plot(
        x_center, mean, marker="D", markersize=size,
        markerfacecolor=color, markeredgecolor=color_dark,
        markeredgewidth=1.5, zorder=5, linestyle="none", # Solid dark outline
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(df: pd.DataFrame, output_path: Path, title: str | None = None):
    """
    Two-panel raincloud figure:
      D.1: reward_rate               -> "Reward rate"
      D.2: model_specialization_index_rate     -> "Specialization index"  (floored at 0)

    X-axis (each panel): low s / high s, labelled Open / Partially-blocked,
    with HA / MA drawn side-by-side at each position.
    """
    panel_specs = [
        ("D.1", "reward_rate",           "Reward Rate",                   False),
        ("D.2", "model_specialization_index_rate",  "Specialization Index",   True),
    ]

    # Create figure with high DPI setting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.24, wspace=0.40)

    if title:
        fig.suptitle(title, fontsize=20, y=0.97, color="#444444")

    for ax, (panel_label, metric_key, ylabel, floor_zero) in zip(axes, panel_specs):

        # ── y-axis range from all data in this panel ─────────────────────
        if metric_key == "model_specialization_index_rate":
            # Force limits to 0 and 100 for Specialization Index
            y_lo, y_hi = 0.0, 100.0
            ax.set_ylim(y_lo, y_hi)
            
            # Format tick labels to include the '%' symbol
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
            
        else:
            # Original dynamic scaling for Reward Rate
            all_vals = df[metric_key].to_numpy(dtype=float)
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

        # ── draw per switching-cost condition × ability type ─────────────
        mean_pts = {at: [] for at in _ABILITY_ORDER}

        for switch in _SWITCH_ORDER:
            switch_x = _SWITCH_X[switch]
            for at in _ABILITY_ORDER:
                sel = df[(df["switch_cost_condition"] == switch) & (df["ability_type"] == at)]
                mean, sem, n, vals = _aggregate(sel[metric_key].to_numpy())
                if n == 0:
                    mean_pts[at].append((switch_x + _SIDE_OFFSET[at], np.nan))
                    continue

                x_center = switch_x + _SIDE_OFFSET[at]
                color, color_dark = _COLOR[at], _COLOR_DARK[at]
                direction = _VIOLIN_DIR[at]

                # Pass color_dark for solid outlines
                _draw_half_violin(ax, x_center, vals, color, direction, color_dark=color_dark, y_min=y_lo, y_max=y_hi)
                _draw_boxplot(ax, x_center, vals, color, color_dark)
                _draw_diamond_mean(ax, x_center, mean, color, color_dark)

                mean_pts[at].append((x_center, mean))

        # ── connecting lines (one per ability group, across switch conditions) ──
        for at in _ABILITY_ORDER:
            pts = mean_pts[at]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if all(not np.isnan(y) for y in ys):
                # Using the actual salmon/teal colors for connecting lines
                actual_color = _COLOR_HA if at == "HA" else _COLOR_MA
                ax.plot(xs, ys, color=actual_color, linewidth=2.5,
                         zorder=4, solid_capstyle="round", alpha=0.8)

        # ── x-axis ─────────────────────────────────────────────────────
        ax.set_xticks([_SWITCH_X[s] for s in _SWITCH_ORDER])
        ax.set_xticklabels([_SWITCH_LABELS[s] for s in _SWITCH_ORDER], fontsize=_FS_TICK_X)
        #ax.set_xlabel("Switching cost condition (Map)", fontsize=_FS_AXIS)
        ax.set_xlim(0.4, 2.6)

        # ── y-axis ─────────────────────────────────────────────────────
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=_FS_AXIS)
        ax.tick_params(axis="y", labelsize=_FS_TICK_Y)

        # ── panel label ────────────────────────────────────────────────
        #ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes,
        #         fontsize=_FS_PANEL, fontweight="bold", va="bottom")

        # ── grid / spines ──────────────────────────────────────────────
        ax.yaxis.grid(True, color="lightgray", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # ── shared legend ──────────────────────────────────────────────────
    legend_handles = []
    for at in _ABILITY_ORDER:
        # High contrast handle matching Reference B/E: solid fill, solid black edge
        handle = matplotlib.lines.Line2D(
            [], [], marker="D", markersize=10,
            markerfacecolor=_COLOR[at], markeredgecolor="black",
            markeredgewidth=1.5, linewidth=0, label=_LEGEND_LABELS[at],
            alpha=1.0 # Full visibility in legend
        )
        # Box patch matching enhanced boxplot style
        box_patch = matplotlib.patches.Patch(
            facecolor=_COLOR[at], edgecolor="black",
            alpha=_ALPHA_BOX, linewidth=1.3,
        )
        legend_handles.append((box_patch, handle))

    # Save main figure with high DPI and font embedding (global rcParams)
    plt.savefig(output_path, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Save legend as a separate figure
    # ------------------------------------------------------------------

    legend_fig = plt.figure(figsize=(5.5, 0.9))

    legend = legend_fig.legend(
        legend_handles,
        [_LEGEND_LABELS[at] for at in _ABILITY_ORDER],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        #title=r"Ability",
        #title_fontsize=_FS_LEG,
        fontsize=_FS_LEG,
        frameon=False,
        loc="center",
        ncol=2,
        handlelength=2.5,
        columnspacing=2.5,
    )

    legend_output = output_path.with_name(output_path.stem + "_legend.png")

    # Legend figure also saved with global 600 DPI and font embedding
    legend_fig.savefig(
        legend_output,
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(legend_fig)

    plt.close(fig)
    print(f"\n✅ Figure saved (High Resolution, PNAS style) → {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    metrics = ["reward_rate", "model_specialization_index_rate", "switching_parameter_a"]
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Group':<32}{'N':>6}   " + "   ".join(f"{m:>26}" for m in metrics)
    print(header)
    print("-" * len(header))
    for switch in _SWITCH_ORDER:
        for at in _ABILITY_ORDER:
            sel = df[(df["switch_cost_condition"] == switch) & (df["ability_type"] == at)]
            label = f"{_SWITCH_LABELS[switch].splitlines()[0]} / {at}"
            n = len(sel)
            stat_strs = []
            for m in metrics:
                mean, sem, _, _ = _aggregate(sel[m].to_numpy())
                stat_strs.append(f"{mean:.3f} ± {sem:.3f}".rjust(26))
            print(f"{label:<32}{n:>6}   " + "   ".join(stat_strs))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HA vs MA comparison from the figure3 long-format CSV."
    )
    parser.add_argument("--csv", default="data/analytic_model/figure3_specialization_index_long.csv",
                         help="Path to the long-format CSV (default: figure3_specialization_index_long.csv)")
    parser.add_argument("--output_dir", default="./figures",
                         help="Where to save the figure (default: current directory)")
    parser.add_argument("--output_name", default="figure3_HA_MA_Analytic_Model.png",
                         help="Output filename (default: figure3_HA_MA_Analytic_Model.png)")
    parser.add_argument("--title", default=None,
                         help="Optional figure suptitle")
    args = parser.parse_args()

    df = load_data(Path(args.csv))
    print_summary(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    make_figure(df, output_path, title=args.title)


if __name__ == "__main__":
    main()