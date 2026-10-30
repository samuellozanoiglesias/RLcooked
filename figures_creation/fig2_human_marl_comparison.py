#!/usr/bin/env python3
"""
Figure 2 -- Map configurations and HA vs MA comparison (humans, MARL, theory).

Merges the three per-source scripts
    analysis_simulations_HA_MA-Human.py
    analysis_simulations_HA_MA-RL.py
    analysis_simulations_HA_MA-Analytic_Model.py
into a single script that builds the complete compound figure in ONE
matplotlib figure (PDF + PNG), reproducing the Fig2 layout:

    Map configuration          | Human Experiments
      a  Open map              |   c  Game score         d  Specialization index
      b  Partially-blocked map |
    MARL Simulations           | MARL Theoretical Solutions
      e  Game score            |   g  Reward Rate
      f  Specialization index  |   h  Specialization Index
    [ legend band: High ability (100%, 100%; rho = 1) | Mixed ability (40%, 20%; rho = 0.85) ]
      (speed percentages for humans / MARL simulations; rho for MARL theory)

Inputs
  ./figures/open_map.png, ./figures/PB_map.png      (map screenshots, --maps_dir)
  Human   : the per-team file written by fig_sm_ability_pairs_humans_rl.py
            (--human_teams_csv, default ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv):
            the first 70 teams per condition (test phase), specialization index with the
            Methods formula, i.e. the same teams as Figs. 4, S-pairs and S-pretraining.
            Required: the script stops if the file is missing.
  MARL    : simulated games of the simulation tree (see marl_simulations.py, same folder)
              <base>/data/samuel_lozano/cooked/<game_version>/simulations/<init_type>/map_<map>/
                  synergy_<s>/specialized_<l>/abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
            HA = abilities_1.0_1.0, MA = abilities_0.4_0.2 (walk1 = walking speed of ai_rl_1,
            cut2 = cutting speed of ai_rl_2, the other speeds 1.0); every Training_* and
            simulation_* folder is used. Or a cached CSV written by a previous run (--rl_csv)
  Theory  : figure3_specialization_index_long.csv   (long format)

Every panel keeps the raincloud style of the original scripts (half-violin +
box + diamond mean + connecting line). Panel positions, sizes, fonts and
line widths were measured from Fig2.pdf: each 2-panel group is drawn with
its original 10 x 5 in design and uniformly scaled by the same factor used
when the figure was assembled, and the page is written at the same size
(804.96 x 446.52 pt) without bbox_inches='tight'.

Section titles ("Map configuration", "Human Experiments", ...) are centred
automatically over the panels of their group (see TITLES), so renaming them
or changing axis labels keeps them centred.

Usage:
nohup python3 fig2_human_marl_comparison.py --human_teams_csv ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv --analytic_csv ./data/analytic_model/figure3_specialization_index_long.csv --map_baseline baseline_division_of_labor_large --map_encouraged encouraged_division_of_labor_large --synergy 1.35 --specialization 0.25 --init_type empty_init --checkpoint 753 --game_version classic_collision --cluster cuenca > log_fig2.out 2>&1 &

    # re-plot quickly from the MARL records cached by a previous run
    python3 fig2_human_marl_comparison.py --rl_csv ./figures/fig2_rl_records.csv

    # layout test with synthetic data
    python3 fig2_human_marl_comparison.py --demo

Author: Samuel Lozano
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_stats as ps                                   # noqa: E402
import marl_simulations as ms                              # noqa: E402

# ---------------------------------------------------------------------------
# Global Style Configuration (PNAS requirements) -- identical in all 3 scripts
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
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ===========================================================================
# 1. HUMAN DATA  (from analysis_simulations_HA_MA-Human.py)
# ===========================================================================
def _parse_column_label(col: str) -> tuple[str, str]:
    """'Open High' / 'Partially Blocked Mixed Ability' -> (low/high, HA/MA)."""
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


_HUMAN_TEAMS_DEFAULT = "./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv"


def load_human_teams(csv_path: Path) -> pd.DataFrame:
    """Per-team human file -> long df (switch_cost_condition, ability_type, total_deliveries,
    specialization_index): HA = (100%, 100%), MA = (40%, 20%)."""
    t = pd.read_csv(csv_path)
    ha = np.isclose(t["walk"], 1.0) & np.isclose(t["cut"], 1.0)
    ma = np.isclose(t["walk"], 0.4) & np.isclose(t["cut"], 0.2)
    t = t[ha | ma].copy()
    t["ability_type"] = np.where(np.isclose(t["walk"], 1.0), "HA", "MA")
    t["switch_cost_condition"] = np.where(t["map"].astype(str).str.lower().str.startswith("open"), "low", "high")
    return t


def write_stats(human_df: pd.DataFrame | None, rl_df: pd.DataFrame | None, path: Path):
    """Main-text statistics of Fig. 2 c-f with the conventions of paper_stats.py."""
    lines = ["=" * 100, "FIG. 2 STATISTICS -- conventions of paper_stats.py (higher group first, Welch's t(df) = |t|;",
             "two-way between-subjects ANOVA ability x map, effect coding, Type III SS, partial eta^2)", "=" * 100]
    lab = {"low": "Open", "high": "PB"}
    blocks = []
    if human_df is not None:
        blocks.append(("HUMANS (unit = team)", human_df))
    if rl_df is not None and not rl_df.empty:
        r = rl_df.copy()
        for c in ("total_deliveries", "specialization_index"):
            r[c] = pd.to_numeric(r[c], errors="coerce")
        blocks.append(("MARL SIMULATIONS (unit = training; mean of its simulated games)",
                       r.groupby(["switch_cost_condition", "ability_type", "training_id"], as_index=False)
                        [["total_deliveries", "specialization_index"]].mean()))
        blocks.append(("MARL SIMULATIONS (unit = simulated game)", r))
    for title, d in blocks:
        lines.append(f"\n{title}")
        for (sw, at), g in d.groupby(["switch_cost_condition", "ability_type"]):
            lines.append(f"  n {lab[sw]:<4} {at}: {len(g)}")
        for col, name, pct in (("total_deliveries", "Game score", False), ("specialization_index", "Specialization index", True)):
            a = ps.anova_type3(d, col, "switch_cost_condition", "ability_type")
            lines.append(f"  {name}: Map x Ability ({ps.fmt_F(a.iloc[2])}); map ({ps.fmt_F(a.iloc[0])}); "
                         f"ability ({ps.fmt_F(a.iloc[1])})")
            for sw in _SWITCH_ORDER:
                sel = d["switch_cost_condition"] == sw
                txt, _ = ps.fmt_compare("HA", d.loc[sel & (d["ability_type"] == "HA"), col],
                                        "MA", d.loc[sel & (d["ability_type"] == "MA"), col], pct)
                lines.append(f"    {lab[sw]:<4} {txt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nStatistics saved -> {path}")


def groups_from_long(df: pd.DataFrame, value_col: str = "value") -> dict:
    """{(switch, ability): np.ndarray} from a long dataframe."""
    out = {}
    for switch in _SWITCH_ORDER:
        for at in _ABILITY_ORDER:
            sel = df[(df["switch_cost_condition"] == switch) & (df["ability_type"] == at)]
            out[(switch, at)] = sel[value_col].to_numpy(dtype=float)
    return out


# ===========================================================================
# 2. MARL SIMULATION DATA  (from analysis_simulations_HA_MA-RL.py)
# ===========================================================================
# Simulated games are read from the abilities_<walk1>_<cut2> folders of the simulation tree
# (marl_simulations.py): walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2,
# the other two speeds are 1.0.
#   HA = abilities_1.0_1.0 ; MA = abilities_0.4_0.2 (ai_rl_1 slow walker, ai_rl_2 slow cutter)
ABILITY_PAIRS = {"HA": (1.0, 1.0), "MA": (0.4, 0.2)}
CLUSTER_DIRS = ms.CLUSTER_DIRS


def collect_group_metrics(map_root: Path, checkpoint, agent_type: str, verbose: bool = True) -> list[dict]:
    """Every Training_*/checkpoint_<ck>/simulation_* of the abilities folder of agent_type."""
    pair = ABILITY_PAIRS[agent_type]
    adir = ms.find_ability_dir(map_root, *pair)
    if adir is None:
        print(f"  [WARN] no abilities_{pair[0]:g}_{pair[1]:g} folder in {map_root}")
        return []
    print(f"  Abilities folder : {adir}")
    df = ms.collect_records(map_root, pairs=[pair], checkpoint=checkpoint, verbose=verbose)
    if df.empty:
        print(f"  [WARN] no simulations with usable data under {adir}/Training_*/checkpoint_{checkpoint}")
        return []
    return df[["training_id", "checkpoint", "simulation", "total_deliveries",
               "specialization_index"]].to_dict("records")


def collect_rl_records(args) -> pd.DataFrame:
    """Crawl the simulation tree -> long df (switch_cost_condition, ability_type, metrics)."""
    base_cluster_dir = ms.cluster_base_dir(args.cluster)
    rows = []
    for switch, map_nr in (("low", args.map_baseline), ("high", args.map_encouraged)):
        root = ms.map_root(base_cluster_dir, args.game_version, map_nr, args.synergy, args.specialization,
                           init_type=args.init_type, study_name=args.study_name or None)
        for at in _ABILITY_ORDER:
            print(f"\n{'=' * 60}\nCollecting {at} data for map {map_nr}\n"
                  f"  Simulations root : {root}\n"
                  f"  Checkpoint folder: checkpoint_{args.checkpoint}\n{'=' * 60}")
            recs = collect_group_metrics(root, args.checkpoint, at, verbose=args.verbose)
            for r in recs:
                rows.append({"switch_cost_condition": switch, "ability_type": at, "map": map_nr, **r})
            print(f"  -> {len(recs)} simulations collected from "
                  f"{len({r['training_id'] for r in recs})} trainings")
    return pd.DataFrame(rows, columns=["switch_cost_condition", "ability_type", "map", "training_id",
                                       "checkpoint", "simulation", "total_deliveries",
                                       "specialization_index"])


# ===========================================================================
# 3. ANALYTIC MODEL DATA  (from analysis_simulations_HA_MA-Analytic_Model.py)
# ===========================================================================
RHO_TOL = 1e-6


def load_analytic_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"rho", "switch_cost_condition", "reward_rate", "model_specialization_index_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["ability_type"] = np.where(np.abs(df["rho"] - 1.0) < RHO_TOL, "HA", "MA")
    df["model_specialization_index_rate"] = df["model_specialization_index_rate"] * 100
    df["switch_cost_condition"] = df["switch_cost_condition"].astype(str).str.strip().str.lower()
    unknown = set(df["switch_cost_condition"].unique()) - set(_SWITCH_ORDER)
    if unknown:
        raise ValueError(f"Unexpected switch_cost_condition values: {sorted(unknown)}")
    return df


# ===========================================================================
# 4. RAINCLOUD STYLE  (shared by the three sources)
# ===========================================================================
_SWITCH_ORDER = ["low", "high"]
_SWITCH_X = {"low": 1.0, "high": 2.0}
_ABILITY_ORDER = ["HA", "MA"]
_SIDE_OFFSET = {"HA": -0.18, "MA": +0.18}
_VIOLIN_DIR = {"HA": -1, "MA": +1}

_COLOR_HA = '#E24A33'   # Reddish-Salmon
_COLOR_MA = '#348ABD'   # Teal-Blue
_COLOR = {"HA": _COLOR_HA, "MA": _COLOR_MA}
_COLOR_DARK = {"HA": "black", "MA": "black"}
_LEGEND_LABELS = {
    # speed percentages (humans, MARL simulations) and the theory parameter rho
    "HA": r"\textbf{High ability (100\%, 100\%; $\boldsymbol{\rho = 1}$)}",
    "MA": r"\textbf{Mixed ability (40\%, 20\%; $\boldsymbol{\rho = 0.85}$)}",
}

_ALPHA_VIOLIN = 0.55
_ALPHA_BOX = 0.85
_VIOLIN_WIDTH = 0.30
_BOX_WIDTH = 0.10
_WHISK_CAP = 0.06
_FS = 30            # axis labels, tick labels and legend (before scaling)


def _aggregate(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0, arr
    sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return arr.mean(), sem, arr.size, arr


def _draw_half_violin(ax, x_center, arr, color, direction, s, y_min=None, y_max=None):
    if arr.size < 2:
        return
    if np.std(arr) < 1e-10:
        arr = arr + np.random.default_rng(0).normal(0, 1e-8, size=arr.size)
    kde = gaussian_kde(arr, bw_method="scott")
    ys = np.linspace(arr.min() if y_min is None else y_min,
                     arr.max() if y_max is None else y_max, 200)
    density = kde(ys)
    density_norm = density / density.max() * _VIOLIN_WIDTH * direction if density.max() > 0 else density * 0
    xs_outer = x_center + density_norm
    ax.fill_betweenx(ys, np.full_like(xs_outer, x_center), xs_outer, color=color,
                     alpha=_ALPHA_VIOLIN, linewidth=0, zorder=2)
    ax.plot(xs_outer, ys, color="black", linewidth=1.0 * s, zorder=2.5)


def _draw_boxplot(ax, x_center, arr, color, s):
    if arr.size < 2:
        return
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_whisk = max(arr.min(), q1 - 1.5 * iqr)
    hi_whisk = min(arr.max(), q3 + 1.5 * iqr)
    bw, cw = _BOX_WIDTH / 2, _WHISK_CAP / 2
    ax.add_patch(plt.Rectangle((x_center - bw, q1), _BOX_WIDTH, iqr, facecolor=color,
                               edgecolor="black", linewidth=1.3 * s, alpha=_ALPHA_BOX, zorder=3))
    ax.plot([x_center - bw, x_center + bw], [med, med], color="black", linewidth=1.8 * s, zorder=4)
    for seg in ([x_center, x_center], [lo_whisk, q1]), ([x_center, x_center], [q3, hi_whisk]), \
               ([x_center - cw, x_center + cw], [lo_whisk, lo_whisk]), \
               ([x_center - cw, x_center + cw], [hi_whisk, hi_whisk]):
        ax.plot(*seg, color="black", linewidth=1.2 * s, zorder=3)


def _draw_diamond_mean(ax, x_center, mean, color, s):
    ax.plot(x_center, mean, marker="D", markersize=10 * s, markerfacecolor=color,
            markeredgecolor="black", markeredgewidth=1.5 * s, zorder=5, linestyle="none")


def draw_raincloud_panel(ax, groups: dict, *, ylabel: str, xticklabels: dict, ylim,
                         violin_full_range: bool, percent: bool, s: float):
    """
    groups            : {(switch, ability): values}
    ylim              : (lo, hi)
    violin_full_range : True  -> KDE evaluated over the whole y-range (MARL / theory scripts)
                        False -> KDE evaluated over the data range (human script)
    """
    y_lo, y_hi = ylim
    ax.set_ylim(y_lo, y_hi)
    mean_pts = {at: [] for at in _ABILITY_ORDER}
    for switch in _SWITCH_ORDER:
        for at in _ABILITY_ORDER:
            x_center = _SWITCH_X[switch] + _SIDE_OFFSET[at]
            mean, _, n, arr = _aggregate(groups.get((switch, at), []))
            if n == 0:
                mean_pts[at].append((x_center, np.nan))
                continue
            span = (y_lo, y_hi) if violin_full_range else (None, None)
            _draw_half_violin(ax, x_center, arr, _COLOR[at], _VIOLIN_DIR[at], s, *span)
            _draw_boxplot(ax, x_center, arr, _COLOR[at], s)
            _draw_diamond_mean(ax, x_center, mean, _COLOR[at], s)
            mean_pts[at].append((x_center, mean))

    for at in _ABILITY_ORDER:
        xs, ys = zip(*mean_pts[at])
        if all(np.isfinite(ys)):
            ax.plot(xs, ys, color=_COLOR[at], linewidth=2.5 * s, zorder=4,
                    solid_capstyle="round", alpha=0.8)

    ax.set_xticks([_SWITCH_X[k] for k in _SWITCH_ORDER])
    ax.set_xticklabels([xticklabels[k] for k in _SWITCH_ORDER])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

    ax.yaxis.grid(True, color="lightgray", linewidth=0.7 * s, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(mpl.rcParams["axes.linewidth"] * s)
    for which in ("major", "minor"):
        ax.tick_params(axis="both", which=which, labelsize=_FS * s,
                       length=mpl.rcParams[f"xtick.{which}.size"] * s,
                       width=mpl.rcParams[f"xtick.{which}.width"] * s,
                       pad=mpl.rcParams[f"xtick.{which}.pad"] * s)
    ax.yaxis.label.set_size(_FS * s)
    ax.yaxis.labelpad = mpl.rcParams["axes.labelpad"] * s


def padded_ylim(values_by_group: dict, frac: float = 0.15):
    """Dynamic y-range of the theory script: data range padded by 15 %."""
    allv = np.concatenate([np.asarray(v, float) for v in values_by_group.values()] or [np.array([])])
    allv = allv[np.isfinite(allv)]
    if not allv.size:
        return 0.0, 1.0
    v_min, v_max = allv.min(), allv.max()
    pad = (v_max - v_min) * frac if v_max != v_min else 1.0
    return v_min - pad, v_max + pad


# ===========================================================================
# 5. FIGURE 2 LAYOUT (points, y measured from the TOP of the page)
# ===========================================================================
PAGE_W, PAGE_H = 804.96, 446.52

# axes rectangles (left, right, top, bottom) and per-group scale factor
PANEL_GROUPS = {
    'human':    dict(s=0.57079, axes={'c': (431.02, 573.28, 37.24, 164.64),
                                      'd': (658.63, 800.89, 37.24, 164.64)}),
    'rl':       dict(s=0.54298, axes={'e': (40.99, 173.31, 235.16, 364.17),
                                      'f': (252.70, 385.02, 235.16, 364.17)}),
    'analytic': dict(s=0.55060, axes={'g': (432.96, 573.23, 232.52, 355.41),
                                      'h': (657.40, 797.68, 232.52, 355.41)}),
}

MAP_RECTS = {                          # (left, right, top, bottom)
    'open': (39.95, 171.09, 30.90, 161.86),
    'pb':   (213.72, 343.83, 30.89, 161.87),
}
MAP_FILES = {'open': 'open_map', 'pb': 'PB_map'}

# free text: (TeX font, size pt, text, x, y[, ha]) -- baseline anchor, left-aligned
# unless ha is given (ha='center': x is the horizontal centre of the text)
_MAP_CAP_Y = 181.76                    # common baseline of the two map captions
_MAP_CAP_SIZE = 16                     # common size of the two map captions
TEXTS = {
    'a': ('cmbx10', 20, 'a', 18.15, 48.48),
    'b': ('cmbx10', 20, 'b', 191.21, 48.51),
    'c': ('cmbx10', 20, 'c', 392.76, 48.48),
    'd': ('cmbx10', 20, 'd', 581.83, 52.01),
    'e': ('cmbx10', 20, 'e', 17.77, 239.02),
    'f': ('cmbx10', 20, 'f', 181.15, 238.99),
    'g': ('cmbx10', 20, 'g', 391.82, 239.07),
    'h': ('cmbx10', 20, 'h', 582.00, 241.43),
    # map captions centred under their map (MAP_RECTS), same size and baseline
    'cap_open':   ('cmr10', _MAP_CAP_SIZE, 'Open', (39.95 + 171.09) / 2, _MAP_CAP_Y, 'center'),
    'cap_pb':     ('cmr10', _MAP_CAP_SIZE, 'Partially-blocked (PB)', (213.72 + 343.83) / 2, _MAP_CAP_Y, 'center'),
}

# Section titles: the horizontal position is computed automatically (centred
# over the panels of their group), so only font, size, text and baseline are
# set here. Keys must be 'map' or a key of PANEL_GROUPS.
_TITLE_Y_TOP = 21.27        # baseline of the top-row titles (pt from top)
_TITLE_Y_BOTTOM = 217.41    # baseline of the bottom-row titles
TITLES = {  # group -> (TeX font, size pt, text, baseline y)
    'map':      ('cmbx10', 18, 'Map configuration', _TITLE_Y_TOP),
    'human':    ('cmbx10', 18, 'Human Experiments', _TITLE_Y_TOP),
    'rl':       ('cmbx10', 18, 'MARL Simulations',  _TITLE_Y_BOTTOM),
    'analytic': ('cmbx10', 18, 'MARL Theory',       _TITLE_Y_BOTTOM),
}
# True  -> centre over the visible content (axes + tick labels + y-labels)
# False -> centre over the plotting areas only
TITLE_CENTER_ON_CONTENT = True

LEGEND_BOX = dict(x0=118.23, x1=683.64, top=403.44, bottom=444.79,
                  face='#f2f2f2', edge='#595959', lw=2.25)
LEGEND_CENTER = (PAGE_W / 2, 424.835)   # legend band centred on the page
LEGEND_SCALE = 0.60231
LEGEND_PAD = 14.0          # horizontal padding of the legend band (pt)


def tex(font: str, size: float, text: str) -> str:
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def _page_transform(fig):
    return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans


def _rect_to_fig(rect):
    L, R, T, B = rect
    return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]


def _group_center_x_pt(fig, axes, renderer, use_content: bool = True) -> float:
    """Horizontal centre (page points) of the union of the given axes.

    use_content=True uses the tight bbox (tick labels, y-labels included);
    hidden axes (e.g. the map images) always use their plotting area.
    """
    bbs = []
    for ax in axes:
        if use_content and ax.axison:
            bbs.append(ax.get_tightbbox(renderer))
        else:
            bbs.append(ax.get_window_extent(renderer))
    bb = Bbox.union(bbs)
    return 0.5 * (bb.x0 + bb.x1) / fig.dpi * 72.0


def _load_map_image(maps_dir: Path, subfolder: str, stem: str):
    for ext in ('.png', '.jpg', '.jpeg'):
        p = maps_dir / subfolder / f'{stem}{ext}'
        if p.exists():
            return plt.imread(str(p))
    print(f'[fig2] WARNING: map image {maps_dir / subfolder / (stem + ".jpg")} not found')
    return None


def build_panel_specs(human: dict, rl: dict, theory: dict) -> dict:
    """Per-panel data + axis settings (one entry per letter c-h)."""
    open_pb = {"low": "Open", "high": "PB"}
    with_s = {"low": "Open\n($s=0.05$)", "high": "PB\n($s=0.5$)"}
    return {
        'c': dict(groups=human['score'], ylabel="Game score", ylim=(0, 16), percent=False,
                  xt=open_pb, full=False),
        'd': dict(groups=human['spec'], ylabel="Specialization index", ylim=(0, 100), percent=True,
                  xt=open_pb, full=False),
        'e': dict(groups=rl['score'], ylabel="Game score", ylim=(0, 16), percent=False,
                  xt=open_pb, full=True),
        'f': dict(groups=rl['spec'], ylabel="Specialization index", ylim=(0, 100), percent=True,
                  xt=open_pb, full=True),
        'g': dict(groups=theory['reward'], ylabel="Reward Rate", ylim=padded_ylim(theory['reward']),
                  percent=False, xt=with_s, full=True),
        'h': dict(groups=theory['spec'], ylabel="Specialization Index", ylim=(0, 100), percent=True,
                  xt=with_s, full=True),
    }


def make_figure(human: dict, rl: dict, theory: dict, maps_dir: Path, output_path: Path):
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
    page = _page_transform(fig)
    specs = build_panel_specs(human, rl, theory)
    axes_by_group = {name: [] for name in list(PANEL_GROUPS) + ['map']}

    # -- raincloud panels ------------------------------------------------
    for gname, group in PANEL_GROUPS.items():
        s = group['s']
        for key, rect in group['axes'].items():
            sp = specs[key]
            ax = fig.add_axes(_rect_to_fig(rect))
            draw_raincloud_panel(ax, sp['groups'], ylabel=sp['ylabel'], xticklabels=sp['xt'],
                                 ylim=sp['ylim'], violin_full_range=sp['full'],
                                 percent=sp['percent'], s=s)
            axes_by_group[gname].append(ax)

    # -- map screenshots ---------------------------------------------------
    for key, rect in MAP_RECTS.items():
        img = _load_map_image(maps_dir, 'subfigures', MAP_FILES[key])
        ax = fig.add_axes(_rect_to_fig(rect))
        ax.set_axis_off()
        if img is not None:
            ax.imshow(img, aspect='auto', interpolation='lanczos')
        axes_by_group['map'].append(ax)

    # -- free text (panel letters, map captions) ---------------------------
    for font, size, text, x, y, *ha in TEXTS.values():
        fig.text(x, y, tex(font, size, text), transform=page, ha=(ha[0] if ha else 'left'),
                 va='baseline', fontsize=size)

    # -- section titles, centred automatically over their panels -----------
    renderer = fig.canvas.get_renderer()
    for gname, (font, size, text, y) in TITLES.items():
        x = _group_center_x_pt(fig, axes_by_group[gname], renderer, TITLE_CENTER_ON_CONTENT)
        fig.text(x, y, tex(font, size, text), transform=page, ha='center',
                 va='baseline', fontsize=size)

    # -- legend band ---------------------------------------------------------
    lb = LEGEND_BOX
    sl = LEGEND_SCALE
    handles = []
    for at in _ABILITY_ORDER:
        box = matplotlib.patches.Patch(facecolor=_COLOR[at], edgecolor="black",
                                       alpha=_ALPHA_BOX, linewidth=1.3 * sl)
        dia = matplotlib.lines.Line2D([], [], marker="D", markersize=10 * sl,
                                      markerfacecolor=_COLOR[at], markeredgecolor="black",
                                      markeredgewidth=1.5 * sl, linewidth=0)
        handles.append((box, dia))
    cx, cy = LEGEND_CENTER
    leg = fig.legend(handles, [_LEGEND_LABELS[at] for at in _ABILITY_ORDER],
                     handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=_FS * sl,
                     frameon=False, loc="center", ncol=2, handlelength=2.0, columnspacing=1.5,
                     handletextpad=0.6, bbox_to_anchor=(cx / PAGE_W, 1.0 - cy / PAGE_H),
                     bbox_transform=fig.transFigure)
    # grey band sized to the legend (never narrower than the original band), centred on it
    bb = leg.get_window_extent(renderer)
    width_pt = max(bb.width / fig.dpi * 72.0 + 2 * LEGEND_PAD, lb['x1'] - lb['x0'])
    if width_pt > PAGE_W:
        print(f"WARNING: legend band ({width_pt:.0f} pt) is wider than the page ({PAGE_W:.0f} pt)")
    fig.add_artist(matplotlib.patches.Rectangle(
        (cx - width_pt / 2, lb['top']), width_pt, lb['bottom'] - lb['top'], transform=page,
        facecolor=lb['face'], edgecolor=lb['edge'], linewidth=lb['lw'], zorder=0))

    # NB: no bbox_inches='tight' -- the page size is part of the layout.
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    plt.close(fig)
    print(f"\nFigure 2 saved -> {output_path} (+ .png)")


# ===========================================================================
# 6. SUMMARY / DEMO / CLI
# ===========================================================================
def print_summary(name: str, groups: dict):
    print(f"\n-- {name} --")
    print(f"{'Group':<14}{'N':>6}   {'mean ± SEM':>18}")
    for switch in _SWITCH_ORDER:
        for at in _ABILITY_ORDER:
            mean, sem, n, _ = _aggregate(groups.get((switch, at), []))
            label = f"{'Open' if switch == 'low' else 'PB'} / {at}"
            print(f"{label:<14}{n:>6}   {f'{mean:.3f} ± {sem:.3f}':>18}")


def demo_data(seed: int = 0):
    rng = np.random.default_rng(seed)

    def g(params, lo, hi, n=80):
        return {k: np.clip(rng.normal(m, sd, n), lo, hi) for k, (m, sd) in params.items()}
    human = {'score': g({('low', 'HA'): (9, 2.5), ('low', 'MA'): (5.8, 2), ('high', 'HA'): (6.3, 2.5),
                         ('high', 'MA'): (8, 2.2)}, 1, 15),
             'spec': g({('low', 'HA'): (58, 30), ('low', 'MA'): (53, 30), ('high', 'HA'): (75, 25),
                        ('high', 'MA'): (93, 8)}, 0, 100)}
    rl = {'score': g({('low', 'HA'): (7.5, 3), ('low', 'MA'): (5.6, 2), ('high', 'HA'): (5.2, 2.5),
                      ('high', 'MA'): (6.4, 2)}, 1, 14, 300),
          'spec': g({('low', 'HA'): (45, 28), ('low', 'MA'): (42, 25), ('high', 'HA'): (60, 30),
                     ('high', 'MA'): (91, 8)}, 0, 100, 300)}
    theory = {'reward': {('low', 'HA'): np.full(50, 0.91), ('low', 'MA'): np.full(50, 0.84),
                         ('high', 'HA'): np.clip(rng.normal(0.79, 0.08, 50), 0.66, 0.98),
                         ('high', 'MA'): np.clip(rng.normal(0.85, 0.09, 50), 0.61, 0.98)},
              'spec': {('low', 'HA'): np.clip(rng.normal(0.5, 0.3, 50), 0, 2),
                       ('low', 'MA'): np.clip(rng.normal(4, 2, 50), 0, 9),
                       ('high', 'HA'): np.clip(rng.normal(49, 30, 50), 0, 97),
                       ('high', 'MA'): np.clip(rng.normal(72, 25, 50), 3, 97)}}
    return human, rl, theory


def main():
    parser = argparse.ArgumentParser(description="Figure 2: maps + HA vs MA (humans, MARL, theory).")
    # human
    parser.add_argument("--human_teams_csv", default=_HUMAN_TEAMS_DEFAULT,
                        help="Per-team human data from fig_sm_ability_pairs_humans_rl.py (required)")
    # analytic model
    parser.add_argument("--analytic_csv", default="data/analytic_model/figure3_specialization_index_long.csv")
    # MARL simulations
    parser.add_argument("--rl_csv", default=None,
                        help="Load MARL records from this CSV instead of crawling the simulation tree")
    parser.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    parser.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    parser.add_argument("--synergy", default="1.35")
    parser.add_argument("--specialization", default="0.25")
    parser.add_argument("--checkpoint", default=ms.DEFAULT_CHECKPOINT,
                        help="Checkpoint folder checkpoint_<n> of the simulated games (default 753), "
                             "'latest' or 'all'")
    parser.add_argument("--game_version", default="classic_collision",
                        choices=["classic", "competition", "classic_collision", "competition_collision"])
    parser.add_argument("--init_type", default=ms.DEFAULT_INIT_TYPE, choices=["empty_init", "random_init"])
    parser.add_argument("--study_name", default="")
    parser.add_argument("--cluster", default="cuenca",
                        help=f"{list(CLUSTER_DIRS)} or a literal base folder")
    parser.add_argument("--verbose", action="store_true")
    # output
    parser.add_argument("--maps_dir", default="./figures", help="Folder with open_map.png and PB_map.png")
    parser.add_argument("--output_dir", default="./figures")
    parser.add_argument("--output_name", default="fig2_human_marl_comparison.pdf")
    parser.add_argument("--demo", action="store_true", help="Synthetic data (layout test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        human, rl, theory = demo_data()
    else:
        # humans
        if not Path(args.human_teams_csv).exists():
            raise FileNotFoundError(
                f"Per-team human file not found: {args.human_teams_csv}. Run fig_sm_ability_pairs_humans_rl.py "
                f"first (it writes <output>_human_teams.csv) or pass --human_teams_csv.")
        human_df = load_human_teams(Path(args.human_teams_csv))
        print(f"Humans: {len(human_df)} teams from {args.human_teams_csv}")
        human = {'score': groups_from_long(human_df, "total_deliveries"),
                 'spec': groups_from_long(human_df, "specialization_index")}

        # MARL
        if args.rl_csv:
            rl_df = pd.read_csv(args.rl_csv)
        else:
            rl_df = collect_rl_records(args)
            cache = output_dir / "fig2_rl_records.csv"
            rl_df.to_csv(cache, index=False)
            print(f"\nMARL records cached -> {cache}  (re-use with --rl_csv)")
        rl = {'score': groups_from_long(rl_df, "total_deliveries"),
              'spec': groups_from_long(rl_df, "specialization_index")}
        write_stats(human_df, rl_df, output_dir / (Path(args.output_name).stem + "_stats.txt"))

        # theory
        th = load_analytic_csv(Path(args.analytic_csv))
        theory = {'reward': groups_from_long(th, "reward_rate"),
                  'spec': groups_from_long(th, "model_specialization_index_rate")}

    for name, d in (("Human game score", human['score']), ("Human specialization", human['spec']),
                    ("MARL game score", rl['score']), ("MARL specialization", rl['spec']),
                    ("Theory reward rate", theory['reward']), ("Theory specialization", theory['spec'])):
        print_summary(name, d)

    make_figure(human, rl, theory, Path(args.maps_dir), output_dir / args.output_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())