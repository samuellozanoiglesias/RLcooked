#!/usr/bin/env python3
"""
Figure S (SI Appendix, Sec. "Ablation") -- Necessity of the specialization and
synergy reward-shaping terms in the MARL simulations.

Derived from fig2_human_marl_comparison.py: same raincloud style (half-violin +
box + diamond mean + connecting line), same fonts and the same page size
(804.96 x 446.52 pt), so that it matches Fig. 2 when both are included at the
same relative width. Panel positions, titles and panel letters were measured on
the reference SM_ablation.pdf.

Layout (2 x 2 groups, each group = game score | specialization index):

      Neither specialization nor synergy   |   Specialization only
        a  Game score   b  Spec. index     |     c  Game score   d  Spec. index
      Synergy only                         |   Specialization and synergy
        e  Game score   f  Spec. index     |     g  Game score   h  Spec. index
      [ legend band: High ability (100%, 100%) | Mixed ability (40%, 20%) ]
      (team speeds as percentages of full speed; rho is reserved for MARL theory)

Reward tiers (folder layout of the simulation tree, see marl_simulations.py)
  <base>/data/samuel_lozano/cooked/<game_version>/simulations/<init_type>/map_<map>/
      synergy_<s>/specialized_<l>/abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
  HA = abilities_1.0_1.0, MA = abilities_0.4_0.2 (walk1 = walking speed of ai_rl_1,
  cut2 = cutting speed of ai_rl_2, the other two speeds 1.0); every Training_* and
  simulation_* folder of checkpoint_753 is used.
  Tier 1  neither term      -> --tier_synergy 0      --tier_specialization 0
  Tier 2  specialization    -> --tier_synergy 0      --tier_specialization 0.25
  Tier 3  synergy           -> --tier_synergy 1.35   --tier_specialization 0
  Tier 4  both (full model) -> --tier_synergy 1.35   --tier_specialization 0.25
Folder values are matched numerically (synergy_0 == synergy_0.0). A value 'none' drops
that nesting level (synergy_<x>/ or specialized_<y>/). If your ablation runs live
elsewhere, pass explicit folders per tier (the ones containing abilities_*) with
--tier_paths (4 templates, '{map}' is replaced by the map name), e.g.
    --tier_paths /data/.../simulations/empty_init/map_{map}/ablation_none ...
Unlike Fig. 2, the resolver here is STRICT: a missing folder is reported and
the tier is left empty instead of silently falling back to the parent folder
(which would plot the full model under the wrong title).

Besides the figure, the script writes <output>_stats.csv and <output>_stats.txt
with the statistics quoted in the SI text, with the conventions of paper_stats.py
(the same as in the main text): per tier, a two-way between-subjects ANOVA
(ability x map; effect coding, Type III SS, partial eta^2) and, per map, mean +- SEM
of HA and MA with Welch's t-test written with the higher group first and |t|, e.g.
  MA > HA: ($6.44 \pm 0.11$ vs.\ $5.28 \pm 0.15$; Welch's $t(89.9) = 6.24$, $p < .001$)
Every test is reported twice: with one value per simulated game (unit 'simulation',
as reported so far) and with one value per training seed (mean of its games; unit
'training', the independent unit, as in the pretraining and five-pair analyses).
--stats_unit selects which one is written first (CSV column 'primary').

Usage:
# every tier from checkpoint_753 of the simulation tree
# (or add --both_from_fig2_csv ./figures/fig2_rl_records.csv to reuse exactly the Fig. 2e-f simulations)
nohup python3 fig_sm_marl_synergy_and_specialization_ablation.py --game_version classic_collision --init_type empty_init --cluster cuenca > log_fig_sm_ablation.out 2>&1 &

    # re-plot from the records cached by a previous run
    python3 fig_sm_marl_synergy_and_specialization_ablation.py \\
        --rl_csv ./figures/fig_sm_ablation_rl_records.csv

    # synthetic data drawn to match the statistics reported in the SI
    python3 fig_sm_marl_synergy_and_specialization_ablation.py --demo

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
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_stats as ps                                   # noqa: E402
import marl_simulations as ms                              # noqa: E402

# ---------------------------------------------------------------------------
# Global Style Configuration (PNAS requirements) -- identical to Fig. 2
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

def groups_from_long(df: pd.DataFrame, value_col: str) -> dict:
    """{(switch, ability): np.ndarray} from a long dataframe."""
    out = {}
    for switch in _SWITCH_ORDER:
        for at in _ABILITY_ORDER:
            sel = df[(df["switch_cost_condition"] == switch) & (df["ability_type"] == at)]
            out[(switch, at)] = pd.to_numeric(sel[value_col], errors="coerce").dropna().to_numpy(float)
    return out


# ===========================================================================
# 1. MARL SIMULATION DATA  (loaders identical to Fig. 2)
# ===========================================================================
# Simulated games are read from the abilities_<walk1>_<cut2> folders of each tier's
# simulation tree (marl_simulations.py, same loaders as Fig. 2): HA = abilities_1.0_1.0,
# MA = abilities_0.4_0.2 (walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2).
ABILITY_PAIRS = {"HA": (1.0, 1.0), "MA": (0.4, 0.2)}
CLUSTER_DIRS = ms.CLUSTER_DIRS


def collect_group_metrics(map_root: Path, checkpoint, agent_type: str, verbose: bool = True) -> list[dict]:
    """Every Training_*/checkpoint_<ck>/simulation_* of the abilities folder of agent_type."""
    if not map_root.exists():
        print(f"  [WARN] simulations root not found: {map_root}")
        return []
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


# ===========================================================================
# 3. RAINCLOUD STYLE  (identical to Fig. 2)
# ===========================================================================
_SWITCH_ORDER = ["low", "high"]          # low = Open map, high = Partially-blocked map
_SWITCH_X = {"low": 1.0, "high": 2.0}
_ABILITY_ORDER = ["HA", "MA"]
_SIDE_OFFSET = {"HA": -0.18, "MA": +0.18}
_VIOLIN_DIR = {"HA": -1, "MA": +1}

_COLOR_HA = '#E24A33'   # Reddish-Salmon
_COLOR_MA = '#348ABD'   # Teal-Blue
_COLOR = {"HA": _COLOR_HA, "MA": _COLOR_MA}
_COLOR_DARK = {"HA": "black", "MA": "black"}
_LEGEND_LABELS = {
    "HA": r"\textbf{High ability (100\%, 100\%)}",
    "MA": r"\textbf{Mixed ability (40\%, 20\%)}",
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


# ===========================================================================
# 2. ABLATION TIERS  (one simulation tree per reward configuration)
# ===========================================================================
TIERS = ["neither", "spec", "syn", "both"]
TIER_TITLES = {
    "neither": "Neither specialization nor synergy",
    "spec": "Specialization only",
    "syn": "Synergy only",
    "both": "Specialization and synergy",
}
TIER_DEFAULT_SYNERGY = ["0", "0", "1.35", "1.35"]
TIER_DEFAULT_SPECIALIZATION = ["0", "0.25", "0", "0.25"]
# Checkpoint FOLDER per tier (checkpoint_<name>): checkpoint_753 for every tier.
TIER_DEFAULT_CHECKPOINT = [ms.DEFAULT_CHECKPOINT] * 4


def _norm_ck(v):
    """'753' -> checkpoint_753 (literal folder); 'latest' = highest-numbered folder; 'all'."""
    return str(v).strip()


def _is_none(v) -> bool:
    return ms.is_none(v)


def resolve_tier_root(base_cluster_dir: str, map_nr: str, game_version: str,
                      synergy, specialization, study_name: str = "",
                      path_template: str | None = None, init_type: str = ms.DEFAULT_INIT_TYPE) -> Path:
    """Folder holding the abilities_* folders of one tier and map:
    .../<game_version>/simulations/<init_type>/map_<map>/synergy_<s>/specialized_<l>.
    STRICT: a missing folder is returned as is and reported (never a parent folder).
    'none' drops a level; path_template ('{map}' substituted) replaces the resolution."""
    return ms.map_root(base_cluster_dir, game_version, map_nr, synergy, specialization,
                       init_type=init_type, study_name=study_name or None, path_template=path_template)


def collect_ablation_records(args) -> pd.DataFrame:
    """Crawl the four tiers x two maps x two compositions -> one long dataframe."""
    base_cluster_dir = ms.cluster_base_dir(args.cluster)
    tier_ck = [args.checkpoint] * 4 if args.checkpoint is not None else args.tier_checkpoints
    templates = args.tier_paths or [None] * 4
    rows = []
    for t_idx, tier in enumerate(TIERS):
        ck = _norm_ck(tier_ck[t_idx])
        for switch, map_nr in (("low", args.map_baseline), ("high", args.map_encouraged)):
            root = resolve_tier_root(base_cluster_dir, map_nr, args.game_version,
                                     args.tier_synergy[t_idx], args.tier_specialization[t_idx],
                                     args.study_name, templates[t_idx], args.init_type)
            for at in _ABILITY_ORDER:
                print(f"\n{'=' * 60}\n[{TIER_TITLES[tier]}] {at} on map {map_nr}\n"
                      f"  Simulations root : {root}\n  Checkpoint folder: "
                      f"{ck if ck.lower() in ('latest', 'all') else 'checkpoint_' + ck}\n{'=' * 60}")
                recs = collect_group_metrics(root, ck, at, verbose=args.verbose)
                for r in recs:
                    rows.append({"tier": tier, "switch_cost_condition": switch,
                                 "ability_type": at, "map": map_nr, **r})
                used = sorted({r["checkpoint"] for r in recs})
                print(f"  -> {len(recs)} simulations from {len({r['training_id'] for r in recs})} "
                      f"trainings, checkpoint folder(s): {used}")
    return pd.DataFrame(rows, columns=["tier", "switch_cost_condition", "ability_type", "map",
                                       "training_id", "checkpoint", "simulation",
                                       "total_deliveries", "specialization_index"])


def tier_groups(df: pd.DataFrame) -> dict:
    """{tier: {'score': groups, 'spec': groups}}"""
    out = {}
    for tier in TIERS:
        sub = df[df["tier"] == tier]
        out[tier] = {"score": groups_from_long(sub, "total_deliveries"),
                     "spec": groups_from_long(sub, "specialization_index")}
    return out


# ===========================================================================
# 4. STATISTICS  (numbers quoted in the SI text)
# ===========================================================================
_MAP_LABEL = {"low": "Open", "high": "PB"}


def _unit_frame(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """unit 'simulation': one row per simulated game; 'training': mean per training seed."""
    d = df.copy()
    for c in ("total_deliveries", "specialization_index"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if unit == "training":
        d = (d.groupby(["tier", "switch_cost_condition", "ability_type", "training_id"], as_index=False)
               [["total_deliveries", "specialization_index"]].mean())
    return d


def ablation_stats(df: pd.DataFrame, units=("training", "simulation")):
    """-> (table, lines) for every unit, tier and measure."""
    rows, lines = [], []
    for unit in units:
        d_all = _unit_frame(df, unit)
        lines.append(f"\n{'#' * 100}\nUNIT = {unit.upper()}"
                     + (" (one value per training seed; independent units)" if unit == "training"
                        else " (one value per simulated game)") + f"\n{'#' * 100}")
        for tier in TIERS:
            d = d_all[d_all["tier"] == tier]
            lines.append(f"\n[{TIER_TITLES[tier]}]")
            for col, metric, name, pct in (("total_deliveries", "score", "Game score", False),
                                           ("specialization_index", "spec", "Specialization index", True)):
                a = ps.anova_type3(d, col, "switch_cost_condition", "ability_type")
                inter = a.iloc[2]
                lines.append(f"  {name}: Map x Ability ({ps.fmt_F(inter)}); "
                             f"map ({ps.fmt_F(a.iloc[0])}); ability ({ps.fmt_F(a.iloc[1])})")
                for switch in _SWITCH_ORDER:
                    sel = d["switch_cost_condition"] == switch
                    ha = d.loc[sel & (d["ability_type"] == "HA"), col].to_numpy(float)
                    ma = d.loc[sel & (d["ability_type"] == "MA"), col].to_numpy(float)
                    txt, res = ps.fmt_compare("HA", ha, "MA", ma, pct)
                    lines.append(f"    {_MAP_LABEL[switch]:<4} {txt}")
                    dh, dm = ps.describe(ha), ps.describe(ma)
                    rows.append({"unit": unit, "tier": tier, "metric": metric, "map": _MAP_LABEL[switch],
                                 "N_HA": dh["n"], "mean_HA": dh["mean"], "sem_HA": dh["sem"],
                                 "N_MA": dm["n"], "mean_MA": dm["mean"], "sem_MA": dm["sem"],
                                 "higher": {"a": "HA", "b": "MA"}.get(res["higher"]),
                                 "welch_t_abs": res["t"], "welch_t_signed_HA_minus_MA": res["t_signed"],
                                 "welch_df": res["df"], "welch_p": res["p"],
                                 "interaction_F": inter["F"], "interaction_df": f"{inter['df']},{inter['df_error']}",
                                 "interaction_p": inter["p"], "interaction_eta_p2": inter["eta_p2"]})
    return pd.DataFrame(rows), lines


# ===========================================================================
# 5. LAYOUT (points, y measured from the TOP of the page; from SM_ablation.pdf)
# ===========================================================================
PAGE_W, PAGE_H = 804.96, 446.52
S_PANEL = 0.54298                      # same scale factor as the MARL panels of Fig. 2

_COLS = [(40.99, 173.31), (252.70, 385.02), (455.20, 587.52), (667.10, 799.42)]
_ROWS = [(41.50, 171.10), (237.20, 366.90)]
# tier -> (row, first column)  ;  each tier occupies two adjacent columns
TIER_POS = {"neither": (0, 0), "spec": (0, 2), "syn": (1, 0), "both": (1, 2)}
PANEL_LETTERS = {"neither": "ab", "spec": "cd", "syn": "ef", "both": "gh"}

# letters: (x, baseline y)   -- left/baseline anchor, cmbx10 at 20 pt
LETTER_XY = {'a': (17.53, 45.73), 'b': (178.26, 45.76), 'c': (432.18, 45.73), 'd': (592.40, 45.76),
             'e': (17.77, 241.44), 'f': (181.14, 241.48), 'g': (431.46, 241.44), 'h': (592.40, 241.48)}
# group titles: (x left, baseline y)  -- cmbx10 at 18 pt, left/baseline anchor
TITLE_XY = {"neither": (45.17, 21.30), "spec": (510.48, 21.30),
            "syn": (131.93, 217.41), "both": (474.84, 217.41)}

LEGEND_BOX = dict(x0=128.23, x1=673.64, top=403.44, bottom=444.79,
                  face='#f2f2f2', edge='#595959', lw=2.25)
LEGEND_CENTER = (402.50, 424.835)
LEGEND_SCALE = 0.60231
LEGEND_PAD = 14.0          # horizontal padding of the legend band (pt)


def tex(font: str, size: float, text: str) -> str:
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def _page_transform(fig):
    return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans


def _rect_to_fig(rect):
    L, R, T, B = rect
    return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]


def make_figure(tg: dict, output_path: Path):
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
    page = _page_transform(fig)
    open_pb = {"low": "Open", "high": "PB"}

    for tier in TIERS:
        row, col0 = TIER_POS[tier]
        T, B = _ROWS[row]
        for k, (metric, ylabel, ylim, pct) in enumerate((
                ("score", "Game score", (0, 16), False),
                ("spec", "Specialization index", (0, 100), True))):
            L, R = _COLS[col0 + k]
            ax = fig.add_axes(_rect_to_fig((L, R, T, B)))
            draw_raincloud_panel(ax, tg[tier][metric], ylabel=ylabel, xticklabels=open_pb,
                                 ylim=ylim, violin_full_range=True, percent=pct, s=S_PANEL)
            letter = PANEL_LETTERS[tier][k]
            x, y = LETTER_XY[letter]
            fig.text(x, y, tex('cmbx10', 20, letter), transform=page, ha='left', va='baseline', fontsize=20)
        x, y = TITLE_XY[tier]
        fig.text(x, y, tex('cmbx10', 18, TIER_TITLES[tier]), transform=page, ha='left',
                 va='baseline', fontsize=18)

    # -- legend band (identical to Fig. 2) ---------------------------------------
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
                     frameon=False, loc="center", ncol=2, handlelength=2.5, columnspacing=2.5,
                     bbox_to_anchor=(cx / PAGE_W, 1.0 - cy / PAGE_H), bbox_transform=fig.transFigure)
    # grey band sized to the legend (never narrower than the Fig. 2 band), centred on it
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    width_pt = max(bb.width / fig.dpi * 72.0 + 2 * LEGEND_PAD, lb['x1'] - lb['x0'])
    fig.add_artist(matplotlib.patches.Rectangle(
        (cx - width_pt / 2, lb['top']), width_pt, lb['bottom'] - lb['top'], transform=page,
        facecolor=lb['face'], edgecolor=lb['edge'], linewidth=lb['lw'], zorder=0))

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    plt.close(fig)
    print(f"\nFigure saved -> {output_path} (+ .png)")


# ===========================================================================
# 6. DEMO / CLI
# ===========================================================================
# (mean, SEM) reported in the SI text, N = 50 per group  -> synthetic data
_DEMO_TARGETS = {
    #            Open-HA        Open-MA        PB-HA          PB-MA
    "neither": {"score": [(8.54, .26), (6.41, .18), (7.17, .30), (3.94, .27)],
                "spec":  [(48.0, 4.0), (47.0, 3.5), (62.1, 4.7), (66.4, 3.2)]},
    "spec":    {"score": [(9.10, .28), (4.83, .21), (7.30, .21), (7.07, .25)],
                "spec":  [(51.0, 4.0), (34.0, 4.0), (64.9, 3.4), (92.8, 2.0)]},
    "syn":     {"score": [(7.71, .46), (5.09, .17), (5.43, .22), (4.26, .26)],
                "spec":  [(50.0, 4.0), (31.0, 3.0), (57.6, 2.9), (64.2, 2.9)]},
    "both":    {"score": [(7.58, .16), (5.68, .10), (5.28, .15), (6.44, .11)],
                "spec":  [(45.3, 1.5), (42.7, 1.2), (60.5, 2.0), (91.1, 0.8)]},
}


def demo_records(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Synthetic records whose group means/SEMs follow the values reported in the SI.
    Scores are integer deliveries, specialization indices are clipped to [0, 100]."""
    rng = np.random.default_rng(seed)
    keys = [("low", "HA"), ("low", "MA"), ("high", "HA"), ("high", "MA")]
    rows = []
    for tier, tgt in _DEMO_TARGETS.items():
        for (switch, at), (m_score, ss), (mp, sp) in zip(keys, tgt["score"], tgt["spec"]):
            score = np.clip(np.rint(rng.normal(m_score, ss * np.sqrt(n), n)), 0, None)
            if mp > 85:   # saturated specialization: skewed towards 100 %
                spec = 100 - rng.gamma(1.2, (100 - mp) / 1.2, n)
            else:
                spec = rng.normal(mp, sp * np.sqrt(n), n)
            spec = np.clip(spec, 0, 100)
            for i in range(n):
                rows.append({"tier": tier, "switch_cost_condition": switch, "ability_type": at,
                             "map": switch, "training_id": f"demo_{i % 8}", "checkpoint": "demo",
                             "simulation": f"simulation_{i}", "total_deliveries": score[i],
                             "specialization_index": spec[i]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="SI figure: ablation of the specialization and synergy terms.")
    p.add_argument("--rl_csv", default=None,
                   help="Load records from this CSV instead of crawling the simulation trees")
    p.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    p.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    p.add_argument("--tier_synergy", nargs=4, default=TIER_DEFAULT_SYNERGY,
                   metavar=("NEITHER", "SPEC", "SYN", "BOTH"),
                   help="synergy_<x> folder per tier ('none' = no synergy level)")
    p.add_argument("--tier_specialization", nargs=4, default=TIER_DEFAULT_SPECIALIZATION,
                   metavar=("NEITHER", "SPEC", "SYN", "BOTH"),
                   help="specialized_<y> folder per tier ('none' = no specialization level)")
    p.add_argument("--tier_paths", nargs=4, default=None,
                   metavar=("NEITHER", "SPEC", "SYN", "BOTH"),
                   help="Explicit folder per tier that contains the abilities_* folders "
                        "('{map}' is substituted)")
    p.add_argument("--tier_checkpoints", nargs=4, default=TIER_DEFAULT_CHECKPOINT,
                   metavar=("NEITHER", "SPEC", "SYN", "BOTH"),
                   help="Checkpoint folder per tier (literal checkpoint_<name>), 'latest' "
                        "(highest-numbered folder) or 'all'. Default: 753 753 753 753")
    p.add_argument("--init_type", default=ms.DEFAULT_INIT_TYPE, choices=["empty_init", "random_init"])
    p.add_argument("--both_from_fig2_csv", default=None,
                   help="Take Tier 4 (full model) from the MARL records cached by "
                        "fig2_human_marl_comparison.py (fig2_rl_records.csv), so that panels g-h are "
                        "exactly the simulations of main-text Fig. 2e-f")
    p.add_argument("--checkpoint", default=None,
                   help="Same checkpoint for ALL tiers (overrides --tier_checkpoints)")
    p.add_argument("--game_version", default="classic_collision",
                   choices=["classic", "competition", "classic_collision", "competition_collision"])
    p.add_argument("--study_name", default="")
    p.add_argument("--cluster", default="cuenca", help=f"{list(CLUSTER_DIRS)} or a literal base folder")
    p.add_argument("--stats_unit", choices=["training", "simulation"], default="training",
                   help="Unit reported first (both are always reported): one value per training seed "
                        "(default, independent units) or per simulated game")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output_dir", default="./figures")
    p.add_argument("--output_name", default="fig_sm_marl_synergy_and_specialization_ablation.pdf")
    p.add_argument("--demo", action="store_true", help="Synthetic data matching the SI statistics")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / args.output_name

    if args.demo:
        df = demo_records()
    elif args.rl_csv:
        df = pd.read_csv(args.rl_csv)
    else:
        df = collect_ablation_records(args)
        cache = out_dir / "fig_sm_ablation_rl_records.csv"
        df.to_csv(cache, index=False)
        print(f"\nMARL records cached -> {cache}  (re-use with --rl_csv)")

    if args.both_from_fig2_csv:
        f2 = pd.read_csv(args.both_from_fig2_csv)
        f2 = f2[f2["ability_type"].isin(_ABILITY_ORDER)].copy()
        f2.insert(0, "tier", "both")
        df = pd.concat([df[df["tier"] != "both"], f2[[c for c in df.columns if c in f2.columns]]],
                       ignore_index=True)
        print(f"\nTier 4 (full model) taken from {args.both_from_fig2_csv}: {len(f2)} simulations")

    # which checkpoint folders ended up in each tier (also catches a stale --rl_csv cache)
    if "checkpoint" in df.columns:
        print("\nCheckpoint folders per tier:")
        for tier in TIERS:
            cks = df.loc[df["tier"] == tier, "checkpoint"].astype(str).value_counts().to_dict()
            print(f"  {TIER_TITLES[tier]:<36} {cks}")

    tg = tier_groups(df)
    units = (args.stats_unit, "simulation" if args.stats_unit == "training" else "training")
    st, lines = ablation_stats(df, units)
    st["primary"] = st["unit"] == args.stats_unit
    header = ["=" * 100,
              "ABLATION STATISTICS -- conventions of paper_stats.py (higher group first, Welch's t(df) = |t|;",
              "two-way between-subjects ANOVA ability x map, effect coding, Type III SS, partial eta^2)",
              "=" * 100]
    print("\n".join(header + lines))
    st_path = out_pdf.with_name(out_pdf.stem + "_stats.csv")
    st.to_csv(st_path, index=False)
    st_path.with_suffix(".txt").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"\nStatistics saved -> {st_path} (+ .txt)")

    make_figure(tg, out_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())