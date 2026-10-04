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
      a  Open (O) map          |   c  Game score         d  Specialization index
      b  Partially-blocked map |
    MARL Simulations           | MARL Theoretical Solutions
      e  Game score            |   g  Reward Rate
      f  Specialization index  |   h  Specialization Index
    [ legend band: High ability (rho = 1) | Mixed ability (rho < 1) ]

Inputs
  ./figures/open_map.png, ./figures/PB_map.png      (map screenshots, --maps_dir)
  Human   : scores_human.csv, specialization_index_human.csv  (wide format)
  MARL    : SLURM simulation tree (as in the RL script), or a cached CSV
            written by a previous run (--rl_csv)
  Theory  : figure3_specialization_index_long.csv   (long format)

Every panel keeps the raincloud style of the original scripts (half-violin +
box + diamond mean + connecting line). Panel positions, sizes, fonts and
line widths were measured from Fig2.pdf: each 2-panel group is drawn with
its original 10 x 5 in design and uniformly scaled by the same factor used
when the figure was assembled, and the page is written at the same size
(804.96 x 446.52 pt) without bbox_inches='tight'.

Usage:
nohup python3 fig2_human_marl_comparison.py \
    --scores_csv ./data/human/scores_human.csv \
    --specialization_csv ./data/human/specialization_index_human.csv \
    --analytic_csv ./data/analytic_model/figure3_specialization_index_long.csv \
    --map_baseline baseline_division_of_labor_large \
    --map_encouraged encouraged_division_of_labor_large \
    --synergy 1.35 --specialization 0.25 --checkpoint 753 \
    --game_version classic_collision --cluster cuenca > log_fig2.out 2>&1 &

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
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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


def load_wide_human_csv(csv_path: Path) -> pd.DataFrame:
    """Wide (participant x condition) CSV -> long df: switch_cost_condition, ability_type, value."""
    df = pd.read_csv(csv_path).reset_index(names="subject_id")
    frames = []
    for col in df.columns:
        if col == "subject_id":
            continue
        switch, ability = _parse_column_label(col)
        sub = df[["subject_id", col]].rename(columns={col: "value"})
        sub["switch_cost_condition"] = switch
        sub["ability_type"] = ability
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.dropna(subset=["value"])


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
HA_WALKING = {"ai_rl_1": 1.0, "ai_rl_2": 1.0}
HA_CUTTING = {"ai_rl_1": 1.0, "ai_rl_2": 1.0}
# MA: one agent is a fast cutter (C=1.0, W low), the other fast walker (C low, W=1.0)
MA_SPEED_SETS = [
    ({"ai_rl_1": 1.0, "ai_rl_2": 0.4}, {"ai_rl_1": 0.2, "ai_rl_2": 1.0}),
    ({"ai_rl_1": 0.4, "ai_rl_2": 1.0}, {"ai_rl_1": 1.0, "ai_rl_2": 0.2}),
]
CLUSTER_DIRS = {
    "cuenca": "",
    "brigit": "/mnt/lustre/home/samuloza/",
    "local": "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado",
}


def _parse_speed_dict(raw: str) -> dict:
    result = {}
    for m in re.finditer(r"['\"]?(ai_rl_\d+)['\"]?\s*:\s*([\d.]+)", raw.strip()):
        result[m.group(1)] = float(m.group(2))
    return result


def _speeds_match(parsed: dict, reference: dict, tol: float = 1e-6) -> bool:
    if set(parsed.keys()) != set(reference.keys()):
        return False
    return all(abs(parsed[k] - reference[k]) < tol for k in reference)


def classify_config(config_path: Path):
    """Return ('HA' | 'MA' | None, checkpoint) from a simulation config.txt."""
    if not config_path.exists():
        return None, None
    text = config_path.read_text(errors="replace")
    m_walk = re.search(r"WALKING_SPEEDS\s*:\s*(\{[^}]+\})", text)
    m_cut = re.search(r"CUTTING_SPEEDS\s*:\s*(\{[^}]+\})", text)
    if not m_walk or not m_cut:
        return None, None
    walking = _parse_speed_dict(m_walk.group(1))
    cutting = _parse_speed_dict(m_cut.group(1))

    if _speeds_match(walking, HA_WALKING) and _speeds_match(cutting, HA_CUTTING):
        agent_type = "HA"
    else:
        agent_type = None
        for w_ref, c_ref in MA_SPEED_SETS:
            if _speeds_match(walking, w_ref) and _speeds_match(cutting, c_ref):
                agent_type = "MA"
                break
    m_ck = re.search(r"CHECKPOINT_NUMBER\s*:\s*(\S+)", text)
    return agent_type, (m_ck.group(1) if m_ck else None)


def resolve_simulations_root(base_cluster_dir: str, map_nr: str, game_version: str,
                             synergy: str | None, specialization: str | None,
                             num_agents: int = 2, study_name: str = "") -> Path:
    def _with_nested(root: Path) -> Path | None:
        candidates = []
        if synergy and specialization:
            candidates.append(root / f"synergy_{synergy}" / f"specialized_{specialization}")
        elif synergy:
            candidates.append(root / f"synergy_{synergy}")
        elif specialization:
            candidates.append(root / f"specialized_{specialization}")
        candidates.append(root)
        for c in candidates:
            if c.exists():
                return c
        return None

    sub = "pretraining/" if num_agents == 1 else ""
    roots = [
        Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/{sub}{game_version}/map_{map_nr}"),
        Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/{sub}map_{map_nr}"),
    ]
    for root in roots:
        found = _with_nested(root)
        if found is not None:
            break
    else:
        found = roots[0]
    sims_root = found / "simulations"
    return sims_root / study_name if study_name else sims_root


def _extract_agent_id_from_filename(file_path: Path, prefixes: list[str]) -> str:
    stem = file_path.stem
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def load_positions(simulation_path: Path) -> pd.DataFrame | None:
    sim_csv = simulation_path / "simulation.csv"
    if sim_csv.exists():
        df = pd.read_csv(sim_csv)
        for old, new in [("tile_x", "x"), ("tile_y", "y"), ("tick", "frame")]:
            if new not in df.columns and old in df.columns:
                df[new] = df[old]
        return df

    pos_files = sorted(simulation_path.glob("positions_*.csv")) or \
        sorted(simulation_path.glob("human_like_positions_*.csv"))
    if not pos_files:
        return None
    frames = []
    for pf in pos_files:
        try:
            d = pd.read_csv(pf)
        except Exception:
            continue
        if "agent_id" not in d.columns:
            d["agent_id"] = _extract_agent_id_from_filename(pf, ["positions_", "human_like_positions_"])
        for old, new in [("tile_x", "x"), ("tile_y", "y"), ("tick", "frame")]:
            if new not in d.columns and old in d.columns:
                d[new] = d[old]
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


def _normalize_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "agent_id" not in df.columns:
        df["agent_id"] = df["player_id"] if "player_id" in df.columns else "unknown"
    if "action_category_name" not in df.columns:
        df["action_category_name"] = np.nan
    for col in ["action_long", "action_name", "action_type", "action"]:
        if col in df.columns:
            df["action_category_name"] = df["action_category_name"].fillna(df[col])
    df["action_category_name"] = df["action_category_name"].fillna("unknown")
    return df


def load_actions(simulation_path: Path) -> pd.DataFrame:
    meaningful = simulation_path / "meaningful_actions.csv"
    if meaningful.exists():
        try:
            return _normalize_actions(pd.read_csv(meaningful))
        except Exception:
            pass
    hl_files = sorted(simulation_path.glob("human_like_actions_*.csv"))
    if hl_files:
        frames = []
        for af in hl_files:
            try:
                d = pd.read_csv(af)
            except Exception:
                continue
            if "agent_id" not in d.columns:
                d["agent_id"] = d["player_id"] if "player_id" in d.columns else \
                    _extract_agent_id_from_filename(af, ["human_like_actions_"])
            frames.append(d)
        if frames:
            return _normalize_actions(pd.concat(frames, ignore_index=True, sort=False))
    actions_csv = simulation_path / "actions.csv"
    if actions_csv.exists():
        try:
            return _normalize_actions(pd.read_csv(actions_csv))
        except Exception:
            pass
    return pd.DataFrame(columns=["agent_id", "action_category_name"])


def extract_metrics(simulation_path: Path, initialization_period: float = 0.0) -> dict | None:
    pos = load_positions(simulation_path)
    if pos is None or pos.empty or "score" not in pos.columns or "agent_id" not in pos.columns:
        return None
    pos = pos.copy()
    pos["agent_id"] = pos["agent_id"].astype(str)
    for _col in ("second", "frame", "score"):
        if _col in pos.columns:
            pos[_col] = pd.to_numeric(pos[_col], errors="coerce")
    if "second" in pos.columns:
        pos = pos.dropna(subset=["second"])
    elif "frame" in pos.columns:
        pos = pos.dropna(subset=["frame"])
    pos = pos.dropna(subset=["score"])
    if pos.empty:
        return None

    if "second" in pos.columns:
        pos["adj_sec"] = (pos["second"] - initialization_period).clip(lower=0)
    elif "frame" in pos.columns:
        pos["adj_sec"] = (pos["frame"] / 30.0 - initialization_period).clip(lower=0)
    else:
        pos["adj_sec"] = 0.0

    score_ts = pos.groupby("adj_sec")["score"].max().sort_index()
    team_deliveries = int((score_ts.diff().fillna(0) > 0).sum())

    deliveries_per_agent: dict[str, int] = {}
    for agent_id, grp in pos.groupby("agent_id"):
        diffs = grp.sort_values("adj_sec")["score"].diff().fillna(0)
        deliveries_per_agent[agent_id] = int((diffs > 0).sum())
    if sum(deliveries_per_agent.values()) == 0 and team_deliveries > 0:
        agents = sorted(pos["agent_id"].unique())
        deliveries_per_agent = {a: team_deliveries // len(agents) for a in agents}

    actions = load_actions(simulation_path)
    cuts_per_agent: dict[str, int] = {}
    if not actions.empty and "action_category_name" in actions.columns:
        actions["agent_id"] = actions["agent_id"].astype(str)
        cut_mask = actions["action_category_name"].astype(str).str.contains("cut", case=False, na=False)
        for agent_id, grp in actions.groupby("agent_id"):
            cuts_per_agent[agent_id] = int(cut_mask[grp.index].sum())

    return {"total_deliveries": team_deliveries,
            "deliveries_per_agent": deliveries_per_agent,
            "cuts_per_agent": cuts_per_agent}


def read_init_period(config_path: Path) -> float:
    if not config_path.exists():
        return 0.0
    m = re.search(r"AGENT_INITIALIZATION_PERIOD\s*:\s*([\d.]+)", config_path.read_text(errors="replace"))
    return float(m.group(1)) if m else 0.0


def specialization_index(metrics: dict) -> float | None:
    """|(cuts_a1 - cuts_a2)/total_cuts + (del_a2 - del_a1)/total_del| * 100 / 2"""
    per_agent_cuts = metrics.get("cuts_per_agent", {})
    per_agent_del = metrics.get("deliveries_per_agent", {})
    if len(per_agent_cuts) < 2 or len(per_agent_del) < 2:
        return None
    vals_cuts = list(per_agent_cuts.values())[:2]
    vals_del = list(per_agent_del.values())[:2]
    total_cuts, total_del = sum(vals_cuts), sum(vals_del)
    if total_cuts == 0 or total_del == 0:
        return 0.0
    cuts_term = (vals_cuts[0] - vals_cuts[1]) / total_cuts
    del_term = (vals_del[1] - vals_del[0]) / total_del
    return abs(del_term + cuts_term) * 100 / 2


def collect_group_metrics(simulations_root: Path, checkpoint_filter: str | None,
                          agent_type_filter: str, verbose: bool = True) -> list[dict]:
    results = []
    if not simulations_root.exists():
        print(f"  [WARN] simulations root not found: {simulations_root}")
        return results
    training_dirs = sorted(simulations_root.glob("Training_*"))
    if not training_dirs:
        print(f"  [WARN] no Training_* dirs found in {simulations_root}")
        return results

    for training_dir in training_dirs:
        for ck_dir in sorted(training_dir.glob("checkpoint_*")):
            ck_number = ck_dir.name.replace("checkpoint_", "")
            if checkpoint_filter is not None and ck_number != str(checkpoint_filter):
                continue
            for sim_dir in sorted(ck_dir.glob("simulation_*")):
                config_path = sim_dir / "config.txt"
                agent_type, config_ck = classify_config(config_path)
                effective_ck = config_ck if config_ck is not None else ck_number
                if checkpoint_filter is not None and str(effective_ck) != str(checkpoint_filter):
                    continue
                if agent_type != agent_type_filter:
                    continue
                metrics = extract_metrics(sim_dir, initialization_period=read_init_period(config_path))
                if metrics is None:
                    if verbose:
                        print(f"  [SKIP] no usable data in {sim_dir}")
                    continue
                spec = specialization_index(metrics)
                results.append({
                    "training_id": training_dir.name,
                    "checkpoint": ck_number,
                    "simulation": sim_dir.name,
                    "total_deliveries": metrics["total_deliveries"],
                    "specialization_index": spec,
                })
                if verbose:
                    spec_str = f"{spec:.3f}" if spec is not None else "N/A"
                    print(f"  {sim_dir.name}  deliveries={metrics['total_deliveries']}  spec_idx={spec_str}")
    return results


def collect_rl_records(args) -> pd.DataFrame:
    """Crawl the simulation tree -> long df (switch_cost_condition, ability_type, metrics)."""
    base_cluster_dir = CLUSTER_DIRS[args.cluster]
    checkpoint_filter = None if str(args.checkpoint).lower() == "all" else args.checkpoint
    rows = []
    for switch, map_nr in (("low", args.map_baseline), ("high", args.map_encouraged)):
        sims_root = resolve_simulations_root(base_cluster_dir, map_nr, args.game_version,
                                             args.synergy, args.specialization,
                                             args.num_agents, args.study_name)
        for at in _ABILITY_ORDER:
            print(f"\n{'=' * 60}\nCollecting {at} data for map {map_nr}\n"
                  f"  Simulations root : {sims_root}\n"
                  f"  Checkpoint filter: {checkpoint_filter or 'all'}\n{'=' * 60}")
            recs = collect_group_metrics(sims_root, checkpoint_filter, at, verbose=args.verbose)
            for r in recs:
                rows.append({"switch_cost_condition": switch, "ability_type": at, "map": map_nr, **r})
            print(f"  -> {len(recs)} simulations collected")
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
    "HA": r"\textbf{High ability ($\boldsymbol{\rho = 1}$)}",
    "MA": r"\textbf{Mixed ability ($\boldsymbol{\rho = 0.85}$)}",
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

# free text: (TeX font, size pt, text, x, y) -- left/baseline anchor
TEXTS = {
    'a': ('cmbx10', 20, 'a', 18.15, 48.48),
    'b': ('cmbx10', 20, 'b', 191.21, 48.51),
    'c': ('cmbx10', 20, 'c', 392.76, 48.48),
    'd': ('cmbx10', 20, 'd', 581.83, 52.01),
    'e': ('cmbx10', 20, 'e', 17.77, 239.02),
    'f': ('cmbx10', 20, 'f', 181.15, 238.99),
    'g': ('cmbx10', 20, 'g', 391.82, 239.07),
    'h': ('cmbx10', 20, 'h', 582.00, 241.43),
    'title_map':  ('cmbx10', 18, 'Map configuration', 108.38, 21.24),
    'title_hum':  ('cmbx10', 18, 'Human Experiments', 502.02, 21.30),
    'title_marl': ('cmbx10', 18, 'MARL Simulations', 104.11, 217.41),
    'title_theo': ('cmbx10', 18, 'MARL Theoretical Solutions', 464.85, 217.41),
    'cap_open':   ('cmr10', 18, 'Open (O)', 65.69, 183.73),
    'cap_pb':     ('cmr10', 16, 'Partially-blocked (PB)', 200.83, 181.76),
}

LEGEND_BOX = dict(x0=128.23, x1=673.64, top=403.44, bottom=444.79,
                  face='#f2f2f2', edge='#595959', lw=2.25)
LEGEND_CENTER = (402.50, 424.835)
LEGEND_SCALE = 0.60231


def tex(font: str, size: float, text: str) -> str:
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def _page_transform(fig):
    return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans


def _rect_to_fig(rect):
    L, R, T, B = rect
    return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]



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

    # -- raincloud panels ------------------------------------------------
    for group in PANEL_GROUPS.values():
        s = group['s']
        for key, rect in group['axes'].items():
            sp = specs[key]
            ax = fig.add_axes(_rect_to_fig(rect))
            draw_raincloud_panel(ax, sp['groups'], ylabel=sp['ylabel'], xticklabels=sp['xt'],
                                 ylim=sp['ylim'], violin_full_range=sp['full'],
                                 percent=sp['percent'], s=s)

    # -- map screenshots ---------------------------------------------------
    for key, rect in MAP_RECTS.items():
        img = _load_map_image(maps_dir, 'subfigures', MAP_FILES[key])
        ax = fig.add_axes(_rect_to_fig(rect))
        ax.set_axis_off()
        if img is not None:
            ax.imshow(img, aspect='auto', interpolation='lanczos')

    # -- free text ---------------------------------------------------------
    for font, size, text, x, y in TEXTS.values():
        fig.text(x, y, tex(font, size, text), transform=page, ha='left', va='baseline',
                 fontsize=size)

    # -- legend band ---------------------------------------------------------
    lb = LEGEND_BOX
    fig.add_artist(matplotlib.patches.Rectangle(
        (lb['x0'], lb['top']), lb['x1'] - lb['x0'], lb['bottom'] - lb['top'], transform=page,
        facecolor=lb['face'], edgecolor=lb['edge'], linewidth=lb['lw'], zorder=0))
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
    fig.legend(handles, [_LEGEND_LABELS[at] for at in _ABILITY_ORDER],
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=_FS * sl,
               frameon=False, loc="center", ncol=2, handlelength=2.5, columnspacing=2.5,
               bbox_to_anchor=(cx / PAGE_W, 1.0 - cy / PAGE_H), bbox_transform=fig.transFigure)

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
    parser.add_argument("--scores_csv", default="data/human/scores_human.csv")
    parser.add_argument("--specialization_csv", default="data/human/specialization_index_human.csv")
    # analytic model
    parser.add_argument("--analytic_csv", default="data/analytic_model/figure3_specialization_index_long.csv")
    # MARL simulations
    parser.add_argument("--rl_csv", default=None,
                        help="Load MARL records from this CSV instead of crawling the simulation tree")
    parser.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    parser.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    parser.add_argument("--synergy", default="1.35")
    parser.add_argument("--specialization", default="0.25")
    parser.add_argument("--checkpoint", default="753", help="Checkpoint number or 'all'")
    parser.add_argument("--game_version", default="classic_collision",
                        choices=["classic", "competition", "classic_collision", "competition_collision"])
    parser.add_argument("--num_agents", type=int, default=2, choices=[1, 2])
    parser.add_argument("--study_name", default="")
    parser.add_argument("--cluster", default="cuenca", choices=list(CLUSTER_DIRS))
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
        scores = load_wide_human_csv(Path(args.scores_csv))
        spec = load_wide_human_csv(Path(args.specialization_csv))   # already in %
        human = {'score': groups_from_long(scores), 'spec': groups_from_long(spec)}

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
