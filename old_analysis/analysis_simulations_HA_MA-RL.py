#!/usr/bin/env python3
"""
HA vs MA Comparison Analysis for SpoiledBroth experiments.

Produces a two-subplot figure comparing Homogeneous Ability (HA) and Mixed Ability (MA) agent pairs
on the Baseline and Encouraged maps, across matched synergy/specialization/checkpoint conditions.

Subplot 1 (left):  Mean total deliveries  ± SEM  (four x-axis groups: HA-Baseline, MA-Baseline, HA-Encouraged, MA-Encouraged)
Subplot 2 (right): Specialization index   ± SEM  defined as |deliveries_agent1 - deliveries_agent2| / total_deliveries
                   (only meaningful actions / cuts counted per agent when available)

Agent-type classification (from config.txt [AGENT_SPEEDS]):
  HA: WALKING_SPEEDS == {ai_rl_1: 1.0, ai_rl_2: 1.0}  AND  CUTTING_SPEEDS == {ai_rl_1: 1.0, ai_rl_2: 1.0}
  MA: WALKING_SPEEDS matches {ai_rl_1: 1.0, ai_rl_2: 0.4}  AND  CUTTING_SPEEDS matches {ai_rl_1: 0.2, ai_rl_2: 1.0}
      (C1=1.0, W1=1.0, C2=0.2, W2=0.4  →  ai_rl_1 is the fast cutter / slow walker;
       ai_rl_2 is the slow cutter / fast walker — or vice-versa, both orientations accepted)

Usage:

nohup python3 analysis_simulations_HA_MA-RL.py --map_baseline baseline_division_of_labor_large --map_encouraged encouraged_division_of_labor_large --synergy 1.35 --specialization 0.25 --checkpoint 753 --game_version classic_collision --cluster cuenca > log_analysis_HA_MA.out 2>&1 & 

Author: Samuel Lozano
"""

import argparse
import re
import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
# Speed fingerprints
# ---------------------------------------------------------------------------
HA_WALKING  = {"ai_rl_1": 1.0, "ai_rl_2": 1.0}
HA_CUTTING  = {"ai_rl_1": 1.0, "ai_rl_2": 1.0}
 
# MA: one agent is a fast cutter (C=1.0, W low), the other fast walker (C low, W=1.0)
# Accept either assignment order.
MA_SPEED_SETS = [
    # (walking, cutting)
    ({"ai_rl_1": 1.0, "ai_rl_2": 0.4}, {"ai_rl_1": 0.2, "ai_rl_2": 1.0}),
    ({"ai_rl_1": 0.4, "ai_rl_2": 1.0}, {"ai_rl_1": 1.0, "ai_rl_2": 0.2}),
]
 
 
# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------
 
def _parse_speed_dict(raw: str) -> dict:
    """Parse a speed dict string like \"{'ai_rl_1': 1.0, 'ai_rl_2': 0.4}\"."""
    raw = raw.strip()
    result = {}
    for m in re.finditer(r"['\"]?(ai_rl_\d+)['\"]?\s*:\s*([\d.]+)", raw):
        result[m.group(1)] = float(m.group(2))
    return result
 
 
def _speeds_match(parsed: dict, reference: dict, tol: float = 1e-6) -> bool:
    if set(parsed.keys()) != set(reference.keys()):
        return False
    return all(abs(parsed[k] - reference[k]) < tol for k in reference)
 
 
def classify_config(config_path: Path):
    """
    Read config.txt and return 'HA', 'MA', or None if unclassifiable.
    Also returns the checkpoint number found in the config.
    """
    if not config_path.exists():
        return None, None
 
    text = config_path.read_text(errors="replace")
 
    # --- Walking speeds ---
    m_walk = re.search(r"WALKING_SPEEDS\s*:\s*(\{[^}]+\})", text)
    m_cut  = re.search(r"CUTTING_SPEEDS\s*:\s*(\{[^}]+\})", text)
 
    if not m_walk or not m_cut:
        return None, None
 
    walking = _parse_speed_dict(m_walk.group(1))
    cutting = _parse_speed_dict(m_cut.group(1))
 
    # HA check
    if _speeds_match(walking, HA_WALKING) and _speeds_match(cutting, HA_CUTTING):
        agent_type = "HA"
    else:
        agent_type = None
        for w_ref, c_ref in MA_SPEED_SETS:
            if _speeds_match(walking, w_ref) and _speeds_match(cutting, c_ref):
                agent_type = "MA"
                break
 
    # Checkpoint from config
    m_ck = re.search(r"CHECKPOINT_NUMBER\s*:\s*(\S+)", text)
    checkpoint = m_ck.group(1) if m_ck else None
 
    return agent_type, checkpoint
 
 
# ---------------------------------------------------------------------------
# Path resolution (mirrors ComprehensiveAnalysisOrchestrator logic)
# ---------------------------------------------------------------------------
 
def resolve_simulations_root(
    base_cluster_dir: str,
    map_nr: str,
    game_version: str,
    synergy: str | None,
    specialization: str | None,
    num_agents: int = 2,
    study_name: str = "",
) -> Path:
    """Return the simulations/ root path for a given map and parameter set."""
 
    def _with_nested(root: Path) -> Path | None:
        """Apply synergy/specialization nesting if requested."""
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
 
    if num_agents == 1:
        roots = [
            Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/{game_version}/map_{map_nr}"),
            Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/map_{map_nr}"),
        ]
    else:
        roots = [
            Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_version}/map_{map_nr}"),
            Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/map_{map_nr}"),
        ]
 
    for root in roots:
        found = _with_nested(root)
        if found is not None:
            break
    else:
        found = roots[0]  # fallback; will likely not exist
 
    sims_root = found / "simulations"
    if study_name:
        sims_root = sims_root / study_name
    return sims_root
 
 
# ---------------------------------------------------------------------------
# Data loading helpers (mirrors orchestrator _load_simulation_dataframe etc.)
# ---------------------------------------------------------------------------
 
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
 
    pos_files = sorted(simulation_path.glob("positions_*.csv"))
    if not pos_files:
        pos_files = sorted(simulation_path.glob("human_like_positions_*.csv"))
    if not pos_files:
        return None
 
    frames = []
    for pf in pos_files:
        try:
            d = pd.read_csv(pf)
        except Exception:
            continue
        if "agent_id" not in d.columns:
            d["agent_id"] = _extract_agent_id_from_filename(
                pf, ["positions_", "human_like_positions_"]
            )
        for old, new in [("tile_x", "x"), ("tile_y", "y"), ("tick", "frame")]:
            if new not in d.columns and old in d.columns:
                d[new] = d[old]
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else None
 
 
def load_actions(simulation_path: Path) -> pd.DataFrame:
    """Return a normalized actions dataframe (may be empty)."""
    meaningful = simulation_path / "meaningful_actions.csv"
    if meaningful.exists():
        try:
            df = pd.read_csv(meaningful)
            return _normalize_actions(df)
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
                if "player_id" in d.columns:
                    d["agent_id"] = d["player_id"]
                else:
                    d["agent_id"] = _extract_agent_id_from_filename(af, ["human_like_actions_"])
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
 
 
def _normalize_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "agent_id" not in df.columns:
        df["agent_id"] = "unknown"
        if "player_id" in df.columns:
            df["agent_id"] = df["player_id"]
    if "action_category_name" not in df.columns:
        df["action_category_name"] = np.nan
    for col in ["action_long", "action_name", "action_type", "action"]:
        if col in df.columns:
            df["action_category_name"] = df["action_category_name"].fillna(df[col])
    df["action_category_name"] = df["action_category_name"].fillna("unknown")
    return df
 
 
# ---------------------------------------------------------------------------
# Per-simulation metric extraction
# ---------------------------------------------------------------------------
 
def extract_metrics(simulation_path: Path, initialization_period: float = 0.0) -> dict | None:
    """
    Return dict with:
        total_deliveries     : int
        deliveries_per_agent : dict[agent_id -> int]
        cuts_per_agent       : dict[agent_id -> int]  (actions containing 'cut')
    """
    pos = load_positions(simulation_path)
    if pos is None or pos.empty:
        return None
    if "score" not in pos.columns or "agent_id" not in pos.columns:
        return None
 
    pos = pos.copy()
    pos["agent_id"] = pos["agent_id"].astype(str)
 
    # Coerce numeric columns
    for _col in ("second", "frame", "score"):
        if _col in pos.columns:
            pos[_col] = pd.to_numeric(pos[_col], errors="coerce")
 
    # Drop any rows where time or score could not be parsed
    if "second" in pos.columns:
        pos = pos.dropna(subset=["second"])
    elif "frame" in pos.columns:
        pos = pos.dropna(subset=["frame"])
    if "score" in pos.columns:
        pos = pos.dropna(subset=["score"])
 
    if pos.empty:
        return None
 
    # Adjusted time
    if "second" in pos.columns:
        pos["adj_sec"] = (pos["second"] - initialization_period).clip(lower=0)
    elif "frame" in pos.columns:
        pos["adj_sec"] = (pos["frame"] / 30.0 - initialization_period).clip(lower=0)
    else:
        pos["adj_sec"] = 0.0
 
    # Team score deliveries
    score_ts = pos.groupby("adj_sec")["score"].max().sort_index()
    team_deliveries = int((score_ts.diff().fillna(0) > 0).sum())
 
    # Per-agent deliveries
    deliveries_per_agent: dict[str, int] = {}
    for agent_id, grp in pos.groupby("agent_id"):
        grp_sorted = grp.sort_values("adj_sec")
        diffs = grp_sorted["score"].diff().fillna(0)
        deliveries_per_agent[agent_id] = int((diffs > 0).sum())
 
    # Fallback team split approach
    if sum(deliveries_per_agent.values()) == 0 and team_deliveries > 0:
        agents = sorted(pos["agent_id"].unique())
        deliveries_per_agent = {a: team_deliveries // len(agents) for a in agents}
 
    # Per-agent cuts
    actions = load_actions(simulation_path)
    cuts_per_agent: dict[str, int] = {}
    if not actions.empty and "action_category_name" in actions.columns:
        actions["agent_id"] = actions["agent_id"].astype(str)
        cut_mask = actions["action_category_name"].str.contains("cut", case=False, na=False)
        for agent_id, grp in actions.groupby("agent_id"):
            cuts_per_agent[agent_id] = int(cut_mask[grp.index].sum())
 
    return {
        "total_deliveries": team_deliveries,
        "deliveries_per_agent": deliveries_per_agent,
        "cuts_per_agent": cuts_per_agent,
    }
 
 
def read_init_period(config_path: Path) -> float:
    if not config_path.exists():
        return 0.0
    text = config_path.read_text(errors="replace")
    m = re.search(r"AGENT_INITIALIZATION_PERIOD\s*:\s*([\d.]+)", text)
    return float(m.group(1)) if m else 0.0
 
 
# ---------------------------------------------------------------------------
# Specialization index
# ---------------------------------------------------------------------------
 
def specialization_index(metrics: dict) -> float | None:
    """
    Calculates specialization as the averaged percentage asymmetry across 
    both deliveries and cuts:
    ((|del_a1 - del_a2| / total_del) + (|cuts_a1 - cuts_a2| / total_cuts)) * 100 / 2
    """
    per_agent_cuts = metrics.get("cuts_per_agent", {})
    per_agent_del = metrics.get("deliveries_per_agent", {})

    # Ensure data exists for at least two agents
    if len(per_agent_cuts) < 2 or len(per_agent_del) < 2:
        return None

    vals_cuts = list(per_agent_cuts.values())[:2]
    vals_del = list(per_agent_del.values())[:2]

    total_cuts = sum(vals_cuts)
    total_del = sum(vals_del)

    # Case 1: No activity at all
    if total_cuts == 0 or total_del == 0:
        return 0.0

    # Case 2: Both actions occurred (Standard execution of your formula)
    if total_cuts > 0 and total_del > 0:
        cuts_term = (vals_cuts[0] - vals_cuts[1]) / total_cuts
        del_term = (vals_del[1] - vals_del[0]) / total_del
        return abs(del_term + cuts_term) * 100 / 2

    return None
 
# ---------------------------------------------------------------------------
# Main data collection
# ---------------------------------------------------------------------------
 
def collect_group_metrics(
    simulations_root: Path,
    checkpoint_filter: str | None,
    agent_type_filter: str,
    verbose: bool = True,
) -> list[dict]:
    """
    Traverse all Training_*/checkpoint_*/simulation_* directories under
    simulations_root.
    """
    results = []
 
    if not simulations_root.exists():
        if verbose:
            print(f"  [WARN] simulations root not found: {simulations_root}")
        return results
 
    training_dirs = sorted(simulations_root.glob("Training_*"))
    if not training_dirs:
        if verbose:
            print(f"  [WARN] no Training_* dirs found in {simulations_root}")
        return results
 
    for training_dir in training_dirs:
        checkpoint_dirs = sorted(training_dir.glob("checkpoint_*"))
        for ck_dir in checkpoint_dirs:
            ck_number = ck_dir.name.replace("checkpoint_", "")
            # Optionally filter by checkpoint
            if checkpoint_filter is not None and ck_number != str(checkpoint_filter):
                continue
 
            sim_dirs = sorted(ck_dir.glob("simulation_*"))
            for sim_dir in sim_dirs:
                config_path = sim_dir / "config.txt"
                agent_type, config_ck = classify_config(config_path)
 
                # Config checkpoint takes precedence for filtering if available
                effective_ck = config_ck if config_ck is not None else ck_number
                if checkpoint_filter is not None and str(effective_ck) != str(checkpoint_filter):
                    continue
 
                if agent_type != agent_type_filter:
                    continue
 
                init_period = read_init_period(config_path)
                metrics = extract_metrics(sim_dir, initialization_period=init_period)
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
                    print(
                        f"  {sim_dir.name}  deliveries={metrics['total_deliveries']}  "
                        f"spec_idx={spec:.3f if spec is not None else 'N/A'}"
                    )
 
    return results
 
 
# ---------------------------------------------------------------------------
# Plotting  –  High-Contrast Raincloud Style
# ---------------------------------------------------------------------------

# High contrast colors Reference Style scheme
_COLOR_HA = '#E24A33' # Reddish-Salmon
_COLOR_MA = '#348ABD' # Teal-Blue
_COLOR = {"HA": _COLOR_HA, "MA": _COLOR_MA}
_COLOR_DARK = {"HA": "black", "MA": "black"}

# Increased alpha values based on Reference style requirements
_ALPHA_VIOLIN = 0.55  # Increased density
_ALPHA_BOX    = 0.85  # Much more solid, less pastel
_VIOLIN_WIDTH = 0.30   # half-violin max half-width in data-x units
_BOX_WIDTH    = 0.10
_WHISK_CAP    = 0.06
 
# x-positions of the two maps on the axis
_MAP_X = {"Baseline": 1.0, "Encouraged": 2.0}
# side offset: HA drawn slightly left, MA slightly right of the map x
_SIDE_OFFSET = {"HA": -0.18, "MA": +0.18}
# violin fans outward from the box (HA to the left, MA to the right)
_VIOLIN_DIR  = {"HA": -1, "MA": +1}
 
_FS_AXIS  = 30
_FS_TICK_X  = 30
_FS_TICK_Y  = 30
_FS_PANEL = 28
_FS_LEG   = 30
 
 
def _aggregate(records: list[dict], key: str):
    """Return (mean, sem, n, values) for a numeric key, ignoring None."""
    vals = [r[key] for r in records if r.get(key) is not None]
    if not vals:
        return np.nan, np.nan, 0, []
    arr = np.array(vals, dtype=float)
    mean = arr.mean()
    sem  = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, sem, len(arr), arr.tolist()
 
 
def _draw_half_violin(ax, x_center, values, color, direction, color_dark='black', y_min=None, y_max=None):
    """Draw a high-contrast half-violin fanning in `direction` (+1=right, -1=left)"""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return
    if np.std(arr) < 1e-10:
        arr = arr + np.random.default_rng(0).normal(0, 1e-8, size=len(arr))
    kde = gaussian_kde(arr, bw_method="scott")
    y_lo = arr.min() if y_min is None else y_min
    y_hi = arr.max() if y_max is None else y_max
    ys = np.linspace(y_lo, y_hi, 200)
    density = kde(ys)
    density_norm = (density / density.max() * _VIOLIN_WIDTH * direction
                    if density.max() > 0 else density * 0)
    xs_outer = x_center + density_norm
    xs_inner = np.full_like(xs_outer, x_center)
    # Area fill (transparent)
    ax.fill_betweenx(ys, xs_inner, xs_outer, color=color, alpha=_ALPHA_VIOLIN, linewidth=0, zorder=2)
    # Add solid dark path outline
    ax.plot(xs_outer, ys, color=color_dark, linewidth=1.0, alpha=1.0, zorder=2.5)
 
 
def _draw_boxplot(ax, x_center, values, color, color_dark):
    """High-contrast IQR box + whiskers (1.5×IQR) centred at x_center."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_whisk = max(arr.min(), q1 - 1.5 * iqr)
    hi_whisk = min(arr.max(), q3 + 1.5 * iqr)
    bw = _BOX_WIDTH / 2
    # IQR box
    rect = plt.Rectangle(
        (x_center - bw, q1), _BOX_WIDTH, iqr,
        facecolor=color, edgecolor=color_dark,
        linewidth=1.3, alpha=_ALPHA_BOX, zorder=3,
    )
    ax.add_patch(rect)
    # Thicker black median line
    ax.plot([x_center - bw, x_center + bw], [med, med],
            color=color_dark, linewidth=1.8, zorder=4)
    # Pure black whiskers
    ax.plot([x_center, x_center], [lo_whisk, q1],
            color=color_dark, linewidth=1.2, zorder=3)
    ax.plot([x_center, x_center], [q3, hi_whisk],
            color=color_dark, linewidth=1.2, zorder=3)
    # Pure black caps
    cw = _WHISK_CAP / 2
    ax.plot([x_center - cw, x_center + cw], [lo_whisk, lo_whisk],
            color=color_dark, linewidth=1.2, zorder=3)
    ax.plot([x_center - cw, x_center + cw], [hi_whisk, hi_whisk],
            color=color_dark, linewidth=1.2, zorder=3)
 
 
def _draw_diamond_mean(ax, x_center, mean, color, color_dark, size=10):
    """Large filled diamond marker for the mean with dark outline."""
    ax.plot(x_center, mean,
            marker="D", markersize=size,
            markerfacecolor=color, markeredgecolor=color_dark,
            markeredgewidth=1.5, zorder=5, linestyle="none")
 
 
def make_figure(
    data: dict,
    synergy: str,
    specialization: str,
    checkpoint: str,
    output_path: Path,
):
    maps        = ["Baseline", "Encouraged"]
    map_labels  = ["Open", "PB"]   # x-tick labels as in the reference
    agent_types = ["HA", "MA"]
    legend_labels = {"HA": "High", "MA": "Mixed"}
 
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # DPI handled by config
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.22, wspace=0.6)
 
    metric_keys  = ["total_deliveries", "specialization_index"]
    ylabels      = ["Game score", "Specialization index"]
 
    for ax_idx, ax in enumerate(axes):
        metric_key = metric_keys[ax_idx]
        is_spec    = (ax_idx == 1)
 
        if is_spec:
            # Force limits to 0.0 and 1.0 (0-100%) for Specialization Index
            y_lo = 0.0
            y_hi = 100.0
        else:
            # Game score axis matched limits
            y_lo = 0.0
            y_hi = 16.0
 
        ax.set_ylim(y_lo, y_hi)
 
        # ── Draw per map × agent_type ────────────────────────────────────
        mean_pts = {at: [] for at in agent_types}
 
        for map_name, map_x in _MAP_X.items():
            for at in agent_types:
                recs = data.get((map_name, at), [])
                mean, sem, n, vals = _aggregate(recs, metric_key)
                if not vals:
                    mean_pts[at].append((map_x + _SIDE_OFFSET[at], np.nan))
                    continue
 
                x_center = map_x + _SIDE_OFFSET[at]
                color      = _COLOR[at]
                color_dark = _COLOR_DARK[at]
                direction  = _VIOLIN_DIR[at]
 
                # High-contrast primitives
                _draw_half_violin(
                    ax, x_center, vals, color, direction, color_dark=color_dark,
                    y_min=y_lo, y_max=y_hi,
                )
                _draw_boxplot(ax, x_center, vals, color, color_dark)
                _draw_diamond_mean(ax, x_center, mean, color, color_dark)
 
                mean_pts[at].append((x_center, mean))
 
        # ── Connecting lines between map positions ───────────────────────
        for at in agent_types:
            pts = mean_pts[at]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            valid = [not np.isnan(y) for y in ys]
            if all(valid):
                actual_color = _COLOR_HA if at == "HA" else _COLOR_MA
                ax.plot(xs, ys, color=actual_color, linewidth=2.5,
                        zorder=4, solid_capstyle="round", alpha=0.8)
 
        # ── x-axis ticks / labels ────────────────────────────────────────
        ax.set_xticks(list(_MAP_X.values()))
        ax.set_xticklabels(map_labels, fontsize=_FS_TICK_X)
        ax.set_xlim(0.4, 2.6)
 
        # ── y-axis Formatting (PNAS Style) ──────────────────────────────────────────────────────
        ax.set_ylabel(ylabels[ax_idx], fontweight="bold", fontsize=_FS_AXIS) # BOLD Y-Label
        ax.tick_params(axis="y", labelsize=_FS_TICK_Y)
        if is_spec:
            # Format y as percentage
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
 
        # ── Grid (light horizontal lines as in reference) ────────────────
        ax.yaxis.grid(True, color="lightgray", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
 
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
 
    # ── Shared legend (High-Contrast HandlerTuple style) ───────────────────────
    legend_handles = []
    for at in agent_types:
        handle = matplotlib.lines.Line2D(
            [], [], marker="D", markersize=10,
            markerfacecolor=_COLOR[at], markeredgecolor="black",
            markeredgewidth=1.5, linewidth=0, label=legend_labels[at],
            alpha=1.0 # Legend solid
        )
        box_patch = matplotlib.patches.Patch(
            facecolor=_COLOR[at], edgecolor="black",
            alpha=_ALPHA_BOX, linewidth=1.3,
        )
        legend_handles.append((box_patch, handle))
 
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Figure saved (High Resolution, PNAS style) → {output_path}")
 
 
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Compare HA vs MA agent pairs on Baseline and Encouraged maps."
    )
 
    # Map names
    parser.add_argument("--map_baseline",   required=True,
                        help='Map name for Baseline, e.g. "baseline_division_of_labor_large"')
    parser.add_argument("--map_encouraged", required=True,
                        help='Map name for Encouraged, e.g. "encouraged_division_of_labor_large"')
 
    # Experiment parameters
    parser.add_argument("--synergy",        required=True,
                        help="Synergy folder value, e.g. 1.35")
    parser.add_argument("--specialization", required=True,
                        help="Specialization folder value, e.g. 0.25")
    parser.add_argument("--checkpoint",     required=True,
                        help="Checkpoint number to filter (e.g. 753). Use 'all' to include all checkpoints.")
    parser.add_argument("--game_version",   default="classic_collision",
                        choices=["classic", "competition", "classic_collision", "competition_collision"],
                        help="Game version (default: classic_collision)")
    parser.add_argument("--num_agents",     type=int, default=2, choices=[1, 2])
    parser.add_argument("--study_name",     default="",
                        help="Optional study subdirectory under simulations/")
 
    # Infrastructure
    parser.add_argument("--cluster",        default="cuenca",
                        choices=["cuenca", "brigit", "local"],
                        help="Cluster preset (default: cuenca)")
    parser.add_argument("--output_dir",     default='./figures',
                        help="Where to save the figure (default: current directory)")
    parser.add_argument("--output_name",    default="figure3_HA_MA_comparison_RL.png",
                        help="Output filename (default: figure3_HA_MA_comparison_RL.png)")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print per-simulation details")
 
    args = parser.parse_args()
 
    # Resolve cluster base dir
    cluster_map = {
        "cuenca": "",
        "brigit": "/mnt/lustre/home/samuloza/",
        "local":  "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado",
    }
    base_cluster_dir = cluster_map[args.cluster]
 
    checkpoint_filter = None if args.checkpoint.lower() == "all" else args.checkpoint
 
    # ------------------------------------------------------------------ #
    # Collect data for all four groups                                    #
    # ------------------------------------------------------------------ #
    group_configs = [
        ("Baseline",   args.map_baseline),
        ("Encouraged", args.map_encouraged),
    ]
 
    data: dict[tuple[str, str], list[dict]] = {}
 
    for map_label, map_nr in group_configs:
        sims_root = resolve_simulations_root(
            base_cluster_dir=base_cluster_dir,
            map_nr=map_nr,
            game_version=args.game_version,
            synergy=args.synergy,
            specialization=args.specialization,
            num_agents=args.num_agents,
            study_name=args.study_name,
        )
 
        for agent_type in ("HA", "MA"):
            print(f"\n{'='*60}")
            print(f"Collecting {agent_type} data for {map_label} map")
            print(f"  Simulations root : {sims_root}")
            print(f"  Checkpoint filter: {checkpoint_filter or 'all'}")
            print(f"{'='*60}")
 
            records = collect_group_metrics(
                simulations_root=sims_root,
                checkpoint_filter=checkpoint_filter,
                agent_type_filter=agent_type,
                verbose=args.verbose,
            )
 
            data[(map_label, agent_type)] = records
            print(f"  → {len(records)} simulations collected for {map_label}/{agent_type}")
 
    # ------------------------------------------------------------------ #
    # Summary table                                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    header = f"{'Group':<22}  {'N':>4}  {'Deliveries mean±SEM':>22}  {'Spec.idx mean±SEM':>20}"
    print(header)
    print("-" * len(header))
    for (map_label, agent_type), records in data.items():
        n = len(records)
        dm, ds, dn, _ = _aggregate(records, "total_deliveries")
        sm, ss, sn, _ = _aggregate(records, "specialization_index")
        d_str = f"{dm:.2f} ± {ds:.2f}" if not np.isnan(dm) else "N/A"
        s_str = f"{sm:.3f} ± {ss:.3f}" if not np.isnan(sm) else "N/A"
        print(f"{map_label}/{agent_type:<10}  {n:>4}  {d_str:>22}  {s_str:>20}")
 
    # ------------------------------------------------------------------ #
    # Plot                                                                #
    # ------------------------------------------------------------------ #
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name is None:
        args.output_name = f"HA_MA_comparison-RL_synergy_{args.synergy}_specialization_{args.specialization}.png"

    output_path = output_dir / args.output_name
 
    make_figure(
        data=data,
        synergy=args.synergy,
        specialization=args.specialization,
        checkpoint=args.checkpoint,
        output_path=output_path,
    )
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())