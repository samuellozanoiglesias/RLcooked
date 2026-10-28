#!/usr/bin/env python3
"""
Figure S + statistics -- Game score and specialization index of human teams and
MARL agents at the five ability pairs of main-text Fig. 4:

    (100%,100%)  (70%,40%)  (40%,20%)  (20%,100%)  (10%,10%)
    labelled (walking speed of player 1, cutting speed of player 2)

on the open and partially-blocked (PB) maps.

Layout (same fonts, panel letters, grid and legend band as Fig. 2 and the SI
ablation figures):

                         Human Experiments
        a  Game score            b  Specialization index
                         MARL Simulations
        c  Game score            d  Specialization index
      [ legend band: the five ability pairs ]

Each panel shows condition means +- SEM (humans: across teams; RL: across
trainings); lines join the open and PB means of the same ability pair.

MARL data (--rl_source):
  'sims' = SIMULATED GAMES of the episode-750 policies (checkpoint_750 folders), crawled with
         the same loader and folder tree (synergy_1.35/specialized_0.25/simulations) as
         Fig. 2e-f, Fig. 3d-f and the SI ablation, so that the (100%,100%) and (40%,20%) cells
         equal Fig. 2e-f wherever the same trainings are simulated. Requires simulations at
         every ability pair.
  'fig4' = the TRAINING data of main-text Fig. 4 (episodes 740-760), described below.
Training data of main-text Fig. 4 ('fig4'), loaded THROUGH
fig4_human_vs_rl_gridmap.py (imported with --fig4_script, default: next to this
file), exactly as fig_sm_marl_ablation.py does: same AbilityGridAnalyzer, same
raw dir, same DataProcessor.load_experiment_data call, same episode window
(prepare_episode_data; default = the Fig. 4 command line: --episode_range
specific --target_episode 750 --num_episodes 20 --synergy 1.35
--specialization 0.25 --init_type empty_init), same speed filter (agent 1 walking
= walk, agent 2 cutting = cut, the other two speeds 1.0) and same
total_deliveries. The specialization index is computed per episode from the
per-agent cut and delivery counters (same function as fig_sm_marl_ablation.py),
then averaged per training: ONE VALUE PER TRAINING is the unit of every RL test
and of the RL means +- SEM, because episodes of one training are not independent.

Statistics (written to <output>_stats.txt and <output>_stats.csv), with the
reporting conventions of paper_stats.py (higher group first, |t|, Welch df,
p = .059 / p < .001, F(df1,df2) with eta_p^2):
  * Main experiment, 2 x 2 (HA (100%,100%) vs MA (40%,20%) x map): two-way
    between-subjects ANOVA (effect coding, Type III SS), and Welch's t-tests per map.
  * All five ability pairs, 5 x 2: omnibus ANOVA (ability pair x map); Welch
    contrasts of every MA pair against (100%,100%) within each map (Holm over the 8
    contrasts of each measure; Mann-Whitney U as rank-based check); Map x Ability
    interaction for each MA pair vs (100%,100%) (Holm over the 4 pairs).
  * The same analyses on the RL trainings, and a human-vs-RL summary of the
    MA - HA differences (sign agreement, Pearson r over the 8 cells).

Specialization index (in %, same formula as the MARL scripts):
    S = 100 * | (D1 - D2)/(D1 + D2) + (C2 - C1)/(C1 + C2) | / 2
with D = deliveries and C = cuts per player; independent of which player is 1.
Teams/episodes with no deliveries or no cuts get S = 0 (as in the MARL scripts).

Human data selection
  * condition suffix '_ability_hints_large_map_with_collision' (the '_large_map'
    sessions without collision are an earlier game version and are excluded, as are
    the small v2 maps and stray ability pairs),
  * maps baseline_division_of_labor_large (Open), encouraged_division_of_labor_large (PB),
  * the first --n_teams teams (default 70) of each ability pair x map, ordered by
    game_start_time; games present in several files are counted once.
  Required files in human_data/: combined_actions_with_demographics_csv.zip and
  oct2026_new_condition_encouraged.zip (+ any later batch).

Usage
    # humans + RL simulated games (episode-750 policies, as Fig. 2e-f)
    nohup python3 fig_sm_ability_pairs_humans_rl.py --human_data ./data/human_data/*.zip --rl_source sims > log_fig_sm_pairs.out 2>&1 &

    # humans + RL training episodes 740-760 (as Fig. 4; run where fig4_human_vs_rl_gridmap.py runs)
    nohup python3 fig_sm_ability_pairs_humans_rl.py --human_data ./human_data/*.zip \\
        --rl_source fig4 > log_fig_sm_pairs.out 2>&1 &

    # re-plot from the caches written by a previous run
    python3 fig_sm_ability_pairs_humans_rl.py \\
        --human_csv ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv \\
        --rl_csv ./figures/fig_sm_ability_pairs_humans_rl_rl_records.csv

The per-team human CSV written here (<output>_human_teams.csv) is also the human
input of fig_sm_marl_ablation.py (pretraining) and fig4_human_vs_rl_gridmap.py
(--human_teams_csv), so the three figures use the same human teams.

Author: Samuel Lozano
"""

import argparse
import importlib.util
import io
import re
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_stats as ps                                   # noqa: E402

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

# ===========================================================================
# 0. CONDITIONS
# ===========================================================================
# (walking speed of player 1, cutting speed of player 2), as fractions of full speed
PAIRS = [(1.0, 1.0), (0.7, 0.4), (0.4, 0.2), (0.2, 1.0), (0.1, 0.1)]
HA = (1.0, 1.0)
MAIN_MA = (0.4, 0.2)
MA_PAIRS = [p for p in PAIRS if p != HA]
SIM_PAIRS = (HA, MAIN_MA)      # pairs with simulated games of the episode-750 policies
MAPS = ["Open", "PB"]
MAP_NAMES = {"baseline_division_of_labor_large": "Open",
             "encouraged_division_of_labor_large": "PB"}
HUMAN_SUFFIX = "_ability_hints_large_map_with_collision"

_PAIR_COLOR = {
    (1.0, 1.0): "#E24A33",   # HA, same red as Fig. 2
    (0.7, 0.4): "#9ECAE1",
    (0.4, 0.2): "#348ABD",   # MA of the main experiment, same blue as Fig. 2
    (0.2, 1.0): "#1F5A96",
    (0.1, 0.1): "#0B2545",
}


def pair_label(p, tex=False):
    a, b = int(round(p[0] * 100)), int(round(p[1] * 100))
    return rf"({a}\%, {b}\%)" if tex else f"({a}%, {b}%)"


def _pair_key(w, c):
    """Snap a (walk, cut) pair to the canonical PAIRS entry (or None)."""
    for p in PAIRS:
        if abs(p[0] - float(w)) < 1e-6 and abs(p[1] - float(c)) < 1e-6:
            return p
    return None


# ===========================================================================
# 1. HUMAN DATA
# ===========================================================================
_COND_RE = re.compile(r"\[p\((\d+), (\d+)\)_cs\(([\d.]+)\)_ws\(([\d.]+)\)\]_"
                      r"\[p\((\d+), (\d+)\)_cs\(([\d.]+)\)_ws\(([\d.]+)\)\](.*)$")
_USECOLS = ["game_id", "player_id", "prolific_id", "phase", "action", "action_long",
            "map_name", "condition", "game_start_time"]
_DEMO_COLS = ["game_id", "prolific_id", "Age", "Sex"]


def parse_condition(cond: str):
    """'[p(1, 2)_cs(1.0)_ws(0.4)]_[p(5, 2)_cs(0.2)_ws(1.0)]_suffix' -> ((0.4, 0.2), suffix).
    Player 1 is the one whose cutting speed is 1.0 (slowed walker), player 2 the one
    whose walking speed is 1.0 (slowed cutter)."""
    m = _COND_RE.match(str(cond))
    if not m:
        return None, None
    cs1, ws1, cs2, ws2 = map(float, (m.group(3), m.group(4), m.group(7), m.group(8)))
    suffix = m.group(9)
    if cs1 == 1.0 and ws2 == 1.0:
        return (ws1, cs2), suffix
    if cs2 == 1.0 and ws1 == 1.0:
        return (ws2, cs1), suffix
    return None, suffix


def _iter_csv_sources(paths):
    """Yield (label, file-like) for every CSV given directly or inside a zip.
    Inside each zip (or folder) the slimmest variant is used for actions:
    combined_actions.csv > *_with_history.csv > *_with_demographics.csv > ..."""
    pref = ["combined_actions.csv", "combined_actions_with_history.csv",
            "combined_actions_with_demographics.csv",
            "combined_actions_with_history_and_demographics.csv"]
    for p in map(Path, paths):
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                by_base = {Path(n).name: n for n in names}
                chosen = next((by_base[b] for b in pref if b in by_base), names[0] if names else None)
                if chosen is None:
                    continue
                yield f"{p.name}:{chosen}", io.BytesIO(zf.read(chosen))
        elif p.suffix.lower() == ".csv":
            yield str(p), p


def _iter_demographic_sources(paths):
    for p in map(Path, paths):
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if "demographics" in Path(n).name and n.lower().endswith(".csv"):
                        yield f"{p.name}:{n}", io.BytesIO(zf.read(n))
                        break
        elif p.suffix.lower() == ".csv" and "demographics" in p.name:
            yield str(p), p


def specialization_from_counts(D1, D2, C1, C2, undefined="zero"):
    if D1 + D2 == 0 or C1 + C2 == 0:
        return 0.0 if undefined == "zero" else np.nan
    return abs((D1 - D2) / (D1 + D2) + (C2 - C1) / (C1 + C2)) * 100 / 2


def load_human_teams(paths, n_teams=70, cut_def="start_cutting", spec_undefined="zero",
                     audit=None):
    """Return one row per selected team (test game) and fill `audit` (list of lines)."""
    audit = audit if audit is not None else []
    frames, seen, file_rows = [], set(), []
    for label, fh in _iter_csv_sources(paths):
        df = pd.read_csv(fh, usecols=lambda c: c in _USECOLS, low_memory=False)
        games = set(df["game_id"].unique())
        new_games = games - seen
        file_rows.append((label, len(df), len(games), len(games - new_games)))
        frames.append(df[df["game_id"].isin(new_games)])
        seen |= new_games
    if not frames:
        raise SystemExit("No human CSV found in --human_data")
    raw = pd.concat(frames, ignore_index=True)

    audit.append("Files read (rows, games, games already loaded from an earlier file):")
    for label, nr, ng, ndup in file_rows:
        audit.append(f"  {label:<75} rows={nr:>7}  games={ng:>5}  duplicates={ndup:>4}")
    audit.append(f"  -> {raw['game_id'].nunique()} unique games in total")

    # ---- classify every game -------------------------------------------------
    g0 = raw.groupby("game_id").agg(condition=("condition", "first"), map_name=("map_name", "first"),
                                    start=("game_start_time", "first"))
    parsed = g0["condition"].map(parse_condition)
    g0["pair"] = parsed.map(lambda x: x[0])
    g0["suffix"] = parsed.map(lambda x: x[1])
    g0["map"] = g0["map_name"].map(MAP_NAMES)
    g0["pair_key"] = g0["pair"].map(lambda p: _pair_key(*p) if p else None)
    ok = (g0["suffix"] == HUMAN_SUFFIX) & g0["map"].notna() & g0["pair_key"].notna()

    audit.append("\nAll game versions found (games per condition x map); [USED] = analysed version:")
    tab = g0.assign(used=ok).groupby(["condition", "map_name", "used"]).size()
    for (cond, mp, used), n in tab.items():
        audit.append(f"  {'[USED]' if used else '      '} {n:>4}  {mp:<36} {cond}")

    sel = g0[ok].copy()
    sel["start"] = sel["start"].fillna("9999")   # games without a start time go last
    sel = sel.sort_values("start")
    sel["rank"] = sel.groupby(["pair_key", "map"]).cumcount()
    kept = sel[sel["rank"] < n_teams]

    # ---- per-team metrics ------------------------------------------------------
    raw = raw[raw["game_id"].isin(kept.index)]
    players = raw.groupby("game_id")["player_id"].unique()
    pid_of = raw.groupby("player_id")["prolific_id"].first()
    test = raw[raw["phase"] == "test"]
    if cut_def == "start_cutting":
        test = test.assign(_cut=test["action"].eq("start cutting"))
    else:
        test = test.assign(_cut=test["action_long"].astype(str).str.contains("cut", case=False))
    test = test.assign(_del=test["action"].eq("deliver"))
    cnt = test.groupby(["game_id", "player_id"])[["_del", "_cut"]].sum()

    rows = []
    for gid, r in kept.iterrows():
        pl = sorted(players[gid])
        D = [int(cnt["_del"].get((gid, p), 0)) for p in pl] + [0, 0]
        C = [int(cnt["_cut"].get((gid, p), 0)) for p in pl] + [0, 0]
        rows.append({
            "game_id": gid, "walk": r["pair_key"][0], "cut": r["pair_key"][1], "map": r["map"],
            "game_start_time": r["start"], "n_players": len(pl),
            "prolific_ids": ";".join(str(pid_of.get(p)) for p in pl),
            "total_deliveries": sum(D[:2]),
            "D1": D[0], "D2": D[1], "C1": C[0], "C2": C[1],
            "specialization_index": specialization_from_counts(D[0], D[1], C[0], C[1], spec_undefined),
            "spec_undefined": (D[0] + D[1] == 0) or (C[0] + C[1] == 0),
        })
    teams = pd.DataFrame(rows)

    audit.append(f"\nSelection: first {n_teams} teams per ability pair x map (by game_start_time):")
    avail = sel.groupby(["pair_key", "map"]).size()
    for p in PAIRS:
        for mp in MAPS:
            a = int(avail.get((p, mp), 0))
            k = int(((teams["walk"] == p[0]) & (teams["cut"] == p[1]) & (teams["map"] == mp)).sum())
            flag = "" if k == n_teams else f"   <-- only {k} teams available"
            sub = kept[(kept["pair_key"] == p) & (kept["map"] == mp)]
            span = f"{str(sub['start'].min())[:10]} .. {str(sub['start'].max())[:10]}" if len(sub) else ""
            audit.append(f"  {pair_label(p):<12} {mp:<4}  available={a:>4}  used={k:>3}  dates {span}{flag}")
    return teams


def load_demographics(paths, teams):
    """Per-participant age / sex for the selected teams (first record per participant)."""
    frames = []
    for _, fh in _iter_demographic_sources(paths):
        frames.append(pd.read_csv(fh, usecols=lambda c: c in _DEMO_COLS, low_memory=False))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "prolific_id"])
    d = d[d["game_id"].isin(teams["game_id"])]
    d = d.merge(teams[["game_id", "walk", "cut", "map"]], on="game_id")
    d = d[~d["prolific_id"].astype(str).str.lower().isin(["undefined", "nan", "none", ""])]
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    return d


# ===========================================================================
# 2. RL DATA -- the Fig. 4 training data, through fig4_human_vs_rl_gridmap.py
#    (identical to fig_sm_marl_ablation.py)
# ===========================================================================
_FIG4_DEFAULT = Path(__file__).resolve().with_name("fig4_human_vs_rl_gridmap.py")
_GRID_VALUES = dict(walk1_start=0.1, walk1_end=1.0, walk1_step=0.1,
                    cut2_start=0.1, cut2_end=1.0, cut2_step=0.1)


def load_fig4_module(path):
    """Import fig4_human_vs_rl_gridmap.py (this also puts the project root on sys.path,
    exactly as when Fig. 4 is run)."""
    path = Path(path) if path else _FIG4_DEFAULT
    if not path.exists():
        raise FileNotFoundError(f"Fig. 4 script not found: {path} (pass --fig4_script)")
    spec = importlib.util.spec_from_file_location("fig4_human_vs_rl_gridmap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_cut_cols(df: pd.DataFrame, override=None):
    """Per-agent cut counters (e.g. cut_ai_rl_1); 'cutting_speed_*' columns are ignored."""
    if override:
        return tuple(override)
    for a1, a2 in (("cut_ai_rl_1", "cut_ai_rl_2"), ("cuts_ai_rl_1", "cuts_ai_rl_2"),
                   ("cut_count_ai_rl_1", "cut_count_ai_rl_2")):
        if a1 in df.columns and a2 in df.columns:
            return a1, a2
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("cut") and "speed" not in lc and c.endswith("ai_rl_1") and c[:-1] + "2" in df.columns:
            return c, c[:-1] + "2"
    return None


def add_specialization(df: pd.DataFrame, cut_cols) -> pd.DataFrame:
    """Per-episode specialization index (Fig. 2 formula), as in fig_sm_marl_ablation.py."""
    df = df.copy()
    if cut_cols is None:
        df["specialization_index"] = np.nan
        return df
    d1 = pd.to_numeric(df.get("deliver_ai_rl_1", 0), errors="coerce").fillna(0)
    d2 = pd.to_numeric(df.get("deliver_ai_rl_2", 0), errors="coerce").fillna(0)
    c1 = pd.to_numeric(df[cut_cols[0]], errors="coerce").fillna(0)
    c2 = pd.to_numeric(df[cut_cols[1]], errors="coerce").fillna(0)
    tc, td = c1 + c2, d1 + d2
    ok = (tc > 0) & (td > 0)
    spec = ((c1 - c2) / tc.where(tc > 0) + (d2 - d1) / td.where(td > 0)).abs() * 100 / 2
    df["specialization_index"] = spec.where(ok, 0.0)
    return df


def collect_rl_fig4(args, log: list) -> pd.DataFrame:
    """One row per training EPISODE of the Fig. 4 window, for every pair of PAIRS."""
    fig4 = load_fig4_module(args.fig4_script)
    rows = []
    for mp, map_name in (("Open", args.map_baseline), ("PB", args.map_encouraged)):
        an = fig4.AbilityGridAnalyzer(
            map_name=map_name, **_GRID_VALUES, init_type=args.init_type, synergy=args.synergy,
            synergy_provided=True, specialization=args.specialization, game_type=args.game_type,
            cluster=args.cluster, study_name=args.study_name, fixed_cut1=1.0, fixed_walk2=1.0, demo=False)
        msg = (f"[RL] {map_name} ({mp}): raw dir {an.raw_dir}; window {args.episode_range}, "
               f"{args.num_episodes} ep" + (f", target {args.target_episode}" if args.episode_range == "specific" else ""))
        print(msg)
        log.append(msg)
        df = an.load_experimental_data()
        if df is None or len(df) == 0:
            log.append("    [WARN] DataProcessor returned no data")
            continue
        df = an.prepare_episode_data(df, episode_selection=args.episode_range,
                                     num_episodes=args.num_episodes, target_episode=args.target_episode)
        an._create_combined_metrics(df)                       # total_deliveries, as in Fig. 4
        cut_cols = find_cut_cols(df, args.cut_cols)
        if cut_cols is None:
            log.append(f"    [WARN] no per-agent cut columns -> specialization index unavailable "
                       f"(pass --cut_cols). Columns: {sorted(df.columns)}")
        df = add_specialization(df, cut_cols)
        tcol = an._resolve_training_id_column(df)
        for pair in PAIRS:
            sub = an._filter_speed_configuration(df, pair[0], pair[1])
            for _, r in sub.iterrows():
                rows.append({"walk": pair[0], "cut": pair[1], "map": mp,
                             "training_id": str(r[tcol]) if tcol else "unknown",
                             "episode": r.get("episode", np.nan),
                             "total_deliveries": r["total_deliveries"],
                             "specialization_index": r["specialization_index"]})
    return pd.DataFrame(rows, columns=["walk", "cut", "map", "training_id", "episode",
                                       "total_deliveries", "specialization_index"])


CLUSTER_DIRS = {
    "cuenca": "",  # same base folders as fig2_human_marl_comparison.py
    "brigit": "/mnt/lustre/home/samuloza/",
    "local": "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado",
}


def _parse_speed_dict(raw: str) -> dict:
    result = {}
    for m in re.finditer(r"['\"]?(ai_rl_\d+)['\"]?\s*:\s*([\d.]+)", raw.strip()):
        result[m.group(1)] = float(m.group(2))
    return result


def classify_config_pair(config_path: Path):
    """Return ((walk, cut) pair | None, checkpoint) from a simulation config.txt.
    A valid team has one agent slowed only in walking and the other only in cutting
    (either may be ai_rl_1); the HA team has every speed at 1.0."""
    if not config_path.exists():
        return None, None
    text = config_path.read_text(errors="replace")
    m_walk = re.search(r"WALKING_SPEEDS\s*:\s*(\{[^}]+\})", text)
    m_cut = re.search(r"CUTTING_SPEEDS\s*:\s*(\{[^}]+\})", text)
    m_ck = re.search(r"CHECKPOINT_NUMBER\s*:\s*(\S+)", text)
    ck = m_ck.group(1) if m_ck else None
    if not m_walk or not m_cut:
        return None, ck
    walking, cutting = _parse_speed_dict(m_walk.group(1)), _parse_speed_dict(m_cut.group(1))
    agents = sorted(walking)
    if len(agents) != 2 or set(agents) != set(cutting):
        return None, ck
    if any(walking[a] < 1.0 and cutting[a] < 1.0 for a in agents):
        return None, ck      # one agent slowed in both abilities: not one of our pairs
    return _pair_key(min(walking.values()), min(cutting.values())), ck


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
    """|(cuts_a1 - cuts_a2)/total_cuts + (del_a2 - del_a1)/total_del| * 100 / 2  (as in Fig. 2)"""
    per_agent_cuts = metrics.get("cuts_per_agent", {})
    per_agent_del = metrics.get("deliveries_per_agent", {})
    if len(per_agent_cuts) < 2 or len(per_agent_del) < 2:
        return None
    agents = sorted(set(per_agent_cuts) & set(per_agent_del))[:2]
    if len(agents) < 2:
        return None
    vals_cuts = [per_agent_cuts[a] for a in agents]
    vals_del = [per_agent_del[a] for a in agents]
    total_cuts, total_del = sum(vals_cuts), sum(vals_del)
    if total_cuts == 0 or total_del == 0:
        return 0.0
    cuts_term = (vals_cuts[0] - vals_cuts[1]) / total_cuts
    del_term = (vals_del[1] - vals_del[0]) / total_del
    return abs(del_term + cuts_term) * 100 / 2


def _latest_numbered_checkpoint(ck_dirs: list[Path]) -> list[Path]:
    numbered = [p for p in ck_dirs if p.name.replace("checkpoint_", "").isdigit()]
    return [max(numbered, key=lambda p: int(p.name.replace("checkpoint_", "")))] if numbered else []


def resolve_rl_root(base_cluster_dir: str, map_nr: str, game_version: str, synergy, specialization,
                    study_name: str = "", path_template: str | None = None) -> Path:
    if path_template:
        root = Path(path_template.replace("{map}", map_nr))
        return root / study_name if study_name else root
    nest = Path()
    if str(synergy).lower() not in ("", "none", "0"):
        nest = nest / f"synergy_{synergy}"
    if str(specialization).lower() not in ("", "none", "0"):
        nest = nest / f"specialized_{specialization}"
    candidates = [
        Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_version}/map_{map_nr}") / nest,
        Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/map_{map_nr}") / nest,
    ]
    found = next((c for c in candidates if c.exists()), candidates[0])
    sims_root = found / "simulations"
    return sims_root / study_name if study_name else sims_root


def collect_rl_sims(args, log: list) -> pd.DataFrame:
    """SIMULATED GAMES (same loader and folder tree as Fig. 2e-f and the SI ablation):
    one row per simulated game of the episode-750 policies, for every pair of PAIRS found."""
    base = CLUSTER_DIRS[args.cluster]
    rows = []
    for mp, map_nr in (("Open", args.map_baseline), ("PB", args.map_encouraged)):
        root = resolve_rl_root(base, map_nr, args.game_type, args.synergy,
                               args.specialization, args.study_name or "", args.rl_path_template)
        msg = f"[RL sims] {map_nr} ({mp}): root {root}; checkpoint folder {args.rl_checkpoint}"
        print(msg)
        log.append(msg)
        if not root.exists():
            log.append(f"    [WARN] simulations root not found: {root}")
            continue
        for training_dir in sorted(root.glob("Training_*")):
            ck_dirs = sorted(training_dir.glob("checkpoint_*"))
            if str(args.rl_checkpoint).lower() == "latest":
                ck_dirs = _latest_numbered_checkpoint(ck_dirs)
            elif str(args.rl_checkpoint).lower() != "all":
                ck_dirs = [c for c in ck_dirs if c.name.replace("checkpoint_", "") == str(args.rl_checkpoint)]
            for ck_dir in ck_dirs:
                for sim_dir in sorted(ck_dir.glob("simulation_*")):
                    cfg = sim_dir / "config.txt"
                    pair, _ = classify_config_pair(cfg)
                    if pair is None:
                        continue
                    met = extract_metrics(sim_dir, initialization_period=read_init_period(cfg))
                    if met is None:
                        if args.verbose:
                            print(f"  [SKIP] no usable data in {sim_dir}")
                        continue
                    rows.append({"walk": pair[0], "cut": pair[1], "map": mp, "episode": np.nan,
                                 "training_id": training_dir.name,
                                 "checkpoint": ck_dir.name.replace("checkpoint_", ""),
                                 "simulation": sim_dir.name,
                                 "total_deliveries": met["total_deliveries"],
                                 "specialization_index": specialization_index(met)})
        sub = [r for r in rows if r["map"] == mp]
        for p in PAIRS:
            k = [r for r in sub if (r["walk"], r["cut"]) == p]
            log.append(f"    {pair_label(p):<12} {len(k):>4} simulations from "
                       f"{len({r['training_id'] for r in k})} trainings")
    return pd.DataFrame(rows, columns=["walk", "cut", "map", "training_id", "checkpoint", "simulation", "episode",
                                       "total_deliveries", "specialization_index"])


def normalize_rl_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the cache written by this script (or any table with walk, cut, map,
    training_id, total_deliveries, specialization_index)."""
    df = df.copy()
    if "map" not in df.columns and "switch_cost_condition" in df.columns:
        df["map"] = df["switch_cost_condition"]
    df["map"] = df["map"].replace({"open": "Open", "pb": "PB", "low": "Open", "high": "PB"})
    if df["walk"].max() > 1.5:
        df["walk"], df["cut"] = df["walk"] / 100, df["cut"] / 100
    keys = [_pair_key(w, c) for w, c in zip(df["walk"], df["cut"])]
    df = df[[k is not None for k in keys]].copy()
    df["walk"] = [k[0] for k in keys if k is not None]
    df["cut"] = [k[1] for k in keys if k is not None]
    if "training_id" not in df.columns:
        df["training_id"] = "unknown"
    return df


def training_means(rl: pd.DataFrame) -> pd.DataFrame:
    """One row per training (mean over its episode window)."""
    return (rl.groupby(["walk", "cut", "map", "training_id"], as_index=False)
              [["total_deliveries", "specialization_index"]].mean())


# ===========================================================================
# 3. ANALYSIS (conventions of paper_stats.py)
# ===========================================================================
METRICS = [("total_deliveries", "Game score", False), ("specialization_index", "Specialization index", True)]


def _cell(df, pair, mp, col):
    return df.loc[(df["walk"] == pair[0]) & (df["cut"] == pair[1]) & (df["map"] == mp), col].to_numpy(float)


def _two(df, pair):
    """HA + one MA pair, with an 'ability' column (HA | MA)."""
    two = df[((df["walk"] == HA[0]) & (df["cut"] == HA[1])) |
             ((df["walk"] == pair[0]) & (df["cut"] == pair[1]))].copy()
    two["ability"] = np.where((two["walk"] == HA[0]) & (two["cut"] == HA[1]), "HA", "MA")
    return two


def analyse(df: pd.DataFrame, who: str, unit: str, out: list, csv_rows: list):
    df = df.copy()
    df["pair"] = [pair_label((w, c)) for w, c in zip(df["walk"], df["cut"])]
    present = [p for p in PAIRS if ((df["walk"] == p[0]) & (df["cut"] == p[1])).any()]

    out.append(f"\n--- {who}: descriptives (unit = {unit}; mean +- SEM) ---")
    for col, name, pct in METRICS:
        out.append(f"  {name}:")
        for p in present:
            for mp in MAPS:
                d = ps.describe(_cell(df, p, mp, col))
                if d["n"]:
                    out.append(f"    {pair_label(p):<12} {mp:<4} n={d['n']:>3}  {ps.fmt_ms(d, pct):<26} "
                               f"SD={d['sd']:6.2f}  median={d['median']:6.2f}")
                    csv_rows.append(dict(source=who, unit=unit, test="descriptive", measure=name,
                                         pair=pair_label(p), map=mp, **d))

    # ---- main experiment ---------------------------------------------------------
    if HA in present and MAIN_MA in present:
        main = _two(df, MAIN_MA)
        out.append(f"\n--- {who}: main experiment, HA {pair_label(HA)} vs MA {pair_label(MAIN_MA)} "
                   f"(two-way ANOVA: ability x map, between-{unit} factors) ---")
        for col, name, pct in METRICS:
            a = ps.anova_type3(main, col, "map", "ability")
            out.append(f"  {name}:")
            for _, r in a.iterrows():
                out.append(f"    {r['term']:<16} ({ps.fmt_F(r)})")
                csv_rows.append(dict(source=who, unit=unit, test="ANOVA 2x2", measure=name, term=r["term"],
                                     F=r["F"], df1=r["df"], df2=r["df_error"], p=r["p"], eta_p2=r["eta_p2"]))
            for mp in MAPS:
                txt, res = ps.fmt_compare("HA", _cell(main, HA, mp, col), "MA", _cell(main, MAIN_MA, mp, col), pct)
                out.append(f"    {mp:<4} {txt}")
                csv_rows.append(dict(source=who, unit=unit, test="Welch HA vs MA", measure=name,
                                     pair=pair_label(MAIN_MA), map=mp, t=res["t"], t_signed_HA_minus_MA=res["t_signed"],
                                     df2=res["df"], p=res["p"]))

    if len(present) < 3:
        return
    sub = df[df["pair"].isin([pair_label(p) for p in present])]
    out.append(f"\n--- {who}: all ability pairs ({len(present)} x 2) ---")
    for col, name, pct in METRICS:
        out.append(f"  {name}: omnibus two-way ANOVA, ability pair x map")
        for _, r in ps.anova_type3(sub, col, "map", "pair").iterrows():
            term = r["term"].replace("pair", "ability pair")
            out.append(f"    {term:<22} ({ps.fmt_F(r)})")
            csv_rows.append(dict(source=who, unit=unit, test="ANOVA omnibus", measure=name, term=term, F=r["F"],
                                 df1=r["df"], df2=r["df_error"], p=r["p"], eta_p2=r["eta_p2"]))

        con = []
        for p in present:
            if p == HA:
                continue
            for mp in MAPS:
                ha, ma = _cell(df, HA, mp, col), _cell(df, p, mp, col)
                txt, res = ps.fmt_compare(pair_label(p, tex=True).join("$$"), ma, "HA", ha, pct)
                con.append((p, mp, txt, res, ps.mann_whitney(ma, ha), ps.describe(ma)["mean"] - ps.describe(ha)["mean"]))
        ph, pu = ps.holm([c[3]["p"] for c in con]), ps.holm([c[4] for c in con])
        out.append(f"  {name}: each MA pair vs HA {pair_label(HA)} within map "
                   f"(Welch; Holm over {len(con)}; Mann-Whitney U, Holm, as rank-based check)")
        for (p, mp, txt, res, pmw, diff), padj, pmwa in zip(con, ph, pu):
            out.append(f"    {mp:<4} {txt}; {ps.fmt_p(padj, 'Holm')}; MWU {ps.fmt_p(pmwa, 'Holm')}")
            csv_rows.append(dict(source=who, unit=unit, test="Welch MA pair vs HA", measure=name, pair=pair_label(p),
                                 map=mp, diff_MA_minus_HA=diff, t=res["t"], t_signed_MA_minus_HA=res["t_signed"],
                                 df2=res["df"], p=res["p"], p_holm=padj, p_mwu=pmw, p_mwu_holm=pmwa))

        inter = [(p, ps.anova_type3(_two(df, p), col, "map", "ability").iloc[2]) for p in present if p != HA]
        ih = ps.holm([r["p"] for _, r in inter])
        out.append(f"  {name}: Map x Ability interaction, each MA pair vs HA (2 x 2 ANOVA; Holm over {len(inter)})")
        for (p, r), padj in zip(inter, ih):
            out.append(f"    {pair_label(p):<12} ({ps.fmt_F(r)}; {ps.fmt_p(padj, 'Holm')})")
            csv_rows.append(dict(source=who, unit=unit, test="Interaction per pair", measure=name, pair=pair_label(p),
                                 F=r["F"], df1=r["df"], df2=r["df_error"], p=r["p"], p_holm=padj, eta_p2=r["eta_p2"]))


def agreement(h: pd.DataFrame, r: pd.DataFrame, out: list, csv_rows: list, by_pair=None):
    out.append("\n--- Humans vs RL: MA - HA difference per map x ability cell (humans: teams; RL: trainings) ---")
    for col, name, pct in METRICS:
        xs, ys = [], []
        out.append(f"  {name}:")
        for p in MA_PAIRS:
            for mp in MAPS:
                rp = by_pair[p] if by_pair else r          # HA of the same MARL data type as the pair
                hh, hm, rh, rm = _cell(h, HA, mp, col), _cell(h, p, mp, col), _cell(rp, HA, mp, col), _cell(rp, p, mp, col)
                if min(map(len, (hh, hm, rh, rm))) == 0:
                    continue
                dh, dr = hm.mean() - hh.mean(), rm.mean() - rh.mean()
                xs.append(dh)
                ys.append(dr)
                same = np.sign(dh) == np.sign(dr)
                out.append(f"    {pair_label(p):<12} {mp:<4} humans {dh:+7.2f}   RL {dr:+7.2f}   "
                           f"{'same direction' if same else 'OPPOSITE DIRECTION'}")
                csv_rows.append(dict(source="Humans vs RL", test="MA - HA difference", measure=name,
                                     pair=pair_label(p), map=mp, diff_human=dh, diff_rl=dr, same_sign=bool(same)))
        if len(xs) >= 3:
            rr, pp = stats.pearsonr(xs, ys)
            agree = int(np.sum(np.sign(xs) == np.sign(ys)))
            out.append(f"    same direction in {agree}/{len(xs)} cells; Pearson $r = {ps._nolead(rr, 2)}$, {ps.fmt_p(pp)} "
                       f"(n = {len(xs)} cells)")


# ===========================================================================
# 4. FIGURE
# ===========================================================================
PAGE_W, PAGE_H = 804.96, 574.0
S_PANEL = 0.54298                      # same scale factor as the MARL panels of Fig. 2
_FS = 30
_COLS = [(86.0, 372.0), (506.0, 792.0)]
_ROWS = [(46.0, 222.0), (284.0, 460.0)]
LETTER_DX, LETTER_DY = -74.0, 4.0      # panel letter relative to (left, top) of the axes
TITLE_Y = {0: 24.0, 1: 262.0}          # baselines of the centred row titles (as in Fig. 3)
TITLE_SIZE = 20                         # same size as the panel letters, as in Fig. 3
LEGEND_BOX = dict(top=511.0, bottom=552.0, face='#f2f2f2', edge='#595959', lw=2.25)
LEGEND_CENTER = (PAGE_W / 2, 531.5)
LEGEND_SCALE = 0.50
LEGEND_PAD = 14.0
LEGEND_ROW_DY = 24.0               # vertical distance of the two legend rows (mixed data types)
_DODGE = dict(zip(PAIRS, np.linspace(-0.24, 0.24, len(PAIRS))))
_MAP_X = {"Open": 1.0, "PB": 2.0}


def tex(font: str, size: float, text: str) -> str:
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def _page_transform(fig):
    return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans


def _rect_to_fig(rect):
    L, R, T, B = rect
    return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]


_DASH = (0, (3.2, 2.2))
_HA_REF_DX = -0.36            # x offset of the hollow (100%,100%) training-episode reference


def draw_means_panel(ax, df, col, *, ylabel, ylim, percent, s, empty_msg=None,
                     dashed_pairs=(), ref_df=None):
    """dashed_pairs: pairs whose MARL values are training episodes (dashed lines);
    ref_df: (100%,100%) training episodes, drawn as a hollow diamond with a dashed line
    (the matched reference of the dashed pairs)."""
    ax.set_ylim(*ylim)
    if df is None or df.empty:
        ax.text(0.5, 0.5, empty_msg or "no data", transform=ax.transAxes, ha="center", va="center",
                fontsize=_FS * s, color="gray")
    else:
        if ref_df is not None and not ref_df.empty:
            pts = []
            for mp in MAPS:
                d = ps.describe(_cell(ref_df, HA, mp, col))
                x = _MAP_X[mp] + _HA_REF_DX
                pts.append((x, d["mean"]))
                if d["n"]:
                    ax.plot(x, d["mean"], marker="D", markersize=11 * s, markerfacecolor="white",
                            markeredgecolor=_PAIR_COLOR[HA], markeredgewidth=2.0 * s, linestyle="none", zorder=5)
                    if np.isfinite(d["sem"]):
                        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor="black",
                                    elinewidth=1.6 * s, capsize=7 * s, capthick=1.6 * s, zorder=6)
            xs, ys = zip(*pts)
            if all(np.isfinite(ys)):
                ax.plot(xs, ys, color=_PAIR_COLOR[HA], linewidth=2.0 * s, zorder=4, linestyle=_DASH, alpha=0.85)
        for p in PAIRS:
            pts = []
            for mp in MAPS:
                d = ps.describe(_cell(df, p, mp, col))
                x = _MAP_X[mp] + _DODGE[p]
                pts.append((x, d["mean"]))
                if d["n"] == 0:
                    continue
                ax.plot(x, d["mean"], marker="D", markersize=11 * s, markerfacecolor=_PAIR_COLOR[p],
                        markeredgecolor="black", markeredgewidth=1.5 * s, linestyle="none", zorder=5)
                if np.isfinite(d["sem"]):      # +- SEM, drawn over the diamond with caps
                    ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor="black",
                                elinewidth=1.6 * s, capsize=7 * s, capthick=1.6 * s, zorder=6)
            xs, ys = zip(*pts)
            if all(np.isfinite(ys)):
                ax.plot(xs, ys, color=_PAIR_COLOR[p], linewidth=2.5 * s, zorder=4,
                        solid_capstyle="round", alpha=0.85,
                        linestyle=_DASH if p in dashed_pairs else "-")
    ax.set_xticks([_MAP_X[m] for m in MAPS])
    ax.set_xticklabels(MAPS)
    ax.set_xlim(0.45, 2.55)
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


def _common_ylim(dfs, col, pad_frac=0.08, floor=0.0):
    vals = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for p in PAIRS:
            for mp in MAPS:
                d = ps.describe(_cell(df, p, mp, col))
                if d["n"]:
                    vals += [d["mean"] - (d["sem"] if np.isfinite(d["sem"]) else 0),
                             d["mean"] + (d["sem"] if np.isfinite(d["sem"]) else 0)]
    if not vals:
        return (0, 1)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * pad_frac
    return (max(floor, np.floor(lo - pad)), np.ceil(hi + pad))


def make_figure(human: pd.DataFrame, rl: pd.DataFrame | None, output_path: Path,
                titles=("Humans", "RL"), rl_training_pairs=(), rl_ha_training=None):
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
    page = _page_transform(fig)
    score_lim = _common_ylim([human, rl, rl_ha_training], "total_deliveries")
    spec_lim = (0, 100)
    letters = iter("abcd")
    for row, (df, title) in enumerate(((human, titles[0]), (rl, titles[1]))):
        T, B = _ROWS[row]
        for k, (col, ylabel, ylim, pct) in enumerate((
                ("total_deliveries", "Game score", score_lim, False),
                ("specialization_index", "Specialization index", spec_lim, True))):
            L, R = _COLS[k]
            ax = fig.add_axes(_rect_to_fig((L, R, T, B)))
            draw_means_panel(ax, df, col, ylabel=ylabel, ylim=ylim, percent=pct, s=S_PANEL,
                             empty_msg="RL data not loaded",
                             dashed_pairs=rl_training_pairs if row == 1 else (),
                             ref_df=rl_ha_training if row == 1 else None)
            fig.text(L + LETTER_DX, T + LETTER_DY, tex('cmbx10', 20, next(letters)), transform=page,
                     ha='left', va='baseline', fontsize=20)
        fig.text(PAGE_W / 2, TITLE_Y[row], tex('cmbx10', TITLE_SIZE, title), transform=page,
                 ha='center', va='baseline', fontsize=TITLE_SIZE)

    # -- legend band (same look as Fig. 2); a second row explains the data types when mixed --
    lb, sl = LEGEND_BOX, LEGEND_SCALE
    handles = [matplotlib.lines.Line2D([], [], marker="D", markersize=13 * sl, markerfacecolor=_PAIR_COLOR[p],
                                       markeredgecolor="black", markeredgewidth=1.5 * sl,
                                       color=_PAIR_COLOR[p], linewidth=2.5 * sl) for p in PAIRS]
    labels = [rf"\textbf{{{pair_label(p, tex=True)}}}" for p in PAIRS]
    mixed = rl_ha_training is not None and not rl_ha_training.empty
    cx, cy = LEGEND_CENTER
    rows = [(handles, labels)]
    if mixed:
        h2 = [matplotlib.lines.Line2D([], [], color="black", linewidth=2.5 * sl),
              matplotlib.lines.Line2D([], [], color="black", linewidth=2.5 * sl, linestyle=_DASH),
              matplotlib.lines.Line2D([], [], marker="D", markersize=13 * sl, markerfacecolor="white",
                                      markeredgecolor=_PAIR_COLOR[HA], markeredgewidth=2.0 * sl,
                                      color=_PAIR_COLOR[HA], linewidth=2.0 * sl, linestyle=_DASH)]
        l2 = [r"\textbf{Simulated games}", r"\textbf{Training episodes}",
              rf"\textbf{{{pair_label(HA, tex=True)}, training}}"]
        rows.append((h2, l2))
    dy = LEGEND_ROW_DY if mixed else 0.0
    widths = []
    for i, (h, l) in enumerate(rows):
        yy = cy - dy / 2 + i * dy
        leg = fig.legend(h, l, fontsize=_FS * sl, frameon=False, loc="center", ncol=len(h),
                         handlelength=1.6 if i == 0 else 2.4, columnspacing=1.4, handletextpad=0.5,
                         bbox_to_anchor=(cx / PAGE_W, 1.0 - yy / PAGE_H), bbox_transform=fig.transFigure)
        widths.append(leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 72.0)
    width_pt = max(widths) + 2 * LEGEND_PAD
    top, bottom = lb['top'] - dy / 2, lb['bottom'] + dy / 2
    fig.add_artist(matplotlib.patches.Rectangle(
        (cx - width_pt / 2, top), width_pt, bottom - top, transform=page,
        facecolor=lb['face'], edgecolor=lb['edge'], linewidth=lb['lw'], zorder=0))

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    plt.close(fig)
    print(f"\nFigure saved -> {output_path} (+ .png)")


# ===========================================================================
# 5. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="SI figure + statistics: five ability pairs, humans and RL.")
    ap.add_argument("--human_data", nargs="*", default=[],
                    help="Human zips / CSVs (duplicate games are counted once)")
    ap.add_argument("--human_csv", default=None, help="Per-team CSV written by a previous run")
    ap.add_argument("--n_teams", type=int, default=70, help="Teams per ability pair x map (first by start time)")
    ap.add_argument("--cut_def", default="start_cutting", choices=["start_cutting", "contains_cut"],
                    help="Human cut count C: 'start cutting' actions (default) or any action containing 'cut'")
    ap.add_argument("--spec_undefined", default="zero", choices=["zero", "nan"],
                    help="Spec. index of teams with no deliveries or no cuts (MARL scripts use 0)")
    ap.add_argument("--exclude_repeat_participants", action="store_true",
                    help="Drop teams with a participant who already played in an earlier selected team")
    # --- RL: exactly the Fig. 4 arguments ---
    ap.add_argument("--rl_source", choices=["hybrid", "sims", "fig4", "none"], default="hybrid",
                    help="'hybrid' (default): simulated games of the episode-750 policies for (100%%,100%%) and "
                         "(40%%,20%%) (as Fig. 2e-f) and training episodes 740-760 (as Fig. 4) for the other "
                         "pairs, which have no simulated games; 'sims' / 'fig4': one data type for every pair")
    ap.add_argument("--rl_csv_training", default=None,
                    help="hybrid: cached training-episode records (with --rl_csv = cached simulation records)")
    ap.add_argument("--rl_checkpoint", default="753",
                    help="sims: checkpoint_<n> folder of the simulated games (episode 750, as everywhere)")
    ap.add_argument("--rl_path_template", default=None,
                    help="sims: explicit simulations root ('{map}' substituted)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rl_csv", default=None, help="Cached RL episode records (skips loading the trainings)")
    ap.add_argument("--fig4_script", default=None,
                    help="Path to fig4_human_vs_rl_gridmap.py (default: next to this script)")
    ap.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    ap.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    ap.add_argument("--game_type", default="classic_collision", choices=["classic", "classic_collision"])
    ap.add_argument("--init_type", default="empty_init", choices=["random_init", "empty_init"])
    ap.add_argument("--synergy", type=float, default=1.35)
    ap.add_argument("--specialization", type=float, default=0.25)
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--cluster", default="cuenca")
    ap.add_argument("--episode_range", choices=["all", "final", "average", "specific"], default="specific")
    ap.add_argument("--num_episodes", type=int, default=20)
    ap.add_argument("--target_episode", type=int, default=753)
    ap.add_argument("--cut_cols", nargs=2, default=None, metavar=("AGENT1", "AGENT2"),
                    help="Per-agent cut-count columns (auto-detected)")
    # --- output ---
    ap.add_argument("--titles", nargs=2, default=["Human Experiments", "MARL Simulations"])
    ap.add_argument("--output_dir", default="./figures")
    ap.add_argument("--output_name", default="fig_sm_ability_pairs_humans_rl.pdf")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / args.output_name
    stem = out_pdf.with_suffix("")
    report, csv_rows = [], []

    report.append("=" * 100)
    report.append("STATISTICS: game score and specialization index at the five ability pairs (humans and RL)")
    report.append("=" * 100)
    report.append(
        "Conventions (paper_stats.py, identical in every figure script):\n"
        "  mean +- SEM; game score 2 decimals, specialization index 1 decimal (%).\n"
        "  Welch's t-tests: the group with the HIGHER mean is written first ('A > B: (A vs. B; ...)') and t is\n"
        "    reported as |t|, as in the main text; Welch-Satterthwaite df. Signed t values are in the CSV.\n"
        "  Two-way between-subjects ANOVAs: effect coding, Type III SS, partial eta^2.\n"
        "  p = .059 / p < .001.\n"
        "Definitions\n"
        "  Game score = deliveries in the 180-s test game (humans) / in a 180-s simulated game ('sims') or\n"
        "    training episode ('fig4') of the MARL agents.\n"
        f"  RL source: {args.rl_source if not args.rl_csv else 'cached CSV ' + args.rl_csv}.\n"
        "  Specialization index = 100 * |(D1 - D2)/(D1 + D2) + (C2 - C1)/(C1 + C2)| / 2 (D deliveries,\n"
        f"    C cuts per player); no deliveries or no cuts -> {'0' if args.spec_undefined == 'zero' else 'excluded'}.\n"
        f"    Human cuts: {args.cut_def!r}. RL cuts: per-agent cut counters of the training data.\n"
        "  Units: humans = team; RL = training (mean over its episode window), as in the SI pretraining section.")

    # ---------------- humans ---------------------------------------------------
    human = None
    if args.human_csv:
        human = pd.read_csv(args.human_csv)
        report.append(f"\nHuman teams read from {args.human_csv}")
    elif args.human_data:
        audit = []
        human = load_human_teams(args.human_data, args.n_teams, args.cut_def, args.spec_undefined, audit)
        report.append("\n" + "=" * 100 + "\nDATA AUDIT (humans)\n" + "=" * 100)
        report += audit

        human = human.sort_values("game_start_time").reset_index(drop=True)
        seen, rep = set(), []
        unknown = {"undefined", "nan", "none", ""}
        n_unknown = 0
        for i, r in human.iterrows():
            all_ids = str(r["prolific_ids"]).split(";")
            ids = [x for x in all_ids if x.strip().lower() not in unknown]
            n_unknown += len(all_ids) - len(ids)
            if seen & set(ids):
                rep.append(i)
            seen |= set(ids)
        human["repeat_participant"] = human.index.isin(rep)
        report.append(f"\nPlayer seats without a valid Prolific ID ('undefined'): {n_unknown}")
        report.append(f"Teams with a participant who already played in an earlier selected team: {len(rep)}")
        for (w, c, mp), n in human[human.repeat_participant].groupby(["walk", "cut", "map"]).size().items():
            report.append(f"    {pair_label((w, c)):<12} {mp:<4} {n}")
        if args.exclude_repeat_participants:
            human = human[~human.repeat_participant]
            report.append(f"  -> EXCLUDED; {len(human)} teams remain")
        else:
            report.append("  (kept; --exclude_repeat_participants for the sensitivity check)")
        report.append("\nTeams with no deliveries or no cuts (specialization index "
                      f"{'0' if args.spec_undefined == 'zero' else 'NaN'}):")
        for (w, c, mp), n in human.groupby(["walk", "cut", "map"])["spec_undefined"].sum().items():
            if n:
                report.append(f"    {pair_label((w, c)):<12} {mp:<4} {int(n)}")

        report.append("\nCollection batches (cells whose games span more than 30 days):")
        for (w, c, mp), g in human.groupby(["walk", "cut", "map"]):
            dt = pd.to_datetime(g["game_start_time"], errors="coerce")
            if dt.notna().any() and (dt.max() - dt.min()).days > 30:
                split = dt.sort_values()[dt.sort_values().diff().idxmax()]
                b1, b2 = g[dt < split], g[dt >= split]
                report.append(f"  {pair_label((w, c))} {mp}: batch 1 n={len(b1)} "
                              f"({str(b1.game_start_time.min())[:10]}..{str(b1.game_start_time.max())[:10]}), "
                              f"batch 2 n={len(b2)} ({str(b2.game_start_time.min())[:10]}.."
                              f"{str(b2.game_start_time.max())[:10]})")
                for col, name, pct in METRICS:
                    txt, _ = ps.fmt_compare("batch 1", b1[col], "batch 2", b2[col], pct)
                    report.append(f"      {name}: {txt}")

        demo = load_demographics(args.human_data, human)
        if demo is not None:
            report.append("\nDemographics (unique valid Prolific IDs of the selected teams; age numeric only):")
            demo["round"] = np.where(((demo["walk"] == 1.0) & (demo["cut"] == 1.0)) |
                                     ((demo["walk"] == MAIN_MA[0]) & (demo["cut"] == MAIN_MA[1])),
                                     "main experiment", "second round")
            for rnd, g in demo.groupby("round"):
                a = g["Age"].dropna()
                report.append(f"  {rnd:<16} participants={g.prolific_id.nunique()}  age mean={a.mean():.2f} "
                              f"SD={a.std(ddof=1):.2f} (n with age={len(a)})  "
                              + ", ".join(f"{k}={v}" for k, v in g["Sex"].value_counts().items()))
        human.to_csv(f"{stem}_human_teams.csv", index=False)
        print(f"Per-team human data -> {stem}_human_teams.csv")

    if human is not None:
        report.append("\n" + "=" * 100 + "\nHUMANS (unit = team)\n" + "=" * 100)
        analyse(human, "Humans", "team", report, csv_rows)
        report.append("\nLaTeX table (all ability pairs, mean +- SEM across teams):")
        report.append("\\begin{tabular}{llccc}\n\\toprule\nMap & (walk, cut) & Game score & Spec.\\ index & Teams \\\\\n\\midrule")
        for mp in MAPS:
            for p in PAIRS:
                ds, dp = ps.describe(_cell(human, p, mp, "total_deliveries")), ps.describe(_cell(human, p, mp, "specialization_index"))
                if ds["n"]:
                    report.append(f"{mp:<4} & ${pair_label(p, True)}$ & {ps.fmt_ms(ds, False)} & {ps.fmt_ms(dp, True)} & {ds['n']} \\\\")
        report.append("\\bottomrule\n\\end{tabular}")

    # ---------------- RL -------------------------------------------------------
    # ------------------------------------------------------------------
    # RL data. 'hybrid': simulated games for SIM_PAIRS, training episodes for the rest.
    # Every test compares MARL groups of the SAME data type.
    # ------------------------------------------------------------------
    rl_sims_ep = rl_train_ep = None
    if args.rl_source in ("hybrid", "sims"):
        if args.rl_csv:
            rl_sims_ep = normalize_rl_csv(pd.read_csv(args.rl_csv))
            report.append(f"\nRL simulated-game records read from {args.rl_csv}")
        else:
            log = []
            rl_sims_ep = collect_rl_sims(args, log)
            report.append("\n" + "=" * 100 + "\nRL DATA: MARL simulations (simulated games of the episode-750 policies, as Fig. 2e-f)\n"
                          + "=" * 100)
            report += log
            rl_sims_ep.to_csv(f"{stem}_rl_records_sims.csv", index=False)
            report.append(f"Cached -> {stem}_rl_records_sims.csv (re-use with --rl_csv)")
    if args.rl_source in ("hybrid", "fig4"):
        cached = args.rl_csv_training if args.rl_source == "hybrid" else args.rl_csv
        if cached:
            rl_train_ep = normalize_rl_csv(pd.read_csv(cached))
            report.append(f"\nRL training-episode records read from {cached}")
        else:
            log = []
            rl_train_ep = collect_rl_fig4(args, log)
            report.append("\n" + "=" * 100 + "\nRL DATA: training episodes 740-760 (as Fig. 4)\n" + "=" * 100)
            report += log
            rl_train_ep.to_csv(f"{stem}_rl_records_training.csv", index=False)
            report.append(f"Cached -> {stem}_rl_records_training.csv (re-use with --rl_csv_training)")
    if args.rl_source == "hybrid" and rl_sims_ep is not None:
        rl_sims_ep = rl_sims_ep[[(w, c) in SIM_PAIRS for w, c in zip(rl_sims_ep["walk"], rl_sims_ep["cut"])]]

    def _coverage(ep_df, kind):
        report.append(f"\nRL coverage, {kind} (trainings; games or episodes per training; episode numbers):")
        for p in PAIRS:
            for mp in MAPS:
                g = ep_df[(ep_df["walk"] == p[0]) & (ep_df["cut"] == p[1]) & (ep_df["map"] == mp)]
                if g.empty:
                    report.append(f"    {pair_label(p):<12} {mp:<4} --")
                    continue
                per = g.groupby("training_id").size()
                ep = pd.to_numeric(g["episode"], errors="coerce") if "episode" in g.columns else pd.Series(dtype=float)
                rng = f"{int(ep.min())}..{int(ep.max())}" if ep.notna().any() else "n/a"
                report.append(f"    {pair_label(p):<12} {mp:<4} trainings={len(per):>3}  per training="
                              f"{per.min()}..{per.max()}  episode numbers {rng}")

    rl_sims = rl_train = None
    if rl_sims_ep is not None and not rl_sims_ep.empty:
        _coverage(rl_sims_ep, "simulated games")
        rl_sims = training_means(rl_sims_ep)
        report.append("\n" + "=" * 100 + "\nMARL SIMULATIONS (unit = training; mean of its simulated games)\n" + "=" * 100)
        analyse(rl_sims, "MARL simulations", "training", report, csv_rows)
    if rl_train_ep is not None and not rl_train_ep.empty:
        _coverage(rl_train_ep, "training episodes")
        rl_train = training_means(rl_train_ep)
        report.append("\n" + "=" * 100 + "\nRL, TRAINING EPISODES 740-760 (unit = training; all pairs; the contrasts of "
                      "(70%,40%), (20%,100%) and (10%,10%) with (100%,100%) use these data)\n" + "=" * 100)
        analyse(rl_train, "RL training episodes", "training", report, csv_rows)

    # data plotted in panels c-d, and the per-pair data used for the human-vs-RL comparison
    rl_plot, train_pairs, ha_ref = None, (), None
    if args.rl_source == "hybrid" and rl_sims is not None and rl_train is not None:
        is_sim_pair = [(w, c) in SIM_PAIRS for w, c in zip(rl_train["walk"], rl_train["cut"])]
        rl_plot = pd.concat([rl_sims, rl_train[[not x for x in is_sim_pair]]], ignore_index=True)
        train_pairs = tuple(p for p in PAIRS if p not in SIM_PAIRS)
        ha_ref = rl_train[(rl_train["walk"] == HA[0]) & (rl_train["cut"] == HA[1])]
        by_pair = {p: (rl_sims if p in SIM_PAIRS else rl_train) for p in MA_PAIRS}
    else:
        rl_plot = rl_sims if rl_sims is not None else rl_train
        by_pair = {p: rl_plot for p in MA_PAIRS} if rl_plot is not None else None

    if rl_plot is not None and human is not None:
        report.append("\n" + "=" * 100 + "\nHUMANS vs RL\n" + "=" * 100)
        if args.rl_source == "hybrid":
            report.append("  MA - HA differences of the agents are taken within one data type: simulated games for "
                          f"{pair_label(MAIN_MA)}, training episodes for the other pairs.")
        agreement(human, rl_plot, report, csv_rows, by_pair=by_pair)
    elif rl_plot is None:
        report.append("\n" + "=" * 100 + "\nRL: no data loaded\n" + "=" * 100)

    txt = f"{stem}_stats.txt"
    Path(txt).write_text("\n".join(report) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(f"{stem}_stats.csv", index=False)
    print("\n".join(report))
    print(f"\nStatistics saved -> {txt} (+ _stats.csv)")

    if human is not None or rl_plot is not None:
        make_figure(human, rl_plot, out_pdf, titles=tuple(args.titles))
    return 0


if __name__ == "__main__":
    sys.exit(main())