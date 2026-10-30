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

MARL data (--rl_source sims, default): SIMULATED GAMES read from the simulation tree
(marl_simulations.py, same loaders as Fig. 2e-f and the SI ablation):
    <base>/data/samuel_lozano/cooked/<game_type>/simulations/<init_type>/map_<map>/
        synergy_1.35/specialized_0.25/abilities_<walk>_<cut>/Training_*/checkpoint_753/simulation_*/
where abilities_<walk>_<cut> holds the walking speed of ai_rl_1 (walk) and the cutting
speed of ai_rl_2 (cut); the other two speeds are 1.0. Every Training_* and every
simulation_* folder of checkpoint_753 is used, for each of the five pairs. The
specialization index is computed per simulated game from the per-agent cut and delivery
counts, then averaged per training: ONE VALUE PER TRAINING is the unit of every RL test
and of the RL means +- SEM, because games of one training are not independent.

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
  * the first --n_teams teams (default 70) of each ability pair x map, in the order in which
    the games appear in the (pre-ordered) CSV files -- no re-ordering; files are read in the
    order given in --human_data; games present in several files are counted once,
  * game score and specialization index from the TEST phase only (phase == 'test'),
  * <o>_selected_games.csv: one column per condition with the game_id of the selected games
    in selection order (row 'team' = 1..70); the check that every game_id is unique is
    written to <o>_stats.txt.
  Required files in human_data/: combined_actions_with_demographics_csv.zip and
  oct2026_new_condition_encouraged.zip (+ any later batch).

Usage
    # humans + RL simulated games (checkpoint_753, every Training_* / simulation_*)
    nohup python3 fig_sm_ability_pairs_humans_rl.py --human_data ./data/human_data/*.zip --rl_source sims --synergy 1.35 --specialization 0.25 --init_type empty_init --rl_checkpoint 753 > log_fig_sm_pairs.out 2>&1 &

    # re-plot from the caches written by a previous run
    python3 fig_sm_ability_pairs_humans_rl.py \\
        --human_csv ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv \\
        --rl_csv ./figures/fig_sm_ability_pairs_humans_rl_rl_records_sims.csv

The per-team human CSV written here (<output>_human_teams.csv) is also the human
input of fig_sm_marl_ablation.py (pretraining) and fig4_human_vs_rl_gridmap.py
(--human_teams_csv), so the three figures use the same human teams.

Author: Samuel Lozano
"""

import argparse
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

# ===========================================================================
# 0. CONDITIONS
# ===========================================================================
# (walking speed of player 1, cutting speed of player 2), as fractions of full speed
PAIRS = [(1.0, 1.0), (0.7, 0.4), (0.4, 0.2), (0.2, 1.0), (0.1, 0.1)]
HA = (1.0, 1.0)
MAIN_MA = (0.4, 0.2)
MA_PAIRS = [p for p in PAIRS if p != HA]
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
    # sort=False keeps the games in the order of the CSV files (files in the order given in
    # --human_data, then row order); the files are pre-ordered, so no re-ordering is done here.
    g0 = raw.groupby("game_id", sort=False).agg(condition=("condition", "first"), map_name=("map_name", "first"),
                                                start=("game_start_time", "first"))
    g0["order"] = np.arange(len(g0))
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

    sel = g0[ok].copy()                         # already in CSV order: no sorting
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

    # only the TEST phase is used for the game score and the specialization index
    no_test = sorted(set(kept.index) - set(test["game_id"].unique()))

    rows = []
    for gid, r in kept.iterrows():
        pl = sorted(players[gid])
        D = [int(cnt["_del"].get((gid, p), 0)) for p in pl] + [0, 0]
        C = [int(cnt["_cut"].get((gid, p), 0)) for p in pl] + [0, 0]
        rows.append({
            "game_id": gid, "walk": r["pair_key"][0], "cut": r["pair_key"][1],
            "map": r["map"], "rank": int(r["rank"]) + 1, "order": int(r["order"]),
            "game_start_time": r["start"], "n_players": len(pl),
            "prolific_ids": ";".join(str(pid_of.get(p)) for p in pl),
            "total_deliveries": sum(D[:2]),
            "D1": D[0], "D2": D[1], "C1": C[0], "C2": C[1],
            "specialization_index": specialization_from_counts(D[0], D[1], C[0], C[1], spec_undefined),
            "spec_undefined": (D[0] + D[1] == 0) or (C[0] + C[1] == 0),
        })
    teams = pd.DataFrame(rows)

    audit.append(f"\nTest phase only: deliveries and cuts are counted in phase == 'test'; "
                 f"selected games without any test-phase row: {len(no_test)}"
                 + (f" -> {', '.join(map(str, no_test))}" if no_test else ""))
    audit.append(f"\nSelection: first {n_teams} teams per ability pair x map, in the order of the CSV files "
                 f"(no re-ordering):")
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


def write_selected_games(teams: pd.DataFrame, path: Path) -> list[str]:
    """CSV table of the selected games: one column per condition (map x ability pair), one row
    per selected game in selection order (column 'team' = 1..70); each cell = game_id.
    Returns the check that every selected game_id is unique."""
    table = {}
    for mp in MAPS:
        for p in PAIRS:
            g = teams[(teams["walk"] == p[0]) & (teams["cut"] == p[1]) & (teams["map"] == mp)].sort_values("rank")
            table[f"{mp} {pair_label(p)}"] = pd.Series(g["game_id"].to_list(), dtype=object)
    df = pd.DataFrame(table)
    df.index = pd.RangeIndex(1, len(df) + 1, name="team")
    df.to_csv(path)
    dup = teams["game_id"][teams["game_id"].duplicated()].unique()
    check = [f"  game_id: all {len(teams)} selected games unique  OK" if not len(dup) else
             f"  game_id: {len(dup)} game_id(s) selected more than once  <-- NOT UNIQUE: {', '.join(map(str, dup))}"]
    print(f"Selected games table -> {path}")
    print(check[0])
    return check


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
# 2. RL DATA -- simulated games of the simulation tree (marl_simulations.py)
#    .../simulations/<init_type>/map_<map>/synergy_<s>/specialized_<l>/
#        abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
#    walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2 (other speeds 1.0),
#    i.e. exactly the (walk, cut) pairs of PAIRS.
# ===========================================================================
CLUSTER_DIRS = ms.CLUSTER_DIRS


def resolve_rl_root(base_cluster_dir: str, map_nr: str, game_version: str, synergy, specialization,
                    study_name: str = "", path_template: str | None = None,
                    init_type: str = ms.DEFAULT_INIT_TYPE) -> Path:
    """Folder holding the abilities_* folders of one map."""
    return ms.map_root(base_cluster_dir, game_version, map_nr, synergy, specialization,
                       init_type=init_type, study_name=study_name or None, path_template=path_template)


def collect_rl_sims(args, log: list) -> pd.DataFrame:
    """SIMULATED GAMES (same loaders and folder tree as Fig. 2e-f and the SI ablation):
    one row per simulated game, for every pair of PAIRS found (every Training_* and
    simulation_* of checkpoint_<rl_checkpoint> inside abilities_<walk>_<cut>)."""
    base = ms.cluster_base_dir(args.cluster)
    frames = []
    for mp, map_nr in (("Open", args.map_baseline), ("PB", args.map_encouraged)):
        root = resolve_rl_root(base, map_nr, args.game_type, args.synergy, args.specialization,
                               args.study_name or "", args.rl_path_template, args.init_type)
        msg = f"[RL sims] {map_nr} ({mp}): root {root}; checkpoint folder checkpoint_{args.rl_checkpoint}"
        print(msg)
        log.append(msg)
        if not root.exists():
            log.append(f"    [WARN] simulations root not found: {root}")
            continue
        df = ms.collect_records(root, pairs=PAIRS, checkpoint=args.rl_checkpoint,
                                verbose=args.verbose, extra={"map": mp})
        for p in PAIRS:
            if ms.find_ability_dir(root, *p) is None:
                log.append(f"    {pair_label(p):<12} [WARN] no abilities_{p[0]:g}_{p[1]:g} folder")
                continue
            k = df[np.isclose(df["walk"], p[0]) & np.isclose(df["cut"], p[1])] if not df.empty else df
            log.append(f"    {pair_label(p):<12} {len(k):>4} simulations from "
                       f"{k['training_id'].nunique() if len(k) else 0} trainings")
        if not df.empty:
            keys = [_pair_key(w, c) for w, c in zip(df["walk"], df["cut"])]
            df["walk"], df["cut"] = [k[0] for k in keys], [k[1] for k in keys]
            df["episode"] = np.nan
            frames.append(df)
    cols = ["walk", "cut", "map", "training_id", "checkpoint", "simulation", "episode",
            "total_deliveries", "specialization_index"]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


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
    ap.add_argument("--n_teams", type=int, default=70, help="Teams per ability pair x map (first ones in the order of the CSV files)")
    ap.add_argument("--cut_def", default="start_cutting", choices=["start_cutting", "contains_cut"],
                    help="Human cut count C: 'start cutting' actions (default) or any action containing 'cut'")
    ap.add_argument("--spec_undefined", default="zero", choices=["zero", "nan"],
                    help="Spec. index of teams with no deliveries or no cuts (MARL scripts use 0)")
    ap.add_argument("--exclude_repeat_participants", action="store_true",
                    help="Drop teams with a participant who already played in an earlier selected team")
    # --- RL: simulated games of the simulation tree ---
    ap.add_argument("--rl_source", choices=["sims", "none"], default="sims",
                    help="'sims' (default): simulated games of every ability pair, read from the "
                         "abilities_<walk>_<cut> folders of the simulation tree; 'none': humans only")
    ap.add_argument("--rl_checkpoint", default=ms.DEFAULT_CHECKPOINT,
                    help="checkpoint_<n> folder of the simulated games (default 753), 'latest' or 'all'")
    ap.add_argument("--rl_path_template", default=None,
                    help="Explicit folder that contains the abilities_* folders ('{map}' substituted)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rl_csv", default=None, help="Cached RL simulation records (skips crawling the tree)")
    ap.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    ap.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    ap.add_argument("--game_type", default="classic_collision",
                    choices=["classic", "classic_collision", "competition", "competition_collision"])
    ap.add_argument("--init_type", default=ms.DEFAULT_INIT_TYPE, choices=["random_init", "empty_init"])
    ap.add_argument("--synergy", type=float, default=1.35)
    ap.add_argument("--specialization", type=float, default=0.25)
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--cluster", default="cuenca", help=f"{list(CLUSTER_DIRS)} or a literal base folder")
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
        "  Game score = deliveries in the 180-s test game (humans) / in a 180-s simulated game of the\n"
        f"    MARL agents (checkpoint_{args.rl_checkpoint}, abilities_<walk>_<cut> folders).\n"
        f"  RL source: {args.rl_source if not args.rl_csv else 'cached CSV ' + args.rl_csv}.\n"
        "  Specialization index = 100 * |(D1 - D2)/(D1 + D2) + (C2 - C1)/(C1 + C2)| / 2 (D deliveries,\n"
        f"    C cuts per player); no deliveries or no cuts -> {'0' if args.spec_undefined == 'zero' else 'excluded'}.\n"
        f"    Human cuts: {args.cut_def!r}. RL cuts: per-agent cut actions of each simulated game.\n"
        "  Units: humans = team; RL = training (mean over its simulated games), as in the SI pretraining section.")

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

        report.append(f"\nSelected games -> {Path(stem).name}_selected_games.csv:")
        report += write_selected_games(human, Path(f"{stem}_selected_games.csv"))

        human = human.sort_values("order").reset_index(drop=True)       # selection (CSV) order
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
    # Simulated games for every ability pair (one abilities_<walk>_<cut> folder per pair).
    rl_sims_ep = None
    if args.rl_source == "sims":
        if args.rl_csv:
            rl_sims_ep = normalize_rl_csv(pd.read_csv(args.rl_csv))
            report.append(f"\nRL simulated-game records read from {args.rl_csv}")
        else:
            log = []
            rl_sims_ep = collect_rl_sims(args, log)
            report.append("\n" + "=" * 100 + f"\nRL DATA: MARL simulations (simulated games, "
                          f"checkpoint_{args.rl_checkpoint}, every Training_* / simulation_*)\n" + "=" * 100)
            report += log
            rl_sims_ep.to_csv(f"{stem}_rl_records_sims.csv", index=False)
            report.append(f"Cached -> {stem}_rl_records_sims.csv (re-use with --rl_csv)")

    def _coverage(ep_df, kind):
        report.append(f"\nRL coverage, {kind} (trainings; games per training):")
        for p in PAIRS:
            for mp in MAPS:
                g = ep_df[(ep_df["walk"] == p[0]) & (ep_df["cut"] == p[1]) & (ep_df["map"] == mp)]
                if g.empty:
                    report.append(f"    {pair_label(p):<12} {mp:<4} --")
                    continue
                per = g.groupby("training_id").size()
                report.append(f"    {pair_label(p):<12} {mp:<4} trainings={len(per):>3}  per training="
                              f"{per.min()}..{per.max()}")

    rl_plot = None
    if rl_sims_ep is not None and not rl_sims_ep.empty:
        _coverage(rl_sims_ep, "simulated games")
        rl_plot = training_means(rl_sims_ep)
        report.append("\n" + "=" * 100 + "\nMARL SIMULATIONS (unit = training; mean of its simulated games)\n" + "=" * 100)
        analyse(rl_plot, "MARL simulations", "training", report, csv_rows)

    if rl_plot is not None and human is not None:
        report.append("\n" + "=" * 100 + "\nHUMANS vs RL\n" + "=" * 100)
        agreement(human, rl_plot, report, csv_rows, by_pair={p: rl_plot for p in MA_PAIRS})
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