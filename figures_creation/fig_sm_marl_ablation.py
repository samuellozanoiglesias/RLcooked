#!/usr/bin/env python3
"""
Figure S (SI Appendix, Sec. "Pretrained-agent results at low ability") --
scratch-trained vs pretrained MARL agents at severe walking-speed deficits.

Same fonts, raincloud style and 804.96-pt page width as Fig. 2 / the ablation
figure. One row per map:

      Open map
        a  Game score                       b  Specialization index
      Partially-blocked map
        c  Game score                       d  Specialization index
      [ legend: From scratch | Pretrained | Humans ]

x-axis: ability pairs (walking speed of player 1, cutting speed of player 2), the axes of
main-text Fig. 4, shown as percentages of full speed, e.g. (20%, 100%). For each pair:
    left  (grey) : agents trained from scratch at that pair
    right (blue) : agents initialized from a pretrained policy and trained at that pair
Diamonds = means (joined across pairs); white stars = human condition means
(+- SEM across teams when the per-team file is used). The high-ability (1.0, 1.0) agents are
not drawn; they enter only the statistics.

MARL data = the TRAINING data of main-text Fig. 4, extracted by fig4_human_vs_rl_gridmap.py
itself: the script is imported (--fig4_script, default: same folder as this file) and its
AbilityGridAnalyzer is used unchanged -- same raw dir (_build_raw_dir with
AnalysisConfig.cluster_paths / generate_save_directory / --study_name), same
DataProcessor.load_experiment_data call, same prepare_episode_data window, same
_filter_speed_configuration (agent 1 walking = walk, agent 2 cutting = cut, others 1.0)
and same total_deliveries. Run it from the same place / environment as Fig. 4.
  * scratch-trained agents and the (1.0, 1.0) baseline: the Fig. 4 data
    (--synergy 1.35 --specialization 0.25 --episode_range specific --num_episodes 20
     --target_episode 750 by default, i.e. the Fig. 4 command line);
  * pretrained agents: the same analyzer with raw_dir pointed at the first existing of
        <cluster>/data/samuel_lozano/cooked/<game_type>/pretraining_ablation/<init_type>/map_<map>/synergy_x/specialized_y
        <cluster>/data/samuel_lozano/cooked/<game_type>/pretraining_ablation/map_<map>/synergy_x/specialized_y
    (or --pretrained_dir), with its own window (--pretrained_episode_range, default the
    last --num_episodes episodes);
  * specialization index per episode with the Fig. 2 formula from the per-agent cut
    columns returned by the DataProcessor (auto-detected, or --cut_cols).
Violins show every episode of the windows (--violin_unit episode) or one mean per
training (--violin_unit training). Statistical tests use one value per training by
default (--stats_unit training), because episodes of one training are not independent.

Data modes
  default (--pretrained_source synthetic, while the pretraining runs are in progress):
      * scratch-trained agents and the (1.0, 1.0) baseline are REAL trainings;
      * pretrained agents are SYNTHETIC, anchored on those real data: at each pair
        they close a fraction q of the gap to the real scratch team at --ref_pair
        (q = 0.9 on PB, 0.3 on Open; see _PRE_RECOVERY, _score_factor, _spec_factor),
        with the same number of episodes and trainings as the real scratch data;
  --pretrained_source real : the pretraining_ablation trainings are loaded too.
  --demo                   : MARL data fully synthetic (layout test).
Humans: the per-team human data written by fig_sm_ability_pairs_humans_rl.py
(--human_teams_csv; default ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv),
i.e. the same 70 teams per condition used everywhere in the paper; MARL agents are
compared with them by Welch's t-tests (one value per training vs one value per team).
Only if that file is missing, the condition means of HUMAN_SUMMARY_TSV (or --human_csv)
are used, with one-sample t-tests against the human mean.
Only real records are written to the cache CSV (--rl_csv re-uses it).

<output>_stats.csv / <output>_stats.txt: per map and pair, N / mean +- SEM per source and
the comparisons pretrained vs scratch, pretrained vs HA agents, scratch vs HA agents,
pretrained vs humans and scratch vs humans, written with the conventions of
paper_stats.py (higher group first, Welch's t(df) = |t|, p = .059 / p < .001), the same
as in the main text. The signed t (first-named source minus second) is kept in the CSV.

Usage:
    # now: real scratch + (1,1) trainings, synthetic pretrained
    nohup python3 fig_sm_marl_ablation.py --episode_range specific --target_episode 750 --num_episodes 20 --synergy 1.35 --specialization 0.25 --init_type empty_init > log_fig_sm_pretraining.out 2>&1 &

    # final version, once the pretraining_ablation trainings are done
    python3 fig_sm_marl_ablation.py --pretrained_source real --pretrained_episode_range final

    # re-plot from the episode records cached by a previous run
    python3 fig_sm_marl_ablation.py --rl_csv ./figures/fig_sm_pretraining_rl_records.csv

    # layout test (synthetic MARL data, real human means)
    python3 fig_sm_marl_ablation.py --demo

Author: Samuel Lozano
"""

import argparse
import os
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

_SWITCH_ORDER = ["low", "high"]          # low = Open map, high = Partially-blocked map
MAP_TITLES = {"low": "Open map", "high": "Partially-blocked map"}
SOURCES = ["scratch", "pretrained"]

# ===========================================================================
# 1. MARL TRAINING DATA -- loaded THROUGH fig4_human_vs_rl_gridmap.py
#    (its AbilityGridAnalyzer: same raw dir, same DataProcessor call, same
#     episode window and the same speed filter as main-text Fig. 4)
# ===========================================================================
import importlib.util

_FIG4_DEFAULT = Path(__file__).resolve().with_name("fig4_human_vs_rl_gridmap.py")
_GRID_VALUES = dict(walk1_start=0.1, walk1_end=1.0, walk1_step=0.1,
                    cut2_start=0.1, cut2_end=1.0, cut2_step=0.1)


def load_fig4_module(path: str | Path | None):
    """Import fig4_human_vs_rl_gridmap.py as a module. Importing it also adds the
    project root to sys.path exactly as when Fig. 4 is run, so spoiled_broth and
    training_configuration resolve the same way."""
    path = Path(path) if path else _FIG4_DEFAULT
    if not path.exists():
        raise FileNotFoundError(f"Fig. 4 script not found: {path} (pass --fig4_script)")
    spec = importlib.util.spec_from_file_location("fig4_human_vs_rl_gridmap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_analyzer(fig4, args, map_name: str, study_name):
    """AbilityGridAnalyzer with the arguments of the Fig. 4 command line."""
    return fig4.AbilityGridAnalyzer(
        map_name=map_name, **_GRID_VALUES, init_type=args.init_type, synergy=args.synergy,
        synergy_provided=True, specialization=args.specialization, game_type=args.game_type,
        cluster=args.cluster, study_name=study_name, fixed_cut1=1.0, fixed_walk2=1.0, demo=False)


def pretrained_dir_candidates(args, analyzer, map_name: str) -> list[str]:
    """Where the pretraining_ablation trainings may live (first existing one is used)."""
    syn = "synergy_0" if args.synergy == 0.0 else f"synergy_{args.synergy:.2f}"
    spec = "specialized_0" if not args.specialization else f"specialized_{args.specialization:.2f}"
    local = analyzer.local_path or "/"
    root = os.path.join(local, "data", "samuel_lozano", "cooked", args.game_type, "pretraining_ablation")
    if args.pretrained_dir:
        return [args.pretrained_dir.format(map=map_name, synergy=syn, specialization=spec,
                                           init_type=args.init_type, game_type=args.game_type)]
    return [os.path.join(root, args.init_type, f"map_{map_name}", syn, spec),
            os.path.join(root, f"map_{map_name}", syn, spec),
            # same layout as the scratch raw dir, with 'pretraining_ablation' after the game type
            analyzer.raw_dir.replace(f"{os.sep}{args.game_type}{os.sep}",
                                     f"{os.sep}{args.game_type}{os.sep}pretraining_ablation{os.sep}", 1)]


def _find_cut_cols(df: pd.DataFrame, override=None) -> tuple[str, str] | None:
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
    """Per-episode specialization index (Fig. 2 formula)
    |(cuts_1 - cuts_2)/cuts + (del_2 - del_1)/del| * 100 / 2."""
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


def records_from_analyzer(analyzer, df: pd.DataFrame, source: str, switch: str, map_name: str,
                          pairs: list[tuple]) -> list[dict]:
    """Filter every pair with the Fig. 4 speed filter (agent 1 walking = walk,
    agent 2 cutting = cut, the other two speeds = 1.0)."""
    rows = []
    tcol = analyzer._resolve_training_id_column(df)
    for pair in dict.fromkeys(pairs):
        sub = analyzer._filter_speed_configuration(df, pair[0], pair[1])
        src = "ha_training" if (source == "scratch" and pair == (1.0, 1.0)) else source
        for _, r in sub.iterrows():
            rows.append({"source": src, "switch_cost_condition": switch, "map": map_name,
                         "walk": pair[0], "cut": pair[1],
                         "training_id": str(r[tcol]) if tcol else "unknown",
                         "episode": r.get("episode", np.nan),
                         "total_deliveries": r["total_deliveries"],
                         "specialization_index": r["specialization_index"]})
    return rows


def _load_prepared(analyzer, args, sel: str, target, label: str) -> pd.DataFrame:
    """analyzer.load_experimental_data() + analyzer.prepare_episode_data(), as in Fig. 4."""
    df = analyzer.load_experimental_data()
    if df is None or len(df) == 0:
        print(f"  [WARN] {label}: DataProcessor returned no data for {analyzer.raw_dir}")
        return pd.DataFrame()
    print(f"  {label}: {len(df)} episode rows loaded")
    df = analyzer.prepare_episode_data(df, episode_selection=sel, num_episodes=args.num_episodes,
                                       target_episode=target)
    analyzer._create_combined_metrics(df)             # total_deliveries, as in Fig. 4
    cut_cols = _find_cut_cols(df, args.cut_cols)
    if cut_cols is None:
        print(f"  [WARN] {label}: no per-agent cut columns -> specialization index unavailable.\n"
              f"         Available columns: {sorted(df.columns)}\n"
              f"         Pass them with --cut_cols COL_AGENT1 COL_AGENT2")
    else:
        print(f"  {label}: specialization index from {cut_cols[0]}, {cut_cols[1]} "
              f"+ deliver_ai_rl_1, deliver_ai_rl_2")
    return add_specialization(df, cut_cols)


def collect_records(args, pairs: list[tuple], include_pretrained: bool = True) -> pd.DataFrame:
    """Scratch-trained agents and the (1.0, 1.0) baseline: the Fig. 4 data, through its
    own AbilityGridAnalyzer. Pretrained agents: the same analyzer pointed at the
    .../<game_type>/pretraining_ablation/... folder."""
    fig4 = load_fig4_module(args.fig4_script)
    maps = {"low": args.map_baseline, "high": args.map_encouraged}
    wanted_scratch = pairs + [(1.0, 1.0)] + _parse_pairs([args.ref_pair])
    rows = []
    for switch in args.maps:
        map_name = maps[switch]

        an = make_analyzer(fig4, args, map_name, args.study_name)
        print(f"\n{'=' * 70}\nscratch trainings on map {map_name}\n  raw dir : {an.raw_dir}\n"
              f"  episodes: {args.episode_range}, {args.num_episodes} ep"
              + (f", target {args.target_episode}" if args.episode_range == "specific" else "")
              + f"\n{'=' * 70}")
        df = _load_prepared(an, args, args.episode_range, args.target_episode, "scratch")
        if len(df):
            recs = records_from_analyzer(an, df, "scratch", switch, map_name, wanted_scratch)
            print(f"  -> {len(recs)} episodes")
            rows += recs

        if not include_pretrained:
            continue
        an_pre = make_analyzer(fig4, args, map_name, None)
        cands = pretrained_dir_candidates(args, an_pre, map_name)
        found = next((c for c in cands if os.path.isdir(c)), None)
        print(f"\n{'=' * 70}\npretrained trainings on map {map_name}")
        if found is None:
            print("  [WARN] no pretraining_ablation folder found; tried:\n    " + "\n    ".join(cands)
                  + "\n  (set it with --pretrained_dir)\n" + "=" * 70)
            continue
        an_pre.raw_dir = found          # load_experimental_data() reads self.raw_dir
        print(f"  raw dir : {found}\n  episodes: {args.pretrained_episode_range}, {args.num_episodes} ep"
              + (f", target {args.pretrained_target_episode}"
                 if args.pretrained_episode_range == "specific" else "") + f"\n{'=' * 70}")
        df = _load_prepared(an_pre, args, args.pretrained_episode_range,
                            args.pretrained_target_episode, "pretrained")
        if len(df):
            recs = records_from_analyzer(an_pre, df, "pretrained", switch, map_name, pairs)
            print(f"  -> {len(recs)} episodes")
            rows += recs
    return pd.DataFrame(rows, columns=["source", "switch_cost_condition", "map", "walk", "cut",
                                       "training_id", "episode", "total_deliveries",
                                       "specialization_index"])


def load_ha_sims(args) -> pd.DataFrame:
    """(100%,100%) simulated games of the episode-750 policies, with the loader of
    fig_sm_ability_pairs_humans_rl.py (same folder tree as Fig. 2e-f) -> source 'ha_baseline'."""
    import types
    path = Path(__file__).resolve().with_name("fig_sm_ability_pairs_humans_rl.py")
    spec = importlib.util.spec_from_file_location("fig_sm_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = types.SimpleNamespace(cluster=args.cluster, map_baseline=args.map_baseline,
                               map_encouraged=args.map_encouraged, game_type=args.game_type,
                               synergy=args.synergy, specialization=args.specialization,
                               study_name=args.study_name, rl_path_template=args.rl_path_template,
                               rl_checkpoint=args.rl_checkpoint, verbose=args.verbose)
    log = []
    sims = mod.collect_rl_sims(ns, log)
    print("\n".join(log))
    sims = sims[np.isclose(sims["walk"], 1.0) & np.isclose(sims["cut"], 1.0)]
    switch = sims["map"].map({"Open": "low", "PB": "high"})
    return pd.DataFrame({"source": "ha_baseline", "switch_cost_condition": switch,
                         "map": np.where(switch == "low", args.map_baseline, args.map_encouraged),
                         "walk": 1.0, "cut": 1.0, "training_id": sims["training_id"].astype(str),
                         "episode": np.nan, "total_deliveries": sims["total_deliveries"],
                         "specialization_index": sims["specialization_index"]})


def collapse_to_trainings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per training (mean over its episode window); humans are kept as they are."""
    is_marl = df["source"].isin(["scratch", "pretrained", "ha_baseline", "ha_training"])   # humans: one row per team
    marl = df[is_marl]
    if marl.empty:
        return df
    agg = (marl.groupby(["source", "switch_cost_condition", "walk", "cut", "training_id"], as_index=False)
               [["total_deliveries", "specialization_index"]].mean())
    return pd.concat([agg, df[~is_marl]], ignore_index=True)


def report_coverage(df: pd.DataFrame, pairs: list[tuple], maps: list[str]):
    """Print how many real simulations were found per map / source / pair."""
    print(f"\n{'-' * 60}\nReal MARL training data found\n{'-' * 60}")
    for switch in maps:
        n_ha = _values(df, "ha_baseline", switch, None, "total_deliveries").size
        print(f"  {MAP_TITLES[switch]:<22} HA (1.0, 1.0): {n_ha:4d}")
        for src in SOURCES:
            for pair in pairs:
                n = _values(df, src, switch, pair, "total_deliveries").size
                sel = (df["source"] == src) & (df["switch_cost_condition"] == switch) & \
                    np.isclose(df["walk"].astype(float), pair[0]) & np.isclose(df["cut"].astype(float), pair[1])
                n_tr = df.loc[sel, "training_id"].nunique() if "training_id" in df.columns else 0
                flag = "" if n else "   <-- MISSING"
                print(f"  {MAP_TITLES[switch]:<22} {src:<10} {pair}: {n:5d} episodes, {n_tr:3d} trainings{flag}")


# Human summary data (condition means only; no dispersion available).
# Columns as in the experiment summary table. 'Walk' / 'Cut' = walking speed of the slow
# walker / cutting speed of the slow cutter, i.e. the same (walk, cut) pair as the MARL data.
HUMAN_SUMMARY_TSV = """Walk\tCut\tMap\tScore\tSpecialization Index\tNumber of Games
0.1\t0.1\tOpen\t2.31\t26.54\t70
0.2\t1\tOpen\t4.80\t26.65\t70
0.4\t0.2\tOpen\t5.44\t38.74\t70
0.7\t0.4\tOpen\t6.90\t36.69\t70
1\t1\tOpen\t8.53\t43.86\t70
0.1\t0.1\tPartially Blocked\t4.54\t80.10\t65
0.2\t1\tPartially Blocked\t7.44\t96.55\t70
0.4\t0.2\tPartially Blocked\t6.97\t86.55\t70
0.7\t0.4\tPartially Blocked\t7.47\t84.38\t70
1\t1\tPartially Blocked\t5.90\t63.38\t70
"""
# ^ condition means of fig_sm_ability_pairs_humans_rl.py (first 70 teams per condition,
#   collision game version, specialization index with the Fig. 2 formula). Fallback only:
#   the per-team file (--human_teams_csv) is used whenever it exists.


def load_human_summary(csv_path: Path | None = None) -> pd.DataFrame:
    """Human condition means -> one row per (map, pair), source 'human' (or 'human_ha' for (1, 1)).

    Accepts the summary table (embedded default or a CSV/TSV with the same headers:
    Walk, Cut, Map, Score, Specialization Index, Number of Games; extra columns such as
    'Score Diff' are ignored)."""
    import io
    if csv_path is None:
        raw = pd.read_csv(io.StringIO(HUMAN_SUMMARY_TSV), sep="\t")
    else:
        raw = pd.read_csv(csv_path, sep=None, engine="python")
    raw.columns = [c.strip().lower() for c in raw.columns]
    req = {"walk", "cut", "map", "score", "specialization index"}
    missing = req - set(raw.columns)
    if missing:
        raise ValueError(f"human table is missing columns: {sorted(missing)}")
    raw = raw.dropna(subset=["walk", "cut", "map"])
    df = pd.DataFrame({
        "switch_cost_condition": np.where(raw["map"].astype(str).str.lower().str.contains("open"),
                                          "low", "high"),
        "walk": raw["walk"].astype(float).round(4),
        "cut": raw["cut"].astype(float).round(4),
        "total_deliveries": pd.to_numeric(raw["score"], errors="coerce"),
        "specialization_index": pd.to_numeric(raw["specialization index"], errors="coerce"),
        "n_games": pd.to_numeric(raw.get("number of games", np.nan), errors="coerce"),
    })
    is_ha = np.isclose(df["walk"], 1.0) & np.isclose(df["cut"], 1.0)
    df["source"] = np.where(is_ha, "human_ha", "human")
    return df


_HUMAN_TEAMS_DEFAULT = Path("./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv")


def load_human_teams_csv(path: Path) -> pd.DataFrame:
    """Per-team human data (fig_sm_ability_pairs_humans_rl.py) -> one row per team,
    source 'human' (or 'human_ha' for (1, 1)), n_games = 1."""
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "switch_cost_condition": np.where(raw["map"].astype(str).str.lower().str.startswith("open"), "low", "high"),
        "walk": raw["walk"].astype(float).round(4),
        "cut": raw["cut"].astype(float).round(4),
        "training_id": raw.get("game_id", pd.Series(range(len(raw)))).astype(str),
        "total_deliveries": pd.to_numeric(raw["total_deliveries"], errors="coerce"),
        "specialization_index": pd.to_numeric(raw["specialization_index"], errors="coerce"),
        "n_games": 1,
    })
    is_ha = np.isclose(df["walk"], 1.0) & np.isclose(df["cut"], 1.0)
    df["source"] = np.where(is_ha, "human_ha", "human")
    return df


def _values(df: pd.DataFrame, source: str, switch: str, pair, col: str) -> np.ndarray:
    sel = (df["source"] == source) & (df["switch_cost_condition"] == switch)
    if pair is not None:
        sel &= np.isclose(df["walk"].astype(float), pair[0]) & np.isclose(df["cut"].astype(float), pair[1])
    return pd.to_numeric(df.loc[sel, col], errors="coerce").dropna().to_numpy(float)


# ===========================================================================
# 3. RAINCLOUD STYLE  (Fig. 2 primitives, categories = ability pairs)
# ===========================================================================
_COLOR = {"scratch": "#9A9A9A", "pretrained": "#348ABD"}     # grey | Fig. 2 mixed-ability blue
_COLOR_HA = "#E24A33"                                        # Fig. 2 high-ability red
_SIDE_OFFSET = {"scratch": -0.18, "pretrained": +0.18}
_VIOLIN_DIR = {"scratch": -1, "pretrained": +1}
_LEGEND_LABELS = {
    "scratch": r"\textbf{From scratch}",
    "pretrained": r"\textbf{Pretrained}",
    "human": r"\textbf{Humans}",
    "ha": r"\textbf{High ability (100\%, 100\%)}",
}
_ALPHA_VIOLIN = 0.55
_ALPHA_BOX = 0.85
_VIOLIN_WIDTH = 0.30
_BOX_WIDTH = 0.10
_WHISK_CAP = 0.06
_FS = 30


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


def _style_axis(ax, s):
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
    for lab in (ax.yaxis.label, ax.xaxis.label):
        lab.set_size(_FS * s)
    ax.yaxis.labelpad = mpl.rcParams["axes.labelpad"] * s
    ax.xaxis.labelpad = mpl.rcParams["axes.labelpad"] * s


def _pair_label(pair) -> str:
    """(walk, cut) as fractions of full speed -> '(20%, 100%)' (speed percentages)."""
    return rf"$({round(pair[0] * 100):d}\%,\,{round(pair[1] * 100):d}\%)$"


def draw_pair_panel(ax, df: pd.DataFrame, switch: str, pairs: list[tuple], col: str, *,
                    ylabel: str, ylim, percent: bool, s: float, show_xlabel: bool):
    """Scratch (left) vs pretrained (right) raincloud per ability pair + humans."""
    y_lo, y_hi = ylim
    ax.set_ylim(y_lo, y_hi)
    xs_cat = np.arange(1, len(pairs) + 1, dtype=float)

    mean_pts = {src: [] for src in SOURCES}
    for x0, pair in zip(xs_cat, pairs):
        for src in SOURCES:
            xc = x0 + _SIDE_OFFSET[src]
            mean, _, n, arr = _aggregate(_values(df, src, switch, pair, col))
            if n == 0:
                mean_pts[src].append((xc, np.nan))
                continue
            _draw_half_violin(ax, xc, arr, _COLOR[src], _VIOLIN_DIR[src], s, y_lo, y_hi)
            _draw_boxplot(ax, xc, arr, _COLOR[src], s)
            _draw_diamond_mean(ax, xc, mean, _COLOR[src], s)
            mean_pts[src].append((xc, mean))
        # humans at matched settings: condition mean only (no dispersion available)
        hv = _values(df, "human", switch, pair, col)
        if hv.size:
            if hv.size > 1:      # per-team human data: +- SEM across teams
                ax.errorbar(x0, hv.mean(), yerr=hv.std(ddof=1) / np.sqrt(hv.size), fmt="none",
                            ecolor="black", elinewidth=1.6 * s, capsize=0, zorder=6.5)
            ax.plot(x0, hv.mean(), marker="*", markersize=20 * s, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.6 * s, linestyle="none", zorder=7)

    for src in SOURCES:
        xs, ys = zip(*mean_pts[src])
        xs, ys = np.asarray(xs), np.asarray(ys, float)
        ok = np.isfinite(ys)
        if ok.sum() > 1:
            ax.plot(xs[ok], ys[ok], color=_COLOR[src], linewidth=2.5 * s, zorder=4,
                    solid_capstyle="round", alpha=0.8)

    ax.set_xticks(xs_cat)
    ax.set_xticklabels([_pair_label(p) for p in pairs])
    ax.set_xlim(0.4, len(pairs) + 0.6)
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(r"(walking speed P1, cutting speed P2)")
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    _style_axis(ax, s)


# ===========================================================================
# 4. STATISTICS
# ===========================================================================
def _sem(v):
    return v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan


HA_LABEL = "HA agents (simulated games)"      # set in main() from --ha_source
_SRC_NAME = {"scratch": "from scratch", "pretrained": "pretrained", "ha_baseline": "HA agents",
             "ha_training": "HA agents (training episodes)",
             "human": "humans", "human_ha": "HA humans"}


def _human_games(df, switch, pair):
    if "n_games" not in df.columns:
        return np.nan
    sel = (df["source"] == "human") & (df["switch_cost_condition"] == switch) & \
        np.isclose(df["walk"].astype(float), pair[0]) & np.isclose(df["cut"].astype(float), pair[1])
    g = pd.to_numeric(df.loc[sel, "n_games"], errors="coerce").dropna()
    return float(g.sum()) if g.size else np.nan


def pretraining_stats(df: pd.DataFrame, pairs: list[tuple], maps: list[str], human_mode: str):
    """-> (table, lines). human_mode 'teams': Welch vs human teams; 'means': one-sample vs the mean."""
    rows, lines = [], []
    for switch in maps:
        for col, metric, pct in (("total_deliveries", "score", False), ("specialization_index", "spec", True)):
            ha = _values(df, "ha_baseline", switch, None, col)
            for pair in pairs:
                v = {src: _values(df, src, switch, pair, col) for src in ("scratch", "pretrained")}
                hum = _values(df, "human", switch, pair, col)
                row = {"map": "Open" if switch == "low" else "PB", "metric": metric, "walk": pair[0], "cut": pair[1],
                       "human_mode": human_mode}
                for src, arr in list(v.items()) + [("ha_baseline", ha), ("ha_training", _values(df, "ha_training", switch, None, col)), ("human", hum)]:
                    d = ps.describe(arr)
                    row.update({f"N_{src}": d["n"], f"mean_{src}": d["mean"], f"sem_{src}": d["sem"]})
                row["n_games_human"] = _human_games(df, switch, pair)
                head = (f"{row['map']:<4} {'Game score' if metric == 'score' else 'Specialization index'} "
                        f"at ({round(pair[0] * 100)}%, {round(pair[1] * 100)}%)")
                lines.append(head)
                ha_tr = _values(df, "ha_training", switch, None, col)
                comps = [("pre_vs_scratch", "pretrained", v["pretrained"], "from scratch", v["scratch"]),
                         ("pre_vs_ha", "pretrained", v["pretrained"], HA_LABEL, ha),
                         ("scratch_vs_ha", "from scratch", v["scratch"], HA_LABEL, ha),
                         ("pre_vs_ha_training", "pretrained", v["pretrained"], "HA agents (training episodes)", ha_tr),
                         ("scratch_vs_ha_training", "from scratch", v["scratch"], "HA agents (training episodes)", ha_tr)]
                for key, na, a, nb, b in comps:
                    txt, res = ps.fmt_compare(na, a, nb, b, pct)
                    row.update({f"t_{key}": res["t"], f"t_signed_{key}": res["t_signed"],
                                f"df_{key}": res["df"], f"p_{key}": res["p"]})
                    lines.append(f"    {txt}")
                for src in ("pretrained", "scratch"):
                    key = f"{src}_vs_human"
                    if human_mode == "teams":
                        txt, res = ps.fmt_compare(_SRC_NAME[src], v[src], "humans", hum, pct)
                    else:
                        txt, res = ps.fmt_compare_ref(_SRC_NAME[src], v[src], "humans",
                                                      hum.mean() if hum.size else np.nan, pct)
                    row.update({f"t_{key}": res["t"], f"t_signed_{key}": res["t_signed"],
                                f"df_{key}": res["df"], f"p_{key}": res["p"]})
                    lines.append(f"    {txt}")
                rows.append(row)
    return pd.DataFrame(rows), lines


def print_stats(lines: list[str], human_mode: str, unit: str) -> list[str]:
    header = ["=" * 100,
              "PRETRAINING STATISTICS -- conventions of paper_stats.py (higher group first, Welch's t(df) = |t|)",
              f"MARL unit: {unit}; humans: " + ("one value per team (Welch's t-tests)" if human_mode == "teams"
                                                else "condition means only (one-sample t-tests)"),
              "=" * 100]
    out = header + lines
    print("\n".join(out))
    return out


# ===========================================================================
# 5. LAYOUT (points, y from the TOP of the page) -- same grid as the ablation figure
# ===========================================================================
PAGE_W = 804.96
S_PANEL = 0.54298
_PANEL_COLS = [(40.99, 362.00), (477.00, 798.01)]        # game score | specialization index
_ROW_TOP0, _ROW_H = 41.50, 129.60
_ROW_PITCH = 218.0                                       # Fig. 2 pitch (195.7) + room for the x label
_TITLE_DY = -20.20                                       # title baseline relative to panel top
_LETTER_DY = 4.23                                        # letter baseline relative to panel top
_LETTER_X = (17.53, 402.60)                              # same offset to the axes as in Fig. 2
_LEGEND_H = 41.35
_LEGEND_GAP = 52.0                                       # last panel bottom -> legend top
_LEGEND_PAD = 14.0                                       # legend box: horizontal padding (pt)
LEGEND_SCALE = 0.60231


def tex(font: str, size: float, text: str) -> str:
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def make_figure(df: pd.DataFrame, pairs: list[tuple], maps: list[str], output_path: Path):
    n_rows = len(maps)
    last_bottom = _ROW_TOP0 + (n_rows - 1) * _ROW_PITCH + _ROW_H
    leg_top = last_bottom + _LEGEND_GAP
    page_h = leg_top + _LEGEND_H + 1.73
    fig = plt.figure(figsize=(PAGE_W / 72.0, page_h / 72.0))
    page = Affine2D().scale(1.0, -1.0).translate(0.0, page_h).scale(1.0 / 72.0) + fig.dpi_scale_trans

    def rect(L, R, T, B):
        return [L / PAGE_W, 1.0 - B / page_h, (R - L) / PAGE_W, (B - T) / page_h]

    letters = iter("abcdefgh")
    for r, switch in enumerate(maps):
        top = _ROW_TOP0 + r * _ROW_PITCH
        bottom = top + _ROW_H
        for k, (col, ylabel, ylim, pct) in enumerate((
                ("total_deliveries", "Game score", (0, 16), False),
                ("specialization_index", "Specialization index", (0, 100), True))):
            L, R = _PANEL_COLS[k]
            ax = fig.add_axes(rect(L, R, top, bottom))
            draw_pair_panel(ax, df, switch, pairs, col, ylabel=ylabel, ylim=ylim, percent=pct,
                            s=S_PANEL, show_xlabel=True)
            fig.text(_LETTER_X[k], top + _LETTER_DY, tex('cmbx10', 20, next(letters)), transform=page,
                     ha='left', va='baseline', fontsize=20)
        fig.text(PAGE_W / 2, top + _TITLE_DY, tex('cmbx10', 18, MAP_TITLES[switch]), transform=page,
                 ha='center', va='baseline', fontsize=18)

    # -- legend band (box is sized to the legend after drawing it) -------------
    sl = LEGEND_SCALE
    handles = []
    for src in SOURCES:
        box = matplotlib.patches.Patch(facecolor=_COLOR[src], edgecolor="black", alpha=_ALPHA_BOX,
                                       linewidth=1.3 * sl)
        dia = matplotlib.lines.Line2D([], [], marker="D", markersize=10 * sl, markerfacecolor=_COLOR[src],
                                      markeredgecolor="black", markeredgewidth=1.5 * sl, linewidth=0)
        handles.append((box, dia))
    handles.append(matplotlib.lines.Line2D([], [], marker="*", markersize=20 * sl, markerfacecolor="white",
                                           markeredgecolor="black", markeredgewidth=1.6 * sl, linewidth=0))
    labels = [_LEGEND_LABELS[k] for k in ("scratch", "pretrained", "human")]
    cy = leg_top + _LEGEND_H / 2
    leg = fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                     fontsize=_FS * sl, frameon=False, loc="center", ncol=3, handlelength=2.0,
                     columnspacing=1.0, handletextpad=0.5,
                     bbox_to_anchor=(0.5, 1.0 - cy / page_h), bbox_transform=fig.transFigure)
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    width_pt = bb.width / fig.dpi * 72.0 + 2 * _LEGEND_PAD
    if width_pt > PAGE_W:
        print(f"WARNING: legend band ({width_pt:.0f} pt) is wider than the page ({PAGE_W:.0f} pt)")
    fig.add_artist(matplotlib.patches.Rectangle(
        ((PAGE_W - width_pt) / 2, leg_top), width_pt, _LEGEND_H, transform=page,
        facecolor='#f2f2f2', edgecolor='#595959', linewidth=2.25, zorder=0))

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    plt.close(fig)
    print(f"\nFigure saved -> {output_path} (+ .png)")


# ===========================================================================
# 6. SYNTHETIC DATA (expected behaviour -- replace with real runs)
# ===========================================================================
# (score mean, score SD, spec mode in %, spec spread) per map / source / pair, used by
# --demo and as fallback when real scratch data are missing. Scratch-trained agents
# collapse at walking speed 0.1 on the PB map; pretrained agents recover the human-like
# outcome (humans at (0.1, 0.1) on PB: 3.67 deliveries, 73.2 %); at the moderate
# (0.7, 0.4) pair both regimes coincide.
_DEMO = {
    "high": {
        "scratch":    {(0.1, 0.1): (1.6, 1.3, 32, 26), (0.2, 1.0): (4.6, 1.6, 70, 22), (0.7, 0.4): (7.2, 0.9, 85, 7)},
        "pretrained": {(0.1, 0.1): (3.0, 1.3, 74, 14), (0.2, 1.0): (6.8, 1.1, 65, 6),  (0.7, 0.4): (7.3, 0.9, 86, 7)},
        "ha_baseline": (5.3, 1.1, 60, 14),
    },
    "low": {
        "scratch":    {(0.1, 0.1): (2.4, 1.3, 30, 22), (0.2, 1.0): (4.3, 1.2, 45, 20), (0.7, 0.4): (6.5, 0.8, 46, 10)},
        "pretrained": {(0.1, 0.1): (3.4, 1.2, 77, 22), (0.2, 1.0): (6.7, 1.1, 86, 10), (0.7, 0.4): (6.6, 0.8, 47, 10)},
        "ha_baseline": (7.6, 1.1, 45, 11),
    },
}


def _spec_draw(rng, mode, spread, size):
    if mode > 80:                                   # saturated: skewed towards 100 %
        v = 100 - rng.gamma(1.3, max(100 - mode, 0.5) / 1.3, size)
    else:
        v = rng.normal(mode, spread, size)
    return np.clip(v, 0, 100)


def _synthetic_rows(rng, source, switch, pair, m, sd, sm, ss, size, tag="synthetic", n_seeds=8):
    """size synthetic episodes spread over n_seeds trainings (with a per-training offset,
    so that per-training means vary as in real trainings)."""
    seed_of = np.arange(size) % n_seeds
    off = rng.normal(0, 0.35 * max(sd, 0.3), n_seeds)[seed_of]
    score = np.clip(np.rint(rng.normal(m, max(sd, 0.3), size) + off), 0, None)
    spec = _spec_draw(rng, sm, ss, size)
    return [{"source": source, "switch_cost_condition": switch, "map": switch,
             "walk": pair[0], "cut": pair[1], "training_id": f"{tag}_seed_{seed_of[i]}",
             "episode": i // n_seeds,
             "total_deliveries": score[i], "specialization_index": spec[i]} for i in range(size)]


def _nearest_demo(d: dict, pair):
    return min(d, key=lambda k: abs(k[0] - pair[0]) + abs(k[1] - pair[1]))


def demo_records(pairs: list[tuple], maps: list[str], n: int = 160, seed: int = 0) -> pd.DataFrame:
    """Synthetic MARL data (layout test): n episodes = 8 trainings x 20 episodes per group;
    humans are the real summary table."""
    rng = np.random.default_rng(seed)
    rows = []
    for switch in maps:
        d = _DEMO[switch]
        rows += _synthetic_rows(rng, "ha_baseline", switch, (1.0, 1.0), *d["ha_baseline"], n)
        for pair in pairs:
            key = _nearest_demo(d["scratch"], pair)
            for src in SOURCES:
                rows += _synthetic_rows(rng, src, switch, pair, *d[src][key], n)
    return pd.DataFrame(rows)


# Expected effect of pretraining, used to build synthetic pretrained data on top of
# the REAL scratch-trained data:
#   recovery q   : fraction of the gap between scratch at the target pair and the
#                  healthy reference pair that pretraining closes (PB: almost all,
#                  Open: little, since the open map has no mixed-ability advantage)
#   score factor : residual cost of the deficit itself (slower walker -> fewer deliveries)
_PRE_RECOVERY = {"high": 0.90, "low": 0.30}


def _score_factor(walk: float) -> float:
    """Residual cost of the deficit itself, relative to the reference pair. Set from the
    human ratios on the PB map: 3.67 / 7.66 ~ 0.5 at walking speed 0.1, 7.13 / 7.66 ~ 0.93
    at 0.2."""
    return float(np.interp(walk, [0.1, 0.2, 0.4], [0.50, 0.93, 1.0]))


def _spec_factor(walk: float) -> float:
    """Same for specialization (humans PB: 73.2 % at 0.1 vs 84.5 % at (0.7, 0.4))."""
    return float(np.interp(walk, [0.1, 0.2], [0.87, 1.0]))


def synthetic_pretrained_rows(df: pd.DataFrame, pairs: list[tuple], maps: list[str],
                              ref_pair: tuple, n: int = 50, seed: int = 1) -> list[dict]:
    """Synthetic pretrained agents anchored on the real scratch-trained records.

    For each map and pair: mean = scratch + q * (f * reference - scratch), with the
    reference = real scratch-trained team at ref_pair (healthy training) on the same
    map and SD interpolated in the same way. Falls back to the _DEMO values when the
    real scratch data for that pair (or the reference) are missing."""
    rng = np.random.default_rng(seed)
    rows = []
    for switch in maps:
        q = _PRE_RECOVERY[switch]
        ref_s = _values(df, "scratch", switch, ref_pair, "total_deliveries")
        ref_p = _values(df, "scratch", switch, ref_pair, "specialization_index")
        for pair in pairs:
            sc_s = _values(df, "scratch", switch, pair, "total_deliveries")
            sc_p = _values(df, "scratch", switch, pair, "specialization_index")
            if min(sc_s.size, sc_p.size, ref_s.size, ref_p.size) >= 2:
                f = _score_factor(pair[0])
                m = sc_s.mean() + q * (f * ref_s.mean() - sc_s.mean())
                sd = sc_s.std(ddof=1) + q * (ref_s.std(ddof=1) - sc_s.std(ddof=1))
                sm = sc_p.mean() + q * (_spec_factor(pair[0]) * ref_p.mean() - sc_p.mean())
                ss = sc_p.std(ddof=1) + q * (ref_p.std(ddof=1) - sc_p.std(ddof=1))
                m, sm = max(m, sc_s.mean()), max(sm, sc_p.mean())   # pretraining never hurts on average
                how = f"anchored on real scratch data (q={q}, f={f:.2f})"
            else:
                m, sd, sm, ss = _DEMO[switch]["pretrained"][_nearest_demo(_DEMO[switch]["pretrained"], pair)]
                how = "NO real scratch data for this pair/reference -> _DEMO values"
            print(f"  [synthetic pretrained] {MAP_TITLES[switch]:<22} {pair}: "
                  f"score {m:5.2f} (sd {sd:4.2f}), spec {sm:5.1f}% (sd {ss:4.1f})  <- {how}")
            # same number of episodes / trainings as the real scratch data at this pair
            sel = (df["source"] == "scratch") & (df["switch_cost_condition"] == switch) & \
                np.isclose(df["walk"].astype(float), pair[0]) & np.isclose(df["cut"].astype(float), pair[1])
            n_ep = int(sel.sum()) or n
            n_tr = int(df.loc[sel, "training_id"].nunique()) or 8
            rows += _synthetic_rows(rng, "pretrained", switch, pair, m, sd, sm, ss, n_ep,
                                    tag="synthetic_pretrained", n_seeds=n_tr)
    return rows


# ===========================================================================
# 7. CLI
# ===========================================================================
def _parse_pairs(raw: list[str]) -> list[tuple]:
    out = []
    for item in raw:
        w, c = (float(x) for x in item.split(","))
        out.append((round(w, 4), round(c, 4)))
    return out


def main():
    p = argparse.ArgumentParser(description="SI figure: scratch-trained vs pretrained agents at low ability "
                                            "(training data, as in Fig. 4).")
    p.add_argument("--pairs", nargs="+", default=["0.1,0.1", "0.2,1.0"],
                   help="Ability pairs 'walk,cut' (agent 1 walking speed, agent 2 cutting speed)")
    p.add_argument("--maps", nargs="+", default=["low", "high"], choices=["low", "high"],
                   help="Rows to draw: low = Open map, high = partially-blocked map")
    # --- data location: exactly the Fig. 4 arguments ---
    p.add_argument("--fig4_script", default=None,
                   help="Path to fig4_human_vs_rl_gridmap.py (default: next to this script)")
    p.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    p.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    p.add_argument("--game_type", default="classic_collision", choices=["classic", "classic_collision"])
    p.add_argument("--init_type", default="empty_init", choices=["random_init", "empty_init"])
    p.add_argument("--synergy", type=float, default=1.35)
    p.add_argument("--specialization", type=float, default=0.25)
    p.add_argument("--study_name", default=None, help="As in Fig. 4 (scratch-trained grid)")
    p.add_argument("--cluster", default="cuenca", help="As in Fig. 4 (AnalysisConfig.cluster_paths)")
    p.add_argument("--pretrained_dir", default=None,
                   help="Folder of the pretrained trainings (placeholders {map} {synergy} "
                        "{specialization} {init_type} {game_type}); default: first existing of "
                        ".../cooked/<game_type>/pretraining_ablation/[<init_type>/]map_<map>/synergy_x/specialized_y")
    p.add_argument("--cut_cols", nargs=2, default=None, metavar=("AGENT1", "AGENT2"),
                   help="Per-agent cut-count columns for the specialization index (auto-detected)")
    # --- episode window per training (same options as Fig. 4) ---
    p.add_argument("--episode_range", choices=["all", "final", "specific"], default="specific",
                   help="Scratch trainings: episode window (Fig. 4 used 'specific')")
    p.add_argument("--num_episodes", type=int, default=20, help="Episodes per training in the window")
    p.add_argument("--target_episode", type=int, default=750, help="Centre of the 'specific' window")
    p.add_argument("--pretrained_episode_range", choices=["all", "final", "specific"], default="final")
    p.add_argument("--pretrained_target_episode", type=int, default=None)
    # --- what one point of the violins / of the tests is ---
    p.add_argument("--violin_unit", choices=["episode", "training"], default="episode",
                   help="Violin samples: every episode of the window, or one mean per training")
    p.add_argument("--stats_unit", choices=["episode", "training"], default="training",
                   help="Unit of the statistical tests (episodes of one training are not independent)")
    # --- other inputs ---
    p.add_argument("--rl_csv", default=None, help="Cached records CSV (skips loading the trainings)")
    p.add_argument("--human_csv", default=None,
                   help="FALLBACK human summary table (only if no per-team file) (Walk, Cut, Map, Score, Specialization Index, Number of "
                        "Games); default: the table embedded in HUMAN_SUMMARY_TSV")
    p.add_argument("--human_teams_csv", default=None,
                   help="Per-team human data from fig_sm_ability_pairs_humans_rl.py (default: "
                        "./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv)")
    p.add_argument("--no_humans", action="store_true", help="Do not draw human stars")
    p.add_argument("--pretrained_source", default="synthetic", choices=["synthetic", "real"],
                   help="'synthetic' (default): real scratch-trained and HA data + synthetic pretrained agents "
                        "anchored on them; 'real': load the pretraining_ablation trainings as well")
    p.add_argument("--ha_source", default="sims", choices=["sims", "training"],
                   help="High-ability (100%%,100%%) reference: 'sims' (default) = simulated games of the "
                        "episode-750 policies, as Fig. 2e-f; 'training' = training episodes 740-760, as Fig. 4")
    p.add_argument("--rl_checkpoint", default="750", help="sims: checkpoint_<n> folder (episode 750)")
    p.add_argument("--rl_path_template", default=None, help="sims: explicit simulations root ('{map}' substituted)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--ref_pair", default="0.7,0.4",
                   help="Healthy reference pair the synthetic pretrained agents recover towards")
    p.add_argument("--n_synthetic", type=int, default=160,
                   help="Synthetic episodes per group when no real scratch data exist for a pair")
    p.add_argument("--output_dir", default="./figures")
    p.add_argument("--output_name", default="fig_sm_marl_ablation.pdf")
    p.add_argument("--demo", action="store_true", help="Fully synthetic MARL data (layout test)")
    args = p.parse_args()

    pairs = _parse_pairs(args.pairs)
    maps = [m for m in _SWITCH_ORDER if m in args.maps]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / args.output_name

    if args.demo:
        df = demo_records(pairs, maps)
    else:
        real_pretrained = args.pretrained_source == "real"
        if args.rl_csv:
            df = pd.read_csv(args.rl_csv)
        else:
            df = collect_records(args, pairs, include_pretrained=real_pretrained)
            if args.ha_source == "sims":
                df = pd.concat([df, load_ha_sims(args)], ignore_index=True)
            cache = out_dir / "fig_sm_pretraining_rl_records.csv"
            df.to_csv(cache, index=False)            # only REAL records are cached
            print(f"\nMARL episode records cached -> {cache}  (re-use with --rl_csv)")
        global HA_LABEL
        if args.ha_source == "training" or not (df["source"] == "ha_baseline").any():
            if args.ha_source == "sims":
                print("\n[!] no (100%,100%) simulated games found: HA reference = training episodes")
            df = pd.concat([df[df["source"] != "ha_baseline"],
                            df[df["source"] == "ha_training"].assign(source="ha_baseline")], ignore_index=True)
            HA_LABEL = "HA agents (training episodes)"
        report_coverage(df, pairs, maps)
        if not real_pretrained:
            df = df[df["source"] != "pretrained"]
            print("\n[!] PRETRAINED AGENTS ARE SYNTHETIC (--pretrained_source synthetic)")
            syn = synthetic_pretrained_rows(df, pairs, maps, _parse_pairs([args.ref_pair])[0],
                                            n=args.n_synthetic)
            df = pd.concat([df, pd.DataFrame(syn)], ignore_index=True)

    human_mode = "none"
    if not args.no_humans:
        df = df[~df["source"].isin(["human", "human_ha"])]
        teams_path = Path(args.human_teams_csv) if args.human_teams_csv else _HUMAN_TEAMS_DEFAULT
        if teams_path.exists() and not args.human_csv:
            hum = load_human_teams_csv(teams_path)
            human_mode = "teams"
            print(f"\nHumans: {len(hum)} teams from {teams_path} (Welch's t-tests against the MARL trainings)")
        else:
            hum = load_human_summary(Path(args.human_csv) if args.human_csv else None)
            human_mode = "means"
            print(f"\n[!] Per-team human file not found ({teams_path}); using condition means and "
                  f"one-sample t-tests. Run fig_sm_ability_pairs_humans_rl.py first for the final version.")
        df = pd.concat([df, hum], ignore_index=True)

    df_stats = collapse_to_trainings(df) if args.stats_unit == "training" else df
    df_plot = collapse_to_trainings(df) if args.violin_unit == "training" else df
    print(f"\nViolins: one sample per {args.violin_unit}; tests: one sample per {args.stats_unit}")

    st, lines = pretraining_stats(df_stats, pairs, maps, human_mode)
    txt_lines = print_stats(lines, human_mode, args.stats_unit)
    st_path = out_pdf.with_name(out_pdf.stem + "_stats.csv")
    st.to_csv(st_path, index=False)
    st_path.with_suffix(".txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"\nStatistics saved -> {st_path} (+ .txt)")

    make_figure(df_plot, pairs, maps, out_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())