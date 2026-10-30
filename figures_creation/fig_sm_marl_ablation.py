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

MARL data
  * scratch-trained agents and the (1.0, 1.0) HA reference: SIMULATED GAMES of the
    simulation tree (marl_simulations.py, same folder; same loaders as Fig. 2 and Figs. 4-5)
        <cluster>/data/samuel_lozano/cooked/<game_type>/simulations/<init_type>/map_<map>/
            synergy_<s>/specialized_<l>/abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
    walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2 (the other two speeds
    1.0); every Training_* and every simulation_* folder of checkpoint_753 (--rl_checkpoint)
    is used. One simulated game = one 180-s game; specialization index per game with the
    Fig. 2 formula.
  * pretrained agents: UNCHANGED -- the TRAINING data of the first existing of
        <cluster>/data/samuel_lozano/cooked/<game_type>/pretraining_ablation/<init_type>/map_<map>/synergy_x/specialized_y
        <cluster>/data/samuel_lozano/cooked/<game_type>/pretraining_ablation/map_<map>/synergy_x/specialized_y
    (or --pretrained_dir), loaded with the training-data analyzer of the previous
    fig4_human_vs_rl_gridmap.py, embedded verbatim in this script (TrainingDataAnalyzer:
    same DataProcessor.load_experiment_data call, prepare_episode_data window and speed
    filter), with its own window (--pretrained_episode_range, default the last
    --num_episodes episodes); specialization index per episode from the per-agent cut
    columns returned by the DataProcessor (auto-detected, or --cut_cols).
Violins show every simulated game / episode (--violin_unit episode) or one mean per
training (--violin_unit training). Statistical tests use one value per training by
default (--stats_unit training), because games / episodes of one training are not independent.

Data modes
  default (--pretrained_source values, while the pretraining runs are in progress):
      * scratch-trained agents and the (1.0, 1.0) baseline are REAL simulated games;
      * pretrained agents are drawn from the fixed placeholder values PRETRAINED_VALUES
        (section 6: score mean / SD, specialization mode / spread per map and pair),
        with the same number of games and trainings as the real scratch data;
  --pretrained_source real : the pretraining_ablation trainings are loaded instead.
  --demo                   : scratch / HA synthetic too (layout test).
Humans: read from the per-team CSV written by fig_sm_ability_pairs_humans_rl.py
(--human_teams_csv; default ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv),
the same file used by Figs. 2 and 4, i.e. the same 70 teams per condition used everywhere
in the paper; MARL agents are compared with them by Welch's t-tests (one value per
training vs one value per team). The script stops if that file is missing (run
fig_sm_ability_pairs_humans_rl.py first, or --no_humans). --human_csv uses a
condition-means table instead (one-sample t-tests against the human mean).
Only real records are written to the cache CSV (--rl_csv re-uses it).

<output>_stats.csv / <output>_stats.txt: per map and pair, N / mean +- SEM per source and
the comparisons pretrained vs scratch, pretrained vs HA agents, scratch vs HA agents,
pretrained vs humans and scratch vs humans (HA agents = (1.0, 1.0) simulated games), written with the conventions of
paper_stats.py (higher group first, Welch's t(df) = |t|, p = .059 / p < .001), the same
as in the main text. The signed t (first-named source minus second) is kept in the CSV.

Usage:
    # now: real scratch + (1,1) simulated games (checkpoint_753), pretrained from PRETRAINED_VALUES
    nohup python3 fig_sm_marl_ablation.py --synergy 1.35 --specialization 0.25 --init_type empty_init --rl_checkpoint 753 > log_fig_sm_pretraining.out 2>&1 &

    # final version, once the pretraining_ablation trainings are done
    python3 fig_sm_marl_ablation.py --pretrained_source real --pretrained_episode_range final

    # re-plot from the records cached by a previous run
    python3 fig_sm_marl_ablation.py --rl_csv ./figures/fig_sm_pretraining_rl_records.csv

    # layout test (synthetic scratch / HA, pretrained from PRETRAINED_VALUES, humans from the CSV)
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
# 1. MARL DATA
#  * scratch-trained agents and the (1.0, 1.0) HA agents: SIMULATED GAMES of the simulation
#    tree (marl_simulations.py, same folder; same loaders as Fig. 2 and Figs. 4-5)
#      <base>/data/samuel_lozano/cooked/<game_type>/simulations/<init_type>/map_<map>/
#          synergy_<s>/specialized_<l>/abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
#    walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2 (other speeds 1.0);
#    every Training_* and simulation_* folder is used.
#  * pretrained agents: UNCHANGED -- training data of the pretraining_ablation folder, loaded
#    with the training-data analyzer of the previous Fig. 4 script (embedded below verbatim).
# ===========================================================================
from decimal import Decimal                                # noqa: E402
from typing import Dict, List, Optional, Tuple             # noqa: E402

import marl_simulations as msim                            # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent))   # project root (spoiled_broth, ...)

_GRID_VALUES = dict(walk1_start=0.1, walk1_end=1.0, walk1_step=0.1,
                    cut2_start=0.1, cut2_end=1.0, cut2_step=0.1)


def _project_imports():
    """Project modules are only needed to load the MARL data (not for --demo)."""
    from spoiled_broth.analysis.utils import DataProcessor, AnalysisConfig
    from training_configuration.path_utils import generate_save_directory
    return DataProcessor, AnalysisConfig, generate_save_directory


def _inclusive_float_range(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError('step must be greater than 0')

    start_dec = Decimal(str(start))
    end_dec = Decimal(str(end))
    step_dec = Decimal(str(step))
    epsilon = step_dec / Decimal('1000000')

    values: List[float] = []
    current = start_dec
    while current <= end_dec + epsilon:
        values.append(float(current))
        current += step_dec

    return values



class TrainingDataAnalyzer:
    """VERBATIM copy of the AbilityGridAnalyzer of the previous fig4_human_vs_rl_gridmap.py
    (training data: DataProcessor.load_experiment_data, prepare_episode_data, speed filter),
    used ONLY for the pretrained agents so that their values stay exactly as before."""

    def __init__(self, map_name: str, walk1_start: float, walk1_end: float, walk1_step: float,
                 cut2_start: float, cut2_end: float, cut2_step: float,
                 init_type: str = 'empty_init', synergy: float = 0.0,
                 synergy_provided: bool = False, specialization: Optional[float] = None,
                 game_type: str = 'classic', cluster: str = 'cuenca',
                 study_name: Optional[str] = None, fixed_cut1: float = 1.0,
                 fixed_walk2: float = 1.0, demo: bool = False):
        self.map_name = map_name
        self.init_type = init_type
        self.synergy = synergy
        self.synergy_provided = synergy_provided
        self.specialization = specialization
        self.game_type = game_type
        self.cluster = cluster
        self.study_name = study_name
        self.fixed_cut1 = fixed_cut1
        self.fixed_walk2 = fixed_walk2
        self.demo = demo

        self.walk1_values = _inclusive_float_range(walk1_start, walk1_end, walk1_step)
        self.cut2_values = _inclusive_float_range(cut2_start, cut2_end, cut2_step)

        if demo:
            return

        DataProcessor, AnalysisConfig, self._generate_save_directory = _project_imports()
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor(self.config)
        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        self.local_path = self.config.cluster_paths[cluster]
        self.raw_dir = self._build_raw_dir()

    def _build_raw_dir(self) -> str:
        synergy_folder = 'synergy_0' if self.synergy == 0.0 else f'synergy_{self.synergy:.2f}'
        if self.specialization is None or self.specialization == 0:
            spec_folder = 'specialized_0'
        else:
            spec_folder = f'specialized_{self.specialization:.2f}'

        if self.study_name:
            base_path = self.local_path if self.local_path else '/'
            return os.path.join(
                base_path,
                'data', 'samuel_lozano', 'cooked', self.study_name,
                self.game_type, self.init_type, f'map_{self.map_name}',
                synergy_folder, spec_folder,
            )

        if self.local_path:
            return self._generate_save_directory(
                self.local_path,
                self.game_type,
                2,
                self.map_name,
                self.init_type,
                self.synergy,
                0.0 if self.specialization is None else self.specialization,
            )

        return os.path.join(
            '/',
            'data', 'samuel_lozano', 'cooked', self.game_type, self.init_type,
            f'map_{self.map_name}', synergy_folder, spec_folder,
        )

    def _resolve_training_id_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in ('training_id', 'timestamp', 'run_id'):
            if col in df.columns:
                return col
        return None

    def load_experimental_data(self) -> pd.DataFrame:
        paths = {
            'raw_dir': self.raw_dir,
            'output_path': f'{self.raw_dir}/training_results.csv',
            'figures_dir': f'{self.raw_dir}/training_figures/',
            'smoothed_figures_dir': f'{self.raw_dir}/training_figures/smoothed_{self.config.smoothing_factor}/',
            'study_name': self.study_name,
        }

        for dir_path in [paths['raw_dir'], paths['figures_dir'], paths['smoothed_figures_dir']]:
            os.makedirs(dir_path, exist_ok=True)

        return self.data_processor.load_experiment_data(paths, num_agents=2)

    def _filter_speed_configuration(self, df: pd.DataFrame, walk1: float, cut2: float) -> pd.DataFrame:
        required_cols = ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        condition = (
            (abs(df['walking_speed_1'] - walk1) < 0.01) &
            (abs(df['cutting_speed_1'] - self.fixed_cut1) < 0.01) &
            (abs(df['walking_speed_2'] - self.fixed_walk2) < 0.01) &
            (abs(df['cutting_speed_2'] - cut2) < 0.01)
        )
        return df[condition].copy()

    def _create_combined_metrics(self, df: pd.DataFrame):
        if 'deliver_ai_rl_1' in df.columns and 'deliver_ai_rl_2' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1'] + df['deliver_ai_rl_2']
        elif 'deliver_ai_rl_1' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1']
        else:
            df['total_deliveries'] = 0

    def _metric_value(self, df: pd.DataFrame, metric_name: str) -> float:
        if len(df) == 0:
            return np.nan
        if metric_name == 'total_deliveries':
            self._create_combined_metrics(df)
            return float(df['total_deliveries'].mean())
        if metric_name in df.columns:
            return float(df[metric_name].mean())
        return np.nan

    def prepare_episode_data(self, df: pd.DataFrame, episode_selection: str = 'all',
                             num_episodes: int = 100, target_episode: Optional[int] = None) -> pd.DataFrame:
        if episode_selection == 'all':
            return df.copy()

        if 'timestamp' not in df.columns or 'episode' not in df.columns:
            return df.copy()

        prepared_frames = []
        for timestamp in df['timestamp'].unique():
            training_df = df[df['timestamp'] == timestamp].copy().sort_values('episode')
            if len(training_df) == 0:
                continue

            if episode_selection == 'final':
                prepared_frames.append(training_df.tail(num_episodes))
            elif episode_selection == 'average':
                numeric_cols = training_df.select_dtypes(include=[np.number]).columns.tolist()
                averaged_row = training_df[numeric_cols].mean().to_dict()
                averaged_row['timestamp'] = timestamp
                prepared_frames.append(pd.DataFrame([averaged_row]))
            elif episode_selection == 'specific':
                if target_episode is None:
                    raise ValueError("target_episode must be provided when episode_selection='specific'")
                if target_episode > len(training_df):
                    continue
                half_window = num_episodes // 2
                target_episode_actual = training_df.iloc[target_episode - 1]['episode']
                start_episode = target_episode_actual - half_window
                end_episode = target_episode_actual + half_window
                prepared_frames.append(training_df[(training_df['episode'] >= start_episode) &
                                                   (training_df['episode'] <= end_episode)])
            else:
                raise ValueError(f"Invalid episode_selection: {episode_selection}")

        if not prepared_frames:
            return df.iloc[0:0].copy()

        return pd.concat(prepared_frames, ignore_index=True)


def make_analyzer(args, map_name: str, study_name):
    """Training-data analyzer (pretrained agents) with the arguments of the old Fig. 4 command line."""
    return TrainingDataAnalyzer(
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
            # same layout as the old scratch raw dir, with 'pretraining_ablation' after the game type
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


def scratch_sim_records(args, switch: str, map_name: str, pairs: list[tuple]) -> list[dict]:
    """Simulated games of the scratch-trained agents (one row per simulation_*); the (1.0, 1.0)
    games become the HA reference (source 'ha_baseline')."""
    root = msim.map_root(msim.cluster_base_dir(args.cluster), args.game_type, map_name, args.synergy,
                         args.specialization, init_type=args.init_type, study_name=args.study_name,
                         path_template=args.rl_path_template)
    print(f"\n{'=' * 70}\nscratch-trained agents (simulated games) on map {map_name}\n"
          f"  {msim.describe_root(root, args.rl_checkpoint)}\n{'=' * 70}")
    df = msim.collect_records(root, pairs=list(dict.fromkeys(pairs)), checkpoint=args.rl_checkpoint,
                              verbose=args.verbose)
    rows = []
    for pair in dict.fromkeys(pairs):
        sub = df[np.isclose(df["walk"], pair[0]) & np.isclose(df["cut"], pair[1])] if len(df) else df
        src = "ha_baseline" if pair == (1.0, 1.0) else "scratch"
        n_tr = sub["training_id"].nunique() if len(sub) else 0
        print(f"  {pair}: {len(sub)} simulated games from {n_tr} trainings"
              + ("" if len(sub) else f"   <-- no abilities_{pair[0]:g}_{pair[1]:g} data"))
        for _, r in sub.iterrows():
            rows.append({"source": src, "switch_cost_condition": switch, "map": map_name,
                         "walk": pair[0], "cut": pair[1], "training_id": str(r["training_id"]),
                         "episode": np.nan, "simulation": r["simulation"],
                         "total_deliveries": r["total_deliveries"],
                         "specialization_index": r["specialization_index"]})
    return rows


def collect_records(args, pairs: list[tuple], include_pretrained: bool = True) -> pd.DataFrame:
    """Scratch-trained agents and the (1.0, 1.0) HA agents: simulated games of the simulation
    tree. Pretrained agents (unchanged): training data of the .../<game_type>/pretraining_ablation/...
    folder through the embedded training-data analyzer."""
    maps = {"low": args.map_baseline, "high": args.map_encouraged}
    wanted_scratch = pairs + [(1.0, 1.0)]
    rows = []
    for switch in args.maps:
        map_name = maps[switch]
        rows += scratch_sim_records(args, switch, map_name, wanted_scratch)

        if not include_pretrained:
            continue
        an_pre = make_analyzer(args, map_name, None)
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
                                       "training_id", "episode", "simulation", "total_deliveries",
                                       "specialization_index"])


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
    print(f"\n{'-' * 60}\nReal MARL data found (scratch / HA: simulated games; pretrained: training episodes)\n{'-' * 60}")
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
                print(f"  {MAP_TITLES[switch]:<22} {src:<10} {pair}: {n:5d} games/episodes, {n_tr:3d} trainings{flag}")


def load_human_summary(csv_path: Path) -> pd.DataFrame:
    """Human condition means from a CSV/TSV (--human_csv) -> one row per (map, pair),
    source 'human' (or 'human_ha' for (1, 1)). Headers: Walk, Cut, Map, Score,
    Specialization Index[, Number of Games]; extra columns are ignored."""
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


HA_LABEL = "HA agents (simulated games)"      # (100%,100%) simulated games of the simulation tree
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
                for src, arr in list(v.items()) + [("ha_baseline", ha), ("human", hum)]:
                    d = ps.describe(arr)
                    row.update({f"N_{src}": d["n"], f"mean_{src}": d["mean"], f"sem_{src}": d["sem"]})
                row["n_games_human"] = _human_games(df, switch, pair)
                head = (f"{row['map']:<4} {'Game score' if metric == 'score' else 'Specialization index'} "
                        f"at ({round(pair[0] * 100)}%, {round(pair[1] * 100)}%)")
                lines.append(head)
                comps = [("pre_vs_scratch", "pretrained", v["pretrained"], "from scratch", v["scratch"]),
                         ("pre_vs_ha", "pretrained", v["pretrained"], HA_LABEL, ha),
                         ("scratch_vs_ha", "from scratch", v["scratch"], HA_LABEL, ha)]
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
# 6. PRETRAINED AGENTS (placeholder values -- replace with the real runs)
# ===========================================================================
# (score mean, score SD, specialization-index mode in %, specialization spread) of the
# PRETRAINED agents per map and ability pair (walking speed of player 1, cutting speed of
# player 2). Edit these numbers to change the pretrained groups. Specialization modes
# above 80 % are drawn skewed towards 100 % (see _spec_draw).
PRETRAINED_VALUES = {
    "Open": {
        "pretrained": {(0.1, 0.1): (3.0, 1.3, 44, 14), (0.2, 1.0): (5.4, 1.1, 37, 7)},
    },
    "PB": {
        "pretrained": {(0.1, 0.1): (4.3, 1.2, 77, 17), (0.2, 1.0): (7.1, 1.1, 91, 10)}
    },
}
_MAP_KEY = {"low": "Open", "high": "PB"}

# --demo only (layout test): scratch-trained and HA agents are synthetic too.
_DEMO_LAYOUT = {
    "high": {
        "scratch":    {(0.1, 0.1): (1.6, 1.3, 32, 26), (0.2, 1.0): (4.6, 1.6, 70, 22), (0.7, 0.4): (7.2, 0.9, 85, 7)},
        "ha_baseline": (5.3, 1.1, 60, 14),
    },
    "low": {
        "scratch":    {(0.1, 0.1): (2.4, 1.3, 30, 22), (0.2, 1.0): (4.3, 1.2, 45, 20), (0.7, 0.4): (6.5, 0.8, 46, 10)},
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
    """size synthetic games spread over n_seeds trainings (with a per-training offset,
    so that per-training means vary as in real trainings)."""
    seed_of = np.arange(size) % n_seeds
    off = rng.normal(0, 0.35 * max(sd, 0.3), n_seeds)[seed_of]
    score = np.clip(np.rint(rng.normal(m, max(sd, 0.3), size) + off), 0, None)
    spec = _spec_draw(rng, sm, ss, size)
    return [{"source": source, "switch_cost_condition": switch, "map": switch,
             "walk": pair[0], "cut": pair[1], "training_id": f"{tag}_seed_{seed_of[i]}",
             "episode": i // n_seeds,
             "total_deliveries": score[i], "specialization_index": spec[i]} for i in range(size)]


def _nearest(d: dict, pair):
    return min(d, key=lambda k: abs(k[0] - pair[0]) + abs(k[1] - pair[1]))


def pretrained_value(switch: str, pair):
    """PRETRAINED_VALUES entry of a map / pair (nearest listed pair, with a warning)."""
    d = PRETRAINED_VALUES[_MAP_KEY[switch]]["pretrained"]
    key = pair if pair in d else _nearest(d, pair)
    if key != pair:
        print(f"  [!] PRETRAINED_VALUES has no entry for {pair} on {_MAP_KEY[switch]}; using {key}")
    return d[key]


def demo_records(pairs: list[tuple], maps: list[str], n: int = 160, seed: int = 0) -> pd.DataFrame:
    """Synthetic MARL data (layout test): n games = 8 trainings x 20 games per group."""
    rng = np.random.default_rng(seed)
    rows = []
    for switch in maps:
        d = _DEMO_LAYOUT[switch]
        rows += _synthetic_rows(rng, "ha_baseline", switch, (1.0, 1.0), *d["ha_baseline"], n)
        for pair in pairs:
            rows += _synthetic_rows(rng, "scratch", switch, pair, *d["scratch"][_nearest(d["scratch"], pair)], n)
            rows += _synthetic_rows(rng, "pretrained", switch, pair, *pretrained_value(switch, pair), n)
    return pd.DataFrame(rows)


def synthetic_pretrained_rows(df: pd.DataFrame, pairs: list[tuple], maps: list[str],
                              n: int = 160, seed: int = 1) -> list[dict]:
    """Pretrained agents drawn from PRETRAINED_VALUES, with the same number of games and
    trainings as the real scratch-trained simulations of each map / pair (n games over
    8 trainings when there are none)."""
    rng = np.random.default_rng(seed)
    rows = []
    for switch in maps:
        for pair in pairs:
            m, sd, sm, ss = pretrained_value(switch, pair)
            sel = (df["source"] == "scratch") & (df["switch_cost_condition"] == switch) & \
                np.isclose(df["walk"].astype(float), pair[0]) & np.isclose(df["cut"].astype(float), pair[1])
            n_ep = int(sel.sum()) or n
            n_tr = int(df.loc[sel, "training_id"].nunique()) or 8
            print(f"  [pretrained] {MAP_TITLES[switch]:<22} {pair}: score {m:5.2f} (sd {sd:4.2f}), "
                  f"spec {sm:5.1f}% (sd {ss:4.1f}) <- PRETRAINED_VALUES; {n_ep} games, {n_tr} trainings")
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
    p = argparse.ArgumentParser(description="SI figure: scratch-trained (simulated games) vs pretrained "
                                            "agents at low ability.")
    p.add_argument("--pairs", nargs="+", default=["0.1,0.1", "0.2,1.0"],
                   help="Ability pairs 'walk,cut' (agent 1 walking speed, agent 2 cutting speed)")
    p.add_argument("--maps", nargs="+", default=["low", "high"], choices=["low", "high"],
                   help="Rows to draw: low = Open map, high = partially-blocked map")
    # --- data location ---
    p.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    p.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    p.add_argument("--game_type", default="classic_collision",
                   choices=["classic", "classic_collision", "competition", "competition_collision"])
    p.add_argument("--init_type", default="empty_init", choices=["random_init", "empty_init"])
    p.add_argument("--synergy", type=float, default=1.35)
    p.add_argument("--specialization", type=float, default=0.25)
    p.add_argument("--study_name", default=None, help="Optional study folder after .../cooked/ (scratch simulations)")
    p.add_argument("--cluster", default="cuenca",
                   help=f"{list(msim.CLUSTER_DIRS)} or a literal base folder (pretrained: AnalysisConfig.cluster_paths)")
    p.add_argument("--pretrained_dir", default=None,
                   help="Folder of the pretrained trainings (placeholders {map} {synergy} "
                        "{specialization} {init_type} {game_type}); default: first existing of "
                        ".../cooked/<game_type>/pretraining_ablation/[<init_type>/]map_<map>/synergy_x/specialized_y")
    p.add_argument("--cut_cols", nargs=2, default=None, metavar=("AGENT1", "AGENT2"),
                   help="Per-agent cut-count columns for the specialization index (auto-detected)")
    # --- pretrained trainings: episode window (unchanged) ---
    p.add_argument("--num_episodes", type=int, default=20, help="Pretrained: episodes per training in the window")
    p.add_argument("--pretrained_episode_range", choices=["all", "final", "specific"], default="final")
    p.add_argument("--pretrained_target_episode", type=int, default=None)
    # --- what one point of the violins / of the tests is ---
    p.add_argument("--violin_unit", choices=["episode", "training"], default="episode",
                   help="Violin samples: every simulated game (scratch, HA) / episode of the window "
                        "(pretrained), or one mean per training")
    p.add_argument("--stats_unit", choices=["episode", "training"], default="training",
                   help="Unit of the statistical tests (games / episodes of one training are not independent)")
    # --- other inputs ---
    p.add_argument("--rl_csv", default=None, help="Cached records CSV (skips loading the MARL data)")
    p.add_argument("--human_csv", default=None,
                   help="Use a human condition-means table instead of the per-team file (Walk, Cut, Map, Score, "
                        "Specialization Index[, Number of Games]); one-sample t-tests")
    p.add_argument("--human_teams_csv", default=None,
                   help="Per-team human data from fig_sm_ability_pairs_humans_rl.py (default: "
                        "./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv)")
    p.add_argument("--no_humans", action="store_true", help="Do not draw human stars")
    p.add_argument("--pretrained_source", default="values", choices=["values", "real"],
                   help="'values' (default): pretrained agents drawn from PRETRAINED_VALUES (top of section 6); "
                        "'real': load the pretraining_ablation trainings")
    p.add_argument("--rl_checkpoint", default=msim.DEFAULT_CHECKPOINT,
                   help="Scratch / HA simulated games: checkpoint_<n> folder (default 753), 'latest' or 'all'")
    p.add_argument("--rl_path_template", default=None,
                   help="Scratch / HA: explicit folder that contains the abilities_* folders ('{map}' substituted)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--n_synthetic", type=int, default=160,
                   help="Pretrained games per group when no scratch simulations exist for a pair")
    p.add_argument("--output_dir", default="./figures")
    p.add_argument("--output_name", default="fig_sm_marl_ablation.pdf")
    p.add_argument("--demo", action="store_true",
                   help="Synthetic scratch / HA data (layout test); pretrained from PRETRAINED_VALUES")
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
            cache = out_dir / "fig_sm_pretraining_rl_records.csv"
            df.to_csv(cache, index=False)            # only REAL records are cached
            print(f"\nMARL records (scratch / HA simulated games, pretrained episodes) cached -> {cache}"
                  f"  (re-use with --rl_csv)")
        if not (df["source"] == "ha_baseline").any():
            print("\n[!] no (100%,100%) simulated games found (abilities_1.0_1.0): HA comparisons are empty")
        report_coverage(df, pairs, maps)
        if not real_pretrained:
            df = df[df["source"] != "pretrained"]
            print("\n[!] PRETRAINED AGENTS FROM PRETRAINED_VALUES (--pretrained_source values)")
            syn = synthetic_pretrained_rows(df, pairs, maps, n=args.n_synthetic)
            df = pd.concat([df, pd.DataFrame(syn)], ignore_index=True)

    human_mode = "none"
    if not args.no_humans:
        df = df[~df["source"].isin(["human", "human_ha"])]
        teams_path = Path(args.human_teams_csv) if args.human_teams_csv else _HUMAN_TEAMS_DEFAULT
        if args.human_csv:
            hum = load_human_summary(Path(args.human_csv))
            human_mode = "means"
            print(f"\nHumans: condition means from {args.human_csv} (one-sample t-tests)")
        elif teams_path.exists():
            hum = load_human_teams_csv(teams_path)
            human_mode = "teams"
            print(f"\nHumans: {len(hum)} teams from {teams_path} (Welch's t-tests against the MARL trainings)")
        else:
            raise FileNotFoundError(
                f"Per-team human file not found: {teams_path}. Run fig_sm_ability_pairs_humans_rl.py first "
                f"(it writes <output>_human_teams.csv), pass --human_teams_csv, or use --no_humans.")
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