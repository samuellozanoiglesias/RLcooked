#!/usr/bin/env python3
"""
2D Grid Cooperative Analysis Figure Generator

This script creates 2D color grid plots combining both map configurations (Y-axis) 
and agent ability configurations (X-axis). The visualization shows how different 
combinations perform across performance metrics.

Map configurations tested (Y-axis):
- baseline_division_of_labor_large (baseline)
- semiencouraged_division_of_labor_large
- 1-semiencouraged_division_of_labor_large
- 2-semiencouraged_division_of_labor_large
- encouraged_division_of_labor_large
- 1-encouraged_division_of_labor_large
- 2-encouraged_division_of_labor_large
- 3-encouraged_division_of_labor_large

Agent ability configurations tested (X-axis):
- X=1.0: Agent 1 (1.0, 1.0), Agent 2 (1.0, 1.0) - baseline abilities
- X=0.8: Agent 1 (0.8, 1.0), Agent 2 (1.0, 0.8)
- X=0.6: Agent 1 (0.6, 1.0), Agent 2 (1.0, 0.6)
- X=0.4: Agent 1 (0.4, 1.0), Agent 2 (1.0, 0.4)
- X=0.2: Agent 1 (0.2, 1.0), Agent 2 (1.0, 0.2)

Generates 6 color grid plots with different comparison baselines:
1. Global baseline comparison (baseline map + X=1.0 abilities)
2. Row-wise comparison (each map vs X=1.0 abilities for that map)
3. Column-wise comparison (baseline map vs each ability configuration)

Color coding:
- White: Baseline condition (difference=0)
- Blue: Better performance than baseline
- Red: Worse performance than baseline

Usage:
    python comparison_human-grid_2D_abilities_cooperative_analysis.py [options]

MARL data: simulated games of the simulation tree (marl_simulations.py, same folder)
    <base>/data/samuel_lozano/cooked/<game_type>/simulations/<init_type>/map_<map>/
        synergy_<s>/specialized_<l>/abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
walk1 = walking speed of agent 1 (ai_rl_1), cut2 = cutting speed of agent 2 (ai_rl_2), the
other two speeds 1.0. Every Training_* and every simulation_* folder is used.

Examples:
nohup python fig3_comparison_human_and_RL-grid_2D_abilities.py --init_type empty_init --cluster brigit --specialization 0.25 --synergy 1.35 --checkpoint 753 > 2D_abilities_grid.log 2>&1 & 
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# This folder (marl_simulations.py) and the project root on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import marl_simulations as ms  # noqa: E402  (same folder)


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


def _format_speed_label(value: float) -> str:
    text = f'{value:.2f}'.rstrip('0').rstrip('.')
    if '.' not in text:
        text += '.0'
    return text


def _format_speed_token(value: float) -> str:
    return _format_speed_label(value).replace('.', 'p')


def _make_diverging_cmap(name: str, neg_color: str, mid_color: str, pos_color: str):
    return mcolors.LinearSegmentedColormap.from_list(name, [neg_color, mid_color, pos_color])


HUMAN_RESULTS = {
    'baseline_division_of_labor_large': {
        'scores': {
            (1.0, 1.0): 9.014,
            (0.7, 0.4): 7.392,
            (0.4, 0.2): 5.743,
        },
        'specialization': {
            (1.0, 1.0): 60,
            (0.7, 0.4): 55,
            (0.4, 0.2): 50,
        },
    },
    'encouraged_division_of_labor_large': {
        'scores': {
            (1.0, 1.0): 70,
            (0.7, 0.4): 80,
            (0.4, 0.2): 90,
        },
        'specialization': {
            (1.0, 1.0): 11.729,
            (0.7, 0.4): 17.514,
            (0.4, 0.2): 17.057,
        },
    },
}
HUMAN_MARKER_OFFSET = 0.0
HUMAN_MARKER_SIZE = 150


def _resolve_human_map_key(map_name: str) -> Optional[str]:
    if map_name in HUMAN_RESULTS:
        return map_name
    for key in HUMAN_RESULTS:
        if key in map_name:
            return key
    return None


class AbilityGridAnalyzer:
    """Load and filter the simulated games of a single map over a 2D ability sweep.

    Data: every simulation_* of every Training_* in
      .../<game_type>/simulations/<init_type>/map_<map>/synergy_<s>/specialized_<l>/
          abilities_<walk1>_<cut2>/Training_*/checkpoint_<checkpoint>/
    (walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2, other speeds 1.0),
    one row per simulated game (see marl_simulations.py)."""

    def __init__(self, map_name: str, walk1_start: float, walk1_end: float, walk1_step: float,
                 cut2_start: float, cut2_end: float, cut2_step: float,
                 init_type: str = 'empty_init', synergy: float = 0.0,
                 synergy_provided: bool = False, specialization: Optional[float] = None,
                 game_type: str = 'classic', cluster: str = 'cuenca',
                 study_name: Optional[str] = None, fixed_cut1: float = 1.0,
                 fixed_walk2: float = 1.0, checkpoint: str = ms.DEFAULT_CHECKPOINT,
                 verbose: bool = False):
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
        self.checkpoint = str(checkpoint)
        self.verbose = verbose
        if abs(fixed_cut1 - ms.FIXED_CUT1) > 1e-9 or abs(fixed_walk2 - ms.FIXED_WALK2) > 1e-9:
            print(f'Warning: the simulation tree only holds cut1 = {ms.FIXED_CUT1} and walk2 = {ms.FIXED_WALK2}; '
                  f'--fixed_cut1 {fixed_cut1} / --fixed_walk2 {fixed_walk2} will match no simulation.')

        self.local_path = ms.cluster_base_dir(cluster)

        self.walk1_values = _inclusive_float_range(walk1_start, walk1_end, walk1_step)
        self.cut2_values = _inclusive_float_range(cut2_start, cut2_end, cut2_step)

        self.raw_dir = str(self._build_raw_dir())

    def _build_raw_dir(self) -> Path:
        """Folder holding the abilities_* folders of this map."""
        spec = 0 if self.specialization is None else self.specialization
        return ms.map_root(self.local_path, self.game_type, self.map_name, self.synergy, spec,
                           init_type=self.init_type, study_name=self.study_name)

    def _resolve_training_id_column(self, df: pd.DataFrame) -> Optional[str]:
        if 'training_id' in df.columns:
            return 'training_id'
        if 'timestamp' in df.columns:
            return 'timestamp'
        if 'run_id' in df.columns:
            return 'run_id'
        return None

    def load_experimental_data(self) -> pd.DataFrame:
        """One row per simulated game of every ability pair of the grid."""
        root = Path(self.raw_dir)
        print(ms.describe_root(root, self.checkpoint))
        pairs = [(w, c) for w in self.walk1_values for c in self.cut2_values]
        df = ms.collect_records(root, pairs=pairs, checkpoint=self.checkpoint, verbose=self.verbose,
                                extra={'map_name': self.map_name})
        n_tr = df['training_id'].nunique() if len(df) else 0
        print(f'Loaded {len(df)} simulated games from {n_tr} trainings ({self.map_name})')
        return df

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
        if 'total_deliveries' in df.columns:      # team deliveries of each simulated game (as Fig. 2)
            pass
        elif 'deliver_ai_rl_1' in df.columns and 'deliver_ai_rl_2' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1'] + df['deliver_ai_rl_2']
        elif 'deliver_ai_rl_1' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1']
        else:
            df['total_deliveries'] = 0

        if 'pure_reward_ai_rl_1' in df.columns and 'pure_reward_ai_rl_2' in df.columns:
            df['pure_reward_total'] = df['pure_reward_ai_rl_1'] + df['pure_reward_ai_rl_2']
        elif 'pure_reward_ai_rl_1' in df.columns:
            df['pure_reward_total'] = df['pure_reward_ai_rl_1']
        else:
            df['pure_reward_total'] = 0

        if 'counter_ai_rl_1' in df.columns and 'counter_ai_rl_2' in df.columns:
            df['total_counters'] = df['counter_ai_rl_1'] + df['counter_ai_rl_2']
        elif 'counter_ai_rl_1' in df.columns:
            df['total_counters'] = df['counter_ai_rl_1']
        else:
            df['total_counters'] = 0

    def _calculate_specialization_index(self, df: pd.DataFrame) -> float:
        """
        Calculates specialization as the averaged percentage asymmetry across 
        both deliveries and cuts:
        abs (del_a1 - del_a2) / total_del + (cuts_a2 - cuts_a1) / total_cuts) * 100 / 2
        """
        required_cols = ['deliver_ai_rl_1', 'deliver_ai_rl_2', 'cut_ai_rl_1', 'cut_ai_rl_2']
        if not all(col in df.columns for col in required_cols):
            return np.nan

        # Get totals for each agent
        del_a1 = df['deliver_ai_rl_1'].sum()
        del_a2 = df['deliver_ai_rl_2'].sum()
        cuts_a1 = df['cut_ai_rl_1'].sum()
        cuts_a2 = df['cut_ai_rl_2'].sum()

        total_del = del_a1 + del_a2
        total_cuts = cuts_a1 + cuts_a2

        # Case 1: No activity at all
        if total_cuts == 0 or total_del == 0:
            return 0.0

        # Case 2: Both actions occurred
        if total_cuts > 0 and total_del > 0:
            del_term = (del_a1 - del_a2) / total_del
            cuts_term = (cuts_a2 - cuts_a1) / total_cuts
            return (abs(del_term + cuts_term) * 100) / 2.0

        return None

    def _metric_value(self, df: pd.DataFrame, metric_name: str) -> float:
        if len(df) == 0:
            return np.nan

        if metric_name == 'total_deliveries':
            self._create_combined_metrics(df)
            return float(df['total_deliveries'].mean())
        if metric_name == 'total_counters':
            self._create_combined_metrics(df)
            return float(df['total_counters'].mean())
        if metric_name == 'specialization':
            return self._calculate_specialization_index(df)

        if metric_name in df.columns:
            return float(df[metric_name].mean())
        return np.nan

    def build_metric_grid(self, df: pd.DataFrame, metric_name: str,
                          difference_to_baseline: bool = True,
                          baseline_walk1: float = 1.0,
                          baseline_cut2: float = 1.0) -> np.ndarray:
        grid = np.full((len(self.cut2_values), len(self.walk1_values)), np.nan)

        baseline_df = self._filter_speed_configuration(df, baseline_walk1, baseline_cut2)
        baseline_value = self._metric_value(baseline_df, metric_name)
        use_difference = difference_to_baseline and np.isfinite(baseline_value)

        if difference_to_baseline and not np.isfinite(baseline_value):
            print(f"Warning: baseline speed pair ({baseline_walk1}, {baseline_cut2}) not found for {metric_name}; plotting raw values instead")

        for i, cut2 in enumerate(self.cut2_values):
            for j, walk1 in enumerate(self.walk1_values):
                config_df = self._filter_speed_configuration(df, walk1, cut2)
                metric_value = self._metric_value(config_df, metric_name)
                if not np.isfinite(metric_value):
                    continue

                if use_difference:
                    grid[i, j] = metric_value - baseline_value
                else:
                    grid[i, j] = metric_value

        return grid

    def summarize_training_ids_by_speed(self, df: pd.DataFrame) -> Tuple[Dict[Tuple[float, float], List[str]], Optional[str]]:
        training_col = self._resolve_training_id_column(df)
        summary: Dict[Tuple[float, float], List[str]] = {}

        for cut2 in self.cut2_values:
            for walk1 in self.walk1_values:
                if training_col is None:
                    summary[(walk1, cut2)] = []
                    continue
                config_df = self._filter_speed_configuration(df, walk1, cut2)
                if len(config_df) == 0:
                    summary[(walk1, cut2)] = []
                    continue
                training_ids = (
                    config_df[training_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                summary[(walk1, cut2)] = sorted(training_ids)

        return summary, training_col


class AbilityGridPlotter:
    """Create 2D heatmaps for walk1 (X) vs cut2 (Y) for a single map."""

    def __init__(self, analyzer: AbilityGridAnalyzer):
        self.analyzer = analyzer
        plt.style.use("default")

        import shutil

        USE_LATEX = shutil.which("latex") is not None

        plt.rcParams.update({
            "text.usetex": USE_LATEX,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        })

        self.delivery_ai_cmap = plt.cm.RdBu_r
        self.gap_ai_cmap = plt.cm.RdYlGn
        self.delivery_human_cmap = self.delivery_ai_cmap
        self.gap_human_cmap = self.gap_ai_cmap

    def save_marker_legend(self, output_path):

        fig, ax = plt.subplots(figsize=(4.5, 1.5))

        handles = [
            plt.Line2D(
                [0], [0],
                marker='s',
                linestyle='None',
                color='black',
                markerfacecolor='white',
                markeredgewidth=1.5,
                markersize=12,
                label='RL',
            ),
            plt.Line2D(
                [0], [0],
                marker='D',
                linestyle='None',
                color='black',
                markerfacecolor='white',
                markeredgewidth=1.5,
                markersize=12,
                label='Human',
            ),
        ]

        ax.legend(
            handles=handles,
            loc='center',
            ncol=2,
            frameon=False,
            fontsize=18,
            handletextpad=0.6,
            columnspacing=1.8,
        )

        ax.axis("off")

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )

        plt.close(fig)

    def _plot_grid(
        self,
        ax,
        grid: np.ndarray,
        title: str,
        cbar_label: str,
        decimals: int = 2,
        cmap=plt.cm.RdBu_r,
        fixed_max_abs: Optional[float] = None,
        cbar_ticks: Optional[List[float]] = None,
        interpolation: str = 'lanczos',
        filterrad: Optional[float] = None,
    ):
        max_abs = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
        if max_abs == 0:
            max_abs = 1.0
        if fixed_max_abs is not None:
            max_abs = fixed_max_abs

        # Display with descending Y so higher cut2 speeds appear at the top.
        grid_display = grid[::-1, :]

        imshow_kwargs = {
            'cmap': cmap,
            'aspect': 'equal',
            'vmin': -max_abs,
            'vmax': max_abs,
            'interpolation': interpolation,
        }
        if filterrad is not None:
            imshow_kwargs['filterrad'] = filterrad

        im = ax.imshow(grid_display, **imshow_kwargs)

        x_labels = [_format_speed_label(v) for v in self.analyzer.walk1_values]
        y_labels = [_format_speed_label(v) for v in reversed(self.analyzer.cut2_values)]

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        ax.set_ylim(len(y_labels) - 0.5, -0.5)
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(1)

        ax.set_xlabel(r'Agent 1 walking ability')
        ax.set_ylabel(r'Agent 2 cutting ability')
        ax.set_title(r'\textbf{' + title + '}', fontsize=21, pad=16)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        cbar.set_label(cbar_label)

        return im

    def _find_index(self, values: List[float], target: float, tol: float = 1e-6) -> Optional[int]:
        for idx, value in enumerate(values):
            if abs(value - target) <= tol:
                return idx
        return None

    def _speed_to_display_coord(self, walk1: float, cut2: float) -> Optional[Tuple[int, int]]:
        walk_idx = self._find_index(self.analyzer.walk1_values, walk1)
        cut_idx = self._find_index(self.analyzer.cut2_values, cut2)
        if walk_idx is None or cut_idx is None:
            return None

        x = walk_idx
        y = len(self.analyzer.cut2_values) - 1 - cut_idx
        return x, y

    def _overlay_human_markers(self, ax, metric_key: str, cmap, norm):
        map_key = _resolve_human_map_key(self.analyzer.map_name)
        if map_key is None:
            return

        human_data = HUMAN_RESULTS.get(map_key, {}).get(metric_key)
        if not human_data:
            return

        baseline_value = human_data.get((1.0, 1.0))
        if baseline_value is None:
            print(f'Human baseline (1.0, 1.0) missing for {metric_key}; using raw values.')

        coords: List[Tuple[int, int]] = []
        values: List[float] = []
        for (walk1, cut2), value in human_data.items():
            coord = self._speed_to_display_coord(walk1, cut2)
            if coord is None:
                print(f'Human marker skipped for walk1={walk1}, cut2={cut2} (not in grid).')
                continue
            coords.append(coord)
            if baseline_value is None:
                values.append(value)
            else:
                values.append(value - baseline_value)

        if not coords:
            return

        offset = HUMAN_MARKER_OFFSET
        xs, ys = zip(*coords)
        xs = [x + offset for x in xs]
        ys = [y + offset for y in ys]

        max_abs = max(abs(min(values)), abs(max(values))) if values else 1.0
        if max_abs == 0:
            max_abs = 1.0
        if norm is None:
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

        ax.scatter(
            xs,
            ys,
            c=values,
            cmap=cmap,
            norm=norm,
            marker='D',
            s=HUMAN_MARKER_SIZE,
            edgecolors='black',
            linewidths=1.6,
            zorder=6,
        )

    def create_color_grid_figure(
        self,
        data: pd.DataFrame,
        output_path: str = None,
        interpolation: str = 'lanczos',
        filterrad: Optional[float] = None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(15, 9.5), sharey=True)

        deliveries_grid = self.analyzer.build_metric_grid(data, 'total_deliveries', difference_to_baseline=True)
        specialization_grid = self.analyzer.build_metric_grid(data, 'specialization', difference_to_baseline=True)

        # Deliveries: keep red/blue diverging style.
        deliveries_im = self._plot_grid(
            axes[0],
            deliveries_grid,
            'Deliveries vs baseline (1.0, 1.0)',
            'Delta deliveries',
            decimals=1,
            cmap=self.delivery_ai_cmap,
            fixed_max_abs=6.0,
            cbar_ticks=[-4, -2, 0, 2, 4],
            interpolation=interpolation,
            filterrad=filterrad,
        )
        # Deliver/cut imbalance: use brown/green diverging style.
        gap_im = self._plot_grid(
            axes[1],
            specialization_grid,
            'Specialization vs baseline (1.0, 1.0)',
            'Delta specialization',
            decimals=2,
            cmap=self.gap_ai_cmap,
            fixed_max_abs=12.0,
            cbar_ticks=[-10, -5, 0, 5, 10],
            interpolation=interpolation,
            filterrad=filterrad,
        )

        self._overlay_human_markers(
            axes[0],
            'scores',
            cmap=self.delivery_human_cmap,
            norm=deliveries_im.norm,
        )
        self._overlay_human_markers(
            axes[1],
            'specialization',
            cmap=self.gap_human_cmap,
            norm=gap_im.norm,
        )

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f'Saved color grid figure to: {output_path}')

        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser for the 2D ability sweep."""
    parser = argparse.ArgumentParser(
        description='Generate 2D color grid plots for agent 1 walking speed vs agent 2 cutting speed',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, default=ms.DEFAULT_CHECKPOINT,
                        help="checkpoint_<n> folder of the simulated games (default 753), 'latest' or 'all'")
    parser.add_argument('--records_csv', type=str, default=None,
                        help='Re-use the per-game records cached by a previous run (skips crawling the tree)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--filename_suffix', type=str, default='')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--map_name', '--map_name_1', dest='map_name', type=str,
                        default='encouraged_division_of_labor_large',
                        help='Map to analyze')
    parser.add_argument('--game_type', type=str, choices=['classic', 'classic_collision', 'competition', 'competition_collision'],
                        default='classic_collision')
    parser.add_argument('--init_type', type=str, choices=['random_init', 'empty_init'], default='empty_init')
    parser.add_argument('--synergy', type=float, default=0.0)
    parser.add_argument('--specialization', type=float, default=None)
    parser.add_argument('--cluster', type=str, default=None, help=f'{list(ms.CLUSTER_DIRS)} or a literal base folder (default cuenca)')

    parser.add_argument('--walk1_min', type=float, default=0.1, help='Minimum value for agent 1 walking speed')
    parser.add_argument('--walk1_max', type=float, default=1.0, help='Maximum value for agent 1 walking speed')
    parser.add_argument('--walk1_step', type=float, default=0.1, help='Step for agent 1 walking speed')
    parser.add_argument('--cut2_min', type=float, default=0.1, help='Minimum value for agent 2 cutting speed')
    parser.add_argument('--cut2_max', type=float, default=1.0, help='Maximum value for agent 2 cutting speed')
    parser.add_argument('--cut2_step', type=float, default=0.1, help='Step for agent 2 cutting speed')
    parser.add_argument('--fixed_cut1', type=float, default=1.0, help='Fixed cutting speed for agent 1')
    parser.add_argument('--fixed_walk2', type=float, default=1.0, help='Fixed walking speed for agent 2')

    return parser


def main():
    """Main function to run the 2D ability-vs-ability analysis."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        cluster = args.cluster if args.cluster else 'cuenca'

        if args.output_dir is None:
            output_dir_base = "./figures/"
        else:
            output_dir_base = args.output_dir

        analyzer = AbilityGridAnalyzer(
            map_name=args.map_name,
            walk1_start=args.walk1_min,
            walk1_end=args.walk1_max,
            walk1_step=args.walk1_step,
            cut2_start=args.cut2_min,
            cut2_end=args.cut2_max,
            cut2_step=args.cut2_step,
            init_type=args.init_type,
            synergy=args.synergy,
            synergy_provided='--synergy' in sys.argv,
            specialization=args.specialization,
            game_type=args.game_type,
            cluster=cluster,
            study_name=args.study_name,
            fixed_cut1=args.fixed_cut1,
            fixed_walk2=args.fixed_walk2,
            checkpoint=args.checkpoint,
            verbose=args.verbose,
        )

        print('=' * 60)
        print('2D ABILITY GRID ANALYSIS - WALKING SPEED 1 vs CUTTING SPEED 2')
        print('=' * 60)
        print(f'Cluster: {cluster}')
        print(f'Map: {args.map_name}')
        print(f'Game type: {args.game_type}')
        print(f'Init type: {args.init_type}')
        print(f'Synergy scaling factor: {args.synergy}')
        print(f'Specialization: {args.specialization}')
        print(f'Agent 1 walking speed range: {args.walk1_min} to {args.walk1_max} (step {args.walk1_step})')
        print(f'Agent 2 cutting speed range: {args.cut2_min} to {args.cut2_max} (step {args.cut2_step})')
        print(f'Fixed speeds: agent 1 cut = {args.fixed_cut1}, agent 2 walk = {args.fixed_walk2}')
        if args.study_name:
            print(f'Study: {args.study_name}')

        print(f'Simulations root: {analyzer.raw_dir}  [checkpoint_{args.checkpoint}]')
        if args.records_csv:
            prepared_data = pd.read_csv(args.records_csv)
            print(f'{len(prepared_data)} simulated games read from {args.records_csv}')
        else:
            prepared_data = analyzer.load_experimental_data()

        plotter = AbilityGridPlotter(analyzer)

        output_dir = Path(output_dir_base)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = [
            'BLUR-comparison_human-ability_grid',
            args.map_name,
            f'walk1_{_format_speed_token(args.walk1_min)}-{_format_speed_token(args.walk1_max)}_step{_format_speed_token(args.walk1_step)}',
            f'cut2_{_format_speed_token(args.cut2_min)}-{_format_speed_token(args.cut2_max)}_step{_format_speed_token(args.cut2_step)}',
        ]

        if args.fixed_cut1 != 1.0:
            filename_parts.append(f'cut1_{_format_speed_token(args.fixed_cut1)}')
        if args.fixed_walk2 != 1.0:
            filename_parts.append(f'walk2_{_format_speed_token(args.fixed_walk2)}')
        if str(args.checkpoint) != ms.DEFAULT_CHECKPOINT:
            filename_parts.append(f'checkpoint{args.checkpoint}')
        if args.init_type != 'empty_init':
            filename_parts.append(args.init_type)
        if args.synergy != 0.0:
            filename_parts.append(f'synergy{_format_speed_token(args.synergy)}')
        if args.specialization is not None:
            filename_parts.append(f'lambda{_format_speed_token(args.specialization)}')
        if args.game_type != 'classic':
            filename_parts.append(args.game_type)
        if args.filename_suffix:
            filename_parts.append(args.filename_suffix)

        base_filename = '_'.join(filename_parts)
        if not args.records_csv:
            cache = output_dir / f'{base_filename}_sim_records.csv'
            prepared_data.to_csv(cache, index=False)
            print(f'Per-game MARL records cached -> {cache}  (re-use with --records_csv)')
        blurred_path = output_dir / f'blurred_{base_filename}.png'
        softer_path = output_dir / f'softer_{base_filename}.png'

        print('\n=== Creating 2D Ability Grid ===\n')
        fig = plotter.create_color_grid_figure(
            prepared_data,
            str(blurred_path),
            interpolation='bilinear',
        )
        plt.close(fig)

        fig = plotter.create_color_grid_figure(
            prepared_data,
            str(softer_path),
            interpolation='nearest',
        )
        plt.close(fig)

        legend_path = output_dir / "legend_markers.png"
        plotter.save_marker_legend(str(legend_path))

        training_summary, training_id_column = analyzer.summarize_training_ids_by_speed(prepared_data)
        print('\n=== Training coverage by speed ===\n')
        if training_id_column is None:
            print('No training id column found in data; cannot report per-speed training IDs.')
        else:
            print(f'Using training id column: {training_id_column}')
            for cut2 in reversed(analyzer.cut2_values):
                for walk1 in reversed(analyzer.walk1_values):
                    training_ids = training_summary[(walk1, cut2)]
                    walk_label = _format_speed_label(walk1)
                    cut_label = _format_speed_label(cut2)
                    if training_ids:
                        print(
                            f'walk1={walk_label}, cut2={cut_label}: '
                            f'{len(training_ids)} training(s) -> {", ".join(training_ids)}'
                        )
                    else:
                        print(f'walk1={walk_label}, cut2={cut_label}: 0 trainings')

        print('\n=== Analysis Complete ===\n')
        print(f'Generated figures: {blurred_path} and {softer_path}')
        print(f'X-axis values: {[ _format_speed_label(v) for v in analyzer.walk1_values ]}')
        print(f'Y-axis values: {[ _format_speed_label(v) for v in analyzer.cut2_values ]}')

    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
