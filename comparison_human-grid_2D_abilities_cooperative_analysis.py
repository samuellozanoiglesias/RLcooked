#!/usr/bin/env python3
"""
2D Grid Cooperative Analysis Figure Generator

This script creates 2D color grid plots combining two map configurations
(side-by-side) and agent ability configurations (X-axis). The visualization
shows how different combinations perform across performance metrics.

Hardcoded map configurations (side-by-side):
- baseline_division_of_labor_large (baseline)
- encouraged_division_of_labor_large (encouraged)

Agent ability configurations tested (X-axis):
- X=1.0: Agent 1 (1.0, 1.0), Agent 2 (1.0, 1.0) - baseline abilities
- X=0.8: Agent 1 (0.8, 1.0), Agent 2 (1.0, 0.8)
- X=0.6: Agent 1 (0.6, 1.0), Agent 2 (1.0, 0.6)
- X=0.4: Agent 1 (0.4, 1.0), Agent 2 (1.0, 0.4)
- X=0.2: Agent 1 (0.2, 1.0), Agent 2 (1.0, 0.2)

Generates 2 color grid figures with shared colorbars:
1. Delta deliveries vs baseline (1.0, 1.0) for both maps (left/right)
2. Delta deliver/cut gap vs baseline (1.0, 1.0) for both maps (left/right)

Color coding:
- White: Baseline condition (difference=0)
- Blue: Better performance than baseline
- Red: Worse performance than baseline

Usage:
    python comparison_human-grid_2D_abilities_cooperative_analysis.py [options]

Examples:
    # Analyze specific episode range around episode 750
    nohup python comparison_human-grid_2D_abilities_cooperative_analysis.py --episode_range specific --init_type empty_init --num_episodes 20 --cluster brigit --specialization 0.25 --synergy 1.35 --target_episode 750 > 2D_abilities_grid.log 2>&1 
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

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import DataProcessor, AnalysisConfig
from training_configuration.path_utils import generate_save_directory


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
        'deliver_cut_gap': {
            (1.0, 1.0): 12.286,
            (0.7, 0.4): 9.038,
            (0.4, 0.2): 7.371,
        },
    },
    'encouraged_division_of_labor_large': {
        'scores': {
            (1.0, 1.0): 6.357,
            (0.7, 0.4): 8.443,
            (0.4, 0.2): 8.049,
        },
        'deliver_cut_gap': {
            (1.0, 1.0): 11.729,
            (0.7, 0.4): 17.514,
            (0.4, 0.2): 17.057,
        },
    },
}
HUMAN_MARKER_OFFSET = 0.0
HUMAN_MARKER_SIZE = 150

COMPARISON_MAPS = (
    'baseline_division_of_labor_large',
    'encouraged_division_of_labor_large',
)

MAP_DISPLAY_NAMES = {
    'baseline_division_of_labor_large': 'Baseline division of labor (large)',
    'encouraged_division_of_labor_large': 'Encouraged division of labor (large)',
}


def _resolve_human_map_key(map_name: str) -> Optional[str]:
    if map_name in HUMAN_RESULTS:
        return map_name
    for key in HUMAN_RESULTS:
        if key in map_name:
            return key
    return None


def _format_map_display_name(map_name: str) -> str:
    if map_name in MAP_DISPLAY_NAMES:
        return MAP_DISPLAY_NAMES[map_name]
    return map_name.replace('_', ' ')


class AbilityGridAnalyzer:
    """Load and filter a single-map training set over a 2D ability sweep."""

    def __init__(self, map_name: str, walk1_start: float, walk1_end: float, walk1_step: float,
                 cut2_start: float, cut2_end: float, cut2_step: float,
                 init_type: str = 'empty_init', synergy: float = 0.0,
                 synergy_provided: bool = False, specialization: Optional[float] = None,
                 game_type: str = 'classic', cluster: str = 'cuenca',
                 study_name: Optional[str] = None, fixed_cut1: float = 1.0,
                 fixed_walk2: float = 1.0):
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor(self.config)
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

        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        self.local_path = self.config.cluster_paths[cluster]

        self.walk1_values = _inclusive_float_range(walk1_start, walk1_end, walk1_step)
        self.cut2_values = _inclusive_float_range(cut2_start, cut2_end, cut2_step)

        self.raw_dir = self._build_raw_dir()

    def _build_raw_dir(self) -> str:
        synergy_folder = 'synergy_0' if self.synergy == 0.0 else f'synergy_{self.synergy:.2f}'
        if self.specialization is None:
            spec_folder = 'specialized_0'
        elif self.specialization == 0:
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
            return generate_save_directory(
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
        if 'training_id' in df.columns:
            return 'training_id'
        if 'timestamp' in df.columns:
            return 'timestamp'
        if 'run_id' in df.columns:
            return 'run_id'
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
        if not all(col in df.columns for col in ['cut_ai_rl_1', 'cut_ai_rl_2', 'deliver_ai_rl_1', 'deliver_ai_rl_2']):
            return 0.0

        n1c = df['cut_ai_rl_1'].sum()
        n2c = df['cut_ai_rl_2'].sum()
        n1d = df['deliver_ai_rl_1'].sum()
        n2d = df['deliver_ai_rl_2'].sum()

        t1 = n1c + n1d
        t2 = n2c + n2d
        if t1 == 0 or t2 == 0:
            return 0.0

        p1 = n1c / t1
        p2 = n2c / t2
        return float(np.abs(p1 - p2))

    def _calculate_deliver_cut_gap(self, df: pd.DataFrame) -> float:
        required_cols = ['deliver_ai_rl_1', 'deliver_ai_rl_2', 'cut_ai_rl_1', 'cut_ai_rl_2']
        if not all(col in df.columns for col in required_cols):
            return np.nan

        gap = (
            (df['deliver_ai_rl_1'] - df['deliver_ai_rl_2']).abs() +
            (df['cut_ai_rl_1'] - df['cut_ai_rl_2']).abs()
        )
        return float(gap.mean())

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
        if metric_name == 'deliver_cut_gap':
            return self._calculate_deliver_cut_gap(df)

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
                prepared_frames.append(training_df[(training_df['episode'] >= start_episode) & (training_df['episode'] <= end_episode)])
            else:
                raise ValueError(f"Invalid episode_selection: {episode_selection}")

        if not prepared_frames:
            return df.iloc[0:0].copy()

        return pd.concat(prepared_frames, ignore_index=True)

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
        plt.style.use('default')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['font.size'] = 11
        self.delivery_ai_cmap = plt.cm.RdBu_r
        self.gap_ai_cmap = plt.cm.RdYlGn
        self.delivery_human_cmap = self.delivery_ai_cmap
        self.gap_human_cmap = self.gap_ai_cmap
        self.specialization_cmap = plt.cm.RdYlGn

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
        show_colorbar: bool = True,
        title_pad: int = 34,
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

        ax.set_xlabel('Agent 1 walking speed (X)')
        ax.set_ylabel('Agent 2 cutting speed (Y)')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=title_pad)

        if show_colorbar:
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
        fig.suptitle(
            f"Ability Grid for map={self.analyzer.map_name}\n"
            "X: walk speed of agent 1 | Y: cut speed of agent 2",
            fontsize=14,
            fontweight='bold',
            y=0.98,
        )

        deliveries_grid = self.analyzer.build_metric_grid(data, 'total_deliveries', difference_to_baseline=True)
        deliver_cut_gap_grid = self.analyzer.build_metric_grid(data, 'deliver_cut_gap', difference_to_baseline=True)

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
            deliver_cut_gap_grid,
            'Deliver/cut gap vs baseline (1.0, 1.0)',
            'Delta abs(deliver diff) + abs(cut diff)',
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
            'deliver_cut_gap',
            cmap=self.gap_human_cmap,
            norm=gap_im.norm,
        )

        handles = [
            plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='none',
                       markeredgecolor='black', markersize=9, label='RL (grid cells)'),
            plt.Line2D([0], [0], marker='D', color='none', markerfacecolor='none',
                       markeredgecolor='black', markersize=9, label='Human (diamonds)'),
        ]
        for ax in axes:
            ax.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.1))

        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f'Saved color grid figure to: {output_path}')

        return fig


def _grid_max_abs(grid: np.ndarray) -> float:
    if np.isfinite(grid).any():
        return float(np.nanmax(np.abs(grid)))
    return 0.0


def _map_filename_token(map_name: str) -> str:
    token = map_name.replace('_division_of_labor_large', '')
    return token.replace('_', '-')


def create_shared_metric_figure(
    entries: List[Tuple[AbilityGridAnalyzer, pd.DataFrame]],
    metric_name: str,
    suptitle: str,
    cbar_label: str,
    cmap,
    fixed_max_abs: Optional[float] = None,
    cbar_ticks: Optional[List[float]] = None,
    interpolation: str = 'lanczos',
    filterrad: Optional[float] = None,
    overlay_human: bool = False,
    human_metric_key: Optional[str] = None,
) -> plt.Figure:
    if not entries:
        raise ValueError('entries must not be empty')

    grids: List[Tuple[AbilityGridAnalyzer, pd.DataFrame, np.ndarray]] = []
    max_abs = 0.0
    for analyzer, data in entries:
        grid = analyzer.build_metric_grid(data, metric_name, difference_to_baseline=True)
        grids.append((analyzer, data, grid))
        max_abs = max(max_abs, _grid_max_abs(grid))

    if max_abs == 0:
        max_abs = 1.0
    if fixed_max_abs is not None:
        max_abs = fixed_max_abs

    fig, axes = plt.subplots(1, len(grids), figsize=(7.5 * len(grids), 9.5), sharey=True)
    if len(grids) == 1:
        axes = [axes]

    im = None
    for idx, (ax, (analyzer, _data, grid)) in enumerate(zip(axes, grids)):
        plotter = AbilityGridPlotter(analyzer)
        im = plotter._plot_grid(
            ax,
            grid,
            _format_map_display_name(analyzer.map_name),
            cbar_label,
            cmap=cmap,
            fixed_max_abs=max_abs,
            cbar_ticks=cbar_ticks,
            interpolation=interpolation,
            filterrad=filterrad,
            show_colorbar=False,
            title_pad=18,
        )
        if idx > 0:
            ax.set_ylabel('')

        if overlay_human:
            metric_key = human_metric_key or metric_name
            plotter._overlay_human_markers(ax, metric_key, cmap=cmap, norm=im.norm)

    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)

    if overlay_human:
        handles = [
            plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='none',
                       markeredgecolor='black', markersize=9, label='RL (grid cells)'),
            plt.Line2D([0], [0], marker='D', color='none', markerfacecolor='none',
                       markeredgecolor='black', markersize=9, label='Human (diamonds)'),
        ]
        fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))

    top_margin = 0.88 if overlay_human else 0.9
    fig.tight_layout(rect=[0.0, 0.0, 0.9, top_margin])

    cbar_left = 0.92
    cbar_width = 0.02
    cbar_bottom = 0.12
    cbar_top = top_margin - 0.02
    if cbar_top <= cbar_bottom:
        cbar_bottom = 0.1
        cbar_top = 0.9
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_top - cbar_bottom])
    cbar = fig.colorbar(im, cax=cax)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label)

    return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser for the 2D ability sweep."""
    parser = argparse.ArgumentParser(
        description='Generate 2D color grid plots for two hardcoded maps with shared colorbars',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--episode_range', choices=['all', 'final', 'average', 'specific'], default='final')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--target_episode', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--filename_suffix', type=str, default='')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--map_name', '--map_name_1', dest='map_name', type=str,
                        default='encouraged_division_of_labor_large',
                        help='(ignored; uses hardcoded maps)')
    parser.add_argument('--game_type', type=str, choices=['classic', 'classic_collision'], default='classic_collision')
    parser.add_argument('--init_type', type=str, choices=['random_init', 'empty_init'], default='empty_init')
    parser.add_argument('--synergy', type=float, default=0.0)
    parser.add_argument('--specialization', type=float, default=None)
    parser.add_argument('--cluster', type=str, default=None)

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
        config = AnalysisConfig()
        if cluster not in config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(config.cluster_paths.keys())}")
        local_path = config.cluster_paths[cluster]

        if args.output_dir is None:
            if local_path:
                output_dir_base = f"{local_path}/data/samuel_lozano/cooked/grid_2d_abilities_figures"
            else:
                output_dir_base = "/data/samuel_lozano/cooked/grid_2d_abilities_figures"
        else:
            output_dir_base = args.output_dir

        comparison_maps = list(COMPARISON_MAPS)

        print('=' * 60)
        print('2D ABILITY GRID COMPARISON - WALKING SPEED 1 vs CUTTING SPEED 2')
        print('=' * 60)
        print(f'Cluster: {cluster}')
        print(f'Maps (hardcoded): {", ".join(comparison_maps)}')
        print(f'Game type: {args.game_type}')
        print(f'Init type: {args.init_type}')
        print(f'Synergy scaling factor: {args.synergy}')
        print(f'Specialization: {args.specialization}')
        print(f'Agent 1 walking speed range: {args.walk1_min} to {args.walk1_max} (step {args.walk1_step})')
        print(f'Agent 2 cutting speed range: {args.cut2_min} to {args.cut2_max} (step {args.cut2_step})')
        print(f'Fixed speeds: agent 1 cut = {args.fixed_cut1}, agent 2 walk = {args.fixed_walk2}')
        if args.study_name:
            print(f'Study: {args.study_name}')

        analysis_entries: List[Tuple[AbilityGridAnalyzer, pd.DataFrame]] = []
        for map_name in comparison_maps:
            analyzer = AbilityGridAnalyzer(
                map_name=map_name,
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
            )

            print(f'Loading data for map: {map_name}')
            df = analyzer.load_experimental_data()
            prepared_data = analyzer.prepare_episode_data(
                df,
                episode_selection=args.episode_range,
                num_episodes=args.num_episodes,
                target_episode=args.target_episode,
            )
            analysis_entries.append((analyzer, prepared_data))

        output_dir = Path(output_dir_base)
        output_dir.mkdir(parents=True, exist_ok=True)

        map_tag = 'vs_'.join(_map_filename_token(name) for name in comparison_maps)

        filename_parts = [
            'BLUR-comparison_human-ability_grid',
            f'maps_{map_tag}',
            f'walk1_{_format_speed_token(args.walk1_min)}-{_format_speed_token(args.walk1_max)}_step{_format_speed_token(args.walk1_step)}',
            f'cut2_{_format_speed_token(args.cut2_min)}-{_format_speed_token(args.cut2_max)}_step{_format_speed_token(args.cut2_step)}',
        ]

        if args.fixed_cut1 != 1.0:
            filename_parts.append(f'cut1_{_format_speed_token(args.fixed_cut1)}')
        if args.fixed_walk2 != 1.0:
            filename_parts.append(f'walk2_{_format_speed_token(args.fixed_walk2)}')
        if args.episode_range != 'final':
            filename_parts.append(args.episode_range)
        if args.num_episodes != 100:
            filename_parts.append(f'{args.num_episodes}ep')
        if args.target_episode is not None:
            filename_parts.append(f'target{args.target_episode}')
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
        deliveries_blurred_path = output_dir / f'blurred_deliveries_{base_filename}.png'
        deliveries_softer_path = output_dir / f'softer_deliveries_{base_filename}.png'
        deliver_cut_gap_blurred_path = output_dir / f'blurred_deliver_cut_gap_{base_filename}.png'
        deliver_cut_gap_softer_path = output_dir / f'softer_deliver_cut_gap_{base_filename}.png'

        print('\n=== Creating shared deliveries grid ===\n')
        deliveries_title = (
            'Ability Grid: Delta Deliveries vs baseline (1.0, 1.0)\n'
            'Left/Right: Baseline vs Encouraged maps'
        )
        fig = create_shared_metric_figure(
            analysis_entries,
            metric_name='total_deliveries',
            suptitle=deliveries_title,
            cbar_label='Delta deliveries',
            cmap=plt.cm.RdBu_r,
            fixed_max_abs=6.0,
            cbar_ticks=[-4, -2, 0, 2, 4],
            interpolation='bilinear',
            overlay_human=True,
            human_metric_key='scores',
        )
        fig.savefig(deliveries_blurred_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved deliveries figure to: {deliveries_blurred_path}')

        fig = create_shared_metric_figure(
            analysis_entries,
            metric_name='total_deliveries',
            suptitle=deliveries_title,
            cbar_label='Delta deliveries',
            cmap=plt.cm.RdBu_r,
            fixed_max_abs=6.0,
            cbar_ticks=[-4, -2, 0, 2, 4],
            interpolation='nearest',
            overlay_human=True,
            human_metric_key='scores',
        )
        fig.savefig(deliveries_softer_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved deliveries figure to: {deliveries_softer_path}')

        print('\n=== Creating shared deliver/cut gap grid ===\n')
        deliver_cut_gap_title = (
            'Ability Grid: Delta Deliver/Cut Gap vs baseline (1.0, 1.0)\n'
            'Left/Right: Baseline vs Encouraged maps'
        )
        fig = create_shared_metric_figure(
            analysis_entries,
            metric_name='deliver_cut_gap',
            suptitle=deliver_cut_gap_title,
            cbar_label='Delta abs(deliver diff) + abs(cut diff)',
            cmap=plt.cm.RdYlGn,
            fixed_max_abs=12.0,
            cbar_ticks=[-10, -5, 0, 5, 10],
            interpolation='bilinear',
            overlay_human=True,
            human_metric_key='deliver_cut_gap',
        )
        fig.savefig(deliver_cut_gap_blurred_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved deliver/cut gap figure to: {deliver_cut_gap_blurred_path}')

        fig = create_shared_metric_figure(
            analysis_entries,
            metric_name='deliver_cut_gap',
            suptitle=deliver_cut_gap_title,
            cbar_label='Delta abs(deliver diff) + abs(cut diff)',
            cmap=plt.cm.RdYlGn,
            fixed_max_abs=12.0,
            cbar_ticks=[-10, -5, 0, 5, 10],
            interpolation='nearest',
            overlay_human=True,
            human_metric_key='deliver_cut_gap',
        )
        fig.savefig(deliver_cut_gap_softer_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved deliver/cut gap figure to: {deliver_cut_gap_softer_path}')

        print('\n=== Training coverage by speed ===\n')
        for analyzer, prepared_data in analysis_entries:
            training_summary, training_id_column = analyzer.summarize_training_ids_by_speed(prepared_data)
            print(f'-- Map: {analyzer.map_name}')
            if training_id_column is None:
                print('No training id column found in data; cannot report per-speed training IDs.')
                continue
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
        print('Generated figures:')
        print(f'- {deliveries_blurred_path}')
        print(f'- {deliveries_softer_path}')
        print(f'- {deliver_cut_gap_blurred_path}')
        print(f'- {deliver_cut_gap_softer_path}')
        if analysis_entries:
            sample_analyzer = analysis_entries[0][0]
            print(f'X-axis values: {[ _format_speed_label(v) for v in sample_analyzer.walk1_values ]}')
            print(f'Y-axis values: {[ _format_speed_label(v) for v in sample_analyzer.cut2_values ]}')

    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
