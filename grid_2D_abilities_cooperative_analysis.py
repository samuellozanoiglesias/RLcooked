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
    python grid_2D_abilities_cooperative_analysis.py [options]

Examples:
    # Default analysis
    nohup python grid_2D_abilities_cooperative_analysis.py --episode_range final --num_episodes 100 > 2D_abilities_grid.log 2>&1 &
    
    # Analyze specific lambda data
    nohup python grid_2D_abilities_cooperative_analysis.py --episode_range final --specialization 0 --num_episodes 100 > 2D_abilities_grid.log 2>&1 &
    
    # Analyze empty_init data
    nohup python grid_2D_abilities_cooperative_analysis.py --episode_range final --init_type empty_init --num_episodes 100 > 2D_abilities_grid.log 2>&1 &

    # Analyze specific episode range around episode 500
    nohup python grid_2D_abilities_cooperative_analysis.py --episode_range specific --init_type empty_init --num_episodes 20 --cluster brigit --specialization 0.05 --synergy 0.40 --target_episode 100 > 2D_abilities_grid.log 2>&1 
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
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

    def _calculate_action_differentiation_index(self, df: pd.DataFrame) -> float:
        action_types = ['deliver', 'cut', 'salad', 'plate', 'raw_food']
        counts_1 = []
        counts_2 = []

        for action in action_types:
            col_1 = f'{action}_ai_rl_1'
            col_2 = f'{action}_ai_rl_2'
            counts_1.append(df[col_1].sum() if col_1 in df.columns else 0.0)
            counts_2.append(df[col_2].sum() if col_2 in df.columns else 0.0)

        t1 = sum(counts_1)
        t2 = sum(counts_2)
        if t1 == 0 or t2 == 0:
            return 0.0

        p1 = [c / t1 for c in counts_1]
        p2 = [c / t2 for c in counts_2]
        return 0.5 * sum(abs(p1[i] - p2[i]) for i in range(len(action_types)))

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
        if metric_name == 'action_differentiation':
            return self._calculate_action_differentiation_index(df)

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


class AbilityGridPlotter:
    """Create 2D heatmaps for walk1 (X) vs cut2 (Y) for a single map."""

    def __init__(self, analyzer: AbilityGridAnalyzer):
        self.analyzer = analyzer
        plt.style.use('default')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['font.size'] = 11

    def _plot_grid(self, ax, grid: np.ndarray, title: str, cbar_label: str, decimals: int = 2, cmap=plt.cm.RdBu_r):
        max_abs = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
        if max_abs == 0:
            max_abs = 1.0

        # Display with descending axes so the baseline (1.0, 1.0) is the upper-left cell.
        grid_display = grid[::-1, ::-1]

        im = ax.imshow(grid_display, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)

        x_labels = [_format_speed_label(v) for v in reversed(self.analyzer.walk1_values)]
        y_labels = [_format_speed_label(v) for v in reversed(self.analyzer.cut2_values)]

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        ax.set_xlabel('Agent 1 walking speed (X)')
        ax.set_ylabel('Agent 2 cutting speed (Y)')
        ax.set_title(title, fontsize=12, fontweight='bold')

        for i in range(grid_display.shape[0]):
            for j in range(grid_display.shape[1]):
                value = grid_display[i, j]
                if not np.isfinite(value):
                    continue
                color = 'white' if abs(value) > 0.5 * max_abs else 'black'
                ax.text(j, i, f'{value:.{decimals}f}', ha='center', va='center', color=color, fontsize=8)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(cbar_label)

    def create_color_grid_figure(self, data: pd.DataFrame, output_path: str = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(
            f"Ability Grid for map={self.analyzer.map_name}\n"
            "X: walk speed of agent 1 | Y: cut speed of agent 2",
            fontsize=14,
            fontweight='bold',
        )

        deliveries_grid = self.analyzer.build_metric_grid(data, 'total_deliveries', difference_to_baseline=True)
        specialization_grid = self.analyzer.build_metric_grid(data, 'specialization', difference_to_baseline=True)
        action_diff_grid = self.analyzer.build_metric_grid(data, 'action_differentiation', difference_to_baseline=True)

        # Deliveries: keep red/blue diverging style.
        self._plot_grid(axes[0], deliveries_grid, 'Deliveries vs baseline (1.0, 1.0)', 'Delta deliveries', decimals=1, cmap=plt.cm.RdBu_r)
        # Specialization and AD: use green/red diverging style as in full-grid analysis.
        self._plot_grid(axes[1], specialization_grid, 'Specialization vs baseline (1.0, 1.0)', 'Delta specialization', decimals=3, cmap=plt.cm.RdYlGn)
        self._plot_grid(axes[2], action_diff_grid, 'Action differentiation vs baseline (1.0, 1.0)', 'Delta action differentiation', decimals=3, cmap=plt.cm.RdYlGn)

        plt.tight_layout()

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

    parser.add_argument('--episode_range', choices=['all', 'final', 'average', 'specific'], default='final')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--target_episode', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--filename_suffix', type=str, default='')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--map_name', '--map_name_1', dest='map_name', type=str,
                        default='encouraged_division_of_labor_large',
                        help='Map to analyze')
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
        local_path = config.cluster_paths[cluster]

        if args.output_dir is None:
            if local_path:
                output_dir_base = f"{local_path}/data/samuel_lozano/cooked/grid_2d_abilities_figures"
            else:
                output_dir_base = "/data/samuel_lozano/cooked/grid_2d_abilities_figures"
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

        df = analyzer.load_experimental_data()
        prepared_data = analyzer.prepare_episode_data(
            df,
            episode_selection=args.episode_range,
            num_episodes=args.num_episodes,
            target_episode=args.target_episode,
        )

        plotter = AbilityGridPlotter(analyzer)

        output_dir = Path(output_dir_base)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = [
            'ability_grid',
            args.map_name,
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
        output_path = output_dir / f'{base_filename}.png'

        print('\n=== Creating 2D Ability Grid ===\n')
        fig = plotter.create_color_grid_figure(prepared_data, str(output_path))
        plt.close(fig)

        print('\n=== Analysis Complete ===\n')
        print(f'Generated figure: {output_path}')
        print(f'X-axis values: {[ _format_speed_label(v) for v in analyzer.walk1_values ]}')
        print(f'Y-axis values: {[ _format_speed_label(v) for v in analyzer.cut2_values ]}')

    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
