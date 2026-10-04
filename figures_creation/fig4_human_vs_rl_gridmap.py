#!/usr/bin/env python3
"""
Figure 4 -- MARL vs. Human score landscape over the 2D ability sweep.

Produces ONE combined figure (PDF + PNG) that reproduces the Fig4 layout
exactly:

    a  "Open Map"               (baseline_division_of_labor_large)
    b  "Partially-Blocked Map"  (encouraged_division_of_labor_large)
    shared colour bar  "Score difference with respect to full speeds (1.0, 1.0)"
    legend band        [square] MARL   [diamond] Human

For each map, the MARL heat-map shows the mean total deliveries of every
(walking speed of player 1, cutting speed of player 2) configuration minus
the (1.0, 1.0) configuration; human results are overlaid as diamonds using
the same colour scale (RdBu_r, clipped to [-4, +4]).

All positions, sizes, line widths and fonts (Computer Modern via LaTeX)
were measured from Fig4.pdf; the figure is written at the same page size
(1332.36 x 737.16 pt) without `bbox_inches='tight'`.

Usage:
    python fig4_human_vs_rl_gridmap.py [options]

Examples:
    # Default analysis
    nohup python fig4_human_vs_rl_gridmap.py --episode_range specific --init_type empty_init \
    --num_episodes 20 --specialization 0.25 --synergy 1.35 --target_episode 750 > fig4.log 2>&1 &

    # Analyze empty_init data for a specific study
    nohup python fig4_human_vs_rl_gridmap.py --episode_range final --init_type empty_init --study_name my_study > fig4.log 2>&1 &

    # Layout test without the project data (synthetic MARL grids)
    python fig4_human_vs_rl_gridmap.py --demo --output_dir ./figures
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D, ScaledTranslation
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _format_speed_label(value: float) -> str:
    text = f'{value:.2f}'.rstrip('0').rstrip('.')
    if '.' not in text:
        text += '.0'
    return text


def _format_speed_token(value: float) -> str:
    return _format_speed_label(value).replace('.', 'p')


# --------------------------------------------------------------------------
# Human results (raw scores; the figure shows score - score(1.0, 1.0)).
# NOTE: the (0.2, 1.0) and (0.1, 0.1) entries are not in the original
# script; their values were ESTIMATED from the diamond colours in Fig4.pdf
# (the Open-map ones are saturated at <= -4). Replace them with the exact
# measured scores.
# --------------------------------------------------------------------------
HUMAN_RESULTS = {
    'baseline_division_of_labor_large': {
        'scores': {
            (1.0, 1.0): 8.63,
            (0.7, 0.4): 6.71,
            (0.4, 0.2): 5.31,
            (0.2, 1.0): 4.8,   # ESTIMATED (colour saturated, <= -4)
            (0.1, 0.1): 4,   # ESTIMATED (colour saturated, <= -4)
        },
    },
    'encouraged_division_of_labor_large': {
        'scores': {
            (1.0, 1.0): 5.99,
            (0.7, 0.4): 7.66,
            (0.4, 0.2): 6.96,
            (0.2, 1.0): 7.13,    # ESTIMATED from colour (~ +1.6)
            (0.1, 0.1): 3.67,    # ESTIMATED from colour (~ +0.6)
        },
    },
}


def _resolve_human_map_key(map_name: str) -> Optional[str]:
    if map_name in HUMAN_RESULTS:
        return map_name
    for key in HUMAN_RESULTS:
        if key in map_name:
            return key
    return None


class AbilityGridAnalyzer:
    """Load and filter a single-map training set over a 2D ability sweep."""

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

    def build_metric_grid(self, df: pd.DataFrame, metric_name: str,
                          difference_to_baseline: bool = True,
                          baseline_walk1: float = 1.0,
                          baseline_cut2: float = 1.0) -> np.ndarray:
        grid = np.full((len(self.cut2_values), len(self.walk1_values)), np.nan)

        baseline_df = self._filter_speed_configuration(df, baseline_walk1, baseline_cut2)
        baseline_value = self._metric_value(baseline_df, metric_name)
        use_difference = difference_to_baseline and np.isfinite(baseline_value)

        if difference_to_baseline and not np.isfinite(baseline_value):
            print(f"Warning: baseline speed pair ({baseline_walk1}, {baseline_cut2}) not found for "
                  f"{metric_name}; plotting raw values instead")

        for i, cut2 in enumerate(self.cut2_values):
            for j, walk1 in enumerate(self.walk1_values):
                config_df = self._filter_speed_configuration(df, walk1, cut2)
                metric_value = self._metric_value(config_df, metric_name)
                if not np.isfinite(metric_value):
                    continue
                grid[i, j] = metric_value - baseline_value if use_difference else metric_value

        return grid

    def demo_grid(self, seed: int = 0) -> np.ndarray:
        """Synthetic score-difference grid (rows = cut2 ascending), layout testing only."""
        rng = np.random.default_rng(seed)
        w = np.array(self.walk1_values)[None, :]
        c = np.array(self.cut2_values)[:, None]
        grid = 6 * (np.minimum(w, 0.55) - 0.55) + 3 * (np.minimum(c, 0.4) - 0.4) \
            + rng.normal(0, 0.6, size=(len(self.cut2_values), len(self.walk1_values)))
        grid[-1, -1] = 0.0
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
                training_ids = config_df[training_col].dropna().astype(str).unique().tolist()
                summary[(walk1, cut2)] = sorted(training_ids)

        return summary, training_col


# ==========================================================================
# FIGURE 4 LAYOUT  (all values in points, y measured from the TOP of the page)
# ==========================================================================
PAGE_W, PAGE_H = 1332.36, 737.16            # 18.50 x 10.24 in

AXES_RECT = {                                # (left, right, top, bottom)
    'a': (123.52, 589.43, 68.10, 534.12),
    'b': (672.33, 1138.13, 68.31, 534.21),
}
CBAR_RECT = (1161.53, 1182.90, 63.04, 545.61)

VMAX = 4.0                                   # colour scale is clipped to [-4, +4]
CMAP = plt.cm.RdBu_r

SPINE_LW = 1.12
TICK_LEN, TICK_W = 8.4, 1.68
CBAR_OUTLINE_LW = 1.19
CBAR_TICK_LEN, CBAR_TICK_W = 5.45, 1.19
XTICK_PAD, YTICK_PAD = 4.46, 4.8
# small nudges of the tick labels (pt) to match the hand-placed labels of Fig4
XTICKLABEL_DX = {'a': 1.4, 'b': 0.35}
YTICKLABEL_DY = -1.5
TICK_FONT = ('cmr10', 30)

HUMAN_MARKER_SIZE = 287                      # scatter 's' (pt^2)
HUMAN_MARKER_LW = 2.24

# Free text: (TeX font, size pt, text, x, y, rotation); (x, y) is the
# left/baseline anchor of the text (rotation_mode='anchor').
TEXTS = {
    'letter_a':  ('cmbx10', 50, 'a', 64.83, 52.20, 0),
    'letter_b':  ('cmbx10', 50, 'b', 613.31, 52.20, 0),
    'title_a':   ('cmbx10', 38, 'Open Map', 255.66, 39.54, 0),
    'title_b':   ('cmbx10', 38, 'Partially-Blocked Map', 689.80, 38.64, 0),
    'xlabel_a':  ('cmr10', 36, 'Walking speed of player 1', 155.25, 615.15, 0),
    'xlabel_b':  ('cmr10', 36, 'Walking speed of player 1', 703.13, 615.06, 0),
    'ylabel_a':  ('cmr10', 36, 'Cutting speed of player 2', 41.84, 500.89, 90),
    'cbar_lab1': ('cmr10', 34.1, 'Score difference with respect', 1274.91, 515.19, 90),
    'cbar_lab2': ('cmr10', 34.1, 'to full speeds (1.0, 1.0)', 1315.78, 473.73, 90),
    'cbar_+4':   ('cmr10', 30, '+4', 1194.91, 73.32, 0),
    'cbar_+2':   ('cmr10', 30, '+2', 1196.03, 189.22, 0),
    'cbar_0':    ('cmr10', 30, '0', 1204.03, 315.95, 0),
    'cbar_-2':   ('cmr10', 30, '-2', 1200.28, 429.58, 0),
    'cbar_-4':   ('cmr10', 30, '-4', 1199.18, 550.18, 0),
    'legend_marl':  ('cmbx10', 40, 'MARL', 502.21, 701.55, 0),
    'legend_human': ('cmbx10', 40, 'Human', 723.02, 700.56, 0),
}

LEGEND_BOX = dict(x0=433.41, x1=887.82, top=648.555, bottom=725.41,
                  face='#f2f2f2', edge='#595959', lw=4.5)
LEGEND_SQUARE = dict(cx=473.58, cy=686.07, side=32.4, lw=4.0)
LEGEND_DIAMOND = dict(cx=695.07, cy=685.98, tip=34.96, lw=3.14)


def configure_style(use_latex: bool = True) -> bool:
    plt.style.use('default')
    mpl.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'axes.unicode_minus': False,
    })
    if use_latex:
        try:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family'] = 'serif'
            probe = plt.figure()
            probe.text(0.5, 0.5, tex('cmr10', 10, 'Test'))
            probe.canvas.draw()
            plt.close(probe)
            return True
        except Exception as exc:
            plt.close('all')
            print(f'[make_figure4] LaTeX unavailable ({type(exc).__name__}); '
                  f'falling back to a matched non-LaTeX serif style.')
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman', 'STIXGeneral', 'DejaVu Serif']
    return False


LATEX_OK = True


def tex(font: str, size: float, text: str) -> str:
    """Text set in a specific Computer Modern font (cmr10 / cmbx10) at `size` pt."""
    if LATEX_OK and mpl.rcParams['text.usetex']:
        return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'
    return text


def _text_kw(font: str, size: float) -> dict:
    kw = {'fontsize': size}
    if not mpl.rcParams['text.usetex'] and font.startswith('cmbx'):
        kw['fontweight'] = 'bold'
    return kw


class Figure4Plotter:
    """Assemble the complete Fig4 in a single matplotlib figure."""

    def __init__(self, walk1_values: List[float], cut2_values: List[float]):
        self.walk1_values = walk1_values
        self.cut2_values = cut2_values

    # -- helpers ----------------------------------------------------------
    def _page(self, fig):
        """Transform from page points (y from top) to display coordinates."""
        return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans

    @staticmethod
    def _rect_to_fig(rect):
        L, R, T, B = rect
        return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]

    def _speed_to_display_coord(self, walk1: float, cut2: float) -> Optional[Tuple[int, int]]:
        def find(values, target):
            for idx, value in enumerate(values):
                if abs(value - target) <= 1e-6:
                    return idx
            return None
        walk_idx = find(self.walk1_values, walk1)
        cut_idx = find(self.cut2_values, cut2)
        if walk_idx is None or cut_idx is None:
            return None
        return walk_idx, len(self.cut2_values) - 1 - cut_idx

    # -- panels -----------------------------------------------------------
    def _plot_grid(self, ax, grid: np.ndarray, show_ylabels: bool, key: str = 'a'):
        # Display with descending Y so higher cut2 speeds appear at the top.
        im = ax.imshow(grid[::-1, :], cmap=CMAP, vmin=-VMAX, vmax=VMAX,
                       interpolation='nearest', aspect='auto')

        nx, ny = len(self.walk1_values), len(self.cut2_values)

        def lab(v):  # label every 0.2 (0.2, 0.4, ..., 1.0), tick every cell
            return tex(*TICK_FONT, _format_speed_label(v)) if round(v * 10) % 2 == 0 else ''

        ax.set_xticks(range(nx))
        ax.set_xticklabels([lab(v) for v in self.walk1_values])
        ax.set_yticks(range(ny))
        ax.set_yticklabels([lab(v) for v in reversed(self.cut2_values)])
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(ny - 0.5, -0.5)

        ax.tick_params(axis='both', which='major', direction='out', color='black',
                       length=TICK_LEN, width=TICK_W, labelsize=TICK_FONT[1])
        ax.tick_params(axis='x', pad=XTICK_PAD)
        ax.tick_params(axis='y', pad=YTICK_PAD, labelleft=show_ylabels)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
            spine.set_edgecolor('black')

        fig = ax.figure
        dx = ScaledTranslation(XTICKLABEL_DX.get(key, 0.0) / 72, 0, fig.dpi_scale_trans)
        dy = ScaledTranslation(0, -YTICKLABEL_DY / 72, fig.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + dx)
        for label in ax.get_yticklabels():
            label.set_transform(label.get_transform() + dy)
        return im

    def _overlay_human_markers(self, ax, map_name: str, norm):
        map_key = _resolve_human_map_key(map_name)
        if map_key is None:
            print(f'No human results for map {map_name}.')
            return
        human_data = HUMAN_RESULTS[map_key].get('scores', {})
        baseline_value = human_data.get((1.0, 1.0))
        if baseline_value is None:
            print('Human baseline (1.0, 1.0) missing; using raw values.')

        xs, ys, values = [], [], []
        for (walk1, cut2), value in human_data.items():
            coord = self._speed_to_display_coord(walk1, cut2)
            if coord is None:
                print(f'Human marker skipped for walk1={walk1}, cut2={cut2} (not in grid).')
                continue
            xs.append(coord[0])
            ys.append(coord[1])
            values.append(value if baseline_value is None else value - baseline_value)

        if xs:
            # Use white diamond outlines when the diamond fill is very dark.
            # This keeps the marker visible over the dark-blue part of RdBu_r.
            edgecolors = []
            for value in values:
                rgba = CMAP(norm(value))
                r, g, b = rgba[:3]
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                edgecolors.append('white' if luminance < 0.42 else 'black')

            ax.scatter(xs, ys, c=values, cmap=CMAP, norm=norm, marker='D',
                       s=HUMAN_MARKER_SIZE, edgecolors=edgecolors,
                       linewidths=HUMAN_MARKER_LW, zorder=6, clip_on=False)

    # -- full figure ------------------------------------------------------
    def create_figure(self, grids: Dict[str, np.ndarray], map_names: Dict[str, str],
                      output_path: Optional[str] = None) -> plt.Figure:
        fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
        page = self._page(fig)

        im = None
        for key in ('a', 'b'):
            ax = fig.add_axes(self._rect_to_fig(AXES_RECT[key]))
            im = self._plot_grid(ax, grids[key], show_ylabels=(key == 'a'), key=key)
            self._overlay_human_markers(ax, map_names[key], im.norm)

        # Shared colour bar (labels are placed as free text, as in Fig4)
        cax = fig.add_axes(self._rect_to_fig(CBAR_RECT))
        cbar = fig.colorbar(im, cax=cax, ticks=[-4, -2, 0, 2, 4])
        cbar.ax.set_yticklabels([])
        cbar.ax.tick_params(length=CBAR_TICK_LEN, width=CBAR_TICK_W, color='black')
        cbar.outline.set_linewidth(CBAR_OUTLINE_LW)
        cbar.outline.set_edgecolor('black')

        # Legend band
        lb = LEGEND_BOX
        fig.add_artist(Rectangle((lb['x0'], lb['top']), lb['x1'] - lb['x0'], lb['bottom'] - lb['top'],
                                 transform=page, facecolor=lb['face'], edgecolor=lb['edge'],
                                 linewidth=lb['lw'], zorder=1))
        sq = LEGEND_SQUARE
        h = sq['side'] / 2
        fig.add_artist(Rectangle((sq['cx'] - h, sq['cy'] - h), sq['side'], sq['side'], transform=page,
                                 facecolor='white', edgecolor='black', linewidth=sq['lw'],
                                 joinstyle='miter', zorder=2))
        dm = LEGEND_DIAMOND
        r = dm['tip'] / 2
        fig.add_artist(Polygon([(dm['cx'], dm['cy'] - r), (dm['cx'] + r, dm['cy']),
                                (dm['cx'], dm['cy'] + r), (dm['cx'] - r, dm['cy'])],
                               closed=True, transform=page, facecolor='white', edgecolor='black',
                               linewidth=dm['lw'], joinstyle='miter', zorder=2))

        # Free text (panel letters, titles, axis labels, colour-bar labels, legend)
        for font, size, text, x, y, rot in TEXTS.values():
            fig.text(x, y, tex(font, size, text), transform=page, ha='left', va='baseline',
                     rotation=rot, rotation_mode='anchor', zorder=3, **_text_kw(font, size))

        if output_path:
            # NB: no bbox_inches='tight' -- the page size is part of the layout.
            fig.savefig(output_path)
            png_path = os.path.splitext(output_path)[0] + '.png'
            fig.savefig(png_path)
            print(f'Saved Figure 4 to: {output_path} and {png_path}')

        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser for the 2D ability sweep."""
    parser = argparse.ArgumentParser(
        description='Generate Figure 4: MARL vs. human score landscape (walking speed 1 vs cutting speed 2)',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--episode_range', choices=['all', 'final', 'average', 'specific'], default='final')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--target_episode', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--filename_suffix', type=str, default='')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--map_open', type=str, default='baseline_division_of_labor_large',
                        help='Map shown in panel a ("Open Map")')
    parser.add_argument('--map_blocked', type=str, default='encouraged_division_of_labor_large',
                        help='Map shown in panel b ("Partially-Blocked Map")')
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
    parser.add_argument('--no_latex', action='store_true', help='Do not use LaTeX for text')
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic MARL grids (layout test, no project data needed)')

    return parser


def main():
    """Main function: build both MARL grids and assemble Figure 4."""
    global LATEX_OK
    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        LATEX_OK = configure_style(not args.no_latex)
        cluster = args.cluster if args.cluster else 'cuenca'

        if args.output_dir is not None:
            output_dir_base = args.output_dir
        else:
            output_dir_base = './figures'

        map_names = {'a': args.map_open, 'b': args.map_blocked}
        common = dict(
            walk1_start=args.walk1_min, walk1_end=args.walk1_max, walk1_step=args.walk1_step,
            cut2_start=args.cut2_min, cut2_end=args.cut2_max, cut2_step=args.cut2_step,
            init_type=args.init_type, synergy=args.synergy,
            synergy_provided='--synergy' in sys.argv, specialization=args.specialization,
            game_type=args.game_type, cluster=cluster, study_name=args.study_name,
            fixed_cut1=args.fixed_cut1, fixed_walk2=args.fixed_walk2, demo=args.demo,
        )

        print('=' * 60)
        print('FIGURE 4 - WALKING SPEED 1 vs CUTTING SPEED 2 (MARL vs HUMAN)')
        print('=' * 60)
        print(f'Cluster: {cluster} | Game type: {args.game_type} | Init type: {args.init_type}')
        print(f'Maps: a = {args.map_open}, b = {args.map_blocked}')
        print(f'Agent 1 walking speed: {args.walk1_min}-{args.walk1_max} (step {args.walk1_step})')
        print(f'Agent 2 cutting speed: {args.cut2_min}-{args.cut2_max} (step {args.cut2_step})')

        grids: Dict[str, np.ndarray] = {}
        analyzers: Dict[str, AbilityGridAnalyzer] = {}
        for i, (key, map_name) in enumerate(map_names.items()):
            analyzer = AbilityGridAnalyzer(map_name=map_name, **common)
            analyzers[key] = analyzer
            if args.demo:
                grids[key] = analyzer.demo_grid(seed=i)
                continue
            df = analyzer.load_experimental_data()
            prepared = analyzer.prepare_episode_data(
                df, episode_selection=args.episode_range,
                num_episodes=args.num_episodes, target_episode=args.target_episode)
            grids[key] = analyzer.build_metric_grid(prepared, 'total_deliveries', difference_to_baseline=True)

            summary, id_col = analyzer.summarize_training_ids_by_speed(prepared)
            print(f'\n=== Training coverage by speed ({map_name}) ===')
            if id_col is None:
                print('No training id column found in data; cannot report per-speed training IDs.')
            else:
                for cut2 in reversed(analyzer.cut2_values):
                    for walk1 in reversed(analyzer.walk1_values):
                        ids = summary[(walk1, cut2)]
                        print(f'walk1={_format_speed_label(walk1)}, cut2={_format_speed_label(cut2)}: '
                              f'{len(ids)} training(s)' + (f' -> {", ".join(ids)}' if ids else ''))

        output_dir = Path(output_dir_base)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = ['figure4_human_vs_rl_gridmap']
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
        if args.demo:
            filename_parts.append('DEMO')
        if args.filename_suffix:
            filename_parts.append(args.filename_suffix)
        out_path = output_dir / ('_'.join(filename_parts) + '.pdf')

        print('\n=== Creating Figure 4 ===\n')
        plotter = Figure4Plotter(analyzers['a'].walk1_values, analyzers['a'].cut2_values)
        fig = plotter.create_figure(grids, map_names, str(out_path))
        plt.close(fig)
        print('\n=== Analysis Complete ===\n')

    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
