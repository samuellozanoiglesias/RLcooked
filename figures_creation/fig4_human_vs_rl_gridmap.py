#!/usr/bin/env python3
"""
Figure 4 -- MARL vs. Human score landscape over the 2D ability sweep.

Produces ONE combined figure (PDF + PNG) that reproduces the Fig4 layout
exactly:

    a  "Open Map"               (baseline_division_of_labor_large)
    b  "Partially-Blocked Map"  (encouraged_division_of_labor_large)
    shared colour bar  "Score difference with respect to full speeds (100%, 100%)"
    legend band        [square] MARL   [diamond] Human

For each map, the MARL heat-map shows the mean total deliveries (simulated games) of
every (walking speed of player 1, cutting speed of player 2) configuration minus
the (100%, 100%) configuration; human results are overlaid as diamonds using
the same colour scale (RdBu_r, clipped to [-4, +4]).

MARL data: simulated games of the simulation tree (marl_simulations.py, same folder)
    <base>/data/samuel_lozano/cooked/<game_type>/simulations/<init_type>/map_<map>/
        synergy_<s>/specialized_<l>/abilities_<walk1>_<cut2>/Training_*/checkpoint_753/simulation_*/
with walk1 = walking speed of ai_rl_1 and cut2 = cutting speed of ai_rl_2 (the other two
speeds are 1.0). Every Training_* and every simulation_* folder is used; each grid cell is the
mean over all simulated games of its abilities folder.

Speeds are given internally as fractions of full speed (0.1 ... 1.0, as in the
folder names) but are DISPLAYED as speed percentages relative to the
high-ability team (10 ... 100 %), the convention used throughout the paper.
Axis labels and colour-bar labels are centred on their axes, so they stay
centred if the wording changes.

All positions, sizes, line widths and fonts (Computer Modern via LaTeX)
were measured from Fig4.pdf; the figure is written at the same page size
(1332.36 x 737.16 pt) without `bbox_inches='tight'`.

Usage:
    python fig4_human_vs_rl_gridmap.py [options]

Examples:
    # Default analysis (simulated games, checkpoint_753, every Training_* / simulation_*)
    nohup python fig4_human_vs_rl_gridmap.py --init_type empty_init --specialization 0.25 --synergy 1.35 --checkpoint 753 > fig4.log 2>&1 &

    # Besides the figure: <name>_coverage.txt (seeds and simulated games per cell) and
    # <name>_per_training.csv (abilities, map, training_id, num_simulations, mean_score)

    # re-plot from the per-game records cached by a previous run
    python fig4_human_vs_rl_gridmap.py --specialization 0.25 --synergy 1.35 --records_csv ./figures/figure4_human_vs_rl_gridmap_synergy1p35_lambda0p25_sim_records.csv

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

# This folder (marl_simulations.py) and the project root on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import marl_simulations as ms                              # noqa: E402  (same folder)


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


def _format_speed_percent(value: float) -> str:
    """Speed fraction -> percentage label, e.g. 0.4 -> '40'."""
    return f'{round(value * 100):d}'


def _format_speed_token(value: float) -> str:
    return _format_speed_label(value).replace('.', 'p')


# --------------------------------------------------------------------------
# Human results (raw scores; the figure shows score - score(1.0, 1.0)).
# Keys are (walking speed of player 1, cutting speed of player 2) as fractions of full speed.
# Filled ONLY from the per-team file written by fig_sm_ability_pairs_humans_rl.py
# (--human_teams_csv, default ./figures/fig_sm_ability_pairs_humans_rl_human_teams.csv):
# condition means of the selected teams, the same teams as Fig. 2 and the SI. No hard-coded values.
# --------------------------------------------------------------------------
HUMAN_RESULTS: Dict[str, Dict[str, Dict[Tuple[float, float], float]]] = {}
_HUMAN_TEAMS_DEFAULT = './figures/fig_sm_ability_pairs_humans_rl_human_teams.csv'
_HUMAN_MAP_OF = {'Open': 'baseline_division_of_labor_large', 'PB': 'encouraged_division_of_labor_large'}


def load_human_results_from_teams(path: str) -> bool:
    """Fill HUMAN_RESULTS with the condition means of the per-team human file."""
    if not path or not os.path.exists(path):
        return False
    HUMAN_RESULTS.clear()
    teams = pd.read_csv(path)
    for (mp, w, c), g in teams.groupby(['map', 'walk', 'cut']):
        key = _HUMAN_MAP_OF.get(mp)
        if key is None:
            continue
        HUMAN_RESULTS.setdefault(key, {'scores': {}})['scores'][(round(float(w), 4), round(float(c), 4))] = \
            float(g['total_deliveries'].mean())
    print(f'Human means taken from {path}:')
    for key, d in HUMAN_RESULTS.items():
        print(f'  {key}: ' + ', '.join(f'{k}={v:.2f}' for k, v in sorted(d['scores'].items()) if v is not None))
    return True


def _resolve_human_map_key(map_name: str) -> Optional[str]:
    if map_name in HUMAN_RESULTS:
        return map_name
    for key in HUMAN_RESULTS:
        if key in map_name:
            return key
    return None


class AbilityGridAnalyzer:
    """Load and filter the simulated games of a single map over the 2D ability sweep.

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
                 fixed_walk2: float = 1.0, demo: bool = False,
                 checkpoint: str = ms.DEFAULT_CHECKPOINT, verbose: bool = False):
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
        self.checkpoint = str(checkpoint)
        self.verbose = verbose

        self.walk1_values = _inclusive_float_range(walk1_start, walk1_end, walk1_step)
        self.cut2_values = _inclusive_float_range(cut2_start, cut2_end, cut2_step)

        if demo:
            return
        if abs(fixed_cut1 - ms.FIXED_CUT1) > 1e-9 or abs(fixed_walk2 - ms.FIXED_WALK2) > 1e-9:
            print(f'Warning: the simulation tree only holds cut1 = {ms.FIXED_CUT1} and walk2 = {ms.FIXED_WALK2}; '
                  f'--fixed_cut1 {fixed_cut1} / --fixed_walk2 {fixed_walk2} will match no simulation.')
        self.local_path = ms.cluster_base_dir(cluster)
        self.raw_dir = str(self._build_raw_dir())

    def _build_raw_dir(self) -> Path:
        """Folder holding the abilities_* folders of this map."""
        spec = 0 if self.specialization is None else self.specialization
        return ms.map_root(self.local_path, self.game_type, self.map_name, self.synergy, spec,
                           init_type=self.init_type, study_name=self.study_name)

    def _resolve_training_id_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in ('training_id', 'timestamp', 'run_id'):
            if col in df.columns:
                return col
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
            return
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

HUMAN_MARKER_SIZE = 287
HUMAN_MARKER_LW = 2.0

# Two-layer outline for visibility on every heatmap colour.
HUMAN_EDGE_COLOR = '#000000'       # Black inner contour
HUMAN_HALO_COLOR = '#FFFFFF'       # White outer halo
HUMAN_HALO_EXTRA_LW = 3.0          # Additional width of the halo

# Free text: (TeX font, size pt, text, x, y, rotation[, ha]); (x, y) is the
# baseline anchor of the text (rotation_mode='anchor'), left-aligned unless a
# 7th element gives ha='center' (then x, y is the centre along the text).
TEXTS = {
    'letter_a':  ('cmbx10', 50, 'a', 64.83, 52.20, 0),
    'letter_b':  ('cmbx10', 50, 'b', 613.31, 52.20, 0),
    'title_a':   ('cmbx10', 38, 'Open Map', 255.66, 39.54, 0),
    'title_b':   ('cmbx10', 38, 'Partially-Blocked Map', 689.80, 38.64, 0),
    'xlabel_a':  ('cmr10', 36, r'Walking speed of player 1 (\%)', (123.52 + 589.43) / 2, 615.15, 0, 'center'),
    'xlabel_b':  ('cmr10', 36, r'Walking speed of player 1 (\%)', (672.33 + 1138.13) / 2, 615.06, 0, 'center'),
    'ylabel_a':  ('cmr10', 36, r'Cutting speed of player 2 (\%)', 41.84, (68.10 + 534.12) / 2, 90, 'center'),
    'cbar_lab':  ('cmr10', 36, 'Relative advantage over HA', 1282.0, (63.04 + 545.61) / 2, 90, 'center'),
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

# ---- gap between panels a and b ------------------------------------------
# Panel b has no y tick labels, so the original 83-pt gap is reduced to PANEL_GAP:
# panel b, its letter/title/x label, the colour bar and its labels move left by
# _B_DX, the page shrinks by the same amount and the legend is re-centred.
PANEL_GAP = 52.0
_B_DX = PANEL_GAP - (AXES_RECT['b'][0] - AXES_RECT['a'][1])
_LETTER_B_OFFSET = 46.0          # letter b: this far left of panel b's frame
_shift_x = lambda r: (r[0] + _B_DX, r[1] + _B_DX, r[2], r[3])
AXES_RECT['b'] = _shift_x(AXES_RECT['b'])
CBAR_RECT = _shift_x(CBAR_RECT)
for _k in ('title_b', 'xlabel_b', 'cbar_lab', 'cbar_+4', 'cbar_+2', 'cbar_0', 'cbar_-2', 'cbar_-4'):
    _t = list(TEXTS[_k])
    _t[3] += _B_DX
    TEXTS[_k] = tuple(_t)
_t = list(TEXTS['letter_b'])
_t[3] = AXES_RECT['b'][0] - _LETTER_B_OFFSET
TEXTS['letter_b'] = tuple(_t)
PAGE_W = PAGE_W + _B_DX
for _k in ('legend_marl', 'legend_human'):
    _t = list(TEXTS[_k])
    _t[3] += _B_DX / 2
    TEXTS[_k] = tuple(_t)
LEGEND_BOX.update(x0=LEGEND_BOX['x0'] + _B_DX / 2, x1=LEGEND_BOX['x1'] + _B_DX / 2)
LEGEND_SQUARE['cx'] += _B_DX / 2
LEGEND_DIAMOND['cx'] += _B_DX / 2

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
    return text.replace('\\%', '%')   # plain-text fallback: '\%' -> '%'


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

        def lab(v):  # label every 20 % (20, 40, ..., 100), tick every cell
            return tex(*TICK_FONT, _format_speed_percent(v)) if round(v * 10) % 2 == 0 else ''

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
            if value is None:
                print(f'Human marker not drawn for walk1={walk1}, cut2={cut2} (no data yet).')
                continue
            coord = self._speed_to_display_coord(walk1, cut2)
            if coord is None:
                print(f'Human marker skipped for walk1={walk1}, cut2={cut2} (not in grid).')
                continue
            xs.append(coord[0])
            ys.append(coord[1])
            values.append(value if baseline_value is None else value - baseline_value)

        if xs:
            # Draw a white halo first, then a black contour.
            ax.scatter(
                xs, ys,
                c='none',
                marker='D',
                s=HUMAN_MARKER_SIZE,
                edgecolors=HUMAN_HALO_COLOR,
                linewidths=HUMAN_MARKER_LW + HUMAN_HALO_EXTRA_LW,
                zorder=6,
                clip_on=False,
            )

            ax.scatter(
                xs, ys,
                c=values,
                cmap=CMAP,
                norm=norm,
                marker='D',
                s=HUMAN_MARKER_SIZE,
                edgecolors=HUMAN_EDGE_COLOR,
                linewidths=HUMAN_MARKER_LW,
                zorder=7,
                clip_on=False,
            )

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
        diamond_vertices = [
            (dm['cx'], dm['cy'] - r),
            (dm['cx'] + r, dm['cy']),
            (dm['cx'], dm['cy'] + r),
            (dm['cx'] - r, dm['cy']),
        ]

        # White halo
        fig.add_artist(Polygon(
            diamond_vertices,
            closed=True,
            transform=page,
            facecolor='white',
            edgecolor=HUMAN_HALO_COLOR,
            linewidth=(
                dm['lw'] * HUMAN_MARKER_LW / 2.24
                + HUMAN_HALO_EXTRA_LW
            ),
            joinstyle='miter',
            zorder=2,
        ))

        # Black contour
        fig.add_artist(Polygon(
            diamond_vertices,
            closed=True,
            transform=page,
            facecolor='white',
            edgecolor=HUMAN_EDGE_COLOR,
            linewidth=dm['lw'] * HUMAN_MARKER_LW / 2.24,
            joinstyle='miter',
            zorder=3,
        ))

        # Free text (panel letters, titles, axis labels, colour-bar labels, legend)
        for font, size, text, x, y, rot, *ha in TEXTS.values():
            fig.text(x, y, tex(font, size, text), transform=page, ha=(ha[0] if ha else 'left'), va='baseline',
                     rotation=rot, rotation_mode='anchor', zorder=3, **_text_kw(font, size))

        if output_path:
            # NB: no bbox_inches='tight' -- the page size is part of the layout.
            fig.savefig(output_path)
            png_path = os.path.splitext(output_path)[0] + '.png'
            fig.savefig(png_path)
            print(f'Saved Figure 4 to: {output_path} and {png_path}')

        return fig


def write_coverage_files(records: List[pd.DataFrame], map_names: Dict[str, str],
                         walk1_values: List[float], cut2_values: List[float], stem: Path,
                         info: str = ''):
    """Seeds and simulated games per grid cell.

    <stem>_coverage.txt : per map, one line per (walk1, cut2) cell with the number of seeds
                          (Training_* folders with at least one usable simulation), the number
                          of simulated games and the games per seed (min..max), plus a
                          seeds grid and a simulations grid (rows = cut2, columns = walk1).
    <stem>_per_training.csv : one row per map x abilities x training, with columns
                          abilities, map, training_id, num_simulations, mean_score."""
    df = pd.concat([r for r in records if r is not None and len(r)], ignore_index=True) \
        if any(r is not None and len(r) for r in records) else pd.DataFrame()

    def ab_name(w, c):
        return f'abilities_{_format_speed_label(w)}_{_format_speed_label(c)}'

    # ---- per-training CSV ----
    cols = ['abilities', 'map', 'training_id', 'num_simulations', 'mean_score']
    if len(df):
        g = (df.groupby(['map_name', 'walk', 'cut', 'training_id'])['total_deliveries']
               .agg(num_simulations='size', mean_score='mean').reset_index())
        g['abilities'] = [ab_name(w, c) for w, c in zip(g['walk'], g['cut'])]
        per_tr = (g.rename(columns={'map_name': 'map'})
                   .sort_values(['map', 'walk', 'cut', 'training_id'])[cols])
    else:
        per_tr = pd.DataFrame(columns=cols)
    csv_path = Path(f'{stem}_per_training.csv')
    per_tr.to_csv(csv_path, index=False)

    # ---- coverage TXT ----
    lines = ['=' * 92, 'FIGURE 4 -- MARL simulations per grid cell (seeds = Training_* folders with data)', '=' * 92]
    if info:
        lines.append(info)
    for key, map_name in map_names.items():
        sub = df[df['map_name'] == map_name] if len(df) else df
        lines += ['', f'Panel {key}: {map_name}',
                  f'  total: {len(sub)} simulated games, '
                  f'{sub["training_id"].nunique() if len(sub) else 0} distinct training ids', '',
                  f'  {"abilities (walk1, cut2)":<26}{"seeds":>6}{"simulations":>13}   games per seed']
        seeds = np.zeros((len(cut2_values), len(walk1_values)), int)
        sims = np.zeros_like(seeds)
        empty = []
        for i, c in enumerate(cut2_values):
            for j, w in enumerate(walk1_values):
                cell = sub[np.isclose(sub['walk'], w) & np.isclose(sub['cut'], c)] if len(sub) else sub
                per = cell.groupby('training_id').size() if len(cell) else pd.Series(dtype=int)
                seeds[i, j], sims[i, j] = len(per), len(cell)
                rng = f'{per.min()}..{per.max()}' if len(per) else '--'
                lines.append(f'  {ab_name(w, c):<26}{len(per):>6}{len(cell):>13}   {rng}')
                if not len(cell):
                    empty.append(ab_name(w, c))
        for title, grid in (('Seeds', seeds), ('Simulations', sims)):
            lines += ['', f'  {title} grid (rows: cut2 = cutting speed of player 2, high to low; '
                          f'columns: walk1 = walking speed of player 1, low to high)',
                      '  cut2 \\ walk1 ' + ''.join(f'{_format_speed_label(w):>6}' for w in walk1_values)]
            for i in reversed(range(len(cut2_values))):
                lines.append(f'  {_format_speed_label(cut2_values[i]):>12} '
                             + ''.join(f'{v:>6d}' for v in grid[i]))
        nz = seeds[seeds > 0]
        lines += ['', f'  cells with data: {int((sims > 0).sum())} / {sims.size}'
                  + (f'; seeds per cell {nz.min()}..{nz.max()}' if nz.size else ''),
                  f'  empty cells: {", ".join(empty) if empty else "none"}']
    txt_path = Path(f'{stem}_coverage.txt')
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nCoverage report -> {txt_path}\nPer-training table -> {csv_path}')


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser for the 2D ability sweep."""
    parser = argparse.ArgumentParser(
        description='Generate Figure 4: MARL vs. human score landscape (walking speed 1 vs cutting speed 2)',
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
    parser.add_argument('--map_open', type=str, default='baseline_division_of_labor_large',
                        help='Map shown in panel a ("Open Map")')
    parser.add_argument('--map_blocked', type=str, default='encouraged_division_of_labor_large',
                        help='Map shown in panel b ("Partially-Blocked Map")')
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
    parser.add_argument('--human_teams_csv', type=str, default=_HUMAN_TEAMS_DEFAULT,
                        help='Per-team human data from fig_sm_ability_pairs_humans_rl.py (required, except '
                             'with --demo); the human diamonds are its condition means')
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
        if not load_human_results_from_teams(args.human_teams_csv):
            msg = (f'Per-team human file not found: {args.human_teams_csv}. Run '
                   f'fig_sm_ability_pairs_humans_rl.py first (it writes <output>_human_teams.csv) or pass --human_teams_csv.')
            if not args.demo:
                raise FileNotFoundError(msg)
            print(f'[demo] {msg} No human markers drawn.')
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
            checkpoint=args.checkpoint, verbose=args.verbose,
        )
        cached = pd.read_csv(args.records_csv) if args.records_csv else None
        records = []

        print('=' * 60)
        print('FIGURE 4 - WALKING SPEED 1 vs CUTTING SPEED 2 (MARL vs HUMAN)')
        print('=' * 60)
        print(f'Cluster: {cluster} | Game type: {args.game_type} | Init type: {args.init_type}')
        print(f'Maps: a = {args.map_open}, b = {args.map_blocked}')
        print(f'Player 1 walking speed: {args.walk1_min:.0%}-{args.walk1_max:.0%} (step {args.walk1_step:.0%})')
        print(f'Player 2 cutting speed: {args.cut2_min:.0%}-{args.cut2_max:.0%} (step {args.cut2_step:.0%})')

        grids: Dict[str, np.ndarray] = {}
        analyzers: Dict[str, AbilityGridAnalyzer] = {}
        for i, (key, map_name) in enumerate(map_names.items()):
            analyzer = AbilityGridAnalyzer(map_name=map_name, **common)
            analyzers[key] = analyzer
            if args.demo:
                grids[key] = analyzer.demo_grid(seed=i)
                continue
            print(f'\nSimulations root ({map_name}): {analyzer.raw_dir}  [checkpoint_{args.checkpoint}]')
            if cached is not None:
                prepared = cached[cached['map_name'] == map_name].copy()
                print(f'{len(prepared)} simulated games read from {args.records_csv}')
            else:
                prepared = analyzer.load_experimental_data()
            records.append(prepared)
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
        if str(args.checkpoint) != ms.DEFAULT_CHECKPOINT:
            filename_parts.append(f'checkpoint{args.checkpoint}')
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
        if not args.demo:
            write_coverage_files(
                records, map_names, analyzers['a'].walk1_values, analyzers['a'].cut2_values,
                output_dir / '_'.join(filename_parts),
                info=(f'cached records: {args.records_csv}' if args.records_csv else
                      f'checkpoint_{args.checkpoint}, synergy {args.synergy}, specialization '
                      f'{args.specialization}, {args.init_type}, {args.game_type}'))
        if not args.demo and cached is None and records:
            cache = output_dir / ('_'.join(filename_parts) + '_sim_records.csv')
            pd.concat(records, ignore_index=True).to_csv(cache, index=False)
            print(f'\nPer-game MARL records cached -> {cache}  (re-use with --records_csv)')

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