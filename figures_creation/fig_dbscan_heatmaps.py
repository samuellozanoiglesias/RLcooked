#!/usr/bin/env python3
"""
Supplementary Figure -- DBSCAN parameter sweep for Human vs. MARL trajectories.

Produces ONE combined figure (PDF + PNG) with a 2 x 3 layout:

    row 1  "Human"   a  Number of clusters
                     b  Noise fraction
                     c  Size of smallest cluster
    row 2  "MARL"    d  Number of clusters
                     e  Noise fraction
                     f  Size of smallest cluster

Every panel is a heat-map over the DBSCAN parameter grid
(x = minimum cluster size, y = epsilon). The black star marks the
parameter pair finally used for each data set.

The visual style (Computer Modern via LaTeX, bold panel letters, tick and
spine geometry, page-level layout in points without `bbox_inches='tight'`)
follows `fig4_human_vs_rl_gridmap.py`.

Input data (one file per data set), each a stacked array of shape
(3, n_eps, n_min_cluster_size) = [n_clusters, noise_fraction, smallest_cluster]:

    sm_clustering_data/human_matrices.npy
    sm_clustering_data/rl_matrices.npy

Usage:
    python fig_dbscan_heatmaps.py [options]

Examples:
    # Default (data in ./sm_clustering_data next to this script)
    python fig_dbscan_heatmaps.py

    # Custom data/output folders, no LaTeX
    python fig_dbscan_heatmaps.py --data_dir /path/to/sm_clustering_data --output_dir ./figures --no_latex
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ==========================================================================
# DATA
# ==========================================================================
DATASETS = {                                 # row key -> (file prefix, row title)
    'human': ('human', 'Human'),
    'rl':    ('rl', 'MARL'),
}
ROW_ORDER = ('human', 'rl')

# DBSCAN parameter pair selected for each data set (marked with a star)
SELECTED_PARAMS = {                          # (minimum cluster size, epsilon)
    'human': (80, 8.2),
    'rl':    (70, 9.0),
}


def load_matrices(data_dir: Path, prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load [n_clusters, noise_fraction, smallest_cluster] from `<prefix>_matrices.npy`."""
    path = data_dir / f'{prefix}_matrices.npy'
    if not path.is_file():
        raise FileNotFoundError(f'Data file not found: {path}')
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f'{path} must have shape (3, n_eps, n_mcs); got {arr.shape}')
    return arr[0], arr[1], arr[2]


# ==========================================================================
# PANEL DEFINITIONS (one per column)
# ==========================================================================
PANELS = [
    dict(name='clusters', label=['Number of Clusters'], cmap='terrain',
         vmin=0, vmax=7, ticks=[0, 3, 6], fmt='{:.0f}'),
    dict(name='noise', label=['Noise Fraction'], cmap='Spectral_r',
         vmin=0.0, vmax=1.0, ticks=[0.0, 0.5, 1.0], fmt='{:.1f}'),
    dict(name='smallest', label=['Size of', 'Smallest Cluster'], cmap='gnuplot_r',
         vmin=0, vmax=400, ticks=[0, 200, 400], fmt='{:.0f}'),
]
PANEL_LETTERS = {'human': ['a', 'b', 'c'], 'rl': ['d', 'e', 'f']}

X_TICKS = [30, 50, 70, 90]
Y_TICKS = [2, 4, 6, 8, 10]


# ==========================================================================
# LAYOUT  (all values in points, y measured from the TOP of the page)
# Same page width, fonts and line widths as Figure 4.
# ==========================================================================
PAGE_W = 1332.36
ROW_H = 685.0
ROW_GAP = 10.0
PAGE_H = 2 * ROW_H + ROW_GAP                 # 18.50 x 19.17 in

COL_LEFT = 150.0                             # left edge of the first column
PANEL_W = 350.0                              # square panels
PANEL_HGAP = 58.0

# vertical positions inside a row (relative to the top of the row)
ROW_TITLE_BASELINE = 44.0
CBAR_LABEL_BASELINE = 140.0                  # baseline of the LAST label line
CBAR_LABEL_LINESKIP = 38.0
CBAR_TOP = 182.0
CBAR_H = 21.0
CBAR_REL_W = 0.80                            # colour bar width / panel width
PANEL_TOP = 222.0
XLABEL_BASELINE = PANEL_TOP + PANEL_W + 92.0

SPINE_LW = 1.12
TICK_LEN, TICK_W = 8.4, 1.68
CBAR_OUTLINE_LW = 1.19
CBAR_TICK_LEN, CBAR_TICK_W = 5.45, 1.19
XTICK_PAD, YTICK_PAD = 4.46, 4.8
TICK_FONT = ('cmr10', 30)
CBAR_TICK_FONT = ('cmr10', 30)
LABEL_FONT = ('cmr10', 36)
CBAR_LABEL_FONT = ('cmr10', 34.1)
TITLE_FONT = ('cmbx10', 38)
LETTER_FONT = ('cmbx10', 50)
EPS_FONTSIZE = 44

GRID_KW = dict(color='white', linewidth=1.3, alpha=0.85)
STAR_KW = dict(marker='*', s=1150, c='black', edgecolors='white', linewidths=1.6, zorder=6)


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
            probe.text(0.5, 0.5, tex('cmr10', 10, 'Test') + r' $\epsilon$')
            probe.canvas.draw()
            plt.close(probe)
            return True
        except Exception as exc:
            plt.close('all')
            print(f'[dbscan_heatmaps] LaTeX unavailable ({type(exc).__name__}); '
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


class DBSCANHeatmapPlotter:
    """Assemble the complete 2 x 3 DBSCAN heat-map figure."""

    def __init__(self, eps_values: np.ndarray, mcs_values: np.ndarray, show_grid: bool = True):
        self.eps_values = eps_values
        self.mcs_values = mcs_values
        self.show_grid = show_grid
        # imshow extent: x = min. cluster size, y = epsilon (increasing downwards)
        self.extent = [mcs_values.min(), mcs_values.max(), eps_values.max(), eps_values.min()]

    # -- helpers ----------------------------------------------------------
    def _page(self, fig):
        """Transform from page points (y from top) to display coordinates."""
        return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans

    @staticmethod
    def _rect_to_fig(rect):
        L, R, T, B = rect
        return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]

    @staticmethod
    def _col_left(col: int) -> float:
        return COL_LEFT + col * (PANEL_W + PANEL_HGAP)

    def _text(self, fig, page, font, size, text, x, y, ha='left', rot=0):
        fig.text(x, y, tex(font, size, text), transform=page, ha=ha, va='baseline',
                 rotation=rot, rotation_mode='anchor', zorder=3, **_text_kw(font, size))

    # -- panels -----------------------------------------------------------
    def _plot_panel(self, ax, data: np.ndarray, spec: dict, show_ylabels: bool,
                    star: Optional[Tuple[float, float]]):
        im = ax.imshow(data, extent=self.extent, cmap=spec['cmap'],
                       vmin=spec['vmin'], vmax=spec['vmax'],
                       interpolation='nearest', aspect='auto')

        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([tex(*TICK_FONT, f'{v:d}') for v in X_TICKS])
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels([tex(*TICK_FONT, f'{v:d}') for v in Y_TICKS])
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        if self.show_grid:
            for v in X_TICKS:
                ax.axvline(v, zorder=2, **GRID_KW)
            for v in Y_TICKS[1:-1]:
                ax.axhline(v, zorder=2, **GRID_KW)

        if star is not None:
            ax.scatter([star[0]], [star[1]], clip_on=False, **STAR_KW)

        ax.tick_params(axis='both', which='major', direction='out', color='black',
                       length=TICK_LEN, width=TICK_W, labelsize=TICK_FONT[1])
        ax.tick_params(axis='x', pad=XTICK_PAD)
        ax.tick_params(axis='y', pad=YTICK_PAD, labelleft=show_ylabels)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
            spine.set_edgecolor('black')
        return im

    def _add_colorbar(self, fig, page, im, spec: dict, col: int, row_top: float):
        left = self._col_left(col)
        cw = PANEL_W * CBAR_REL_W
        cl = left + 0.5 * (PANEL_W - cw)
        cax = fig.add_axes(self._rect_to_fig((cl, cl + cw, row_top + CBAR_TOP,
                                              row_top + CBAR_TOP + CBAR_H)))
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=spec['ticks'])
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cbar.set_ticklabels([tex(*CBAR_TICK_FONT, spec['fmt'].format(t)) for t in spec['ticks']])
        cax.tick_params(length=CBAR_TICK_LEN, width=CBAR_TICK_W, color='black',
                        labelsize=CBAR_TICK_FONT[1], pad=3.0)
        cbar.outline.set_linewidth(CBAR_OUTLINE_LW)
        cbar.outline.set_edgecolor('black')

        # Colour-bar label as free text (multi-line labels grow upwards)
        xc = left + 0.5 * PANEL_W
        for k, line in enumerate(reversed(spec['label'])):
            self._text(fig, page, *CBAR_LABEL_FONT, line, xc,
                       row_top + CBAR_LABEL_BASELINE - k * CBAR_LABEL_LINESKIP, ha='center')

    # -- full figure ------------------------------------------------------
    def create_figure(self, data: Dict[str, Tuple[np.ndarray, ...]],
                      stars: Dict[str, Optional[Tuple[float, float]]],
                      output_path: Optional[str] = None) -> plt.Figure:
        fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
        page = self._page(fig)

        for r, key in enumerate(ROW_ORDER):
            row_top = r * (ROW_H + ROW_GAP)
            for col, spec in enumerate(PANELS):
                left = self._col_left(col)
                rect = (left, left + PANEL_W, row_top + PANEL_TOP, row_top + PANEL_TOP + PANEL_W)
                ax = fig.add_axes(self._rect_to_fig(rect))
                im = self._plot_panel(ax, data[key][col], spec, show_ylabels=(col == 0),
                                      star=stars.get(key))
                self._add_colorbar(fig, page, im, spec, col, row_top)

                # Panel letter
                self._text(fig, page, *LETTER_FONT, PANEL_LETTERS[key][col],
                           left - 58.0, row_top + CBAR_LABEL_BASELINE + 4.0)

                if col == 0:
                    ax.set_ylabel(r'$\epsilon$', fontsize=EPS_FONTSIZE, labelpad=12, rotation=0,
                                  va='center')

            # Row title (centred over the row) and shared x label (centred under it)
            row_l, row_r = self._col_left(0), self._col_left(len(PANELS) - 1) + PANEL_W
            xc = 0.5 * (row_l + row_r)
            self._text(fig, page, *TITLE_FONT, DATASETS[key][1], xc,
                       row_top + ROW_TITLE_BASELINE, ha='center')
            self._text(fig, page, *LABEL_FONT, 'Minimum cluster size', xc,
                       row_top + XLABEL_BASELINE, ha='center')

        if output_path:
            # NB: no bbox_inches='tight' -- the page size is part of the layout.
            fig.savefig(output_path)
            png_path = os.path.splitext(output_path)[0] + '.png'
            fig.savefig(png_path, dpi=150)
            print(f'Saved figure to: {output_path} and {png_path}')

        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate the DBSCAN parameter-sweep heat-maps (Human vs. MARL)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    default_data = Path(__file__).resolve().parent / 'data'
    parser.add_argument('--data_dir', type=str, default=str(default_data),
                        help='Folder containing human_matrices.npy and rl_matrices.npy')
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--filename', type=str, default='dbscan_heatmaps_human_vs_rl',
                        help='Output file name (without extension)')
    parser.add_argument('--eps_min', type=float, default=2.0, help='Smallest epsilon in the sweep')
    parser.add_argument('--eps_max', type=float, default=10.0, help='Largest epsilon in the sweep')
    parser.add_argument('--mcs_min', type=int, default=20, help='Smallest minimum cluster size in the sweep')
    parser.add_argument('--mcs_max', type=int, default=100, help='Largest minimum cluster size in the sweep')
    parser.add_argument('--human_star', type=float, nargs=2, default=list(SELECTED_PARAMS['human']),
                        metavar=('MCS', 'EPS'), help='Selected (min cluster size, epsilon) for Human')
    parser.add_argument('--rl_star', type=float, nargs=2, default=list(SELECTED_PARAMS['rl']),
                        metavar=('MCS', 'EPS'), help='Selected (min cluster size, epsilon) for MARL')
    parser.add_argument('--no_grid', action='store_true', help='Do not draw the white grid lines')
    parser.add_argument('--no_latex', action='store_true', help='Do not use LaTeX for text')
    return parser


def main():
    global LATEX_OK
    args = setup_argument_parser().parse_args()

    try:
        LATEX_OK = configure_style(not args.no_latex)
        data_dir = Path(args.data_dir)

        print('=' * 60)
        print('DBSCAN PARAMETER SWEEP - HUMAN vs MARL')
        print('=' * 60)
        print(f'Data folder: {data_dir}')

        data: Dict[str, Tuple[np.ndarray, ...]] = {}
        for key in ROW_ORDER:
            data[key] = load_matrices(data_dir, DATASETS[key][0])
            n_clu, noise, mins = data[key]
            print(f'{DATASETS[key][1]:>6}: grid {n_clu.shape} | clusters {n_clu.min():.0f}-{n_clu.max():.0f} | '
                  f'noise {noise.min():.2f}-{noise.max():.2f} | smallest {mins.min():.0f}-{mins.max():.0f}')

        n_eps, n_mcs = data[ROW_ORDER[0]][0].shape
        for key in ROW_ORDER:
            if data[key][0].shape != (n_eps, n_mcs):
                raise ValueError('Human and MARL grids have different shapes.')
        eps_values = np.linspace(args.eps_min, args.eps_max, n_eps)
        mcs_values = np.linspace(args.mcs_min, args.mcs_max, n_mcs)
        print(f'epsilon: {args.eps_min}-{args.eps_max} ({n_eps} values) | '
              f'min. cluster size: {args.mcs_min}-{args.mcs_max} ({n_mcs} values)')

        stars = {'human': tuple(args.human_star), 'rl': tuple(args.rl_star)}
        for key, (mcs, eps) in stars.items():
            i = int(np.argmin(np.abs(eps_values - eps)))
            j = int(np.argmin(np.abs(mcs_values - mcs)))
            n_clu, noise, mins = (m[i, j] for m in data[key])
            print(f'Selected {DATASETS[key][1]} (mcs={mcs:g}, eps={eps:g}): '
                  f'{n_clu:.0f} clusters, noise {noise:.3f}, smallest cluster {mins:.0f}')

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (args.filename + '.pdf')

        print('\n=== Creating figure ===\n')
        plotter = DBSCANHeatmapPlotter(eps_values, mcs_values, show_grid=not args.no_grid)
        fig = plotter.create_figure(data, stars, str(out_path))
        plt.close(fig)
        print('\n=== Done ===\n')

    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
