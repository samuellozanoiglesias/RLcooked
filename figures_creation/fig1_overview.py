#!/usr/bin/env python3
"""
Figure 1 -- Overview of the study: human experiments, multi-agent
reinforcement learning (MARL), simulations and theoretical solutions.

Produces ONE combined figure (PDF + PNG):

    a  "Human Experiments" / "Multi-Agent Reinforcement Learning"
       -> both play the same Overcooked layout (game screenshot)
    b  "Simulations"            input features -> neural network -> actions
    c  "Theoretical Solutions"  landscape over (task bias, propensity to
                                switch tasks) with the learning trajectory

The visual style (Computer Modern via LaTeX, bold panel letters / titles,
font sizes, line widths and the page-level layout in points without
`bbox_inches='tight'`) follows `fig4_human_vs_rl_gridmap.py`; the page width
(1332.36 pt) is the same as Figure 4 so that both figures scale identically.

Raster sprites and the human / robot vector icons were cut from the original
Fig1.pdf with `extract_fig1_assets.py` (the two agent sprites are the game's
own player PNGs) and are read from `fig1_assets/`. The landscape of panel c
(contours and gradient-descent trajectory) is generated analytically here:

    map.png, agent_pink.png, agent_yellow.png, item_tomato.png,
    icon_board.png, icon_counter.png, icon_tomatoes.png, icon_plates.png,
    icon_delivery.png, vectors.json

Usage:
    python fig1_overview.py [--assets_dir fig1_assets] [--output_dir ./figures] [--no_latex]
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch, Circle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from matplotlib import colors as mcolors
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ==========================================================================
# LAYOUT  (all values in points, y measured from the TOP of the page)
# ==========================================================================
PAGE_W, PAGE_H = 1332.36, 1150.0              # 18.50 x 15.97 in (same width as Fig. 4)

PANEL_BOX = {                                 # (left, right, top, bottom)
    'a': (18.0, 1314.0, 18.0, 512.0),
    'b': (18.0, 656.0, 530.0, 1132.0),
    'c': (676.0, 1314.0, 530.0, 1132.0),
}
BOX_STYLE = dict(face='#f7f7f7', edge='#595959', lw=3.0, radius=26.0)

# -- colours ---------------------------------------------------------------
ICON_BLUE = '#56abd7'                         # human / robot icons, node edges
DARK_BLUE = '#0071bc'                         # arrows and divider in panel a
NODE_FACE = '#d0ebff'                         # network nodes and feature boxes
TRAJ_BLUE = '#1f8fd1'                         # learning trajectory in panel c
LANDSCAPE_GREYS = ('#4a4a4a', '#f7f7f7')      # low -> high value of the landscape
CONTOUR_EDGE = '#2b2b2b'

# -- line widths (as Fig. 4) -----------------------------------------------
SPINE_LW = 1.12
ARROW_LW = 5.0
DIVIDER_LW = 3.0
NN_EDGE_LW = 1.6
NN_NODE_LW = 2.4
PAREN_LW = 4.0
TRAJ_LW = 4.5
CONTOUR_LW = 1.0

# -- fonts (TeX font, size pt) ---------------------------------------------
LETTER_FONT = ('cmbx10', 50)
TITLE_FONT = ('cmbx10', 38)
LABEL_FONT = ('cmr10', 36)
FEATURE_FONTSIZE = 31
REGION_FONT = ('cmr10', 29)                   # region labels inside panel c (was 24)

# -- panel a ---------------------------------------------------------------
MAP_RECT = (850.0, 1290.0, 45.0, 485.0)
LEFT_CENTER_X = 440.0
TITLE_HUMAN_Y = 82.0
TITLE_MARL_Y = 340.0
DIVIDER_Y = 287.0
DIVIDER_X = (PANEL_BOX['a'][0], 828.0)
HUMAN_ICONS = dict(xs=(330.0, 550.0), top=102.0, height=150.0)
ROBOT_ICONS = dict(xs=(330.0, 550.0), top=362.0, height=128.0)
ARROWS_A = [((650.0, 180.0), (836.0, 228.0)), ((650.0, 440.0), (836.0, 392.0))]
# overlays on the map (normalised map coords, y down): (name, x0, y0, size)
MAP_OVERLAYS = [
    ('item_tomato', 0.2149, 0.2161, 0.0836),
    ('agent_pink',  0.2967, 0.2931, 0.1045),
    ('agent_yellow', 0.2996, 0.7968, 0.1045),
]

# -- panel b ---------------------------------------------------------------
ROW_Y0, ROW_DY, N_ROWS = 655.0, 70.0, 6
INPUT_X, INPUT_W, INPUT_H = 130.0, 112.0, 52.0
INPUT_FEATURES = ['salad', 'tomato', 'plates', 'partner', 'counter']
OUTPUT_X, OUTPUT_ICON = 590.0, 54.0
OUTPUT_ICONS = ['icon_board', 'icon_counter', 'icon_tomatoes', 'icon_plates', 'icon_delivery']
PARENS = {'input': (64.0, 196.0), 'output': (556.0, 624.0)}
PAREN_PAD = 40.0                              # extra height above/below the rows
PAREN_BULGE = 16.0                            # max horizontal offset of a bracket
NN_LAYERS = dict(xs=(262.0, 342.0, 422.0, 502.0), sizes=(3, 5, 5, 2), r=22.0)
IO_LABEL_Y = 1098.0

# -- panel c ---------------------------------------------------------------
# The y-label + landscape group is centred in the panel (x ~ 995 pt) and the
# landscape is a 470 pt square.
LANDSCAPE_RECT = (784.0, 1254.0, 600.0, 1070.0)
YLABEL_C_X = 762.0
XLABEL_C_Y = 1102.0
# Landscape: V(x, y) on x = task bias in [-1, 1], y = propensity to switch in [0, 1].
# A saddle separates a broad "independent work" plateau (high switching, no bias)
# from two low-switching basins; the right (biased) basin is the deepest. The
# trajectory is obtained by gradient descent on V from just below the plateau.
LANDSCAPE = dict(ys=0.47,       # height of the saddle
                 a=3.7,         # curvature across the saddle (vertical)
                 b=0.40,        # curvature along the saddle (horizontal)
                 c=1.10,        # extra depth of the effective-cooperation basin
                 ridge=0.10,    # width of the ridge separating the two basins
                 tilt=0.22,     # small bias that selects the right-hand basin
                 n_levels=24)
TRAJ_START = (0.04, 0.955)                    # (task bias, propensity) initial state
TRAJ_STOP = dict(x_max=0.80, y_min=0.24)
AGENTS_C = [                                  # (sprite, arc-length fraction, size in axes units)
    ('agent_pink',   0.47, 0.12),
    ('agent_yellow', 0.66, 0.12),
]
AGENT_OFFSET = 0.085                          # sideways shift of the agents off the path
REGION_TEXT = [                               # (lines, x, y) in axes fractions (y up)
    (['Independent', 'work'], 0.26, 0.885),
    (['Ineffective', 'division of labor'], 0.25, 0.105),
    (['Effective', 'division of labor'], 0.75, 0.105),
]
REGION_BOX = dict(boxstyle='round,pad=0.22,rounding_size=0.35', facecolor='white',
                  edgecolor='none', alpha=0.78)


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
            probe.text(0.5, 0.5, tex('cmr10', 10, 'Test') + r' $d_{\textit{x}}\vdots$')
            probe.canvas.draw()
            plt.close(probe)
            return True
        except Exception as exc:
            plt.close('all')
            print(f'[make_figure1] LaTeX unavailable ({type(exc).__name__}); '
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


def feature_label(name: str) -> str:
    if mpl.rcParams['text.usetex']:
        return rf'$d_{{\textit{{{name}}}}}$'
    return rf'$d_{{\mathit{{{name}}}}}$'


def vdots() -> str:
    return r'$\vdots$'


def multiline(font: str, size: float, lines) -> str:
    """Centred multi-line label in a given Computer Modern font."""
    if mpl.rcParams['text.usetex']:
        body = r'\\'.join(lines)
        return rf'{{\font\figfont={font} at {size}pt\figfont \begin{{tabular}}{{@{{}}c@{{}}}}{body}\end{{tabular}}}}'
    return '\n'.join(lines)


def _softplus(z, k=10.0):
    return np.logaddexp(0.0, k * z) / k


def landscape(x, y, ys, a, b, c, ridge, tilt, **_):
    """Potential over (task bias x, propensity to switch y); lower = better."""
    s = 1.0 / (1.0 + np.exp(-(y - ys) / 0.05))                 # 1 above the saddle, 0 below
    shape = s + (1.0 - s) * np.exp(-x ** 2 / (2.0 * ridge ** 2))
    return (a * (y - ys) ** 2 * shape - b * x ** 2
            - c * _softplus(x) * (1.0 - y) ** 1.5 - tilt * x)


def descend(start, params, step=0.002, h=1e-4, x_max=0.82, y_min=0.10, n_max=5000):
    """Normalised gradient descent on the landscape (the 'learning' trajectory)."""
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(n_max):
        gx = (landscape(p[0] + h, p[1], **params) - landscape(p[0] - h, p[1], **params)) / (2 * h)
        gy = (landscape(p[0], p[1] + h, **params) - landscape(p[0], p[1] - h, **params)) / (2 * h)
        g = np.array([gx, gy])
        p = p - step * g / (np.linalg.norm(g) + 1e-12)
        path.append(p.copy())
        if p[0] > x_max or p[1] < y_min:
            break
    return np.array(path)


class Figure1Plotter:
    """Assemble the complete Fig. 1 in a single matplotlib figure."""

    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.sprites: Dict[str, np.ndarray] = {}
        names = ['map', 'agent_pink', 'agent_yellow', 'item_tomato'] + OUTPUT_ICONS
        for name in names:
            path = assets_dir / f'{name}.png'
            if not path.is_file():
                raise FileNotFoundError(f'Missing asset: {path} (run extract_fig1_assets.py)')
            self.sprites[name] = np.asarray(Image.open(path).convert('RGBA'))
        vec_path = assets_dir / 'vectors.json'
        if not vec_path.is_file():
            raise FileNotFoundError(f'Missing asset: {vec_path} (run extract_fig1_assets.py)')
        with open(vec_path) as fh:
            self.vectors = json.load(fh)

    # -- helpers ----------------------------------------------------------
    def _page(self, fig):
        """Transform from page points (y from top) to display coordinates."""
        return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans

    @staticmethod
    def _rect_to_fig(rect):
        L, R, T, B = rect
        return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]

    def _text(self, fig, page, font, size, text, x, y, ha='left', rot=0, color='black', zorder=5):
        fig.text(x, y, tex(font, size, text), transform=page, ha=ha, va='baseline', color=color,
                 rotation=rot, rotation_mode='anchor', zorder=zorder, **_text_kw(font, size))

    def _sprite(self, fig, name, left, top, width, height=None, zorder=4):
        """Pixel-art sprite placed in page points (nearest-neighbour, no smoothing)."""
        height = width if height is None else height
        ax = fig.add_axes(self._rect_to_fig((left, left + width, top, top + height)))
        ax.imshow(self.sprites[name], interpolation='nearest', aspect='auto')
        ax.set_axis_off()
        ax.set_zorder(zorder)
        return ax

    @staticmethod
    def _shape_path(shape, x0, y0, w, h) -> MplPath:
        v = np.asarray(shape['verts'], dtype=float)
        v[:, 0] = x0 + v[:, 0] * w
        v[:, 1] = y0 + v[:, 1] * h
        return MplPath(v, np.asarray(shape['codes'], dtype=np.uint8))

    def _vector_icon(self, fig, page, key, cx, top, height, face, hole_face):
        icon = self.vectors[key]
        w = height * icon['aspect']
        for shape in icon['shapes']:
            path = self._shape_path(shape, cx - w / 2, top, w, height)
            colour = hole_face if shape['role'] == 'hole' else face
            fig.add_artist(PathPatch(path, transform=page, facecolor=colour, edgecolor='none',
                                     zorder=3))

    def _panel_box(self, fig, page, key):
        L, R, T, B = PANEL_BOX[key]
        fig.add_artist(FancyBboxPatch(
            (L, T), R - L, B - T, transform=page,
            boxstyle=f"round,pad=0,rounding_size={BOX_STYLE['radius']}",
            facecolor=BOX_STYLE['face'], edgecolor=BOX_STYLE['edge'],
            linewidth=BOX_STYLE['lw'], zorder=0))
        self._text(fig, page, *LETTER_FONT, key, L + 20.0, T + 52.0)

    def _paren(self, fig, page, x, top, bottom, opens: bool):
        """Large round bracket drawn as a quadratic Bezier."""
        s = -1.0 if opens else 1.0
        mid = 0.5 * (top + bottom)
        path = MplPath([(x, top), (x + s * PAREN_BULGE * 2, mid), (x, bottom)],
                       [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        fig.add_artist(PathPatch(path, transform=page, facecolor='none', edgecolor='black',
                                 linewidth=PAREN_LW, capstyle='round', zorder=3))

    # -- panel a ----------------------------------------------------------
    def _panel_a(self, fig, page):
        self._panel_box(fig, page, 'a')

        self._text(fig, page, *TITLE_FONT, 'Human Experiments', LEFT_CENTER_X, TITLE_HUMAN_Y, ha='center')
        self._text(fig, page, *TITLE_FONT, 'Multi-Agent Reinforcement Learning', LEFT_CENTER_X,
                   TITLE_MARL_Y, ha='center')

        for cx in HUMAN_ICONS['xs']:
            self._vector_icon(fig, page, 'human', cx, HUMAN_ICONS['top'], HUMAN_ICONS['height'],
                              ICON_BLUE, BOX_STYLE['face'])
        for cx in ROBOT_ICONS['xs']:
            self._vector_icon(fig, page, 'robot', cx, ROBOT_ICONS['top'], ROBOT_ICONS['height'],
                              ICON_BLUE, 'white')

        fig.add_artist(mpl.lines.Line2D(DIVIDER_X, (DIVIDER_Y, DIVIDER_Y), transform=page,
                                        color=DARK_BLUE, linewidth=DIVIDER_LW,
                                        linestyle=(0, (5, 4)), zorder=2))

        for (x0, y0), (x1, y1) in ARROWS_A:
            rad = -0.25 if y1 > y0 else 0.25
            fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=page,
                                           connectionstyle=f'arc3,rad={rad}',
                                           arrowstyle='-|>,head_length=0.55,head_width=0.32',
                                           mutation_scale=40, color=DARK_BLUE,
                                           linewidth=ARROW_LW, zorder=3))

        # Game layout with the two players and an item
        L, R, T, B = MAP_RECT
        ax = fig.add_axes(self._rect_to_fig(MAP_RECT))
        ax.imshow(self.sprites['map'], extent=(0, 1, 1, 0), interpolation='nearest', aspect='auto')
        for name, x0, y0, size in MAP_OVERLAYS:
            ax.imshow(self.sprites[name], extent=(x0, x0 + size, y0 + size, y0),
                      interpolation='nearest', aspect='auto', zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
            spine.set_edgecolor('black')

    # -- panel b ----------------------------------------------------------
    def _panel_b(self, fig, page):
        self._panel_box(fig, page, 'b')
        L, R, T, B = PANEL_BOX['b']
        self._text(fig, page, *TITLE_FONT, 'MARL Simulations', 0.5 * (L + R), T + 50.0, ha='center')

        rows = [ROW_Y0 + k * ROW_DY for k in range(N_ROWS)]
        top, bottom = rows[0] - PAREN_PAD, rows[-1] + PAREN_PAD

        # Input vector
        for k, y in enumerate(rows):
            is_dots = k == len(rows) - 1
            fig.add_artist(FancyBboxPatch((INPUT_X - INPUT_W / 2, y - INPUT_H / 2), INPUT_W, INPUT_H,
                                          transform=page, boxstyle='round,pad=0,rounding_size=14',
                                          facecolor=NODE_FACE, edgecolor='none', zorder=2))
            label = vdots() if is_dots else feature_label(INPUT_FEATURES[k])
            fig.text(INPUT_X, y + (2 if is_dots else 0), label, transform=page, ha='center',
                     va='center', fontsize=FEATURE_FONTSIZE, zorder=4)
        self._paren(fig, page, PARENS['input'][0], top, bottom, opens=True)
        self._paren(fig, page, PARENS['input'][1], top, bottom, opens=False)

        # Neural network (fully connected)
        cy = 0.5 * (rows[0] + rows[-1])
        layers = []
        for x, n in zip(NN_LAYERS['xs'], NN_LAYERS['sizes']):
            layers.append([(x, cy + (i - (n - 1) / 2) * ROW_DY) for i in range(n)])
        for la, lb in zip(layers[:-1], layers[1:]):
            for (xa, ya) in la:
                for (xb, yb) in lb:
                    fig.add_artist(mpl.lines.Line2D((xa, xb), (ya, yb), transform=page, color='black',
                                                    linewidth=NN_EDGE_LW, zorder=2))
        for layer in layers:
            for (x, y) in layer:
                fig.add_artist(Circle((x, y), NN_LAYERS['r'], transform=page, facecolor=NODE_FACE,
                                      edgecolor=ICON_BLUE, linewidth=NN_NODE_LW, zorder=3))

        # Output vector (actions = interactable tiles)
        for name, y in zip(OUTPUT_ICONS, rows):
            self._sprite(fig, name, OUTPUT_X - OUTPUT_ICON / 2, y - OUTPUT_ICON / 2, OUTPUT_ICON)
        y = rows[-1]
        fig.add_artist(FancyBboxPatch((OUTPUT_X - 22, y - 17), 44, 34, transform=page,
                                      boxstyle='round,pad=0,rounding_size=12',
                                      facecolor=NODE_FACE, edgecolor='none', zorder=2))
        fig.text(OUTPUT_X, y + 1, vdots(), transform=page, ha='center', va='center',
                 fontsize=FEATURE_FONTSIZE - 6, zorder=4)
        self._paren(fig, page, PARENS['output'][0], top, bottom, opens=True)
        self._paren(fig, page, PARENS['output'][1], top, bottom, opens=False)

        self._text(fig, page, *LABEL_FONT, 'Input', INPUT_X, IO_LABEL_Y, ha='center')
        self._text(fig, page, *LABEL_FONT, 'Output', OUTPUT_X, IO_LABEL_Y, ha='center')

    # -- panel c ----------------------------------------------------------
    def _panel_c(self, fig, page):
        self._panel_box(fig, page, 'c')
        L, R, T, B = PANEL_BOX['c']
        self._text(fig, page, *TITLE_FONT, 'MARL Theory', 0.5 * (L + 40 + R), T + 50.0,
                   ha='center')

        ax = fig.add_axes(self._rect_to_fig(LANDSCAPE_RECT))
        params = {k: v for k, v in LANDSCAPE.items() if k != 'n_levels'}

        # Landscape (filled contours + contour lines)
        xg = np.linspace(-1.0, 1.0, 700)
        yg = np.linspace(0.0, 1.0, 500)
        X, Y = np.meshgrid(xg, yg)
        Z = landscape(X, Y, **params)
        # levels halfway between linear and quantile spacing: even contrast across the
        # panel without placing contours in the nearly flat regions
        t = np.linspace(0.0, 1.0, LANDSCAPE['n_levels'])
        levels = 0.5 * (np.quantile(Z, t) + Z.min() + t * (Z.max() - Z.min()))
        cmap = mcolors.LinearSegmentedColormap.from_list('landscape', LANDSCAPE_GREYS)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, zorder=1)
        ax.contour(X, Y, Z, levels=levels[1:-1], colors=CONTOUR_EDGE, linewidths=CONTOUR_LW,
                   linestyles='solid', alpha=0.75, zorder=2)

        # Learning trajectory: gradient descent from independent work
        path = descend(TRAJ_START, params, **TRAJ_STOP)
        ax.plot(path[:-6, 0], path[:-6, 1], color=TRAJ_BLUE, linewidth=TRAJ_LW,
                linestyle=(0, (2.2, 1.6)), dash_capstyle='round', zorder=3)
        ax.annotate('', xy=path[-1], xytext=path[-12],
                    arrowprops=dict(arrowstyle='-|>,head_length=0.7,head_width=0.32',
                                    color=TRAJ_BLUE, linewidth=0, shrinkA=0, shrinkB=0,
                                    mutation_scale=30),
                    zorder=3)
        ax.scatter([path[0, 0]], [path[0, 1]], s=150, facecolor='white', edgecolor=TRAJ_BLUE,
                   linewidth=2.5, zorder=4)

        # Agents walking along the trajectory (offset to the left of the path)
        seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        sx, sy = 2.0, 1.0                                  # axes units per data unit
        for name, frac, size in AGENTS_C:
            i = int(np.searchsorted(seg, frac * seg[-1]))
            d = path[min(i + 3, len(path) - 1)] - path[max(i - 3, 0)]
            n = np.array([-d[1] / sx, d[0] / sy])          # left-hand normal in axes units
            n /= np.linalg.norm(n)
            cx = (path[i, 0] + 1.0) / sx + AGENT_OFFSET * n[0]
            cy = path[i, 1] / sy + AGENT_OFFSET * n[1]
            ax.imshow(self.sprites[name], interpolation='nearest', aspect='auto', zorder=5,
                      extent=((cx - size / 2) * sx - 1.0, (cx + size / 2) * sx - 1.0,
                              (cy - size / 2) * sy, (cy + size / 2) * sy))

        # Region labels
        for lines, xa, ya in REGION_TEXT:
            ax.text(xa, ya, multiline(*REGION_FONT, lines), transform=ax.transAxes, ha='center',
                    va='center', color='#1a1a1a', bbox=REGION_BOX, zorder=6,
                    **_text_kw(*REGION_FONT))

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
            spine.set_edgecolor('black')
            spine.set_zorder(7)

        Lr, Rr, Tr, Br = LANDSCAPE_RECT
        self._text(fig, page, *LABEL_FONT, 'Task bias', 0.5 * (Lr + Rr), XLABEL_C_Y, ha='center')
        self._text(fig, page, *LABEL_FONT, 'Propensity to switch tasks', YLABEL_C_X, 0.5 * (Tr + Br),
                   ha='center', rot=90)

    # -- full figure ------------------------------------------------------
    def create_figure(self, output_path: Optional[str] = None) -> plt.Figure:
        fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
        page = self._page(fig)

        self._panel_a(fig, page)
        self._panel_b(fig, page)
        self._panel_c(fig, page)

        if output_path:
            # NB: no bbox_inches='tight' -- the page size is part of the layout.
            fig.savefig(output_path)
            png_path = os.path.splitext(output_path)[0] + '.png'
            fig.savefig(png_path, dpi=200)
            print(f'Saved Figure 1 to: {output_path} and {png_path}')
        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate Figure 1: overview (human experiments, MARL, simulations, theory)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    default_assets = Path(__file__).resolve().parent / './figures/fig1_assets'
    parser.add_argument('--assets_dir', type=str, default=str(default_assets),
                        help='Folder with the sprites and vectors.json cut from Fig1.pdf')
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--filename', type=str, default='figure1_overview',
                        help='Output file name (without extension)')
    parser.add_argument('--no_latex', action='store_true', help='Do not use LaTeX for text')
    return parser


def main():
    global LATEX_OK
    args = setup_argument_parser().parse_args()
    try:
        LATEX_OK = configure_style(not args.no_latex)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (args.filename + '.pdf')

        print('=' * 60)
        print('FIGURE 1 - OVERVIEW')
        print('=' * 60)
        print(f'Assets: {args.assets_dir}')
        plotter = Figure1Plotter(Path(args.assets_dir))
        fig = plotter.create_figure(str(out_path))
        plt.close(fig)
        print('\n=== Done ===\n')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()