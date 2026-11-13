#!/usr/bin/env python3
"""
Cover-letter figure (simplified Fig. 1).

    Human experiments  --reveal-->  The weaker team can win  <--explain--  RL simulations + theory

Drawn at its final printed size (FIG_W x FIG_H inches): include it in LaTeX with
\\includegraphics[width=\\linewidth]{fig_cover_letter.pdf}  (letter text width = 6.75 in),
so the font sizes below are the sizes on paper. All text is set by LaTeX
(Computer Modern, as in fig1_overview.py); this needs a LaTeX installation with
the cm-super fonts.

Left   : the game (open map) with the two chefs.
Centre : mean game score of human teams, main experiment (paper Fig. 2c):
         high-ability vs mixed-ability teams, open vs partially-blocked map.
Right  : learning landscape of MARL theory, same landscape, dashed trajectory,
         chefs and region labels as panel c of fig1_overview.py.

Assets (fig_cover_assets/): map.png, pink.png, yellow.png, tomato.png.
Usage:  python fig_cover_letter.py   ->  fig_cover_letter.pdf / .png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import colors as mcolors
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'figures' / 'fig_cover_assets'

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#222222',
})

# ---------------------------------------------------------------- style -----
FIG_W, FIG_H = 7.35 , 2.55                       # inches, final printed size
FS_TITLE, FS_CAPTION, FS_TEXT, FS_ARROW, FS_SMALL = 10.5, 10.5, 9.0, 10.0, 8.0   # pt, as printed

INK = '#111111'
PANEL_BG = '#f1f3f5'
ARROW = '#0071bc'                               # arrows (paper's dark blue)
ARROW_TEXT = '#333333'                          # "reveal" / "explain"
HA_RED, MA_BLUE = '#e4675a', '#4c93c6'          # paper's colours for HA / MA teams
GOLD, GOLD_EDGE = '#f2b705', '#9a7400'
TRAJ_BLUE = '#1f8fd1'                           # as in fig1_overview.py
LANDSCAPE_GREYS = ('#4a4a4a', '#f7f7f7')
CONTOUR_EDGE = '#2b2b2b'

# human data, main experiment (paper Fig. 2c): mean game score (HA, MA)
SCORES = {'open': (8.6, 5.1), 'pb': (5.8, 6.9)}

# ---------------------------------------------------------------- layout ----
GAP = 0.50                                      # inches between panels (arrows live here)
PW = (FIG_W - 2 * GAP) / 3                      # panel width
COLS = {k: (i * (PW + GAP), i * (PW + GAP) + PW) for i, k in enumerate('abc')}
TITLE_Y = FIG_H - 0.22
ARROW_Y = 1.18


def ax_in(fig, x, y, w, h):
    """Axes placed in inches from the bottom-left corner of the figure."""
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def text_in(fig, x, y, s, **kw):
    kw.setdefault('color', INK)
    return fig.text(x / FIG_W, y / FIG_H, s, **kw)


def panel(fig, key, title):
    L, R = COLS[key]
    fig.add_artist(FancyBboxPatch((L / FIG_W, 0.0), (R - L) / FIG_W, 1.0,
                                  transform=fig.transFigure,
                                  boxstyle='round,pad=0,rounding_size=0.02',
                                  mutation_aspect=FIG_W / FIG_H,
                                  facecolor=PANEL_BG, edgecolor='none', zorder=0))
    text_in(fig, 0.5 * (L + R), TITLE_Y, rf'\textbf{{{title}}}', ha='center', va='center',
            fontsize=FS_TITLE)


def flow_arrow(fig, x_from, x_to, label):
    m = 0.10                                    # clearance from the panels
    a = x_from + (m if x_to > x_from else -m)
    b = x_to + (-m if x_to > x_from else m)
    fig.add_artist(FancyArrowPatch((a / FIG_W, ARROW_Y / FIG_H), (b / FIG_W, ARROW_Y / FIG_H),
                                   transform=fig.transFigure,
                                   arrowstyle='-|>,head_length=0.45,head_width=0.28',
                                   mutation_scale=16, color=ARROW, linewidth=2.2,
                                   shrinkA=0, shrinkB=0))
    text_in(fig, 0.5 * (x_from + x_to), ARROW_Y + 0.19, rf'\textit{{{label}}}', ha='center',
            va='center', fontsize=FS_ARROW, color=ARROW_TEXT)


def sprite(name):
    return np.asarray(Image.open(ASSETS / f'{name}.png').convert('RGBA'))


# ------------------------------------------------------- MARL landscape -----
# (identical to fig1_overview.py)
LANDSCAPE = dict(ys=0.47, a=3.7, b=0.40, c=1.10, ridge=0.10, tilt=0.22)
N_LEVELS = 24
TRAJ_START = (0.04, 0.955)
TRAJ_STOP = dict(x_max=0.80, y_min=0.24)
AGENTS_C = [('pink', 0.47), ('yellow', 0.66)]   # (sprite, arc-length fraction)
AGENT_SIZE = 0.13                               # sprite height, axes fraction
AGENT_OFFSET = 0.09                             # sideways shift off the path, axes fraction


def _softplus(z, k=10.0):
    return np.logaddexp(0.0, k * z) / k


def landscape(x, y, ys, a, b, c, ridge, tilt):
    s = 1.0 / (1.0 + np.exp(-(y - ys) / 0.05))
    shape = s + (1.0 - s) * np.exp(-x ** 2 / (2.0 * ridge ** 2))
    return (a * (y - ys) ** 2 * shape - b * x ** 2
            - c * _softplus(x) * (1.0 - y) ** 1.5 - tilt * x)


def descend(start, step=0.002, h=1e-4, x_max=0.82, y_min=0.10, n_max=5000):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(n_max):
        g = np.array([
            landscape(p[0] + h, p[1], **LANDSCAPE) - landscape(p[0] - h, p[1], **LANDSCAPE),
            landscape(p[0], p[1] + h, **LANDSCAPE) - landscape(p[0], p[1] - h, **LANDSCAPE),
        ]) / (2 * h)
        p = p - step * g / (np.linalg.norm(g) + 1e-12)
        path.append(p.copy())
        if p[0] > x_max or p[1] < y_min:
            break
    return np.array(path)


def multiline(lines):
    return r'\begin{tabular}{@{}c@{}}' + r'\\'.join(lines) + r'\end{tabular}'


# --------------------------------------------------------------- panels -----
def panel_humans(fig):
    panel(fig, 'a', 'Human experiments')
    L, R = COLS['a']
    side = 1.42
    ax = ax_in(fig, 0.5 * (L + R) - side / 2, 0.58, side, side)
    ax.imshow(sprite('map'), extent=(0, 1, 1, 0), interpolation='nearest')
    for name, x, y, s in [('tomato', 0.215, 0.216, 0.084),
                          ('pink', 0.297, 0.293, 0.105),
                          ('yellow', 0.60, 0.60, 0.105)]:
        ax.imshow(sprite(name), extent=(x, x + s, y + s, y), interpolation='nearest', zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    text_in(fig, 0.5 * (L + R), 0.29, multiline(['700 teams', '1,400 participants']),
            ha='center', va='center', fontsize=FS_CAPTION)


def panel_result(fig):
    panel(fig, 'b', 'The weaker team can win')
    L, R = COLS['b']

    # legend: two rows, left aligned, under the title
    for k, (col, lab) in enumerate([(HA_RED, 'Uniformly strong team'),
                                    (MA_BLUE, 'Weaker, complementary team')]):
        y = 2.02 - 0.20 * k
        x = L + 0.12
        fig.add_artist(Rectangle((x / FIG_W, (y - 0.055) / FIG_H), 0.11 / FIG_W, 0.11 / FIG_H,
                                 transform=fig.transFigure, facecolor=col, edgecolor='none'))
        text_in(fig, x + 0.16, y, lab, ha='left', va='center', fontsize=FS_TEXT)

    ax = ax_in(fig, L + 0.34, 0.50, R - L - 0.44, 1.00)
    w, gap, pos = 0.36, 0.03, (0.0, 1.15)
    for x, key in zip(pos, ['open', 'pb']):
        ha, ma = SCORES[key]
        ax.bar(x - w / 2 - gap / 2, ha, w, color=HA_RED, edgecolor='none')
        ax.bar(x + w / 2 + gap / 2, ma, w, color=MA_BLUE, edgecolor='none')
    # gold star on the winning (mixed-ability) bar of the partially-blocked kitchen
    ax.plot(pos[1] + w / 2 + gap / 2, SCORES['pb'][1] + 1.15, marker='*', ms=15, color=GOLD,
            markeredgecolor=GOLD_EDGE, markeredgewidth=0.7, clip_on=False, zorder=5)
    ax.set_xlim(-0.52, 1.67)
    ax.set_ylim(0, 9.2)
    ax.set_xticks(pos)
    ax.set_xticklabels([multiline(['Open', 'kitchen']),
                        multiline(['Partially-blocked', 'kitchen'])], fontsize=FS_TEXT)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_yticks([])
    ax.set_ylabel('Team score', fontsize=FS_TEXT, labelpad=4)
    ax.patch.set_alpha(0)
    for side, sp in ax.spines.items():
        sp.set_visible(side in ('left', 'bottom'))


def panel_models(fig):
    panel(fig, 'c', 'RL simulations + theory')
    L, R = COLS['c']
    side = 1.50
    ax = ax_in(fig, 0.5 * (L + R) - side / 2 + 0.08, 0.42, side, side)

    xg = np.linspace(-1.0, 1.0, 700)
    yg = np.linspace(0.0, 1.0, 500)
    X, Y = np.meshgrid(xg, yg)
    Z = landscape(X, Y, **LANDSCAPE)
    t = np.linspace(0.0, 1.0, N_LEVELS)
    levels = 0.5 * (np.quantile(Z, t) + Z.min() + t * (Z.max() - Z.min()))
    cmap = mcolors.LinearSegmentedColormap.from_list('landscape', LANDSCAPE_GREYS)
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=mcolors.BoundaryNorm(levels, cmap.N),
                zorder=1)
    ax.contour(X, Y, Z, levels=levels[1:-1], colors=CONTOUR_EDGE, linewidths=0.35,
               linestyles='solid', alpha=0.75, zorder=2)

    # learning trajectory: gradient descent from independent work
    path = descend(TRAJ_START, **TRAJ_STOP)
    ax.plot(path[:-6, 0], path[:-6, 1], color=TRAJ_BLUE, linewidth=1.8,
            linestyle=(0, (2.2, 1.6)), dash_capstyle='round', zorder=3)
    ax.annotate('', xy=path[-1], xytext=path[-12],
                arrowprops=dict(arrowstyle='-|>,head_length=0.6,head_width=0.3', color=TRAJ_BLUE,
                                linewidth=0, shrinkA=0, shrinkB=0, mutation_scale=13), zorder=3)
    ax.scatter([path[0, 0]], [path[0, 1]], s=28, facecolor='white', edgecolor=TRAJ_BLUE,
               linewidth=1.3, zorder=4)

    # the two chefs walking along the trajectory (offset to the left of the path)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    sx, sy = 2.0, 1.0                                       # data units per axes unit
    for name, frac in AGENTS_C:
        img = sprite(name)
        aspect = img.shape[1] / img.shape[0]
        i = int(np.searchsorted(seg, frac * seg[-1]))
        d = path[min(i + 3, len(path) - 1)] - path[max(i - 3, 0)]
        n = np.array([-d[1] / sy, d[0] / sx])               # left-hand normal, axes units
        n /= np.linalg.norm(n)
        cx = (path[i, 0] + 1.0) / sx + AGENT_OFFSET * n[0]
        cy = path[i, 1] / sy + AGENT_OFFSET * n[1]
        hh, ww = AGENT_SIZE, AGENT_SIZE * aspect
        ax.imshow(img, interpolation='nearest', aspect='auto', zorder=5,
                  extent=((cx - ww / 2) * sx - 1.0, (cx + ww / 2) * sx - 1.0,
                          (cy - hh / 2) * sy, (cy + hh / 2) * sy))

    box = dict(boxstyle='round,pad=0.2,rounding_size=0.3', facecolor='white',
               edgecolor='none', alpha=0.8)
    for lines, xa, ya in [(['Independent', 'work'], 0.26, 0.86),
                          (['Ineffective', 'division', 'of labor'], 0.205, 0.145),
                          (['Effective', 'division', 'of labor'], 0.795, 0.145)]:
        ax.text(xa, ya, multiline(lines), transform=ax.transAxes, ha='center', va='center',
                fontsize=FS_SMALL, bbox=box, zorder=6)

    ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor('black'); sp.set_zorder(7)
    ax.set_xlabel('Difference in task propensity', fontsize=FS_SMALL + 0.5, labelpad=4)
    ax.set_ylabel('Propensity to switch tasks', fontsize=FS_SMALL + 0.5, labelpad=4)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_humans(fig)
    panel_result(fig)
    panel_models(fig)
    flow_arrow(fig, COLS['a'][1], COLS['b'][0], 'reveal')
    flow_arrow(fig, COLS['c'][0], COLS['b'][1], 'explain')
    out = HERE / 'figures/fig_cover_letter.pdf'
    fig.savefig(out)                                   # no bbox_inches: size is final
    fig.savefig(out.with_suffix('.png'), dpi=300)
    print('saved', out)


if __name__ == '__main__':
    main()