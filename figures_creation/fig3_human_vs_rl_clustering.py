"""
Figure 3 -- Human vs. MARL behavioral clustering results.

Produces ONE combined, journal-ready figure (PDF + PNG) that reproduces the
Fig3 layout exactly:

    "Human Experiments"   ->  a (action counts) | b (specialization) | c (clusters by map)
    [ shared legend in a grey framed band ]
    "MARL Simulations"    ->  d (action counts) | e (specialization) | f (clusters by map)

Layout notes
------------
Every panel is drawn with its original "native" design (4.8 x 4.0 in figure
for a/c/d/f, 4.2 x 4.0 in for b/e, 24 pt tick labels, 30 pt axis labels,
1.3 pt bar edges, ...) and then uniformly scaled by a per-panel factor
`s` so that it occupies exactly the same place and size as in Fig3.pdf.
All font sizes, line widths, tick lengths, pads, marker sizes and hatch
line widths are multiplied by `s`, so the output is pixel-for-pixel the
same design, but fully vector and produced in a single matplotlib figure.

Requires source CSVs in `csv_directory` (see FILES below). If they are
not found, small synthetic placeholder data are generated automatically
so the script can still be run end-to-end for layout/style testing --
replace with real data for publication.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.ticker import PercentFormatter

# --------------------------------------------------------------------------
# 0. PATHS
# --------------------------------------------------------------------------
csv_directory = './data/human/'
figures_directory = './figures/'

FILES = {
    'action_human':  '3a_human_action_counts.csv',
    'action_rl':     '3d_rl_action_counts.csv',
    'spec_human':    '3b_specialization.csv',
    'spec_rl':       '3e_rl_specialization.csv',
    'cluster_human': '3c_clusters_by_map.csv',
    'cluster_rl':    '3f_clusters_by_map.csv',
}

# Canonical category order used by ALL the 3x CSVs (and everywhere in this
# script): 0 = Independent (I), 1 = Server (CS), 2 = Chef (CC), 3 = Other (O).
#   * 3a / 3d  action counts : 10 rows (actions 0-9) x 4 columns, with a
#                              '# Independent,Server,Chef,Other' header that
#                              is used to map the columns by name.
#   * 3b / 3e  specialization: 4 ragged rows (I, CS, CC, O); the FIRST value
#                              of every row is the row index (0,1,2,3) and is
#                              dropped.
#   * 3c / 3f  clusters      : 4 rows (Open-HA, Open-MA, Blocked-HA,
#                              Blocked-MA) x 4 columns (I, CS, CC, O), raw
#                              counts (normalised per row when plotted).
CATEGORY_KEYS = ['independent', 'server', 'chef', 'other']

# --------------------------------------------------------------------------
# 1. STYLE  (serif/LaTeX typesetting, muted-but-distinct fills + hatches so
#    the figure still reads in black & white print). LaTeX is used if a
#    working TeX installation is available; otherwise the script falls back
#    automatically to a matched non-LaTeX serif style so it still runs.
# --------------------------------------------------------------------------
USE_LATEX = True  # set False to force the non-LaTeX fallback


def configure_style(use_latex=True):
    if use_latex:
        try:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = ['Latin Modern Roman']
            mpl.rcParams['text.latex.preamble'] = r"""
                \usepackage{lmodern}
                \usepackage{amsmath}
                \usepackage{amssymb}
            """
            probe = plt.figure()
            probe.text(0.5, 0.5, r'Test $x^2$')
            probe.canvas.draw()
            plt.close(probe)
            return True
        except Exception as exc:
            plt.close('all')
            print(f"[make_figure3] LaTeX unavailable ({type(exc).__name__}); "
                  f"falling back to a matched non-LaTeX serif style.")
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.serif'] = ['STIX Two Text', 'Times New Roman', 'DejaVu Serif']
    return False


LATEX_OK = configure_style(USE_LATEX)

plt.style.use('ggplot')  # base look, overridden below for a cleaner feel

mpl.rcParams.update({
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor':    '#4d4d4d',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'axes.unicode_minus': False,
    'font.size':          20,
    'axes.titlesize':     20,
    'axes.labelsize':     30,
    'xtick.labelsize':    24,
    'ytick.labelsize':    24,
    'legend.fontsize':    30,
    'axes.linewidth':     1.0,
    'grid.color':         '#d9d9d9',
    'grid.linewidth':     0.7,
    'figure.dpi':         300,
    'savefig.dpi':        600,
    'pdf.fonttype':       42,   # embed real (editable) fonts, not Type-3 bitmaps
    'ps.fonttype':        42,
})


def bold(txt):
    """Bold text that works both with and without LaTeX."""
    return rf"\textbf{{{txt}}}" if LATEX_OK else txt


BOLD_KW = {} if LATEX_OK else {'fontweight': 'bold'}


def heading(txt, size):
    """Row titles and panel letters are set in Computer Modern Bold (cmbx10),
    exactly as in Fig3, rather than the Latin Modern used inside the panels."""
    if LATEX_OK:
        return rf"{{\font\hdfont=cmbx10 at {size}pt\hdfont {txt}}}"
    return txt

# Categories, in a fixed canonical order used everywhere (bars, violins, legend).
CATEGORY_LABELS = [bold(t) for t in (
    "Independent (I)",
    "Cooperative Server (CS)",
    "Cooperative Chef (CC)",
    "Other (O)",
)]

HATCHES = ['.', '//', '\\\\', None]
HUMAN_COLORS = ['#988ED5', '#FBC15E', '#8EBA42', '#777777']

BAR_WIDTH = 0.5

# --------------------------------------------------------------------------
# 1b. PAGE GEOMETRY (all values in points, measured from Fig3.pdf; y is
#     measured from the TOP of the page, as in the original layout).
# --------------------------------------------------------------------------
PAGE_W, PAGE_H = 878.75, 666.125            # 12.20 x 9.25 in

# Axes rectangles (left, right, top, bottom) and per-panel scale factor.
PANELS = {
    'a': dict(rect=(71.56, 284.67, 72.04, 248.50),  s=0.79568),
    'b': dict(rect=(374.79, 575.63, 79.17, 269.21), s=0.85697),
    'c': dict(rect=(640.57, 851.00, 74.86, 249.09), s=0.78568),
    'd': dict(rect=(77.97, 284.84, 438.06, 609.34), s=0.77237),
    'e': dict(rect=(374.79, 575.63, 443.27, 633.31), s=0.85697),
    'f': dict(rect=(640.57, 851.00, 438.96, 613.19), s=0.78568),
}

# Panel letters: (x of text origin, baseline y)
PANEL_LETTERS = {
    'a': (12.0, 64.25), 'b': (304.2, 64.25), 'c': (599.3, 64.25),
    'd': (10.9, 428.35), 'e': (305.8, 428.35), 'f': (601.6, 428.35),
}
LETTER_SIZE = 30.4                            # pt

# Row super-titles: (centre x, baseline y)
TITLES = {
    'Human Experiments': (439.0, 33.47),
    'MARL Simulations':  (439.0, 397.37),
}
TITLE_SIZE = 30.15                             # pt

# Legend band (grey box, full page width) and legend itself.
LEGEND_BOX = dict(top=306.263, bottom=354.75, face='#f2f2f2',
                  edge='#595959', lw=3.0)
LEGEND_CENTER = (439.7, 331.4)
LEGEND_SCALE = 0.6975

# Hatch density. Matplotlib draws hatches on a fixed 1-inch tile; the
# original panels were shrunk by s ~ 0.77-0.86 when assembled, which made
# their hatches correspondingly denser. Scaling the density by 4/3 (6 -> 8
# per inch) reproduces that look in a single, unscaled figure.
HATCH_DENSITY_FACTOR = 4 / 3
_original_hatch = Path.hatch


def _denser_hatch(hatchpattern, density=6):
    return _original_hatch(hatchpattern, int(round(density * HATCH_DENSITY_FACTOR)))


Path.hatch = staticmethod(_denser_hatch)


def _patch_pdf_hatch_fill():
    """The PDF backend fills hatch *shapes* ('.', 'o', '*') with the patch
    face colour, so the dots come out as hollow rings, whereas the raster
    (Agg) backend fills them with the hatch colour. Fig3 was assembled from
    raster panels (solid black dots), so the PDF writer is patched to use
    the hatch colour for the shapes too. Falls back silently if the
    matplotlib internals differ."""
    try:
        from matplotlib.backends import backend_pdf as bpdf
        from matplotlib.transforms import Affine2D

        def writeHatches(self):
            hatchDict = dict()
            sidelen = 72.0
            for hatch_style, name in self._hatch_patterns.items():
                ob = self.reserveObject('hatch pattern')
                hatchDict[name] = ob
                res = {'Procsets': [bpdf.Name(x) for x in
                                    "PDF Text ImageB ImageC ImageI".split()]}
                self.beginStream(
                    ob.id, None,
                    {'Type': bpdf.Name('Pattern'),
                     'PatternType': 1, 'PaintType': 1, 'TilingType': 1,
                     'BBox': [0, 0, sidelen, sidelen],
                     'XStep': sidelen, 'YStep': sidelen,
                     'Resources': res,
                     'Matrix': [1, 0, 0, 1, 0, self.height * 72]})
                stroke_rgb, fill_rgb, hatch, lw = hatch_style
                Op = bpdf.Op
                self.output(stroke_rgb[0], stroke_rgb[1], stroke_rgb[2],
                            Op.setrgb_stroke)
                if fill_rgb is not None:
                    self.output(fill_rgb[0], fill_rgb[1], fill_rgb[2],
                                Op.setrgb_nonstroke,
                                0, 0, sidelen, sidelen, Op.rectangle, Op.fill)
                # shapes are filled with the hatch colour (as in Agg)
                self.output(stroke_rgb[0], stroke_rgb[1], stroke_rgb[2],
                            Op.setrgb_nonstroke)
                self.output(lw, Op.setlinewidth)
                self.output(*self.pathOperations(
                    Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False))
                self.output(Op.fill_stroke)
                self.endStream()
            self.writeObject(self.hatchObject, hatchDict)

        original = bpdf.PdfFile.writeHatches

        def safe_writeHatches(self):
            styles = list(self._hatch_patterns)
            if styles and len(styles[0]) != 4:
                return original(self)
            return writeHatches(self)

        bpdf.PdfFile.writeHatches = safe_writeHatches
    except Exception:
        pass


_patch_pdf_hatch_fill()

# rcParams that encode physical sizes and must be scaled per panel.
_SCALED_RC = [
    'font.size', 'axes.titlesize', 'axes.labelsize', 'xtick.labelsize',
    'ytick.labelsize', 'legend.fontsize', 'axes.linewidth', 'axes.labelpad',
    'grid.linewidth', 'hatch.linewidth', 'lines.linewidth', 'patch.linewidth',
    'lines.markersize', 'lines.markeredgewidth',
    'xtick.major.size', 'xtick.major.width', 'xtick.major.pad',
    'xtick.minor.size', 'xtick.minor.width', 'xtick.minor.pad',
    'ytick.major.size', 'ytick.major.width', 'ytick.major.pad',
    'ytick.minor.size', 'ytick.minor.width', 'ytick.minor.pad',
    'boxplot.whiskerprops.linewidth', 'boxplot.capprops.linewidth',
    'boxplot.boxprops.linewidth', 'boxplot.medianprops.linewidth',
    'boxplot.meanprops.linewidth', 'boxplot.flierprops.linewidth',
]


def scaled_rc(s):
    rc = {}
    for k in _SCALED_RC:
        v = mpl.rcParams[k]
        if isinstance(v, str):  # e.g. 'large' -> absolute points first
            v = mpl.font_manager.FontProperties(size=v).get_size_in_points()
        rc[k] = v * s
    return rc


def pin_tick_sizes(ax, s, base):
    """Ticks can be re-created at draw time from the *global* rcParams, so the
    scaled values are pinned explicitly on the axes."""
    for which in ('major', 'minor'):
        ax.tick_params(axis='both', which=which,
                       labelsize=base['xtick.labelsize'] * s,
                       length=base[f'xtick.{which}.size'] * s,
                       width=base[f'xtick.{which}.width'] * s,
                       pad=base[f'xtick.{which}.pad'] * s,
                       grid_linewidth=base['grid.linewidth'] * s)


# --------------------------------------------------------------------------
# 2. DATA LOADING
# --------------------------------------------------------------------------
def _synthetic_data():
    print("[make_figure3] Source CSVs not found under "
          f"'{csv_directory}' -- generating synthetic placeholder data "
          "for demonstration only. Replace with real data for publication.")
    rng = np.random.default_rng(0)
    d = {}
    d['action_human'] = rng.integers(0, 40, size=(4, 10)).astype(float)
    d['action_rl'] = rng.integers(0, 60, size=(4, 10)).astype(float)

    def ragged(counts):
        return [list(rng.beta(2, 2, size=n)) for n in counts]

    d['spec_human'] = ragged((40, 35, 38, 20))
    d['spec_rl'] = ragged((30, 42, 25, 15))
    d['cluster_human'] = rng.integers(5, 60, size=(4, 4)).astype(float)
    d['cluster_rl'] = rng.integers(5, 50, size=(4, 4)).astype(float)
    return d


def _read_action_counts(path):
    """Return a (4 categories x 10 actions) array in canonical category order."""
    with open(path) as fh:
        header = fh.readline().lstrip('#').strip()
    names = [h.strip().lower() for h in header.split(',')]
    raw = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)   # actions x columns
    if raw.shape[1] != 4 or len(names) != 4:
        raise ValueError(f"{path}: expected 4 category columns, got {raw.shape[1]}")
    order = []
    for key in CATEGORY_KEYS:
        match = [i for i, n in enumerate(names) if n.startswith(key)]
        if len(match) != 1:
            raise ValueError(f"{path}: cannot find column '{key}' in header {names}")
        order.append(match[0])
    return raw[:, order].T


def _read_specialization(path):
    """Ragged rows (I, CS, CC, O); drop the leading row-index column."""
    rows = pd.read_csv(path, header=None).apply(lambda r: r.dropna().tolist(), axis=1).tolist()
    for i, r in enumerate(rows):
        if not r or r[0] != i:
            raise ValueError(f"{path}: row {i} does not start with its index")
    return [r[1:] for r in rows]


def _check_consistency(spec, clusters, label):
    """Number of specialization values per category must equal the cluster totals."""
    n_spec = [len(r) for r in spec]
    n_clu = clusters.sum(axis=0).round().astype(int).tolist()
    if n_spec != n_clu:
        print(f"[make_figure3] WARNING ({label}): specialization counts {n_spec} "
              f"!= cluster totals {n_clu} (categories I, CS, CC, O)")


def load_data():
    if not all(os.path.exists(f'{csv_directory}{fn}') for fn in FILES.values()):
        return _synthetic_data()

    path = {k: f'{csv_directory}{fn}' for k, fn in FILES.items()}
    d = {}
    d['action_human'] = _read_action_counts(path['action_human'])
    d['action_rl'] = _read_action_counts(path['action_rl'])
    d['spec_human'] = _read_specialization(path['spec_human'])
    d['spec_rl'] = _read_specialization(path['spec_rl'])
    d['cluster_human'] = np.loadtxt(path['cluster_human'], delimiter=',')
    d['cluster_rl'] = np.loadtxt(path['cluster_rl'], delimiter=',')

    _check_consistency(d['spec_human'], d['cluster_human'], 'human')
    _check_consistency(d['spec_rl'], d['cluster_rl'], 'MARL')
    return d


# --------------------------------------------------------------------------
# 3. PANEL DRAWING FUNCTIONS  (s = panel scale factor)
# --------------------------------------------------------------------------
def plot_action_counts(ax, data, order=(0, 2, 1, 3), s=1.0):
    n_actions = data.shape[1]
    cum0 = np.zeros(n_actions)
    for o in order:  # stacked bottom -> top: I, CC, CS, O
        ax.bar(range(n_actions), data[o], bottom=cum0, width=BAR_WIDTH,
               facecolor=HUMAN_COLORS[o], hatch=HATCHES[o],
               edgecolor='k', linewidth=1.3 * s)
        cum0 += data[o]

    ax.set_facecolor('w')
    ax.grid(visible=True, axis='both', color=mpl.rcParams['grid.color'])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Action')
    ax.set_xticks(np.arange(n_actions), labels=np.arange(n_actions))
    ax.set_ylim(0, cum0.max() * 1.12)


def plot_specialization(ax, toplot, s=1.0):
    parts = ax.violinplot(toplot, points=500, showmeans=False,
                          showmedians=False, showextrema=False)
    ax.boxplot(toplot, medianprops={"color": 'k', "linewidth": 1.6 * s},
               widths=0.22, boxprops={"linewidth": 1.3 * s}, showfliers=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(HUMAN_COLORS[i])
        pc.set_hatch(HATCHES[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(0)
        pc.set_alpha(0.55)

    means = [np.mean(t) for t in toplot]
    ax.scatter([1, 2, 3, 4], means, marker='d', edgecolor='k',
               linewidths=1.5 * s, s=90 * s ** 2, c=HUMAN_COLORS[:4], zorder=3)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["I", "CS", "CC", "O"], ha="center")
    ax.set_ylabel('Specialization Index')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_facecolor('w')
    ax.grid(visible=True, axis='both', color=mpl.rcParams['grid.color'])


def plot_clusters_by_map(ax, count_of_types, order=(0, 2, 1, 3), s=1.0):
    xs = [0, 1, 2.2, 3.2]
    cum0 = np.zeros(4)
    count_of_types = count_of_types.astype(float)
    count_of_types = count_of_types / count_of_types.sum(axis=1, keepdims=True)
    for o in order:  # rows: Open-HA, Open-MA, Blocked-HA, Blocked-MA
        rows = count_of_types[:, o]
        ax.bar(xs, rows, bottom=cum0, width=BAR_WIDTH,
               facecolor=HUMAN_COLORS[o], hatch=HATCHES[o],
               edgecolor='k', linewidth=1.3 * s)
        cum0 += rows

    ax.set_xticks([0.5, 2.7], labels=['\nOpen', '\nPartially-Blocked'])
    ax.set_xticks(xs, labels=['HA', 'MA', 'HA', 'MA'], minor=True)
    ax.set_ylabel('Frequency', labelpad=3 * s)
    ax.set_facecolor('w')
    ax.grid(visible=True, axis='both', color=mpl.rcParams['grid.color'])
    ax.set_ylim(0, cum0.max() * 1.10)


# --------------------------------------------------------------------------
# 4. FIGURE ASSEMBLY
# --------------------------------------------------------------------------
def _fx(x_pt):
    return x_pt / PAGE_W


def _fy(y_pt_from_top):
    return 1.0 - y_pt_from_top / PAGE_H


def build_figure(data):
    base = {k: mpl.rcParams[k] for k in _SCALED_RC + ['grid.linewidth']}
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))

    draw = {
        'a': lambda ax, s: plot_action_counts(ax, data['action_human'], s=s),
        'b': lambda ax, s: plot_specialization(ax, data['spec_human'], s=s),
        'c': lambda ax, s: plot_clusters_by_map(ax, data['cluster_human'], s=s),
        'd': lambda ax, s: plot_action_counts(ax, data['action_rl'], s=s),
        'e': lambda ax, s: plot_specialization(ax, data['spec_rl'], s=s),
        'f': lambda ax, s: plot_clusters_by_map(ax, data['cluster_rl'], s=s),
    }

    axes = {}
    for key, spec in PANELS.items():
        L, R, T, B = spec['rect']
        s = spec['s']
        with mpl.rc_context(scaled_rc(s)):
            ax = fig.add_axes([_fx(L), _fy(B), (R - L) / PAGE_W, (B - T) / PAGE_H])
            draw[key](ax, s)
            pin_tick_sizes(ax, s, base)
        axes[key] = ax

    # --- panel letters (a-f) ---
    for key, (x, y) in PANEL_LETTERS.items():
        fig.text(_fx(x), _fy(y), heading(key, LETTER_SIZE), fontsize=LETTER_SIZE,
                 ha='left', va='baseline', **BOLD_KW)

    # --- row-group super-titles ---
    for title, (x, y) in TITLES.items():
        fig.text(_fx(x), _fy(y), heading(title, TITLE_SIZE), fontsize=TITLE_SIZE,
                 ha='center', va='baseline', **BOLD_KW)

    # --- grey legend band ---
    top, bot = LEGEND_BOX['top'], LEGEND_BOX['bottom']
    fig.add_artist(Rectangle((0, _fy(bot)), 1, (bot - top) / PAGE_H,
                             transform=fig.transFigure, clip_on=False,
                             facecolor=LEGEND_BOX['face'], edgecolor=LEGEND_BOX['edge'],
                             linewidth=LEGEND_BOX['lw'], zorder=0))

    # --- single shared legend ---
    sl = LEGEND_SCALE
    with mpl.rc_context(scaled_rc(sl)):
        handles = [
            Patch(facecolor=HUMAN_COLORS[i], hatch=HATCHES[i], edgecolor="k",
                  linewidth=1.3 * sl, label=CATEGORY_LABELS[i])
            for i in range(4)
        ]
        fig.legend(handles=handles, ncol=4, loc='center',
                   bbox_to_anchor=(_fx(LEGEND_CENTER[0]), _fy(LEGEND_CENTER[1])),
                   bbox_transform=fig.transFigure, frameon=False,
                   fontsize=24 * sl, handlelength=1.6, handleheight=1,
                   handletextpad=0.8, columnspacing=1.6, borderpad=0.6,
                   labelspacing=1.2)

    return fig, axes


# --------------------------------------------------------------------------
# 5. RUN
# --------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(figures_directory, exist_ok=True)
    data = load_data()

    print("[make_figure3] Assembling combined figure...")
    fig, axes = build_figure(data)

    # NB: no bbox_inches='tight' -- the page size is part of the layout.
    out_path = f'{figures_directory}figure3_human_vs_rl.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'))

    print(f"[make_figure3] Completed. Saved combined figure to: {out_path}")