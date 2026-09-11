"""
Combined Figure 2 — Human vs. RL behavioral clustering results.

Produces 7 individual PNGs (6 panels + 1 legend), and then produces 
a single, journal-ready (PNAS first-tier style) combined figure with 
two row-groups of three panels each:

    Row 1  "Human Results"   ->  Panel A (action counts) | Panel B (specialization) | Panel C (clusters by map)
    Row 2  "RL Results"      ->  Panel D (action counts) | Panel E (specialization) | Panel F (clusters by map)

Requires source CSVs in `csv_directory` (see FILES below). If they are
not found, small synthetic placeholder data are generated automatically
so the script can still be run end-to-end for layout/style testing --
replace with real data for publication.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import os

# --------------------------------------------------------------------------
# 0. PATHS
# --------------------------------------------------------------------------
csv_directory = './human/'
figures_directory = './figures/'

FILES = {
    'action_human':  '2a_human_action_counts.csv',
    'action_rl':     '2d_rl_action_counts.csv',
    'spec_human':    '2b_specialization.csv',
    'spec_rl':       '2e_rl_specialization.csv',
    'cluster_human': '2c_clusters_by_map.csv',
    'cluster_rl':    '2f_rl_clusters_by_map.csv',
}

# --------------------------------------------------------------------------
# 1. STYLE  (PNAS-like: serif/LaTeX typesetting, small type, clean panel
#    labels, muted-but-distinct fills + hatches so the figure still reads
#    in black & white print). LaTeX is used if a working TeX installation
#    is available on this machine; otherwise the script falls back
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
            print(f"[make_figure2] LaTeX unavailable ({type(exc).__name__}); "
                  f"falling back to a matched non-LaTeX serif style.")
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.serif'] = ['STIX Two Text', 'Times New Roman', 'DejaVu Serif']
    return False

LATEX_OK = configure_style(USE_LATEX)

plt.style.use('ggplot')  # base look, overridden below for a cleaner PNAS feel

mpl.rcParams.update({
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'savefig.facecolor':'white',
    'axes.edgecolor':   '#4d4d4d',
    'axes.labelcolor':  'black',
    'xtick.color':      'black',
    'ytick.color':      'black',
    'axes.unicode_minus': False,
    'font.size':         20,
    'axes.titlesize':    20,
    'axes.labelsize':    30,
    'xtick.labelsize':    24,
    'ytick.labelsize':    24,
    'legend.fontsize':    30,
    'axes.linewidth':    1.0,
    'grid.color':        '#d9d9d9',
    'grid.linewidth':    0.7,
    'figure.dpi':        300,
    'savefig.dpi':        600,
    'pdf.fonttype':       42,   # embed real (editable) fonts, not Type-3 bitmaps
    'ps.fonttype':        42,
})

# Categories, in a fixed canonical order used everywhere (bars, violins,
# legend). This is the same order/semantics as the original per-panel
# legends ("Independent", "Cooperative Server", "Cooperative Chef", "Other").
CATEGORY_LABELS = [
    r"\textbf{Independent (I)}",
    r"\textbf{Cooperative Server (CS)}",
    r"\textbf{Cooperative Chef (CC)}",
    r"\textbf{Other (O)}",
]

HATCHES = ['.', '//', '\\\\', None]
HUMAN_COLORS  = ['#988ED5','#FBC15E','#8EBA42','#777777']
RL_COLORS  = ['#FBC15E', '#8EBA42', '#988ED5', '#777777']

['#4C72B0', '#DD8452', '#55A868', '#8C8C8C']  # fill colour per category (index-matched)

['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42']

BAR_WIDTH = 0.5
PANEL_LABEL_KW = dict(transform=None, fontsize=13, fontweight='bold',
                       va='bottom', ha='left')

# --------------------------------------------------------------------------
# 2. DATA LOADING
# --------------------------------------------------------------------------
def _synthetic_data():
    print("[make_figure2] Source CSVs not found under "
          f"'{csv_directory}' -- generating synthetic placeholder data "
          "for demonstration only. Replace with real data for publication.")
    rng = np.random.default_rng(0)
    d = {}
    d['action_human']  = rng.integers(0, 40, size=(4, 10)).astype(float)
    d['action_rl']     = rng.integers(0, 60, size=(4, 10)).astype(float)

    def ragged(counts):
        rows = [rng.beta(2, 2, size=n) for n in counts]
        m = max(len(r) for r in rows)
        return [list(r) for r in rows], m

    rows, _ = ragged((40, 35, 38, 20))
    d['spec_human'] = rows
    rows, _ = ragged((30, 42, 25, 15))
    d['spec_rl'] = rows

    d['cluster_human'] = rng.integers(5, 60, size=(4, 4)).astype(float)
    d['cluster_rl']    = rng.integers(5, 50, size=(4, 4)).astype(float)
    return d


def load_data():
    if not all(os.path.exists(f'{csv_directory}{fn}') for fn in FILES.values()):
        return _synthetic_data()

    d = {}
    d['action_human'] = np.loadtxt(f'{csv_directory}{FILES["action_human"]}', delimiter=',')
    d['action_rl']    = np.loadtxt(f'{csv_directory}{FILES["action_rl"]}', delimiter=',')

    d['spec_human'] = pd.read_csv(f'{csv_directory}{FILES["spec_human"]}', header=None) \
                          .apply(lambda r: r.dropna().tolist(), axis=1).tolist()
    d['spec_rl'] = pd.read_csv(f'{csv_directory}{FILES["spec_rl"]}', header=None) \
                       .apply(lambda r: r.dropna().tolist(), axis=1).tolist()

    d['cluster_human'] = np.loadtxt(f'{csv_directory}{FILES["cluster_human"]}', delimiter=',')
    d['cluster_rl']    = np.loadtxt(f'{csv_directory}{FILES["cluster_rl"]}', delimiter=',')
    return d


# --------------------------------------------------------------------------
# 3. PANEL DRAWING FUNCTIONS
# --------------------------------------------------------------------------
def plot_action_counts(ax, data, order, hatch_by_category, RL=False):
    n_actions = data.shape[1]
    cum0 = np.zeros(n_actions)
    for i, o in enumerate(order):
        hatch_idx = o if hatch_by_category else i
        if RL:
            facecolor = RL_COLORS[o]
        else:
            facecolor = HUMAN_COLORS[o]
        ax.bar(range(n_actions), data[o], bottom=cum0, width=BAR_WIDTH,
               facecolor=facecolor, hatch=HATCHES[hatch_idx],
               edgecolor='k', linewidth=1.3)
        cum0 += data[o]

    ax.set_facecolor('w')
    ax.grid(visible=True, axis='both', color=mpl.rcParams['grid.color'])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Action')
    ax.set_xticks(np.arange(n_actions), labels=np.arange(n_actions))
    ax.set_ylim(0, cum0.max() * 1.12)


def plot_specialization(ax, toplot):
    parts = ax.violinplot(toplot, points=500, showmeans=False,
                           showmedians=False, showextrema=False)
    ax.boxplot(toplot, medianprops={"color": 'k', "linewidth": 1.6},
               widths=0.22, boxprops={"linewidth": 1.3}, showfliers=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(HUMAN_COLORS[i])
        pc.set_hatch(HATCHES[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(0)
        pc.set_alpha(0.55)

    means = [np.mean(t) for t in toplot]
    ax.scatter([1, 2, 3, 4], means, marker='d', edgecolor='k',
               linewidths=1.5, s=90, c=HUMAN_COLORS[:4], zorder=3)

    labels = ["I", "CS", "CC", "O"]

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel('Specialization Index')
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_facecolor('w')
    ax.grid(visible=True, axis='both', color=mpl.rcParams['grid.color'])


def plot_clusters_by_map(ax, count_of_types, order, hatch_by_category, row_index=None, RL=False):
    xs = [0, 1, 2.2, 3.2]
    cum0 = np.zeros(4)
    count_of_types = count_of_types.astype(float)
    count_of_types = count_of_types / count_of_types.sum(axis=1, keepdims=True)
    for i, o in enumerate(order):
        hatch_idx = o if hatch_by_category else i
        if RL:
            facecolor = RL_COLORS[o]
        else:
            facecolor = HUMAN_COLORS[o]
        rows = count_of_types[row_index, o] if row_index is not None else count_of_types[:, o]
        ax.bar(xs, rows, bottom=cum0, width=BAR_WIDTH,
               facecolor=facecolor, hatch=HATCHES[hatch_idx],
               edgecolor='k', linewidth=1.3)
        cum0 += rows

    ax.set_xticks([0.5, 2.7], labels=['\nOpen', '\nPartially-Blocked'])
    ax.set_xticks(xs, labels=['HA', 'MA', 'HA', 'MA'], minor=True)
    ax.set_ylabel('Frequency', labelpad=3)
    ax.set_facecolor('w')
    ax.grid(visible=True, axis='both', color=mpl.rcParams['grid.color'])
    ax.set_ylim(0, cum0.max() * 1.10)


# --------------------------------------------------------------------------
# 4. PREPARE INDIVIDUAL PANELS
# --------------------------------------------------------------------------
def create_individual_pngs(data):
    """Generates and saves the 6 panels and 1 legend separately before the composite."""
    print("[make_figure2] Generating individual panels...")
    
    # Common figure size mapping roughly to the grid allocations
    fs_wide = (4.8, 4.0)
    fs_narr = (4.2, 4.0)
    
    # --- Panel A ---
    fig, ax = plt.subplots(figsize=fs_wide)
    plot_action_counts(ax, data['action_human'], order=[0, 2, 1, 3], hatch_by_category=True)
    fig.savefig(f"{figures_directory}/figure2_panel_A.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- Panel B ---
    fig, ax = plt.subplots(figsize=fs_narr)
    plot_specialization(ax, data['spec_human'])
    fig.savefig(f"{figures_directory}/figure2_panel_B.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- Panel C ---
    fig, ax = plt.subplots(figsize=fs_wide)
    plot_clusters_by_map(ax, data['cluster_human'], order=[0, 2, 1, 3], hatch_by_category=True, row_index=[1, 0, 3, 2])
    fig.savefig(f"{figures_directory}/figure2_panel_C.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- Panel D ---
    fig, ax = plt.subplots(figsize=fs_wide)
    plot_action_counts(ax, data['action_rl'], order=[2, 1, 0, 3], hatch_by_category=False, RL=True)
    fig.savefig(f"{figures_directory}/figure2_panel_D.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- Panel E ---
    fig, ax = plt.subplots(figsize=fs_narr)
    plot_specialization(ax, data['spec_rl'])
    fig.savefig(f"{figures_directory}/figure2_panel_E.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- Panel F ---
    fig, ax = plt.subplots(figsize=fs_wide)
    plot_clusters_by_map(ax, data['cluster_rl'], order=[2, 1, 0, 3], hatch_by_category=False, row_index=None, RL=True)
    fig.savefig(f"{figures_directory}/figure2_panel_F.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- Legend ---
    handles = [
        Patch(facecolor=HUMAN_COLORS[i], hatch=HATCHES[i], edgecolor="k", 
              linewidth=1.3, label=CATEGORY_LABELS[i]) 
        for i in range(4)
    ]
    legend_fig = plt.figure(figsize=(4.8, 2.0))
    legend_fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=24,
        handlelength=1.6,
        handleheight=1,
        handletextpad=0.8,
        columnspacing=1.6,
        borderpad=0.6,
        labelspacing=1.2,
    )
    legend_fig.savefig(f"{figures_directory}/figure2_legend.png", dpi=600, transparent=True, bbox_inches="tight")
    plt.close(legend_fig)


# --------------------------------------------------------------------------
# 5. FIGURE ASSEMBLY
# --------------------------------------------------------------------------
def build_figure(data):
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = GridSpec(nrows=2, ncols=3, figure=fig,
                  left=0.07, right=0.985, top=0.85, bottom=0.20,
                  hspace=1.15, wspace=0.38,
                  width_ratios=[1.0, 0.85, 1.05])

    axes = {
        'A': fig.add_subplot(gs[0, 0]),
        'B': fig.add_subplot(gs[0, 1]),
        'C': fig.add_subplot(gs[0, 2]),
        'D': fig.add_subplot(gs[1, 0]),
        'E': fig.add_subplot(gs[1, 1]),
        'F': fig.add_subplot(gs[1, 2]),
    }

    # --- Human row (A, B, C) ---
    plot_action_counts(axes['A'], data['action_human'], order=[0, 2, 1, 3], hatch_by_category=True)
    plot_specialization(axes['B'], data['spec_human'])
    plot_clusters_by_map(axes['C'], data['cluster_human'], order=[0, 2, 1, 3], hatch_by_category=True, row_index=[1, 0, 3, 2])

    # --- RL row (D, E, F) ---
    plot_action_counts(axes['D'], data['action_rl'], order=[2, 1, 0, 3], hatch_by_category=False, RL=True)
    plot_specialization(axes['E'], data['spec_rl'])
    plot_clusters_by_map(axes['F'], data['cluster_rl'], order=[2, 1, 0, 3], hatch_by_category=False, row_index=None, RL=True)

    # --- panel labels (A-F) ---
    panel_labels = {}
    for letter, ax in axes.items():
        panel_labels[letter] = ax.text(-0.16, 1.14, letter, transform=ax.transAxes,
                                       fontsize=14, fontweight='bold', va='bottom', ha='left')

    # --- row-group super-titles ---
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    x_center = (lefts[0] + rights[-1]) / 2
    fig.text(x_center, tops[0] + 0.055, 'Human Results', ha='center', va='bottom', fontsize=15, fontweight='bold')
    fig.text(x_center, tops[1] + 0.055, 'RL Results', ha='center', va='bottom', fontsize=15, fontweight='bold')

    # --- single shared legend ---
    handles = [
        Patch(facecolor=HUMAN_COLORS[i], hatch=HATCHES[i], edgecolor="k",
              linewidth=1.3, label=CATEGORY_LABELS[i])
        for i in range(4)
    ]
    fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               frameon=False, handlelength=2.0, handleheight=1.5, columnspacing=2.2)

    return fig, panel_labels, handles


# --------------------------------------------------------------------------
# 6. RUN
# --------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(figures_directory, exist_ok=True)
    data = load_data()

    # 1. First create the perfectly proportioned individual PNGs (7 total)
    create_individual_pngs(data)

    # 2. Then assemble and save the full combined figure
    print("[make_figure2] Assembling combined figure...")
    fig, panel_labels, handles = build_figure(data)
    
    out_path = f'{figures_directory}figure2_human_vs_rl.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight')
    
    print(f"[make_figure2] Completed. Saved combined figure to: {out_path}")