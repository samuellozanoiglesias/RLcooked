#!/usr/bin/env python3
"""
Figure S (SI Appendix, Sec. "Training length and non-stationary behavior") --
MARL training dynamics (no human data).

For the five ability configurations of the paper
    (1.0, 1.0)  (0.7, 0.4)  (0.4, 0.2)  (0.2, 1.0)  (0.1, 0.1)
      (agent 1 walking speed, agent 2 cutting speed; same axes as main-text Fig. 4)
the figure shows the game score (deliveries per episode; one episode = one 180-s game)
as a function of the training episode. One panel per map (a: Open map,
b: Partially-blocked map), one line per configuration = mean over seeds of the
per-training score smoothed with a centred --smooth-episode window (default 20, the
Fig. 4 window); shaded band = SEM over seeds; dashed vertical line at the episode
used throughout the paper (750).
--metric specialization plots the specialization index instead.

DATA: the TRAINING data of main-text Fig. 4, extracted by fig4_human_vs_rl_gridmap.py
itself (imported; --fig4_script, default: same folder as this file). Its
AbilityGridAnalyzer is used unchanged (same raw dir, same DataProcessor call, same
_filter_speed_configuration and total_deliveries), with ALL episodes of every training.

Style: Latin Modern via LaTeX, 804.96-pt page width, cmbx10 panel letters / titles and
a grey legend band, as Figs. 2 and S2-S3.

Usage:
    nohup python3 fig_sm_marl_training_dynamics.py --synergy 1.35 --specialization 0.25 --init_type empty_init --game_type classic_collision > log_fig_sm_dynamics.out 2>&1 &

    # re-plot from the per-episode records cached by a previous run
    python3 fig_sm_marl_training_dynamics.py --data_csv ./figures/fig_sm_training_dynamics_records.csv

    # specialization index instead of game score
    python3 fig_sm_marl_training_dynamics.py --data_csv ... --metric specialization

    # layout test with synthetic training curves
    python3 fig_sm_marl_training_dynamics.py --demo

Author: Samuel Lozano
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global Style Configuration (PNAS requirements) -- identical to Fig. 2
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        \usepackage{amssymb}
    """,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

_SWITCH_ORDER = ["low", "high"]                 # low = Open map, high = Partially-blocked map
MAP_TITLES = {"low": "Open map", "high": "Partially-blocked map"}

# Five ability configurations of the paper (walk1, cut2), in legend order
CONFIGS = [(1.0, 1.0), (0.7, 0.4), (0.4, 0.2), (0.2, 1.0), (0.1, 0.1)]
# Fig. 2 palette: high ability red, original mixed ability (0.4, 0.2) blue; others from
# the same (ggplot) palette
_COLOR = {(1.0, 1.0): "#E24A33", (0.7, 0.4): "#988ED5", (0.4, 0.2): "#348ABD",
          (0.2, 1.0): "#E5A000", (0.1, 0.1): "#5E9C3A"}
_MARKER = {(1.0, 1.0): "o", (0.7, 0.4): "s", (0.4, 0.2): "D", (0.2, 1.0): "^", (0.1, 0.1): "v"}

# ===========================================================================
# 1. TRAINING DATA -- through fig4_human_vs_rl_gridmap.py (AbilityGridAnalyzer)
# ===========================================================================
_FIG4_DEFAULT = Path(__file__).resolve().with_name("fig4_human_vs_rl_gridmap.py")
_GRID_VALUES = dict(walk1_start=0.1, walk1_end=1.0, walk1_step=0.1,
                    cut2_start=0.1, cut2_end=1.0, cut2_step=0.1)


def load_fig4_module(path):
    path = Path(path) if path else _FIG4_DEFAULT
    if not path.exists():
        raise FileNotFoundError(f"Fig. 4 script not found: {path} (pass --fig4_script)")
    spec = importlib.util.spec_from_file_location("fig4_human_vs_rl_gridmap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_cut_cols(df: pd.DataFrame, override=None):
    if override:
        return tuple(override)
    for a1, a2 in (("cut_ai_rl_1", "cut_ai_rl_2"), ("cuts_ai_rl_1", "cuts_ai_rl_2"),
                   ("cut_count_ai_rl_1", "cut_count_ai_rl_2")):
        if a1 in df.columns and a2 in df.columns:
            return a1, a2
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("cut") and "speed" not in lc and c.endswith("ai_rl_1") and c[:-1] + "2" in df.columns:
            return c, c[:-1] + "2"
    return None


def _specialization(df: pd.DataFrame, cut_cols) -> pd.Series:
    """Per-episode specialization index (Fig. 2 formula), in %."""
    if cut_cols is None:
        return pd.Series(np.nan, index=df.index)
    d1 = pd.to_numeric(df.get("deliver_ai_rl_1", 0), errors="coerce").fillna(0)
    d2 = pd.to_numeric(df.get("deliver_ai_rl_2", 0), errors="coerce").fillna(0)
    c1 = pd.to_numeric(df[cut_cols[0]], errors="coerce").fillna(0)
    c2 = pd.to_numeric(df[cut_cols[1]], errors="coerce").fillna(0)
    tc, td = c1 + c2, d1 + d2
    spec = ((c1 - c2) / tc.where(tc > 0) + (d2 - d1) / td.where(td > 0)).abs() * 100 / 2
    return spec.where((tc > 0) & (td > 0), 0.0)


def collect_records(args) -> pd.DataFrame:
    fig4 = load_fig4_module(args.fig4_script)
    maps = {"low": args.map_baseline, "high": args.map_encouraged}
    frames = []
    for switch in _SWITCH_ORDER:
        an = fig4.AbilityGridAnalyzer(
            map_name=maps[switch], **_GRID_VALUES, init_type=args.init_type, synergy=args.synergy,
            synergy_provided=True, specialization=args.specialization, game_type=args.game_type,
            cluster=args.cluster, study_name=args.study_name, fixed_cut1=1.0, fixed_walk2=1.0, demo=False)
        print(f"\n{'=' * 70}\n{MAP_TITLES[switch]}: {maps[switch]}\n  raw dir: {an.raw_dir}\n{'=' * 70}")
        df = an.load_experimental_data()
        if df is None or len(df) == 0:
            print("  [WARN] no training data returned")
            continue
        df = an.prepare_episode_data(df, episode_selection="all")
        an._create_combined_metrics(df)
        cut_cols = _find_cut_cols(df, args.cut_cols)
        if cut_cols is None:
            print(f"  [WARN] no per-agent cut columns -> specialization unavailable "
                  f"(columns: {sorted(df.columns)}; use --cut_cols)")
        df["specialization_index"] = _specialization(df, cut_cols)
        tcol = an._resolve_training_id_column(df)
        for cfg in CONFIGS:
            sub = an._filter_speed_configuration(df, cfg[0], cfg[1])
            if sub.empty:
                print(f"  {cfg}: no trainings")
                continue
            rec = pd.DataFrame({
                "switch_cost_condition": switch, "walk": cfg[0], "cut": cfg[1],
                "training_id": sub[tcol].astype(str).values if tcol else "unknown",
                "episode": pd.to_numeric(sub["episode"], errors="coerce").values,
                "score": pd.to_numeric(sub["total_deliveries"], errors="coerce").values,
                "specialization": sub["specialization_index"].values})
            print(f"  {cfg}: {rec['training_id'].nunique()} trainings, "
                  f"episodes {int(rec['episode'].min())}-{int(rec['episode'].max())}")
            frames.append(rec)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ===========================================================================
# 2. CURVES
# ===========================================================================
def config_curve(df: pd.DataFrame, switch: str, cfg, metric: str, smooth: int):
    """Episode grid, mean and SEM over trainings of the smoothed per-episode metric."""
    sel = (df["switch_cost_condition"] == switch) & np.isclose(df["walk"], cfg[0]) & \
        np.isclose(df["cut"], cfg[1])
    sub = df.loc[sel, ["training_id", "episode", metric]].dropna()
    if sub.empty:
        return None
    wide = sub.pivot_table(index="episode", columns="training_id", values=metric, aggfunc="mean")
    wide = wide.sort_index()
    if smooth > 1:
        wide = wide.rolling(smooth, center=True, min_periods=max(1, smooth // 2)).mean()
    n = wide.notna().sum(axis=1)
    mean = wide.mean(axis=1)
    sem = wide.std(axis=1, ddof=1) / np.sqrt(n.clip(lower=1))
    ok = n >= 1
    return wide.index.values[ok], mean.values[ok], sem.fillna(0).values[ok], n.values[ok]


# ===========================================================================
# 3. LAYOUT (points, y from the TOP of the page) -- same scale as Figs. 2 / S2-S3
# ===========================================================================
PAGE_W = 804.96
S_PANEL = 0.54298
_FS = 30
_PANEL_COLS = [(78.00, 378.00), (485.00, 785.00)]       # room for %-tick labels on both panels
_TOP, _H = 41.50, 160.0
_TITLE_DY, _LETTER_DY = -20.20, 4.23
_LETTER_X = (3.60, 410.60)                              # same letter-to-axes offset as Fig. S3 (b)
_LEGEND_GAP, _LEGEND_H, _LEGEND_PAD = 52.0, 41.35, 14.0
LEGEND_SCALE = 0.60231


def tex(font, size, text):
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def _style_axis(ax, s):
    ax.yaxis.grid(True, color="lightgray", linewidth=0.7 * s, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(mpl.rcParams["axes.linewidth"] * s)
    for which in ("major", "minor"):
        ax.tick_params(axis="both", which=which, labelsize=_FS * s,
                       length=mpl.rcParams[f"xtick.{which}.size"] * s,
                       width=mpl.rcParams[f"xtick.{which}.width"] * s,
                       pad=mpl.rcParams[f"xtick.{which}.pad"] * s)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_size(_FS * s)
    ax.xaxis.labelpad = ax.yaxis.labelpad = mpl.rcParams["axes.labelpad"] * s


def _cfg_label(cfg):
    return rf"$({cfg[0]:.1f},\,{cfg[1]:.1f})$"


def draw_panel(ax, df, switch, args, s):
    metric = args.metric
    summary = []
    for cfg in CONFIGS:
        cur = config_curve(df, switch, cfg, metric, args.smooth)
        if cur is None:
            continue
        ep, mean, sem, _ = cur
        col = _COLOR[cfg]
        ax.fill_between(ep, mean - sem, mean + sem, color=col, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(ep, mean, color=col, linewidth=2.5 * s, zorder=3, solid_capstyle="round")
        # a few markers per line, so the lines stay identifiable in grey-scale
        idx = np.linspace(0, len(ep) - 1, 11).astype(int)[1:-1]
        ax.plot(ep[idx], mean[idx], linestyle="none", marker=_MARKER[cfg], markersize=7 * s,
                markerfacecolor=col, markeredgecolor="black", markeredgewidth=1.0 * s, zorder=4)
        at = np.interp(args.mark_episode, ep, mean) if ep.min() <= args.mark_episode <= ep.max() else np.nan
        k = int(np.nanargmax(mean))
        summary.append((cfg, at, ep[k], mean[k], mean[-1], ep[-1]))

    ax.axvline(args.mark_episode, color="black", linestyle=(0, (4, 2.5)), linewidth=1.8 * s, zorder=1.8)
    if metric == "specialization":
        ax.set_ylim(0, 100)
        ax.set_ylabel("Specialization index")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        ax.set_yticks([0, 50, 100])
    else:
        ax.set_ylim(0, args.score_ymax)
        ax.set_ylabel("Game score")
    ax.set_xlim(0, args.max_episode)
    ax.set_xlabel("Training episode")
    ax.xaxis.set_major_locator(mtick.MultipleLocator(500))
    _style_axis(ax, s)
    return summary


def make_figure(df, args, out_pdf: Path):
    bottom = _TOP + _H
    leg_top = bottom + _LEGEND_GAP
    page_h = leg_top + _LEGEND_H + 1.73
    fig = plt.figure(figsize=(PAGE_W / 72.0, page_h / 72.0))
    page = Affine2D().scale(1.0, -1.0).translate(0.0, page_h).scale(1.0 / 72.0) + fig.dpi_scale_trans

    def rect(L, R, T, B):
        return [L / PAGE_W, 1.0 - B / page_h, (R - L) / PAGE_W, (B - T) / page_h]

    summaries = {}
    for k, switch in enumerate(_SWITCH_ORDER):
        L, R = _PANEL_COLS[k]
        ax = fig.add_axes(rect(L, R, _TOP, bottom))
        summaries[switch] = draw_panel(ax, df, switch, args, S_PANEL)
        fig.text(_LETTER_X[k], _TOP + _LETTER_DY, tex('cmbx10', 20, "ab"[k]), transform=page,
                 ha='left', va='baseline', fontsize=20)
        fig.text((L + R) / 2, _TOP + _TITLE_DY, tex('cmbx10', 18, MAP_TITLES[switch]), transform=page,
                 ha='center', va='baseline', fontsize=18)

    # -- legend band (box sized to the legend) ----------------------------------
    sl = LEGEND_SCALE
    handles = [matplotlib.lines.Line2D([], [], color=_COLOR[c], linewidth=2.5 * sl * 1.6,
                                       marker=_MARKER[c], markersize=10 * sl, markerfacecolor=_COLOR[c],
                                       markeredgecolor="black", markeredgewidth=1.3 * sl) for c in CONFIGS]
    labels = [_cfg_label(c) for c in CONFIGS]
    handles.append(matplotlib.lines.Line2D([], [], color="black", linestyle=(0, (4, 2.5)),
                                           linewidth=1.8 * sl * 1.6))
    labels.append(rf"\textbf{{Episode {args.mark_episode}}}")
    cy = leg_top + _LEGEND_H / 2
    leg = fig.legend(handles, labels, fontsize=_FS * sl, frameon=False, loc="center",
                     ncol=len(handles), handlelength=1.7, columnspacing=0.9, handletextpad=0.4,
                     bbox_to_anchor=(0.5, 1.0 - cy / page_h), bbox_transform=fig.transFigure)
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    w = bb.width / fig.dpi * 72.0 + 2 * _LEGEND_PAD
    fig.add_artist(matplotlib.patches.Rectangle(((PAGE_W - w) / 2, leg_top), w, _LEGEND_H, transform=page,
                                                facecolor='#f2f2f2', edgecolor='#595959', linewidth=2.25,
                                                zorder=0))
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix('.png'))
    plt.close(fig)
    print(f"\nFigure saved -> {out_pdf} (+ .png)")
    return summaries


# ===========================================================================
# 4. DEMO DATA (synthetic training curves, layout test only)
# ===========================================================================
def demo_records(n_seeds=8, n_ep=2000, seed=0) -> pd.DataFrame:
    """Mixed-ability teams learn cooperation early; high-ability teams later, so that the
    MA > HA advantage on the PB map is transient."""
    rng = np.random.default_rng(seed)
    ep = np.arange(1, n_ep + 1)
    sig = lambda c, w: 1 / (1 + np.exp(-(ep - c) / w))
    # (final score, rise centre, width, final specialization) per map / config
    spec = {"low":  {(1.0, 1.0): (11.0, 450, 120, 50), (0.7, 0.4): (7.5, 500, 140, 55),
                     (0.4, 0.2): (6.2, 550, 150, 55), (0.2, 1.0): (5.6, 650, 180, 40),
                     (0.1, 0.1): (2.0, 900, 250, 20)},
            "high": {(1.0, 1.0): (9.5, 1250, 220, 95), (0.7, 0.4): (8.4, 650, 160, 92),
                     (0.4, 0.2): (7.6, 600, 160, 93), (0.2, 1.0): (2.5, 900, 300, 50),
                     (0.1, 0.1): (1.2, 1100, 300, 90)}}
    rows = []
    for switch, d in spec.items():
        for cfg, (top, c, w, sp) in d.items():
            base = 0.15 * top
            for s in range(n_seeds):
                cc = c + rng.normal(0, 60)
                if switch == "high" and cfg == (1.0, 1.0):          # HA: independent plateau first
                    mu = 2 + 3.6 * sig(350, 90) + (top - 5.6) * sig(cc, w)
                else:
                    mu = base + (top - base) * sig(cc, w)
                sc = np.clip(rng.normal(mu, 1.3), 0, None).round()
                spc = np.clip(rng.normal(sp * sig(cc, w) + 25 * (1 - sig(cc, w)), 12), 0, 100)
                rows.append(pd.DataFrame({"switch_cost_condition": switch, "walk": cfg[0], "cut": cfg[1],
                                          "training_id": f"demo_{s}", "episode": ep, "score": sc,
                                          "specialization": spc}))
    return pd.concat(rows, ignore_index=True)


# ===========================================================================
# 5. CLI
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description="SI figure: MARL training dynamics (game score vs training episode).")
    p.add_argument("--fig4_script", default=None, help="Path to fig4_human_vs_rl_gridmap.py")
    p.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    p.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    p.add_argument("--game_type", default="classic_collision", choices=["classic", "classic_collision"])
    p.add_argument("--init_type", default="empty_init", choices=["random_init", "empty_init"])
    p.add_argument("--synergy", type=float, default=1.35)
    p.add_argument("--specialization", type=float, default=0.25)
    p.add_argument("--study_name", default=None)
    p.add_argument("--cluster", default="cuenca")
    p.add_argument("--cut_cols", nargs=2, default=None, metavar=("AGENT1", "AGENT2"))
    p.add_argument("--data_csv", default=None, help="Cached per-episode records (skips loading)")
    p.add_argument("--metric", choices=["score", "specialization"], default="score")
    p.add_argument("--smooth", type=int, default=20,
                   help="Centred rolling window (episodes) per training; 20 = the Fig. 4 window")
    p.add_argument("--mark_episode", type=int, default=750)
    p.add_argument("--max_episode", type=int, default=2000)
    p.add_argument("--score_ymax", type=float, default=16)
    p.add_argument("--output_dir", default="./figures")
    p.add_argument("--output_name", default="fig_sm_marl_training_dynamics.pdf")
    p.add_argument("--demo", action="store_true", help="Synthetic training curves (layout test)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        df = demo_records()
    elif args.data_csv:
        df = pd.read_csv(args.data_csv)
    else:
        df = collect_records(args)
        cache = out_dir / "fig_sm_training_dynamics_records.csv"
        df.to_csv(cache, index=False)
        print(f"\nPer-episode records cached -> {cache}  (re-use with --data_csv)")
    if df.empty:
        print("No data -- nothing to plot.")
        return 1

    name = args.output_name
    if args.metric != "score":
        name = name.replace(".pdf", f"_{args.metric}.pdf")
    if args.demo:
        name = name.replace(".pdf", "_DEMO.pdf")
    summaries = make_figure(df, args, out_dir / name)

    unit = "%" if args.metric == "specialization" else " deliveries"
    print(f"\n{'=' * 78}\nSummary ({args.metric}, mean over seeds, smoothed over {args.smooth} episodes)\n{'=' * 78}")
    rows = []
    for switch, summ in summaries.items():
        for cfg, at, ep_max, vmax, vend, ep_end in summ:
            print(f"  {MAP_TITLES[switch]:<22} {str(cfg):<11} episode {args.mark_episode}: {at:6.2f}{unit}"
                  f" | max {vmax:6.2f} at episode {ep_max:.0f} | final (episode {ep_end:.0f}): {vend:6.2f}")
            rows.append({"map": MAP_TITLES[switch], "walk": cfg[0], "cut": cfg[1],
                         f"{args.metric}_at_{args.mark_episode}": at, f"max_{args.metric}": vmax,
                         "episode_of_max": ep_max, f"final_{args.metric}": vend, "final_episode": ep_end})
        ha = {c: (a, e) for c, a, _, _, e, _ in summ}
        if (1.0, 1.0) in ha and (0.4, 0.2) in ha:
            print(f"  {MAP_TITLES[switch]:<22} (0.4, 0.2) - (1.0, 1.0): episode {args.mark_episode}: "
                  f"{ha[(0.4, 0.2)][0] - ha[(1.0, 1.0)][0]:+.2f} | final: {ha[(0.4, 0.2)][1] - ha[(1.0, 1.0)][1]:+.2f}")
    pd.DataFrame(rows).to_csv(out_dir / name.replace(".pdf", "_summary.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())