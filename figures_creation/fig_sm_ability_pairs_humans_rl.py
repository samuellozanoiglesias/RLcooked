#!/usr/bin/env python3
"""
Figure S + statistics -- Game score and specialization index of human teams and
MARL agents at the five ability pairs of main-text Fig. 4:

    (100%,100%)  (70%,40%)  (40%,20%)  (20%,100%)  (10%,10%)
    labelled (walking speed of player 1, cutting speed of player 2)

on the open and partially-blocked (PB) maps.

Layout (same fonts, panel letters, grid and legend band as Fig. 2 and the SI
ablation figures):

                         Human Experiments
        a  Game score            b  Specialization index
                         MARL Simulations
        c  Game score            d  Specialization index
      [ legend band: the five ability pairs ]

Each panel shows condition means +- SEM (humans: across teams; RL: across
simulated games); lines join the open and PB means of the same ability pair.

HUMAN DATA: read from the per-team CSV written by human_data_processing.py
(<prefix>_teams.csv). Team selection, test-phase metrics, the data audit and the
demographics are all done there; this script only analyses and plots.

MARL data (--rl_source sims, default): simulated games read from the simulation tree
(marl_simulations.py, same loaders as Fig. 2e-f and the SI ablation):
    <base>/data/samuel_lozano/cooked/<game_type>/simulations/<init_type>/map_<map>/
        synergy_1.35/specialized_0.25/abilities_<walk>_<cut>/Training_*/checkpoint_750/simulation_*/
or from the cache of a previous run (--rl_csv). Unit of the MARL statistics: one simulated
game (default) or one training (--rl_unit training).

Statistics (written to <output>_stats.txt and <output>_stats.csv), with the
reporting conventions of paper_stats.py (higher group first, |t|, Welch df,
p = .059 / p < .001, F(df1,df2) with eta_p^2):
  * Main experiment, 2 x 2 (HA (100%,100%) vs MA (40%,20%) x map): two-way
    between-subjects ANOVA (effect coding, Type III SS), and Welch's t-tests per map.
  * All five ability pairs, 5 x 2: omnibus ANOVA (ability pair x map); Welch
    contrasts of every MA pair against (100%,100%) within each map (Holm over the 8
    contrasts of each measure; Mann-Whitney U as rank-based check); Map x Ability
    interaction for each MA pair vs (100%,100%) (Holm over the 4 pairs).
  * The same analyses on the MARL simulations, and a human-vs-RL summary of the
    MA - HA differences (sign agreement, Pearson r over the 8 cells).

# 1. human data (once)
    python3 human_data_processing.py --human_data ./data/human_data/ --output_dir ./figures

Usage
# 2. figure + statistics
nohup python3 fig_sm_ability_pairs_humans_rl.py --human_teams ./figures/human_teams.csv --rl_source sims --synergy 1.35 --specialization 0.25 --init_type empty_init --rl_checkpoint 750 > log_fig_sm_ability_pairs_humans_rl.out 2>&1 &
    # re-plot from the RL cache of a previous run
    python3 fig_sm_ability_pairs_humans_rl.py --human_teams ./figures/human_teams.csv \\
        --rl_csv ./figures/fig_sm_ability_pairs_humans_rl_rl_records_sims.csv

Author: Samuel Lozano
"""

import argparse
import re
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_stats as ps                                   # noqa: E402
import marl_simulations as ms                              # noqa: E402
# conditions shared with the human data processing (identical definitions in every script)
from human_data_processing import (PAIRS, HA, MAIN_MA, MA_PAIRS, MAPS, MAP_NAMES,   # noqa: E402
                                   pair_label, pair_key as _pair_key)

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

# ===========================================================================
# 0. CONDITIONS (shared definitions imported from human_data_processing.py) and colors
# ===========================================================================
_PAIR_COLOR = {
    (1.0, 1.0): "#E24A33",   # HA, same red as Fig. 2
    (0.7, 0.4): "#9ECAE1",
    (0.4, 0.2): "#348ABD",   # MA of the main experiment, same blue as Fig. 2
    (0.2, 1.0): "#1F5A96",
    (0.1, 0.1): "#0B2545",
}

# Short names shown in the figure for the two main-experiment conditions
PAIR_ABBREV = {(1.0, 1.0): "HA", (0.4, 0.2): "MA"}


def pair_display(p, tex=False):
    """Figure label: 'HA' for (100%,100%), 'MA' for (40%,20%), otherwise the percentages."""
    for k, name in PAIR_ABBREV.items():
        if abs(p[0] - k[0]) < 1e-6 and abs(p[1] - k[1]) < 1e-6:
            return name
    return pair_label(p, tex=tex)


def read_human_teams(path) -> pd.DataFrame:
    """Per-team CSV of human_data_processing.py; pairs snapped to the canonical PAIRS."""
    h = pd.read_csv(path, dtype={"game_id": str})
    keys = [_pair_key(w, c) for w, c in zip(h["walk"], h["cut"])]
    h = h[[k is not None for k in keys]].copy()
    h["walk"] = [k[0] for k in keys if k is not None]
    h["cut"] = [k[1] for k in keys if k is not None]
    return h


# ===========================================================================
# 2. RL DATA -- simulated games of the simulation tree (marl_simulations.py)
#    .../simulations/<init_type>/map_<map>/synergy_<s>/specialized_<l>/
#        abilities_<walk1>_<cut2>/Training_*/checkpoint_750/simulation_*/
#    walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2 (other speeds 1.0),
#    i.e. exactly the (walk, cut) pairs of PAIRS.
# ===========================================================================
CLUSTER_DIRS = ms.CLUSTER_DIRS


def resolve_rl_root(base_cluster_dir: str, map_nr: str, game_version: str, synergy, specialization,
                    study_name: str = "", path_template: str | None = None,
                    init_type: str = ms.DEFAULT_INIT_TYPE) -> Path:
    """Folder holding the abilities_* folders of one map."""
    return ms.map_root(base_cluster_dir, game_version, map_nr, synergy, specialization,
                       init_type=init_type, study_name=study_name or None, path_template=path_template)


def collect_rl_sims(args, log: list) -> pd.DataFrame:
    """SIMULATED GAMES (same loaders and folder tree as Fig. 2e-f and the SI ablation):
    one row per simulated game, for every pair of PAIRS found (every Training_* and
    simulation_* of checkpoint_<rl_checkpoint> inside abilities_<walk>_<cut>)."""
    base = ms.cluster_base_dir(args.cluster)
    frames = []
    for mp, map_nr in (("Open", args.map_baseline), ("PB", args.map_encouraged)):
        root = resolve_rl_root(base, map_nr, args.game_type, args.synergy, args.specialization,
                               args.study_name or "", args.rl_path_template, args.init_type)
        msg = f"[RL sims] {map_nr} ({mp}): root {root}; checkpoint folder checkpoint_{args.rl_checkpoint}"
        print(msg)
        log.append(msg)
        if not root.exists():
            log.append(f"    [WARN] simulations root not found: {root}")
            continue
        df = ms.collect_records(root, pairs=PAIRS, checkpoint=args.rl_checkpoint,
                                verbose=args.verbose, extra={"map": mp})
        for p in PAIRS:
            if ms.find_ability_dir(root, *p) is None:
                log.append(f"    {pair_label(p):<12} [WARN] no abilities_{p[0]:g}_{p[1]:g} folder")
                continue
            k = df[np.isclose(df["walk"], p[0]) & np.isclose(df["cut"], p[1])] if not df.empty else df
            log.append(f"    {pair_label(p):<12} {len(k):>4} simulations from "
                       f"{k['training_id'].nunique() if len(k) else 0} trainings")
        if not df.empty:
            keys = [_pair_key(w, c) for w, c in zip(df["walk"], df["cut"])]
            df["walk"], df["cut"] = [k[0] for k in keys], [k[1] for k in keys]
            df["episode"] = np.nan
            frames.append(df)
    cols = ["walk", "cut", "map", "training_id", "checkpoint", "simulation", "episode",
            "total_deliveries", "specialization_index"]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def normalize_rl_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the cache written by this script (or any table with walk, cut, map,
    training_id, total_deliveries, specialization_index)."""
    df = df.copy()
    if "map" not in df.columns and "switch_cost_condition" in df.columns:
        df["map"] = df["switch_cost_condition"]
    df["map"] = df["map"].replace({"open": "Open", "pb": "PB", "low": "Open", "high": "PB"})
    if df["walk"].max() > 1.5:
        df["walk"], df["cut"] = df["walk"] / 100, df["cut"] / 100
    keys = [_pair_key(w, c) for w, c in zip(df["walk"], df["cut"])]
    df = df[[k is not None for k in keys]].copy()
    df["walk"] = [k[0] for k in keys if k is not None]
    df["cut"] = [k[1] for k in keys if k is not None]
    if "training_id" not in df.columns:
        df["training_id"] = "unknown"
    return df


def training_means(rl: pd.DataFrame) -> pd.DataFrame:
    """One row per training (mean over its episode window)."""
    return (rl.groupby(["walk", "cut", "map", "training_id"], as_index=False)
              [["total_deliveries", "specialization_index"]].mean())


def _trailing_int(v) -> float:
    """'Training_12' -> 12 (natural sorting); NaN if there is no number."""
    m = re.search(r"(\d+)\s*$", str(v))
    return float(m.group(1)) if m else np.nan


def rl_per_training_table(rl: pd.DataFrame, args, from_tree: bool) -> pd.DataFrame:
    """One row per training (Training_* folder = one seed) x ability pair x map of the MARL
    simulations, averaged over that training's simulated games (the 'training' unit):

      map, map_name                   Open / PB and the map folder
      walk, cut, pair, ability        walking speed of ai_rl_1, cutting speed of ai_rl_2,
                                      '(40%, 20%)'-style label and figure label (HA / MA / %)
      synergy, specialization_level   reward folders synergy_<s> / specialized_<l>
                                      (empty when the records come from --rl_csv)
      checkpoint, training, n_simulations
      score, score_sd                 mean / SD over games of total_deliveries
      specialization_index, specialization_index_sd   mean / SD over games (%)
    """
    cols = ["map", "map_name", "walk", "cut", "pair", "ability", "synergy", "specialization_level",
            "checkpoint", "training", "n_simulations", "score", "score_sd",
            "specialization_index", "specialization_index_sd"]
    if rl is None or rl.empty:
        return pd.DataFrame(columns=cols)
    d = rl.copy()
    for c in ("total_deliveries", "specialization_index"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "checkpoint" not in d.columns:
        d["checkpoint"] = args.rl_checkpoint if from_tree else np.nan
    keys = ["walk", "cut", "map", "training_id"]
    out = (d.groupby(keys, as_index=False, sort=False)
             .agg(checkpoint=("checkpoint", lambda x: ";".join(sorted(map(str, x.dropna().unique())))),
                  n_simulations=("total_deliveries", "size"),
                  score=("total_deliveries", "mean"),
                  score_sd=("total_deliveries", "std"),
                  specialization_index=("specialization_index", "mean"),
                  specialization_index_sd=("specialization_index", "std")))
    out["training"] = out["training_id"].map(
        lambda t: str(t) if str(t).startswith("Training_") else f"Training_{t}")
    out["pair"] = [pair_label((w, c)) for w, c in zip(out["walk"], out["cut"])]
    out["ability"] = [pair_display((w, c)) for w, c in zip(out["walk"], out["cut"])]
    if from_tree:
        out["map_name"] = out["map"].map({"Open": args.map_baseline, "PB": args.map_encouraged})
        out["synergy"], out["specialization_level"] = args.synergy, args.specialization
    else:
        out["map_name"] = out["map"].map({v: k for k, v in MAP_NAMES.items()})
        out["synergy"], out["specialization_level"] = np.nan, np.nan
    # order of the figure: map (Open, PB) -> ability pair (order of PAIRS) -> training (numeric)
    out["_m"] = out["map"].map({m: i for i, m in enumerate(MAPS)})
    out["_p"] = [PAIRS.index(_pair_key(w, c)) if _pair_key(w, c) else len(PAIRS)
                 for w, c in zip(out["walk"], out["cut"])]
    out["_t"] = out["training"].map(_trailing_int)
    out = out.sort_values(["_m", "_p", "_t", "training"], kind="stable", na_position="last")
    return out[cols].reset_index(drop=True)


# ===========================================================================
# 3. ANALYSIS (conventions of paper_stats.py)
# ===========================================================================
METRICS = [("total_deliveries", "Game score", False), ("specialization_index", "Specialization index", True)]


def _cell(df, pair, mp, col):
    return df.loc[(df["walk"] == pair[0]) & (df["cut"] == pair[1]) & (df["map"] == mp), col].to_numpy(float)


def _two(df, pair):
    """HA + one MA pair, with an 'ability' column (HA | MA)."""
    two = df[((df["walk"] == HA[0]) & (df["cut"] == HA[1])) |
             ((df["walk"] == pair[0]) & (df["cut"] == pair[1]))].copy()
    two["ability"] = np.where((two["walk"] == HA[0]) & (two["cut"] == HA[1]), "HA", "MA")
    return two


def analyse(df: pd.DataFrame, who: str, unit: str, out: list, csv_rows: list):
    df = df.copy()
    df["pair"] = [pair_label((w, c)) for w, c in zip(df["walk"], df["cut"])]
    present = [p for p in PAIRS if ((df["walk"] == p[0]) & (df["cut"] == p[1])).any()]

    out.append(f"\n--- {who}: descriptives (unit = {unit}; mean +- SEM) ---")
    for col, name, pct in METRICS:
        out.append(f"  {name}:")
        for p in present:
            for mp in MAPS:
                d = ps.describe(_cell(df, p, mp, col))
                if d["n"]:
                    out.append(f"    {pair_label(p):<12} {mp:<4} n={d['n']:>3}  {ps.fmt_ms(d, pct):<26} "
                               f"SD={d['sd']:6.2f}  median={d['median']:6.2f}")
                    csv_rows.append(dict(source=who, unit=unit, test="descriptive", measure=name,
                                         pair=pair_label(p), map=mp, **d))

    # ---- main experiment ---------------------------------------------------------
    if HA in present and MAIN_MA in present:
        main = _two(df, MAIN_MA)
        out.append(f"\n--- {who}: main experiment, HA {pair_label(HA)} vs MA {pair_label(MAIN_MA)} "
                   f"(two-way ANOVA: ability x map, between-{unit} factors) ---")
        for col, name, pct in METRICS:
            a = ps.anova_type3(main, col, "map", "ability")
            out.append(f"  {name}:")
            for _, r in a.iterrows():
                out.append(f"    {r['term']:<16} ({ps.fmt_F(r)})")
                csv_rows.append(dict(source=who, unit=unit, test="ANOVA 2x2", measure=name, term=r["term"],
                                     F=r["F"], df1=r["df"], df2=r["df_error"], p=r["p"], eta_p2=r["eta_p2"]))
            for mp in MAPS:
                txt, res = ps.fmt_compare("HA", _cell(main, HA, mp, col), "MA", _cell(main, MAIN_MA, mp, col), pct)
                out.append(f"    {mp:<4} {txt}")
                csv_rows.append(dict(source=who, unit=unit, test="Welch HA vs MA", measure=name,
                                     pair=pair_label(MAIN_MA), map=mp, t=res["t"], t_signed_HA_minus_MA=res["t_signed"],
                                     df2=res["df"], p=res["p"]))

    if len(present) < 3:
        return
    sub = df[df["pair"].isin([pair_label(p) for p in present])]
    out.append(f"\n--- {who}: all ability pairs ({len(present)} x 2) ---")
    for col, name, pct in METRICS:
        out.append(f"  {name}: omnibus two-way ANOVA, ability pair x map")
        for _, r in ps.anova_type3(sub, col, "map", "pair").iterrows():
            term = r["term"].replace("pair", "ability pair")
            out.append(f"    {term:<22} ({ps.fmt_F(r)})")
            csv_rows.append(dict(source=who, unit=unit, test="ANOVA omnibus", measure=name, term=term, F=r["F"],
                                 df1=r["df"], df2=r["df_error"], p=r["p"], eta_p2=r["eta_p2"]))

        con = []
        for p in present:
            if p == HA:
                continue
            for mp in MAPS:
                ha, ma = _cell(df, HA, mp, col), _cell(df, p, mp, col)
                txt, res = ps.fmt_compare(pair_label(p, tex=True).join("$$"), ma, "HA", ha, pct)
                con.append((p, mp, txt, res, ps.mann_whitney(ma, ha), ps.describe(ma)["mean"] - ps.describe(ha)["mean"]))
        ph, pu = ps.holm([c[3]["p"] for c in con]), ps.holm([c[4] for c in con])
        out.append(f"  {name}: each MA pair vs HA {pair_label(HA)} within map "
                   f"(Welch; Holm over {len(con)}; Mann-Whitney U, Holm, as rank-based check)")
        for (p, mp, txt, res, pmw, diff), padj, pmwa in zip(con, ph, pu):
            out.append(f"    {mp:<4} {txt}; {ps.fmt_p(padj, 'Holm')}; MWU {ps.fmt_p(pmwa, 'Holm')}")
            csv_rows.append(dict(source=who, unit=unit, test="Welch MA pair vs HA", measure=name, pair=pair_label(p),
                                 map=mp, diff_MA_minus_HA=diff, t=res["t"], t_signed_MA_minus_HA=res["t_signed"],
                                 df2=res["df"], p=res["p"], p_holm=padj, p_mwu=pmw, p_mwu_holm=pmwa))

        inter = [(p, ps.anova_type3(_two(df, p), col, "map", "ability").iloc[2]) for p in present if p != HA]
        ih = ps.holm([r["p"] for _, r in inter])
        out.append(f"  {name}: Map x Ability interaction, each MA pair vs HA (2 x 2 ANOVA; Holm over {len(inter)})")
        for (p, r), padj in zip(inter, ih):
            out.append(f"    {pair_label(p):<12} ({ps.fmt_F(r)}; {ps.fmt_p(padj, 'Holm')})")
            csv_rows.append(dict(source=who, unit=unit, test="Interaction per pair", measure=name, pair=pair_label(p),
                                 F=r["F"], df1=r["df"], df2=r["df_error"], p=r["p"], p_holm=padj, eta_p2=r["eta_p2"]))


def agreement(h: pd.DataFrame, r: pd.DataFrame, out: list, csv_rows: list, by_pair=None):
    out.append("\n--- Humans vs RL: MA - HA difference per map x ability cell (humans: teams; RL: simulated games) ---")
    for col, name, pct in METRICS:
        xs, ys = [], []
        out.append(f"  {name}:")
        for p in MA_PAIRS:
            for mp in MAPS:
                rp = by_pair[p] if by_pair else r          # HA of the same MARL data type as the pair
                hh, hm, rh, rm = _cell(h, HA, mp, col), _cell(h, p, mp, col), _cell(rp, HA, mp, col), _cell(rp, p, mp, col)
                if min(map(len, (hh, hm, rh, rm))) == 0:
                    continue
                dh, dr = hm.mean() - hh.mean(), rm.mean() - rh.mean()
                xs.append(dh)
                ys.append(dr)
                same = np.sign(dh) == np.sign(dr)
                out.append(f"    {pair_label(p):<12} {mp:<4} humans {dh:+7.2f}   RL {dr:+7.2f}   "
                           f"{'same direction' if same else 'OPPOSITE DIRECTION'}")
                csv_rows.append(dict(source="Humans vs RL", test="MA - HA difference", measure=name,
                                     pair=pair_label(p), map=mp, diff_human=dh, diff_rl=dr, same_sign=bool(same)))
        if len(xs) >= 3:
            rr, pp = stats.pearsonr(xs, ys)
            agree = int(np.sum(np.sign(xs) == np.sign(ys)))
            out.append(f"    same direction in {agree}/{len(xs)} cells; Pearson $r = {ps._nolead(rr, 2)}$, {ps.fmt_p(pp)} "
                       f"(n = {len(xs)} cells)")


# ===========================================================================
# 4. FIGURE
# ===========================================================================
PAGE_W, PAGE_H = 804.96, 574.0
S_PANEL = 0.54298                      # same scale factor as the MARL panels of Fig. 2
_FS = 30
_COLS = [(86.0, 372.0), (506.0, 792.0)]
_ROWS = [(46.0, 222.0), (284.0, 460.0)]
LETTER_DX, LETTER_DY = -74.0, 4.0      # panel letter relative to (left, top) of the axes
TITLE_Y = {0: 24.0, 1: 262.0}          # baselines of the centred row titles (as in Fig. 3)
TITLE_SIZE = 20                         # same size as the panel letters, as in Fig. 3
LEGEND_BOX = dict(top=511.0, bottom=552.0, face='#f2f2f2', edge='#595959', lw=2.25)
LEGEND_CENTER = (PAGE_W / 2, 531.5)
LEGEND_SCALE = 0.50
LEGEND_PAD = 14.0
LEGEND_ROW_DY = 24.0               # vertical distance of the two legend rows (mixed data types)
_DODGE = dict(zip(PAIRS, np.linspace(-0.24, 0.24, len(PAIRS))))
_MAP_X = {"Open": 1.0, "PB": 2.0}


def tex(font: str, size: float, text: str) -> str:
    return rf'{{\font\figfont={font} at {size}pt\figfont {text}}}'


def _page_transform(fig):
    return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans


def _rect_to_fig(rect):
    L, R, T, B = rect
    return [L / PAGE_W, 1.0 - B / PAGE_H, (R - L) / PAGE_W, (B - T) / PAGE_H]


_DASH = (0, (3.2, 2.2))
_HA_REF_DX = -0.36            # x offset of the hollow (100%,100%) training-episode reference


def draw_means_panel(ax, df, col, *, ylabel, ylim, percent, s, empty_msg=None,
                     dashed_pairs=(), ref_df=None):
    """dashed_pairs: pairs whose MARL values are training episodes (dashed lines);
    ref_df: (100%,100%) training episodes, drawn as a hollow diamond with a dashed line
    (the matched reference of the dashed pairs)."""
    ax.set_ylim(*ylim)
    if df is None or df.empty:
        ax.text(0.5, 0.5, empty_msg or "no data", transform=ax.transAxes, ha="center", va="center",
                fontsize=_FS * s, color="gray")
    else:
        if ref_df is not None and not ref_df.empty:
            pts = []
            for mp in MAPS:
                d = ps.describe(_cell(ref_df, HA, mp, col))
                x = _MAP_X[mp] + _HA_REF_DX
                pts.append((x, d["mean"]))
                if d["n"]:
                    ax.plot(x, d["mean"], marker="D", markersize=11 * s, markerfacecolor="white",
                            markeredgecolor=_PAIR_COLOR[HA], markeredgewidth=2.0 * s, linestyle="none", zorder=5)
                    if np.isfinite(d["sem"]):
                        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor="black",
                                    elinewidth=1.6 * s, capsize=7 * s, capthick=1.6 * s, zorder=6)
            xs, ys = zip(*pts)
            if all(np.isfinite(ys)):
                ax.plot(xs, ys, color=_PAIR_COLOR[HA], linewidth=2.0 * s, zorder=4, linestyle=_DASH, alpha=0.85)
        for p in PAIRS:
            pts = []
            for mp in MAPS:
                d = ps.describe(_cell(df, p, mp, col))
                x = _MAP_X[mp] + _DODGE[p]
                pts.append((x, d["mean"]))
                if d["n"] == 0:
                    continue
                ax.plot(x, d["mean"], marker="D", markersize=11 * s, markerfacecolor=_PAIR_COLOR[p],
                        markeredgecolor="black", markeredgewidth=1.5 * s, linestyle="none", zorder=5)
                if np.isfinite(d["sem"]):      # +- SEM, drawn over the diamond with caps
                    ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor="black",
                                elinewidth=1.6 * s, capsize=7 * s, capthick=1.6 * s, zorder=6)
            xs, ys = zip(*pts)
            if all(np.isfinite(ys)):
                ax.plot(xs, ys, color=_PAIR_COLOR[p], linewidth=2.5 * s, zorder=4,
                        solid_capstyle="round", alpha=0.85,
                        linestyle=_DASH if p in dashed_pairs else "-")
    ax.set_xticks([_MAP_X[m] for m in MAPS])
    ax.set_xticklabels(MAPS)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.grid(True, color="lightgray", linewidth=0.7 * s, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(mpl.rcParams["axes.linewidth"] * s)
    for which in ("major", "minor"):
        ax.tick_params(axis="both", which=which, labelsize=_FS * s,
                       length=mpl.rcParams[f"xtick.{which}.size"] * s,
                       width=mpl.rcParams[f"xtick.{which}.width"] * s,
                       pad=mpl.rcParams[f"xtick.{which}.pad"] * s)
    ax.yaxis.label.set_size(_FS * s)
    ax.yaxis.labelpad = mpl.rcParams["axes.labelpad"] * s


def _common_ylim(dfs, col, pad_frac=0.08, floor=0.0):
    vals = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for p in PAIRS:
            for mp in MAPS:
                d = ps.describe(_cell(df, p, mp, col))
                if d["n"]:
                    vals += [d["mean"] - (d["sem"] if np.isfinite(d["sem"]) else 0),
                             d["mean"] + (d["sem"] if np.isfinite(d["sem"]) else 0)]
    if not vals:
        return (0, 1)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * pad_frac
    return (max(floor, np.floor(lo - pad)), np.ceil(hi + pad))


def make_figure(human: pd.DataFrame, rl: pd.DataFrame | None, output_path: Path,
                titles=("Humans", "RL"), rl_training_pairs=(), rl_ha_training=None):
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
    page = _page_transform(fig)
    score_lim = _common_ylim([human, rl, rl_ha_training], "total_deliveries")
    spec_lim = (0, 100)
    letters = iter("abcd")
    for row, (df, title) in enumerate(((human, titles[0]), (rl, titles[1]))):
        T, B = _ROWS[row]
        for k, (col, ylabel, ylim, pct) in enumerate((
                ("total_deliveries", "Game score", score_lim, False),
                ("specialization_index", "Specialization index", spec_lim, True))):
            L, R = _COLS[k]
            ax = fig.add_axes(_rect_to_fig((L, R, T, B)))
            draw_means_panel(ax, df, col, ylabel=ylabel, ylim=ylim, percent=pct, s=S_PANEL,
                             empty_msg="RL data not loaded",
                             dashed_pairs=rl_training_pairs if row == 1 else (),
                             ref_df=rl_ha_training if row == 1 else None)
            fig.text(L + LETTER_DX, T + LETTER_DY, tex('cmbx10', 20, next(letters)), transform=page,
                     ha='left', va='baseline', fontsize=20)
        fig.text(PAGE_W / 2, TITLE_Y[row], tex('cmbx10', TITLE_SIZE, title), transform=page,
                 ha='center', va='baseline', fontsize=TITLE_SIZE)

    # -- legend band (same look as Fig. 2); a second row explains the data types when mixed --
    lb, sl = LEGEND_BOX, LEGEND_SCALE
    handles = [matplotlib.lines.Line2D([], [], marker="D", markersize=13 * sl, markerfacecolor=_PAIR_COLOR[p],
                                       markeredgecolor="black", markeredgewidth=1.5 * sl,
                                       color=_PAIR_COLOR[p], linewidth=2.5 * sl) for p in PAIRS]
    labels = [rf"\textbf{{{pair_display(p, tex=True)}}}" for p in PAIRS]
    mixed = rl_ha_training is not None and not rl_ha_training.empty
    cx, cy = LEGEND_CENTER
    rows = [(handles, labels)]
    if mixed:
        h2 = [matplotlib.lines.Line2D([], [], color="black", linewidth=2.5 * sl),
              matplotlib.lines.Line2D([], [], color="black", linewidth=2.5 * sl, linestyle=_DASH),
              matplotlib.lines.Line2D([], [], marker="D", markersize=13 * sl, markerfacecolor="white",
                                      markeredgecolor=_PAIR_COLOR[HA], markeredgewidth=2.0 * sl,
                                      color=_PAIR_COLOR[HA], linewidth=2.0 * sl, linestyle=_DASH)]
        l2 = [r"\textbf{Simulated games}", r"\textbf{Training episodes}",
              rf"\textbf{{{pair_display(HA, tex=True)}, training}}"]
        rows.append((h2, l2))
    dy = LEGEND_ROW_DY if mixed else 0.0
    widths = []
    for i, (h, l) in enumerate(rows):
        yy = cy - dy / 2 + i * dy
        leg = fig.legend(h, l, fontsize=_FS * sl, frameon=False, loc="center", ncol=len(h),
                         handlelength=1.6 if i == 0 else 2.4, columnspacing=1.4, handletextpad=0.5,
                         bbox_to_anchor=(cx / PAGE_W, 1.0 - yy / PAGE_H), bbox_transform=fig.transFigure)
        widths.append(leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 72.0)
    width_pt = max(widths) + 2 * LEGEND_PAD
    top, bottom = lb['top'] - dy / 2, lb['bottom'] + dy / 2
    fig.add_artist(matplotlib.patches.Rectangle(
        (cx - width_pt / 2, top), width_pt, bottom - top, transform=page,
        facecolor=lb['face'], edgecolor=lb['edge'], linewidth=lb['lw'], zorder=0))

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.png'))
    plt.close(fig)
    print(f"\nFigure saved -> {output_path} (+ .png)")


# ===========================================================================
# 5. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="SI figure + statistics: five ability pairs, humans and RL.")
    ap.add_argument("--human_teams", default=None,
                    help="Per-team CSV written by human_data_processing.py (<prefix>_teams.csv)")
    # --- RL: simulated games of the simulation tree ---
    ap.add_argument("--rl_source", choices=["sims", "none"], default="sims",
                    help="'sims' (default): simulated games of every ability pair, read from the "
                         "abilities_<walk>_<cut> folders of the simulation tree; 'none': humans only")
    ap.add_argument("--rl_checkpoint", default=ms.DEFAULT_CHECKPOINT,
                    help="checkpoint_<n> folder of the simulated games (default 750), 'latest' or 'all'")
    ap.add_argument("--rl_path_template", default=None,
                    help="Explicit folder that contains the abilities_* folders ('{map}' substituted)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rl_csv", default=None, help="Cached RL simulation records (skips crawling the tree)")
    ap.add_argument("--rl_unit", choices=["simulation", "training"], default="simulation",
                    help="Unit of the MARL statistics and of the MARL means +- SEM in the figure: one simulated "
                         "game (default, as Fig. 2 and the ablation) or one training (mean of its games)")
    ap.add_argument("--map_baseline", default="baseline_division_of_labor_large")
    ap.add_argument("--map_encouraged", default="encouraged_division_of_labor_large")
    ap.add_argument("--game_type", default="classic_collision",
                    choices=["classic", "classic_collision", "competition", "competition_collision"])
    ap.add_argument("--init_type", default=ms.DEFAULT_INIT_TYPE, choices=["random_init", "empty_init"])
    ap.add_argument("--synergy", type=float, default=1.35)
    ap.add_argument("--specialization", type=float, default=0.25)
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--cluster", default="cuenca", help=f"{list(CLUSTER_DIRS)} or a literal base folder")
    # --- output ---
    ap.add_argument("--titles", nargs=2, default=["Human Experiments", "MARL Simulations"])
    ap.add_argument("--output_dir", default="./figures")
    ap.add_argument("--output_name", default="fig_sm_ability_pairs_humans_rl.pdf")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / args.output_name
    stem = out_pdf.with_suffix("")
    report, csv_rows = [], []

    report.append("=" * 100)
    report.append("STATISTICS: game score and specialization index at the five ability pairs (humans and RL)")
    report.append("=" * 100)
    report.append(
        "Conventions (paper_stats.py, identical in every figure script):\n"
        "  mean +- SEM; game score 2 decimals, specialization index 1 decimal (%).\n"
        "  Welch's t-tests: the group with the HIGHER mean is written first ('A > B: (A vs. B; ...)') and t is\n"
        "    reported as |t|, as in the main text; Welch-Satterthwaite df. Signed t values are in the CSV.\n"
        "  Two-way between-subjects ANOVAs: effect coding, Type III SS, partial eta^2.\n"
        "  p = .059 / p < .001.\n"
        "Definitions\n"
        "  Game score = deliveries in the 180-s test game (humans) / in a 180-s simulated game of the\n"
        f"    MARL agents (checkpoint_{args.rl_checkpoint}, abilities_<walk>_<cut> folders).\n"
        f"  RL source: {args.rl_source if not args.rl_csv else 'cached CSV ' + args.rl_csv}.\n"
        "  Specialization index = 100 * |(D1 - D2)/(D1 + D2) + (C2 - C1)/(C1 + C2)| / 2 (D deliveries,\n"
        "    C cuts per player); no deliveries or no cuts -> 0.\n"
        "    Human teams, selection and test-phase metrics: see the report of human_data_processing.py.\n"
        f"  Units: humans = team; RL = {'simulated game' if args.rl_unit == 'simulation' else 'training (mean over its simulated games)'}.")

    # ---------------- humans ---------------------------------------------------
    human = None
    if args.human_teams:
        human = read_human_teams(args.human_teams)
        report.append(f"\nHuman teams read from {args.human_teams} ({len(human)} teams)")

    if human is not None:
        report.append("\n" + "=" * 100 + "\nHUMANS (unit = team)\n" + "=" * 100)
        analyse(human, "Humans", "team", report, csv_rows)
        report.append("\nLaTeX table (all ability pairs, mean +- SEM across teams):")
        report.append("\\begin{tabular}{llccc}\n\\toprule\nMap & (walk, cut) & Game score & Spec.\\ index & Teams \\\\\n\\midrule")
        for mp in MAPS:
            for p in PAIRS:
                ds, dp = ps.describe(_cell(human, p, mp, "total_deliveries")), ps.describe(_cell(human, p, mp, "specialization_index"))
                if ds["n"]:
                    report.append(f"{mp:<4} & ${pair_label(p, True)}$ & {ps.fmt_ms(ds, False)} & {ps.fmt_ms(dp, True)} & {ds['n']} \\\\")
        report.append("\\bottomrule\n\\end{tabular}")

    # ---------------- RL -------------------------------------------------------
    # Simulated games for every ability pair (one abilities_<walk>_<cut> folder per pair).
    rl_sims_ep = None
    if args.rl_source == "sims":
        if args.rl_csv:
            rl_sims_ep = normalize_rl_csv(pd.read_csv(args.rl_csv))
            report.append(f"\nRL simulated-game records read from {args.rl_csv}")
        else:
            log = []
            rl_sims_ep = collect_rl_sims(args, log)
            report.append("\n" + "=" * 100 + f"\nRL DATA: MARL simulations (simulated games, "
                          f"checkpoint_{args.rl_checkpoint}, every Training_* / simulation_*)\n" + "=" * 100)
            report += log
            rl_sims_ep.to_csv(f"{stem}_rl_records_sims.csv", index=False)
            report.append(f"Cached -> {stem}_rl_records_sims.csv (re-use with --rl_csv)")

    def _coverage(ep_df, kind):
        report.append(f"\nRL coverage, {kind} (trainings; games per training):")
        for p in PAIRS:
            for mp in MAPS:
                g = ep_df[(ep_df["walk"] == p[0]) & (ep_df["cut"] == p[1]) & (ep_df["map"] == mp)]
                if g.empty:
                    report.append(f"    {pair_label(p):<12} {mp:<4} --")
                    continue
                per = g.groupby("training_id").size()
                report.append(f"    {pair_label(p):<12} {mp:<4} trainings={len(per):>3}  per training="
                              f"{per.min()}..{per.max()}")

    rl_plot = None
    if rl_sims_ep is not None and not rl_sims_ep.empty:
        _coverage(rl_sims_ep, "simulated games")
        rl_pt = rl_per_training_table(rl_sims_ep, args, from_tree=not args.rl_csv)
        rl_pt.to_csv(f"{stem}_rl_per_training.csv", index=False)
        report.append(f"\nPer-training RL values -> {Path(stem).name}_rl_per_training.csv "
                      f"({len(rl_pt)} rows = trainings x ability pairs x maps, "
                      f"{int(rl_pt['n_simulations'].sum())} simulated games)")
        print(f"Per-training RL data -> {stem}_rl_per_training.csv ({len(rl_pt)} rows)")
        if args.rl_unit == "simulation":
            # one value per simulated game: the analogue of one human team's game
            rl_plot = rl_sims_ep.copy()
            report.append("\n" + "=" * 100 + "\nMARL SIMULATIONS (unit = simulated game)\n" + "=" * 100)
            analyse(rl_plot, "MARL simulations", "simulated game", report, csv_rows)
        else:
            rl_plot = training_means(rl_sims_ep)
            report.append("\n" + "=" * 100 + "\nMARL SIMULATIONS (unit = training; mean of its simulated games)\n" + "=" * 100)
            analyse(rl_plot, "MARL simulations", "training", report, csv_rows)

    if rl_plot is not None and human is not None:
        report.append("\n" + "=" * 100 + "\nHUMANS vs RL\n" + "=" * 100)
        agreement(human, rl_plot, report, csv_rows, by_pair={p: rl_plot for p in MA_PAIRS})
    elif rl_plot is None:
        report.append("\n" + "=" * 100 + "\nRL: no data loaded\n" + "=" * 100)

    txt = f"{stem}_stats.txt"
    Path(txt).write_text("\n".join(report) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(f"{stem}_stats.csv", index=False)
    print("\n".join(report))
    print(f"\nStatistics saved -> {txt} (+ _stats.csv)")

    if human is not None or rl_plot is not None:
        make_figure(human, rl_plot, out_pdf, titles=tuple(args.titles))
    return 0


if __name__ == "__main__":
    sys.exit(main())