#!/usr/bin/env python3
"""
Human data processing for the five ability pairs (main experiment + second round).

This script does everything that concerns the raw human data, so that the figure scripts
only read its outputs:

    (100%,100%)  (70%,40%)  (40%,20%)  (20%,100%)  (10%,10%)   x   Open / PB map
    labelled (walking speed of player 1, cutting speed of player 2)

Steps
  1. Read the human batches (zips / CSVs / folders) in the canonical batch order.
  2. Select the analysed teams: the first --n_teams (70) games of each ability pair x map, in the
     order of the CSV files (no re-ordering), restricted to the game version
     '_ability_hints_large_map_with_collision' on baseline_division_of_labor_large (Open) and
     encouraged_division_of_labor_large (PB).
     With --selected_games, the game_ids of an existing <prefix>_selected_games.csv are used
     instead (same teams, same order), so the demographics can be computed for exactly the
     teams of a previous run.
  3. Per team (TEST phase only): deliveries and cuts per player, game score, specialization index
         S = 100 * | (D1 - D2)/(D1 + D2) + (C2 - C1)/(C1 + C2) | / 2   (0 if no delivery or no cut)
  4. Per participant (one row per player seat of a selected team): age, sex, ethnicity
     (simplified), country of birth, country of residence, nationality, read from every CSV that
     holds these columns (Prolific export merged into the action logs).
  5. Summaries: data audit, demographics (age: n, mean, SD, median, IQR, range; categorical
     variables: n and % per category, most frequent category), a LaTeX table for the SI and a
     draft sentence for Methods.

Outputs (in --output_dir, file names start with --prefix, default 'human'):
  <prefix>_teams.csv            one row per selected team -- the human input of
                                fig_sm_ability_pairs_humans_rl.py, fig_sm_marl_ablation.py and
                                fig4_human_vs_rl_gridmap.py (same columns as before)
  <prefix>_selected_games.csv   one column per condition with the selected game_ids (row = team 1..70)
  <prefix>_participants.csv     one row per participant (player seat) with condition and demographics
                                (no Prolific IDs unless --keep_prolific_ids)
  <prefix>_demographics.csv     tidy summary: group x variable x category -> n, %, (age statistics)
  <prefix>_demographics_table.tex   LaTeX table (main experiment, second round, all)
  <prefix>_report.txt           audit + demographics in text form

Usage
    python3 human_data_processing.py --human_data ./data/human_data/ --output_dir ./figures
    # same teams as an earlier run:
    python3 human_data_processing.py --human_data ./data/human_data/ --selected_games ./figures/human_selected_games.csv

Author: Samuel Lozano
"""

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_stats as ps                                   # noqa: E402

# ===========================================================================
# 0. CONDITIONS (imported by the figure scripts, so every script uses the same definitions)
# ===========================================================================
PAIRS = [(1.0, 1.0), (0.7, 0.4), (0.4, 0.2), (0.2, 1.0), (0.1, 0.1)]
HA = (1.0, 1.0)
MAIN_MA = (0.4, 0.2)
MA_PAIRS = [p for p in PAIRS if p != HA]
MAPS = ["Open", "PB"]
MAP_NAMES = {"baseline_division_of_labor_large": "Open",
             "encouraged_division_of_labor_large": "PB"}
HUMAN_SUFFIX = "_ability_hints_large_map_with_collision"


def pair_label(p, tex=False):
    a, b = int(round(p[0] * 100)), int(round(p[1] * 100))
    return rf"({a}\%, {b}\%)" if tex else f"({a}%, {b}%)"


def pair_key(w, c):
    """Snap a (walk, cut) pair to the canonical PAIRS entry (or None)."""
    for p in PAIRS:
        if abs(p[0] - float(w)) < 1e-6 and abs(p[1] - float(c)) < 1e-6:
            return p
    return None


def round_of(p):
    return "Main experiment" if p in (HA, MAIN_MA) else "Second round"


# ===========================================================================
# 1. READING THE FILES
# ===========================================================================
_COND_RE = re.compile(r"\[p\((\d+), (\d+)\)_cs\(([\d.]+)\)_ws\(([\d.]+)\)\]_"
                      r"\[p\((\d+), (\d+)\)_cs\(([\d.]+)\)_ws\(([\d.]+)\)\](.*)$")
_USECOLS = ["game_id", "player_id", "prolific_id", "phase", "action", "action_long",
            "map_name", "condition", "game_start_time"]

HUMAN_BATCH_ORDER = [
    "combined_actions_with_demographics",
    "oct2026_new_condition_encouraged",
    "oct2026_new_condition_encouraged_2",
]

# Demographic variables: output name -> accepted column names (compared case-insensitively,
# ignoring spaces / underscores)
DEMO_VARS = {
    "age": ["Age"],
    "sex": ["Sex"],
    "ethnicity": ["Ethnicity simplified", "Ethnicity"],
    "country_of_birth": ["Country of birth"],
    "country_of_residence": ["Country of residence"],
    "nationality": ["Nationality"],
}
DEMO_LABELS = {"age": "Age", "sex": "Sex", "ethnicity": "Ethnicity (simplified)",
               "country_of_birth": "Country of birth", "country_of_residence": "Country of residence",
               "nationality": "Nationality"}
CATEGORICAL = ["sex", "ethnicity", "country_of_residence", "country_of_birth", "nationality"]
# Prolific placeholders for data that are no longer available -> missing
# ('Prefer not to say' is a real answer and is kept as a category)
MISSING_TOKENS = {"", "nan", "none", "null", "undefined", "consent_revoked", "data_expired"}
_ID_COLS = {"prolific_id": ["prolific_id", "Participant id", "participant_id"]}


def _norm(s):
    return re.sub(r"[\s_]+", "", str(s)).lower()


def parse_condition(cond: str):
    """'[p(1, 2)_cs(1.0)_ws(0.4)]_[p(5, 2)_cs(0.2)_ws(1.0)]_suffix' -> ((0.4, 0.2), suffix)."""
    m = _COND_RE.match(str(cond))
    if not m:
        return None, None
    cs1, ws1, cs2, ws2 = map(float, (m.group(3), m.group(4), m.group(7), m.group(8)))
    suffix = m.group(9)
    if cs1 == 1.0 and ws2 == 1.0:
        return (ws1, cs2), suffix
    if cs2 == 1.0 and ws1 == 1.0:
        return (ws2, cs1), suffix
    return None, suffix


def _batch_sort_key(path: Path):
    stem = path.stem
    if stem in HUMAN_BATCH_ORDER:
        return (0, HUMAN_BATCH_ORDER.index(stem), "", 0)
    m = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    return (1, 0, m.group(1), int(m.group(2) or 0))


def order_human_paths(paths, keep_order=False):
    """Expand folders, drop duplicates and, unless keep_order, sort into the canonical batch order."""
    out, seen = [], set()
    for p in map(Path, paths):
        items = (sorted(q for q in p.iterdir() if q.suffix.lower() in (".zip", ".csv"))
                 if p.is_dir() else [p])
        for q in items:
            if not q.exists():
                print(f"[WARN] human data file not found: {q}")
                continue
            r = q.resolve()
            if r not in seen:
                seen.add(r)
                out.append(q)
    return out if keep_order else sorted(out, key=_batch_sort_key)


def _iter_action_sources(paths):
    """(label, file-like) for the action log of every file; inside a zip the slimmest variant."""
    pref = ["combined_actions.csv", "combined_actions_with_history.csv",
            "combined_actions_with_demographics.csv",
            "combined_actions_with_history_and_demographics.csv"]
    for p in map(Path, paths):
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                by_base = {Path(n).name: n for n in names}
                chosen = next((by_base[b] for b in pref if b in by_base), names[0] if names else None)
                if chosen is not None:
                    yield f"{p.name}:{chosen}", io.BytesIO(zf.read(chosen))
        elif p.suffix.lower() == ".csv":
            yield str(p), p


def _iter_all_csvs(paths):
    """(label, bytes) for every CSV, inside zips or given directly."""
    for p in map(Path, paths):
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".csv"):
                        yield f"{p.name}:{n}", zf.read(n)
        elif p.suffix.lower() == ".csv":
            yield str(p), p.read_bytes()


def specialization_from_counts(D1, D2, C1, C2, undefined="zero"):
    if D1 + D2 == 0 or C1 + C2 == 0:
        return 0.0 if undefined == "zero" else np.nan
    return abs((D1 - D2) / (D1 + D2) + (C2 - C1) / (C1 + C2)) * 100 / 2


# ===========================================================================
# 2. TEAM SELECTION AND PER-TEAM METRICS
# ===========================================================================
def read_selected_games(path: Path) -> dict:
    """<prefix>_selected_games.csv -> {game_id: (pair, map, rank)}."""
    df = pd.read_csv(path, dtype=str)
    if "team" in df.columns:
        df = df.set_index("team")
    lab = {f"{mp} {pair_label(p)}": (p, mp) for mp in MAPS for p in PAIRS}
    out = {}
    for col in df.columns:
        key = lab.get(col.strip())
        if key is None:
            print(f"[WARN] unknown column in {path.name}: {col!r} (ignored)")
            continue
        for rank, gid in enumerate(df[col].dropna().str.strip(), start=1):
            out[gid] = (key[0], key[1], rank)
    return out


def load_human_teams(paths, n_teams=70, cut_def="start_cutting", spec_undefined="zero",
                     preselected=None, audit=None):
    """Return (teams, seats): one row per selected team, one row per player seat."""
    audit = audit if audit is not None else []
    frames, seen, file_rows = [], set(), []
    for label, fh in _iter_action_sources(paths):
        df = pd.read_csv(fh, usecols=lambda c: c in _USECOLS, dtype={"game_id": str, "player_id": str},
                         low_memory=False)
        df = df[df["game_id"].notna()]
        df["game_id"] = df["game_id"].str.strip()
        games = set(df["game_id"].unique())
        new_games = games - seen
        file_rows.append((label, len(df), len(games), len(games - new_games)))
        frames.append(df[df["game_id"].isin(new_games)].assign(source=label.split(":")[0]))
        seen |= new_games
    if not frames:
        raise SystemExit("No human CSV found in --human_data")
    raw = pd.concat(frames, ignore_index=True)

    audit.append("Files read (rows, games, games already loaded from an earlier file):")
    for label, nr, ng, ndup in file_rows:
        audit.append(f"  {label:<75} rows={nr:>7}  games={ng:>5}  duplicates={ndup:>4}")
    audit.append(f"  -> {raw['game_id'].nunique()} unique games in total")

    g0 = raw.groupby("game_id", sort=False).agg(condition=("condition", "first"), map_name=("map_name", "first"),
                                                start=("game_start_time", "first"), source=("source", "first"))
    g0["order"] = np.arange(len(g0))
    parsed = g0["condition"].map(parse_condition)
    g0["suffix"] = parsed.map(lambda x: x[1])
    g0["map"] = g0["map_name"].map(MAP_NAMES)
    g0["pair_key"] = parsed.map(lambda x: pair_key(*x[0]) if x[0] else None)
    ok = (g0["suffix"] == HUMAN_SUFFIX) & g0["map"].notna() & g0["pair_key"].notna()

    audit.append("\nAll game versions found (games per condition x map); [USED] = analysed version:")
    for (cond, mp, used), n in g0.assign(used=ok).groupby(["condition", "map_name", "used"]).size().items():
        audit.append(f"  {'[USED]' if used else '      '} {n:>4}  {mp:<36} {cond}")

    sel = g0[ok].copy()
    sel["rank"] = sel.groupby(["pair_key", "map"]).cumcount()
    if preselected is None:
        kept = sel[sel["rank"] < n_teams]
        how = f"first {n_teams} teams per ability pair x map, in the order of the CSV files (no re-ordering)"
    else:
        missing = sorted(set(preselected) - set(g0.index))
        if missing:
            audit.append(f"\n[WARN] {len(missing)} game_id(s) of --selected_games not found in the data: "
                         + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
        kept = g0.loc[[g for g in preselected if g in g0.index]].copy()
        bad = [g for g in kept.index
               if (kept.at[g, "pair_key"], kept.at[g, "map"]) != preselected[g][:2]]
        if bad:
            audit.append(f"[WARN] {len(bad)} game_id(s) whose condition in the data differs from "
                         f"--selected_games: {', '.join(bad[:20])}")
        kept["rank"] = [preselected[g][2] - 1 for g in kept.index]
        kept["pair_key"] = [preselected[g][0] for g in kept.index]
        kept["map"] = [preselected[g][1] for g in kept.index]
        how = "game_ids of --selected_games (same teams and order as the earlier run)"

    raw = raw[raw["game_id"].isin(kept.index)]
    players = raw.groupby("game_id")["player_id"].unique()
    pid_of = raw.groupby(["game_id", "player_id"])["prolific_id"].first()
    test = raw[raw["phase"] == "test"]
    if cut_def == "start_cutting":
        test = test.assign(_cut=test["action"].eq("start cutting"))
    else:
        test = test.assign(_cut=test["action_long"].astype(str).str.contains("cut", case=False))
    test = test.assign(_del=test["action"].eq("deliver"))
    cnt = test.groupby(["game_id", "player_id"])[["_del", "_cut"]].sum()
    no_test = sorted(set(kept.index) - set(test["game_id"].unique()))

    rows, seats = [], []
    for gid, r in kept.iterrows():
        pl = sorted(players[gid])
        D = [int(cnt["_del"].get((gid, p), 0)) for p in pl] + [0, 0]
        C = [int(cnt["_cut"].get((gid, p), 0)) for p in pl] + [0, 0]
        p_key = r["pair_key"]
        rows.append({
            "game_id": gid, "walk": p_key[0], "cut": p_key[1], "map": r["map"],
            "rank": int(r["rank"]) + 1, "order": int(r["order"]),
            "game_start_time": r["start"], "source_file": r["source"], "n_players": len(pl),
            "prolific_ids": ";".join(str(pid_of.get((gid, p))) for p in pl),
            "total_deliveries": sum(D[:2]),
            "D1": D[0], "D2": D[1], "C1": C[0], "C2": C[1],
            "specialization_index": specialization_from_counts(D[0], D[1], C[0], C[1], spec_undefined),
            "spec_undefined": (D[0] + D[1] == 0) or (C[0] + C[1] == 0),
        })
        for k, p in enumerate(pl, start=1):
            seats.append({"game_id": gid, "player_id": str(p), "seat": k,
                          "prolific_id": str(pid_of.get((gid, p))),
                          "map": r["map"], "walk": p_key[0], "cut": p_key[1], "pair": pair_label(p_key),
                          "round": round_of(p_key), "team_rank": int(r["rank"]) + 1})
    teams = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    seats = pd.DataFrame(seats)
    order = dict(zip(teams["game_id"], teams["order"]))
    seats = (seats.assign(_o=seats["game_id"].map(order)).sort_values(["_o", "seat"])
                  .drop(columns="_o").reset_index(drop=True))          # same (CSV) order as teams

    audit.append(f"\nTest phase only: deliveries and cuts are counted in phase == 'test'; "
                 f"selected games without any test-phase row: {len(no_test)}"
                 + (f" -> {', '.join(map(str, no_test))}" if no_test else ""))
    audit.append(f"\nSelection: {how}:")
    avail = sel.groupby(["pair_key", "map"]).size()
    for p in PAIRS:
        for mp in MAPS:
            sub = kept[(kept["pair_key"] == p) & (kept["map"] == mp)]
            k = len(sub)
            flag = "" if k == n_teams else f"   <-- {k} teams (expected {n_teams})"
            span = f"{str(sub['start'].min())[:10]} .. {str(sub['start'].max())[:10]}" if k else ""
            audit.append(f"  {pair_label(p):<12} {mp:<4}  available={int(avail.get((p, mp), 0)):>4}  "
                         f"used={k:>3}  dates {span}{flag}")
            if k:
                audit.append("        from: " + ", ".join(f"{s}={n}" for s, n in
                                                         sub.groupby("source", sort=False).size().items()))
    return teams, seats


def write_selected_games(teams: pd.DataFrame, path: Path) -> str:
    table = {}
    for mp in MAPS:
        for p in PAIRS:
            g = teams[(teams["walk"] == p[0]) & (teams["cut"] == p[1]) & (teams["map"] == mp)].sort_values("rank")
            table[f"{mp} {pair_label(p)}"] = pd.Series(g["game_id"].to_list(), dtype=object)
    df = pd.DataFrame(table)
    df.index = pd.RangeIndex(1, len(df) + 1, name="team")
    df.to_csv(path)
    dup = teams["game_id"][teams["game_id"].duplicated()].unique()
    return (f"  game_id: all {len(teams)} selected games unique  OK" if not len(dup) else
            f"  game_id: {len(dup)} game_id(s) selected more than once  <-- NOT UNIQUE: {', '.join(map(str, dup))}")


# ===========================================================================
# 3. DEMOGRAPHICS
# ===========================================================================
def _match_columns(columns):
    """Map the file's columns to {output name: column} for the demographic and id variables."""
    norm = {_norm(c): c for c in columns}
    found = {}
    for var, names in list(DEMO_VARS.items()) + list(_ID_COLS.items()):
        for n in names:
            if _norm(n) in norm:
                found[var] = norm[_norm(n)]
                break
    for c in ("game_id", "player_id"):
        if c in columns:
            found[c] = c
    return found


def _clean_missing(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(MISSING_TOKENS).fillna(True))


def attach_demographics(paths, seats: pd.DataFrame, audit: list) -> pd.DataFrame:
    """Add the demographic variables to every seat. Each CSV holding at least one demographic
    column is used; it is linked to the seats by (game_id, player_id) when it has a player_id
    column, otherwise by (game_id, prolific_id). Earlier files take precedence."""
    out = seats.copy()
    for v in DEMO_VARS:
        out[v] = pd.Series(pd.NA, index=out.index, dtype="string")
    used = []
    for label, data in _iter_all_csvs(paths):
        header = pd.read_csv(io.BytesIO(data), nrows=0).columns
        cols = _match_columns(header)
        dvars = [v for v in DEMO_VARS if v in cols]
        if not dvars or "game_id" not in cols:
            continue
        key = "player_id" if "player_id" in cols else ("prolific_id" if "prolific_id" in cols else None)
        if key is None:
            continue
        wanted = ["game_id", key] + dvars
        d = pd.read_csv(io.BytesIO(data), usecols=[cols[k] for k in wanted], dtype=str, low_memory=False)
        # select and rename BY NAME: usecols returns the columns in file order, not in the order asked
        d = d[[cols[k] for k in wanted]].rename(columns={cols[k]: k for k in wanted})
        d["game_id"], d[key] = d["game_id"].str.strip(), d[key].str.strip()
        d = d[d["game_id"].isin(out["game_id"])]
        for v in dvars:
            d[v] = _clean_missing(d[v])
        d = d.groupby(["game_id", key], as_index=False)[dvars].first()   # first non-missing value
        m = out[["game_id", key]].merge(d, on=["game_id", key], how="left")
        m.index = out.index
        before = int(out[dvars].notna().any(axis=1).sum())
        for v in dvars:
            out[v] = out[v].fillna(m[v].astype("string"))
        n_match = int(m[dvars].notna().any(axis=1).sum())
        n_new = int(out[dvars].notna().any(axis=1).sum()) - before
        used.append(f"  {label:<75} linked by (game_id, {key}); variables: {', '.join(dvars)}\n"
                    f"      seats with data in this file: {n_match}; newly filled: {n_new}")
    audit.append("\nDemographic sources:")
    audit += used if used else ["  [WARN] no CSV with demographic columns found"]
    n_any = int(out[list(DEMO_VARS)].notna().any(axis=1).sum())
    audit.append(f"  -> seats with at least one demographic value: {n_any} of {len(out)}")
    if used and n_any == 0:
        audit.append("  [WARN] no seat could be linked: check the game_id / player_id formats of the demographic files")
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def _groups(part):
    return [("Main experiment", part[part["round"] == "Main experiment"]),
            ("Second round", part[part["round"] == "Second round"]),
            ("All", part)]


def summarize_demographics(part: pd.DataFrame, top_k: int):
    """Tidy summary rows + report lines. Countries/nationalities: top_k categories of the whole
    sample, the rest pooled as 'Other'."""
    rows, lines = [], []
    tops = {}
    for v in CATEGORICAL:
        vc = part[v].dropna().value_counts()
        tops[v] = list(vc.index) if v in ("sex", "ethnicity") else list(vc.index[:top_k])
    for gname, g in _groups(part):
        n = len(g)
        lines.append(f"\n  {gname}: {n} participants ({g['game_id'].nunique()} teams)")
        a = g["age"].dropna()
        st = dict(n=len(a), mean=a.mean(), sd=a.std(ddof=1), median=a.median(),
                  q1=a.quantile(.25), q3=a.quantile(.75), min=a.min(), max=a.max())
        rows.append(dict(group=gname, variable="age", category="", n_participants=n, **st))
        lines.append(f"    Age: n = {st['n']} (missing {n - st['n']}); mean = {st['mean']:.2f}, SD = {st['sd']:.2f}, "
                     f"median = {st['median']:.0f} (IQR {st['q1']:.0f}-{st['q3']:.0f}), "
                     f"range {st['min']:.0f}-{st['max']:.0f}")
        for v in CATEGORICAL:
            s = g[v]
            n_av = int(s.notna().sum())
            vc = s.dropna().value_counts()
            mode = vc.index[0] if len(vc) else "--"
            lines.append(f"    {DEMO_LABELS[v]}: n = {n_av} (missing {n - n_av}); "
                         f"{len(vc)} categories; most frequent: {mode}")
            cats = [c for c in tops[v] if c in vc.index]
            other = vc.drop(cats, errors="ignore")
            items = [(c, int(vc[c]), np.nan) for c in cats]
            if other.sum():
                items.append((OTHER_POOLED, int(other.sum()), len(other)))
            for c, k, k_cat in items:
                pct = 100 * k / n_av if n_av else np.nan
                shown = f"Other (pooled, {k_cat} categories)" if c == OTHER_POOLED else c
                lines.append(f"      {shown:<40} {k:>5}  ({pct:5.1f}%)")
                rows.append(dict(group=gname, variable=v, category=c, n_participants=n,
                                 n=k, n_available=n_av, pct_of_available=pct, n_pooled_categories=k_cat))
            if n - n_av:
                rows.append(dict(group=gname, variable=v, category="Not available", n_participants=n,
                                 n=n - n_av, n_available=n_av, pct_of_available=np.nan))
    return pd.DataFrame(rows), lines


OTHER_POOLED = "Other (pooled)"


def _tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def demographics_latex(summ: pd.DataFrame) -> str:
    groups = ["Main experiment", "Second round", "All"]
    n_tot = {g: int(summ[summ.group == g]["n_participants"].iloc[0]) for g in groups}
    L = [r"\begin{table}[ht!]", r"\centering",
         r"\caption{Characteristics of the participants of the analyzed teams (Prolific demographic data). "
         r"Age: mean $\pm$ SD (range); other variables: number of participants (percentage of those "
         r"with data). Countries and nationalities: most frequent categories, the rest pooled as Other.}",
         r"\label{tab:si-demographics}", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         " & " + " & ".join(f"{g} ($n = {n_tot[g]:,}$)".replace(",", "{,}") for g in groups) + r" \\",
         r"\midrule"]
    a = summ[summ.variable == "age"].set_index("group")
    L.append("Age (years) & " + " & ".join(
        f"${a.at[g, 'mean']:.1f} \\pm {a.at[g, 'sd']:.1f}$ ({a.at[g, 'min']:.0f}--{a.at[g, 'max']:.0f})"
        for g in groups) + r" \\")
    L.append(r"\quad Not available & " + " & ".join(
        str(int(a.at[g, "n_participants"] - a.at[g, "n"])) for g in groups) + r" \\")
    for v in CATEGORICAL:
        sub = summ[summ.variable == v]
        L.append(r"\addlinespace" + "\n" + rf"\textit{{{DEMO_LABELS[v]}}} & & & \\")
        # same rows in every column: categories in the order of the whole sample, then the pooled
        # 'Other' row, then 'Not available'
        order = [c for c in dict.fromkeys(sub.loc[sub.group == "All", "category"])
                 if c not in (OTHER_POOLED, "Not available")]
        order += [c for c in (OTHER_POOLED, "Not available") if (sub.category == c).any()]
        for c in order:
            cells = []
            for g in groups:
                r = sub[(sub.group == g) & (sub.category == c)]
                if r.empty:
                    cells.append("0")
                elif c == "Not available":
                    cells.append(f"{int(r['n'].iloc[0])}")
                elif c == OTHER_POOLED:
                    cells.append(f"{int(r['n'].iloc[0])} ({r['pct_of_available'].iloc[0]:.1f}\\%; "
                                 f"{int(r['n_pooled_categories'].iloc[0])})")
                else:
                    cells.append(f"{int(r['n'].iloc[0])} ({r['pct_of_available'].iloc[0]:.1f}\\%)")
            label = "Other (number of categories)" if c == OTHER_POOLED else _tex_escape(c)
            L.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def methods_sentence(summ: pd.DataFrame) -> str:
    def age(g):
        r = summ[(summ.group == g) & (summ.variable == "age")].iloc[0]
        return f"mean age = ${r['mean']:.2f}$; SD = ${r['sd']:.2f}$; $n = {int(r['n'])}$ participants with age data"

    def cat(g, v, k=2):
        s = summ[(summ.group == g) & (summ.variable == v) & (summ.category != "Not available")
                 & (summ.category != OTHER_POOLED)].sort_values("n", ascending=False).head(k)
        return ", ".join(f"{_tex_escape(c)}: {int(n)} ({p:.0f}\\%)" for c, n, p in
                         zip(s.category, s.n, s.pct_of_available))
    out = []
    for g in ("Main experiment", "Second round"):
        out.append(f"{g}: ({age(g)}; sex: {cat(g, 'sex', 3)}; most frequent countries of residence: "
                   f"{cat(g, 'country_of_residence')})")
    return "\n".join(out)


# ===========================================================================
# 4. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Human data: team selection, per-team metrics, demographics.")
    ap.add_argument("--human_data", nargs="+", required=True,
                    help="Human zips / CSVs or folders holding them")
    ap.add_argument("--keep_file_order", action="store_true",
                    help="Read the files in the order given instead of the canonical batch order")
    ap.add_argument("--selected_games", default=None,
                    help="Re-use the game_ids of an existing *_selected_games.csv instead of selecting again")
    ap.add_argument("--n_teams", type=int, default=70)
    ap.add_argument("--cut_def", default="start_cutting", choices=["start_cutting", "contains_cut"])
    ap.add_argument("--spec_undefined", default="zero", choices=["zero", "nan"])
    ap.add_argument("--top_k", type=int, default=5,
                    help="Countries / nationalities: number of categories listed before 'Other'")
    ap.add_argument("--keep_prolific_ids", action="store_true",
                    help="Keep the Prolific IDs in <prefix>_participants.csv (dropped by default)")
    ap.add_argument("--output_dir", default="./figures")
    ap.add_argument("--prefix", default="human")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.prefix
    paths = order_human_paths(args.human_data, args.keep_file_order)
    print("Human data files (in reading order):\n  " + "\n  ".join(map(str, paths)))

    pre = read_selected_games(Path(args.selected_games)) if args.selected_games else None
    rep = ["=" * 100, "HUMAN DATA: selection, per-team metrics and demographics", "=" * 100]
    audit = []
    teams, seats = load_human_teams(paths, args.n_teams, args.cut_def, args.spec_undefined, pre, audit)
    rep += audit

    rep.append(f"\nSelected games -> {stem.name}_selected_games.csv:")
    rep.append(write_selected_games(teams, Path(f"{stem}_selected_games.csv")))
    teams.to_csv(f"{stem}_teams.csv", index=False)
    rep.append(f"Per-team data -> {stem.name}_teams.csv ({len(teams)} teams)")

    rep.append(f"\nTeams with no deliveries or no cuts (specialization index "
               f"{'0' if args.spec_undefined == 'zero' else 'NaN'}): {int(teams['spec_undefined'].sum())}")
    for (w, c, mp), n in teams.groupby(["walk", "cut", "map"])["spec_undefined"].sum().items():
        if n:
            rep.append(f"    {pair_label((w, c)):<12} {mp:<4} {int(n)}")

    rep.append("\nCollection batches (cells whose games span more than 30 days):")
    for (w, c, mp), g in teams.groupby(["walk", "cut", "map"]):
        dt = pd.to_datetime(g["game_start_time"], errors="coerce")
        if dt.notna().any() and (dt.max() - dt.min()).days > 30:
            ds = dt.sort_values()
            split = ds[ds.diff().idxmax()]
            b1, b2 = g[dt < split], g[dt >= split]
            rep.append(f"  {pair_label((w, c))} {mp}: batch 1 n={len(b1)}, batch 2 n={len(b2)}")
            for col, name, pct in (("total_deliveries", "Game score", False),
                                   ("specialization_index", "Specialization index", True)):
                txt, _ = ps.fmt_compare("batch 1", b1[col], "batch 2", b2[col], pct)
                rep.append(f"      {name}: {txt}")

    # ---- demographics ----------------------------------------------------------------------
    part = attach_demographics(paths, seats, rep)
    summ, lines = summarize_demographics(part, args.top_k)
    rep.append("\n" + "=" * 100 + "\nDEMOGRAPHICS (one row per participant = player seat of a selected team)\n"
               + "=" * 100)
    rep.append("Prolific placeholders CONSENT_REVOKED / DATA_EXPIRED and empty values count as missing; "
               "'Prefer not to say' is kept as a category.")
    rep += lines
    rep.append("\nDraft for Methods (Participants):\n" + methods_sentence(summ))

    cols = ["round", "map", "pair", "walk", "cut", "game_id", "team_rank", "seat", "player_id"]
    cols += (["prolific_id"] if args.keep_prolific_ids else []) + list(DEMO_VARS)
    part[cols].to_csv(f"{stem}_participants.csv", index=False)
    summ.to_csv(f"{stem}_demographics.csv", index=False)
    Path(f"{stem}_demographics_table.tex").write_text(demographics_latex(summ) + "\n", encoding="utf-8")
    rep.append(f"\nFiles: {stem.name}_participants.csv, {stem.name}_demographics.csv, "
               f"{stem.name}_demographics_table.tex")

    Path(f"{stem}_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\nReport -> {stem}_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())