#!/usr/bin/env python3
"""
simulations_per_condition.py -- inventory of the MARL simulation tree.

Walks

    <root>/[with_pretraining/]map_<map>/synergy_<s>/specialized_<l>/abilities_<walk>_<cut>/
        Training_<id>/checkpoint_<n>/simulation_<id>/

(default root: /data/samuel_lozano/cooked/classic_collision/simulations/empty_init)
and counts, for every condition, the trainings and the simulated games, saving their ids.
Folders under <root>/with_pretraining/ are flagged with with_pretraining = True.
abilities_<walk>_<cut>: walking speed of ai_rl_1 and cutting speed of ai_rl_2.

Outputs (in --output_dir, prefix --output_name):
  <prefix>_simulations.csv  one row per simulation_* folder: with_pretraining, map, synergy,
                            specialization, walk, cut, abilities, training_id, checkpoint,
                            simulation_id, path (+ n_files with --check_contents)
  <prefix>_trainings.csv    one row per Training_* x checkpoint (trainings with no checkpoint
                            folder or no simulations are kept, with n_simulations = 0)
  <prefix>_conditions.csv   one row per condition (with_pretraining x map x synergy x
                            specialization x abilities x checkpoint): n_trainings,
                            n_simulations, simulations per training (min..max), and the
                            ';'-separated training ids
  <prefix>_summary.txt      readable summary + folders that did not match the structure

Usage
    python3 simulations_per_condition.py
    python3 simulations_per_condition.py --checkpoint 750 --output_dir ./figures
    python3 simulations_per_condition.py --root /other/simulations/empty_init --check_contents
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = "/data/samuel_lozano/cooked/classic_collision/simulations/empty_init"
PRETRAINING_DIR = "with_pretraining"

_RE = {
    "map": re.compile(r"^map_(.+)$"),
    "synergy": re.compile(r"^synergy_(.+)$"),
    "specialization": re.compile(r"^specialized_(.+)$"),
    "abilities": re.compile(r"^abilities_([0-9.]+)_([0-9.]+)$"),
    "training": re.compile(r"^Training_(.+)$"),
    "checkpoint": re.compile(r"^checkpoint_(.+)$"),
    "simulation": re.compile(r"^simulation_(.+)$"),
}


def _subdirs(path: Path):
    """Sorted sub-folders of path (empty list if unreadable)."""
    try:
        with os.scandir(path) as it:
            return sorted((Path(e.path) for e in it if e.is_dir(follow_symlinks=True)),
                          key=lambda p: p.name)
    except OSError as exc:
        print(f"[WARN] cannot read {path}: {exc}", file=sys.stderr)
        return []


def _n_files(path: Path) -> int:
    try:
        with os.scandir(path) as it:
            return sum(1 for e in it if e.is_file())
    except OSError:
        return -1


def _num(v):
    """'1.35' -> 1.35; keep the string if it is not a number."""
    try:
        return float(v)
    except ValueError:
        return v


def _match(level: str, path: Path, skipped: list):
    m = _RE[level].match(path.name)
    if m is None:
        skipped.append((level, str(path)))
    return m


def crawl(root: Path, checkpoint: str | None, check_contents: bool):
    """Returns (simulation rows, training rows, skipped folders)."""
    sims, trainings, skipped = [], [], []

    tops = [(False, root)]
    pre = root / PRETRAINING_DIR
    if pre.is_dir():
        tops.append((True, pre))

    for with_pre, top in tops:
        for map_dir in _subdirs(top):
            if not with_pre and map_dir.name == PRETRAINING_DIR:
                continue
            m_map = _match("map", map_dir, skipped)
            if not m_map:
                continue
            for syn_dir in _subdirs(map_dir):
                m_syn = _match("synergy", syn_dir, skipped)
                if not m_syn:
                    continue
                for spec_dir in _subdirs(syn_dir):
                    m_spec = _match("specialization", spec_dir, skipped)
                    if not m_spec:
                        continue
                    for ab_dir in _subdirs(spec_dir):
                        m_ab = _match("abilities", ab_dir, skipped)
                        if not m_ab:
                            continue
                        cond = dict(with_pretraining=with_pre,
                                    map=m_map.group(1),
                                    synergy=_num(m_syn.group(1)),
                                    specialization=_num(m_spec.group(1)),
                                    walk=float(m_ab.group(1)),
                                    cut=float(m_ab.group(2)),
                                    abilities=ab_dir.name.replace("abilities_", ""))
                        for tr_dir in _subdirs(ab_dir):
                            if not _match("training", tr_dir, skipped):
                                continue
                            ck_dirs = [d for d in _subdirs(tr_dir) if _match("checkpoint", d, skipped)]
                            if checkpoint is not None:
                                ck_dirs = [d for d in ck_dirs if d.name == f"checkpoint_{checkpoint}"]
                            if not ck_dirs:      # training without (the requested) checkpoint
                                trainings.append(dict(cond, training_id=tr_dir.name,
                                                      checkpoint=None, n_simulations=0,
                                                      simulation_ids="", path=str(tr_dir)))
                                continue
                            for ck_dir in ck_dirs:
                                ck = ck_dir.name.replace("checkpoint_", "")
                                sim_ids = []
                                for sim_dir in _subdirs(ck_dir):
                                    if not _match("simulation", sim_dir, skipped):
                                        continue
                                    row = dict(cond, training_id=tr_dir.name, checkpoint=ck,
                                               simulation_id=sim_dir.name, path=str(sim_dir))
                                    if check_contents:
                                        row["n_files"] = _n_files(sim_dir)
                                    sims.append(row)
                                    sim_ids.append(sim_dir.name)
                                trainings.append(dict(cond, training_id=tr_dir.name, checkpoint=ck,
                                                      n_simulations=len(sim_ids),
                                                      simulation_ids=";".join(sim_ids),
                                                      path=str(tr_dir)))
    return sims, trainings, skipped


COND_KEYS = ["with_pretraining", "map", "synergy", "specialization", "walk", "cut", "abilities"]


def condition_table(trainings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = trainings.copy()
    t["checkpoint"] = t["checkpoint"].fillna("none")
    for key, g in t.groupby(COND_KEYS + ["checkpoint"], sort=False, dropna=False):
        with_sims = g[g["n_simulations"] > 0]
        rows.append(dict(zip(COND_KEYS + ["checkpoint"], key),
                         n_trainings=g["training_id"].nunique(),
                         n_trainings_with_simulations=with_sims["training_id"].nunique(),
                         n_simulations=int(g["n_simulations"].sum()),
                         sims_per_training_min=int(g["n_simulations"].min()),
                         sims_per_training_max=int(g["n_simulations"].max()),
                         training_ids=";".join(g["training_id"])))
    return pd.DataFrame(rows)


def _sort(df: pd.DataFrame, extra=()) -> pd.DataFrame:
    keys = [k for k in COND_KEYS + list(extra) if k in df.columns]
    # HA-like (1.0, 1.0) first: abilities sorted by walk, cut descending
    asc = [False if k in ("walk", "cut") else True for k in keys]
    return df.sort_values(keys, ascending=asc, kind="stable", na_position="last").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Count trainings and simulations per condition of the simulation tree.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"Simulation root (default {DEFAULT_ROOT})")
    ap.add_argument("--checkpoint", default=None,
                    help="Only this checkpoint_<n> folder (e.g. 750); default: every checkpoint")
    ap.add_argument("--check_contents", action="store_true",
                    help="Also count the files inside every simulation folder (n_files; 0 = empty folder)")
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--output_name", default="simulations_per_condition")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[ERROR] root not found: {root}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.output_name

    print(f"Crawling {root} (checkpoint: {args.checkpoint or 'all'}) ...")
    sims, trainings, skipped = crawl(root, args.checkpoint, args.check_contents)

    sim_cols = COND_KEYS + ["training_id", "checkpoint", "simulation_id"] + \
        (["n_files"] if args.check_contents else []) + ["path"]
    sims_df = _sort(pd.DataFrame(sims, columns=sim_cols), ["training_id", "checkpoint", "simulation_id"])
    tr_cols = COND_KEYS + ["training_id", "checkpoint", "n_simulations", "simulation_ids", "path"]
    tr_df = _sort(pd.DataFrame(trainings, columns=tr_cols), ["training_id", "checkpoint"])
    cond_df = _sort(condition_table(tr_df), ["checkpoint"]) if len(tr_df) else pd.DataFrame()

    sims_df.to_csv(f"{prefix}_simulations.csv", index=False)
    tr_df.to_csv(f"{prefix}_trainings.csv", index=False)
    cond_df.to_csv(f"{prefix}_conditions.csv", index=False)

    # ---------------- readable summary ----------------
    lines = ["=" * 100,
             f"SIMULATIONS PER CONDITION  --  {root}",
             f"checkpoint: {args.checkpoint or 'all'}",
             "=" * 100,
             f"simulation folders: {len(sims_df)}   trainings (x checkpoint): {len(tr_df)}   "
             f"conditions: {len(cond_df)}", ""]
    if len(cond_df):
        for pre, g in cond_df.groupby("with_pretraining", sort=True):
            lines.append(f"--- {'WITH PRETRAINING' if pre else 'without pretraining'} "
                         f"({int(g.n_simulations.sum())} simulations, {int(g.n_trainings.sum())} trainings) ---")
            lines.append(f"  {'map':<42} {'syn':>5} {'spec':>5} {'abilities':>10} {'ckpt':>5} "
                         f"{'train':>6} {'sims':>6} {'sims/train':>11}")
            for _, r in g.iterrows():
                lines.append(f"  {r['map']:<42} {str(r.synergy):>5} {str(r.specialization):>5} "
                             f"{r.abilities:>10} {str(r.checkpoint):>5} {r.n_trainings:>6} "
                             f"{r.n_simulations:>6} {r.sims_per_training_min:>5}..{r.sims_per_training_max:<5}")
            lines.append("")
        empty = tr_df[tr_df["n_simulations"] == 0]
        lines.append(f"Trainings without simulations (or without the requested checkpoint): {len(empty)}")
        lines += [f"  {p}" for p in empty["path"]]
        if args.check_contents and len(sims_df):
            e = sims_df[sims_df["n_files"] == 0]
            lines.append(f"\nEmpty simulation folders: {len(e)}")
            lines += [f"  {p}" for p in e["path"]]
    lines.append(f"\nFolders that do not match the expected structure: {len(skipped)}")
    lines += [f"  [expected {lvl}] {p}" for lvl, p in skipped]
    lines += ["", f"-> {prefix}_simulations.csv", f"-> {prefix}_trainings.csv",
              f"-> {prefix}_conditions.csv", f"-> {prefix}_summary.txt"]

    text = "\n".join(lines) + "\n"
    Path(f"{prefix}_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
